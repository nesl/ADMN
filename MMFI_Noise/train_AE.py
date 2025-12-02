import numpy as np
import os
from tqdm import trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset, transform_noise, transform_discrete_noise, transform_finite_noise
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from models.layer_controller import Conv_Controller_AE
from torchvision.transforms import Resize
from einops import rearrange
import config
import random
import argparse
import sys
import time
from torchvision.utils import make_grid, save_image
import os

def get_args_parser():
    parser = argparse.ArgumentParser(description='GTDM Controller Training, load config file and override params')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default=config.base_root, help="Base directory for datasets")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--train_type', type=str, default='discrete', choices=['continuous', 'discrete', 'finite'])

    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
   
    return args

def save_reconstructions(recon_batch, gt_batch, save_path, nrow=8, value_range=(0, 1)):
    """
    Save reconstructed and ground truth images side by side.

    Args:
        recon_batch (torch.Tensor): Reconstructed images of shape (B, C, H, W).
        gt_batch (torch.Tensor): Ground truth images of shape (B, C, H, W).
        save_path (str): Path to save the image.
        nrow (int): Number of images in a row in the grid.
        value_range (tuple): Min and max value to clip images.
    """
    assert recon_batch.shape == gt_batch.shape, "Reconstruction and ground truth batches must have the same shape"

    # Clip reconstructed images
    recon_batch = torch.clamp(recon_batch, min=value_range[0], max=value_range[1])

    # Combine reconstructed and ground truth images
    # For each image, stack GT and Recon vertically (along height)
    combined = []
    for recon, gt in zip(recon_batch, gt_batch):
        combined_img = torch.cat([gt, recon], dim=-2)  # concatenate along height
        combined.append(combined_img)

    combined_batch = torch.stack(combined)

    # Make a grid of combined images
    grid = make_grid(combined_batch, nrow=nrow, value_range=value_range, normalize=True)

    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image
    save_image(grid, save_path)


def main(args):
    # Set seed
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seedVal)
    torch.manual_seed(args.seedVal)
    torch.cuda.manual_seed(args.seedVal)
    np.random.seed(args.seedVal)
    # Create based on noise type and number of layers
    # dt_string = "Controller_" + str(args.train_type) + '_Layer_' + str(args.total_layers) + '_Seed_' + str(args.seedVal)
    # os.mkdir('./logs/' + dt_string)
    dt_string = 'AE_Model'
    os.mkdir('./logs/' + dt_string)
    
    
    #PickleDataset inherits from a Pytorch Dataset, creates train and val datasets
    trainset = PickleDataset(args.base_root, dataset_type='train')
    valset = PickleDataset(args.base_root, dataset_type='val')
    batch_size = args.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")




    # Create the overall model and load on appropriate device
    model = Conv_Controller_AE(embed_dim=256)
    
    model.to(device)
    

    optimizer = Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=args.num_epochs)
    writer = SummaryWriter(log_dir='./logs/' + dt_string) # Implement tensorboard
    loss_fn = torch.nn.MSELoss()
    # Training loop
    train_start = time.time()
    for epoch in trange(args.num_epochs, desc="Training"):
        
        batch_num = 0
        epoch_train_loss = 0
        model.train()
        
        # Change the temperature with the CosineAnnealer if we are doing progressive gumbel softmax with decreasing temperature
        #print("Changed temperature to ", annealer.forward(epoch))
        resizer = Resize((100, 100))
        
        for batch in train_dataloader:
            batch_num += 1 

            if args.train_type == 'continuous':
                data, gt_noise = transform_noise(batch, args.batch_size, img_std_max=2, depth_std_max=4)
            elif args.train_type == 'finite':
                data, gt_noise = transform_finite_noise(batch, args.batch_size, img_std_max=2, depth_std_max=3)
            elif args.train_type == 'discrete':
                data, gt_noise = transform_discrete_noise(batch, args.batch_size, img_std_candidates=[0, 0.75, 1.5, 2], depth_std_candidates=[0, 2, 3, 4])
            else:
                raise Exception('Invalid test type specified')

            rgb_data = data['rgb'].cuda()[:, 0:3] # Get only the first frame
            depth_data = data['depth'].cuda()[:, 0:3]
            b_size = rgb_data.shape[0]
            rgb_data = rearrange(rgb_data, 'b n c h w -> (b n) c h w')
            depth_data = rearrange(depth_data, 'b n c h w -> (b n) c h w')
            rgb_data = resizer(rgb_data)
            depth_data = resizer(depth_data)
            rgb_data = rearrange(rgb_data, '(b n) c h w -> b n c h w', b=b_size)
            depth_data = rearrange(depth_data, '(b n) c h w -> b n c h w', b=b_size)

          
            # Forward pass gives us the loc results and the predicted noise of each modality
            img_recon, depth_recon = model(rgb_data, depth_data) #Dictionary
            
            train_loss = loss_fn(img_recon, rgb_data[:, 0]) + loss_fn(depth_recon, depth_data[:, 0]) # Compare to the first frame of the data
           
            with torch.no_grad():
                epoch_train_loss += train_loss
                print("Batch Num: ", batch_num, 'Train Loss', train_loss.detach().cpu().item())
                print('\n')

            train_loss.backward()
            optimizer.step() 
            optimizer.zero_grad()           
            
        
        print('TRAIN LOSS', epoch_train_loss / batch_num)
        writer.add_scalar("Training loss", epoch_train_loss / batch_num, epoch)
        scheduler.step()
        print(scheduler.get_last_lr()[0])
        
        batch_num = 0
        epoch_val_loss = 0
        with torch.no_grad():
            model.eval()

            for batch in val_dataloader:
                batch_num += 1
                if args.train_type == 'continuous':
                    data, gt_noise = transform_noise(batch, args.batch_size, img_std_max=2, depth_std_max=4)
                elif args.train_type == 'finite':
                    data, gt_noise = transform_finite_noise(batch, args.batch_size, img_std_max=2, depth_std_max=3)
                elif args.train_type == 'discrete':
                    data, gt_noise = transform_discrete_noise(batch, args.batch_size, img_std_candidates=[0, 0.75, 1.5, 2], depth_std_candidates=[0, 2, 3, 4])
                else:
                    raise Exception('Invalid test type specified')
                # Each batch is a dictionary containing all the sensor data, and the ground truth positions
                rgb_data = data['rgb'].cuda()[:, 0:3] # Get only the first frame
                depth_data = data['depth'].cuda()[:, 0:3]
                b_size = rgb_data.shape[0]
                rgb_data = rearrange(rgb_data, 'b n c h w -> (b n) c h w')
                depth_data = rearrange(depth_data, 'b n c h w -> (b n) c h w')
                rgb_data = resizer(rgb_data)
                depth_data = resizer(depth_data)
                rgb_data = rearrange(rgb_data, '(b n) c h w -> b n c h w', b=b_size)
                depth_data = rearrange(depth_data, '(b n) c h w -> b n c h w', b=b_size)
            
                # Forward pass gives us the loc results and the predicted noise of each modality
                img_recon, depth_recon = model(rgb_data, depth_data) #Dictionary

                val_loss = loss_fn(img_recon, rgb_data[:, 0]) + loss_fn(depth_recon, depth_data[:, 0]) # Compare to the first frame of the data
                if batch_num == 2:
                    save_reconstructions(img_recon, rgb_data[:, 0], './output_images/' + str(epoch) + 'img.png')
                    save_reconstructions(depth_recon, depth_data[:, 0], './output_images/' + str(epoch) + 'depth.png')
                print('Batch Number', batch_num, "Val Loss", val_loss)
                epoch_val_loss += val_loss
 
            
        with open( './logs/' + dt_string + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(epoch_train_loss) + ' | Val Accuracy ' + str(epoch_val_loss)
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + dt_string + '/last.pt')
                
    print(time.time() - train_start)
    
if __name__ == '__main__':
    args = get_args_parser()
    main(args)

