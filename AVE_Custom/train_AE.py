import numpy as np
import os
from tqdm import trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from cacher import cache_data
from models.layer_controller import Conv_Controller_AE
from sklearn.metrics import accuracy_score
from torchvision.transforms import Resize
from einops import rearrange

import random
import argparse
import time
from torchvision.utils import make_grid, save_image
import os



def get_args_parser():
    parser = argparse.ArgumentParser(description='AVE Controller Training, load config file and override params')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default = '/mnt/ssd_8t/redacted/AVE_Dataset/', help="Base dataset root")
    parser.add_argument("--cached_root", type=str, default = '/mnt/ssd_8t/redacted/AVE_Dataset_Cached/', help="Base dataset root")
    parser.add_argument("--valid_mods", type=str, nargs="+", default=['image', 'audio'], help="List of valid modalities")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--discretization_method', type=str, default='admn', choices=['admn', 'straight_through', 'progressive'])
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
    
    dt_string = 'AE_Model'
    os.mkdir('./logs/' + dt_string)
    
    cache_data(args.base_root, args.cached_root)
    #PickleDataset inherits from a Pytorch Dataset, creates train and val datasets
    trainset = PickleDataset(data_root = args.cached_root, type='train')
    valset = PickleDataset(data_root = args.cached_root, type='val')
    batch_size = args.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")


    # Create the overall model and load on appropriate device
    model = Conv_Controller_AE()
    
    model.to(device)
    

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
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
        resizer_img = Resize((100, 100))
        resizer_aud = Resize((128, 512))
        for batch in train_dataloader:
            batch_num += 1
            batch, noise_label = batch # Format is data, label for noise (0-3)
            for i in range(len(batch)):
                batch[i] = batch[i].cuda()
            
            audio_data, img_data, label = batch
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w') # Compress stack dimension into batch
            img_data = resizer_img(img_data)
            img_data = rearrange(img_data, '(b s) c h w -> b s c h w', b = audio_data.shape[0])
            audio_data = resizer_aud(audio_data)
            batch[0] = audio_data
            batch[1] = img_data
            # Forward pass gives us the loc results and the predicted noise of each modality
            img_recon, audio_recon = model(batch, valid_mods=args.valid_mods) #Dictionary
            train_loss = loss_fn(img_recon, img_data[:, 0]) + loss_fn(audio_recon, audio_data) # Compare to the first frame of the image data
           
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
            gt_labels = []
            pred_labels = []
          
            model.eval()
            for batch in val_dataloader:
                batch_num += 1
                # Each batch is a dictionary containing all the sensor data, and the ground truth positions
                batch, noise_label = batch
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda()
                
                audio_data, img_data, label = batch
                img_data = rearrange(img_data, 'b s c h w -> (b s) c h w') # Compress stack dimension into batch
                img_data = resizer_img(img_data)
                img_data = rearrange(img_data, '(b s) c h w -> b s c h w', b = audio_data.shape[0])
                audio_data = resizer_aud(audio_data)
                batch[0] = audio_data
                batch[1] = img_data
                gt_labels.append(label.cpu())
                img_recon, aud_recon = model(batch, valid_mods=args.valid_mods) #Dictionary
                val_loss = loss_fn(img_recon, img_data[:, 0]) + loss_fn(audio_data, aud_recon)
                if batch_num == 2:
                    save_reconstructions(img_recon, img_data[:, 0], './output_images/' + str(epoch) + 'img.png')
                    save_reconstructions(aud_recon, audio_data, './output_images/' + str(epoch) + 'aud.png')
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

