import numpy as np
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset, transform_noise, transform_set_noise, transform_finite_noise, transform_mask, transform_discrete_noise
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from cache_datasets import cache_data
from models.MMFI_Model import Conv_MMFI_Controller
from sklearn.metrics import accuracy_score, f1_score
import argparse
import random
import argparse
import sys

class Tee:
    def __init__(self, *file_objects):
        self.file_objects = file_objects

    def write(self, message):
        for file in self.file_objects:
            file.write(message)
            file.flush()  # Ensure immediate write

    def flush(self):
        for file in self.file_objects:
            file.flush()


class CosineAnnealer:
    def __init__(self, num_epochs=100, max=5, min=1):
        self.num_epochs = num_epochs
        self.max = max
        self.min = min
    def forward(self, step):
        if step > self.num_epochs:
            step = self.num_epochs
        return (self.max - self.min) * ( np.cos(np.pi/(2 * self.num_epochs) * step) + self.min/(self.max - self.min))

def mseloss(t1, t2):
    sum = 0
    for i in range(len(t1)):
        sum += (t1[i].item() - t2[i].item()) ** 2
    return sum ** 0.5 

def get_args_parser():
    parser = argparse.ArgumentParser(description='GTDM Controller Training, load config file and override params')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default='/mnt/ssd_8t/jason/MMFI_Pickles_Img_DepthColorized', help="Base directory for datasets")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--train_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])

    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
   
    return args



def main(args):
    
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seedVal)
    torch.manual_seed(args.seedVal)
    torch.cuda.manual_seed(args.seedVal)
    np.random.seed(args.seedVal)
    # Get current date and time to create new training directory within ./logs/ to store model weights
    now = datetime.now()
    dt_string = "Controller_" + str(args.train_type) + '_Layer_' + str(args.total_layers) + '_Seed_' + str(args.seedVal)
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
    model = Conv_MMFI_Controller(total_layers=args.total_layers)

    print(model.load_state_dict(torch.load('./logs/MMFI_Base_Model_Img2_Depth3_LD02/last.pt'), strict=False))
    # model_template = Conv_GTDM_Controller(args.adapter_hidden_dim, valid_mods=args.valid_mods, valid_nodes = args.valid_nodes, total_layers=args.total_layers)

    # model_template.load_state_dict(torch.load('./logs/Conv_Controller_Reference/last.pt'))
    # model.controller = model_template.controller
    ae_weights = torch.load('./logs/AE_MMFI/last.pt')
    new_ae_weights = {}
    for key in ae_weights.keys():
        if 'encoder_dict' in key:
            number = int(key.split('.')[2])
            split_key = key.split('.')
            split_key[2] = str(number + 1)
            new_key = ".".join(split_key)
            new_ae_weights[new_key] = ae_weights[key]
        else:
            new_ae_weights[key] = ae_weights[key]
    print(model.controller.load_state_dict(new_ae_weights, strict=False))

    model.to(device)
    for param in model.parameters():
        param.requires_grad=False

    for param in model.controller.output_head.parameters():
        param.requires_grad = True
   

    # for param in model.vision.parameters():
    #     param.requires_grad = False

    # for param in model.depth.parameters():
    #     param.requires_grad = False

    # for param in model.vision.blocks[0].parameters():
    #     param.requires_grad = True
    # for param in model.depth.blocks[0].parameters():
    #     param.requires_grad = True

    params = [
        {"params": [p for name, p in model.controller.named_parameters() if "output_head" not in name], "lr": args.learning_rate},
        {"params": model.controller.output_head.parameters(), "lr": args.learning_rate},
    ]
    optimizer = Adam(params)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=args.num_epochs - 5)
    annealer = CosineAnnealer(25, 2, 1)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir='./logs/' + dt_string) # Implement tensorboard
    
    # Training loop
    for epoch in trange(args.num_epochs, desc="Training"):
        
        batch_num = 0
        epoch_train_loss = 0
        model.train()
        # if epoch % 3 == 2:
        #     model.controller.decrease_model_layers(min_layers=8)
        #print("Changed temperature to ", annealer.forward(epoch))
        #import pdb; pdb.set_trace()
        train_pred_labels = []
        train_gt_labels = []
        for batch in train_dataloader:
            batch_num += 1

            train_loss = 0.0
            # Each batch is a dictionary containing all the sensor data, and the ground truth positions
            # Data itself is a dictionary with keys ('modality', 'node') that points to data of dimension batch_size
            # print('Img 0', data[('img_std', 'img_std')][0])
            # print("Depth 0", data[('depth_std', 'depth_std')][0])
            if args.train_type == 'continuous':
                data, gt_noise = transform_noise(batch, args.batch_size, img_std_max=2, depth_std_max=4)
            elif args.train_type == 'finite':
                data, gt_noise = transform_finite_noise(batch, args.batch_size, img_std_max=2, depth_std_max=3)
            elif args.train_type == 'discrete':
                data, gt_noise = transform_discrete_noise(batch, args.batch_size, img_std_candidates=[0, 0.75, 1.5, 2], depth_std_candidates=[0, 2, 3, 4])
            else:
                raise Exception('Invalid test type specified')
            # Perform forward pass
            batch_results, pred_noise = model(data, controller_temperature = 1) #Dictionary
            print(pred_noise[0], gt_noise[0])
            # key is still ('modality', 'node') with a distribution estimated by the model
            train_loss = loss_fn(batch_results, data['labels'])
            train_pred_labels.extend(torch.argmax(batch_results, dim=-1).cpu().tolist())
            train_gt_labels.extend(data['labels'].cpu().tolist())
            

            with torch.no_grad():
                # Print one sample from the batch to see prediction result and loss
                print('Batch Number', batch_num)
                print('\tGT', data['labels'][0], 'predicted', torch.argmax(batch_results[0]), 'with loss', train_loss)
                print('-------------------------------------------------------------------------------------------------------------------------------')
                epoch_train_loss += train_loss # Accumulate batch loss into overall epoch loss

            noise_loss = torch.mean(torch.abs(gt_noise[:, 0] - pred_noise[:, 0])) + torch.mean(torch.abs(gt_noise[:, 1] - pred_noise[:, 1])) * 3


            print("Noise loss", noise_loss)
            
            
            train_loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

            
               
        #     # Backprop and update
           
            
        epoch_train_loss /= batch_num
        print('TRAIN LOSS', epoch_train_loss)
        if epoch >= 5: 
            scheduler.step()
            print(scheduler.get_last_lr()[0])

        writer.add_scalar("Training loss", epoch_train_loss, epoch)

        batch_num = 0
        epoch_val_loss = 0
        with torch.no_grad():
            log_file = open('./logs/' + dt_string + '/validation.txt', "w")
            temp_std_out = sys.stdout
            sys.stdout = Tee(sys.stdout, log_file)
            model.eval()
            pred_labels = []
            gt_labels = []
            for batch in val_dataloader:

                batch_num += 1
                val_loss = 0.0
                
                if args.train_type == 'continuous':
                    data, _ = transform_noise(batch, args.batch_size, img_std_max=2, depth_std_max=4)
                elif args.train_type == 'finite':
                    data, _ = transform_finite_noise(batch, args.batch_size, img_std_max=2, depth_std_max=3)
                elif args.train_type == 'discrete':
                    data, _ = transform_discrete_noise(batch, args.batch_size, img_std_candidates=[0, 0.75, 1.5, 2], depth_std_candidates=[0, 2, 3, 4])
                else:
                    raise Exception('Invalid test type specified')
                # Perform forward pass
                batch_results, pred_noise = model(data, controller_temperature=1) #Dictionary
                print(pred_noise[0])
                val_loss = loss_fn(batch_results, data['labels'])
                pred_labels.extend(torch.argmax(batch_results, dim=-1).cpu().tolist())
                gt_labels.extend(data['labels'].cpu().tolist())
                epoch_val_loss += val_loss
            epoch_val_loss /= batch_num
            print("Validation loss", epoch_val_loss)
            print("Accuracy: ", accuracy_score(gt_labels, pred_labels))
            log_file.close()
            sys.stdout = temp_std_out
            
        with open( './logs/' + dt_string + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(epoch_train_loss) + 
                   ' | Train Accuracy ' + str(accuracy_score(train_gt_labels, train_pred_labels)) + 
                  ' | Val Loss ' + str(epoch_val_loss) + 
                  ' | Val Accuracy ' + str(accuracy_score(gt_labels, pred_labels)) + 
                  ' | Dropout ' + str(model.vision.layerdrop_rate) + 
                  ' ' + str(model.depth.layerdrop_rate) +
                  ' | LR ' + str(scheduler.get_lr())
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + dt_string + '/last.pt')
                

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

