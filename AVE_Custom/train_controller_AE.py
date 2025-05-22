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
from models.AVE_Model import Conv_AVE_Controller
from sklearn.metrics import accuracy_score

import random
import argparse
import sys
import time

# Helps us write data to a file
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

# Anneals loss in a cosine manner
class CosineAnnealer:
    def __init__(self, num_epochs=100, max=5, min=1):
        self.num_epochs = num_epochs
        self.max = max
        self.min = min
    def forward(self, step):
        if step > self.num_epochs:
            step = self.num_epochs
        return (self.max - self.min) * ( np.cos(np.pi/(2 * self.num_epochs) * step) + self.min/(self.max - self.min))



def get_args_parser():
    parser = argparse.ArgumentParser(description='AVE Controller Training, load config file and override params')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default = '/mnt/ssd_8t/jason/AVE_Dataset/', help="Base dataset root")
    parser.add_argument("--cached_root", type=str, default = '/mnt/ssd_8t/jason/AVE_Dataset_Cached/', help="Base dataset root")
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
    parser.add_argument("--temp", type=float, default=1, help="Learning rate for training")
    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
    
    return args



def main(args):
    # Set seed
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seedVal)
    torch.manual_seed(args.seedVal)
    torch.cuda.manual_seed(args.seedVal)
    np.random.seed(args.seedVal)
    # Create based on noise type and number of layers
    dt_string = 'AE_Controller_Layer_' + str(args.total_layers) + '_Seed_' + str(args.seedVal)
    os.mkdir('./logs/' + dt_string)
    
    cache_data(base_root=args.base_root, cached_root=args.cached_root)
    #PickleDataset inherits from a Pytorch Dataset, creates train and val datasets
    trainset = PickleDataset(data_root = args.cached_root, type='val', valid_noise_types=[1, 2])
    valset = PickleDataset(data_root = args.cached_root, type='val', valid_noise_types=[1, 2])
    batch_size = args.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")


    # Create the overall model and load on appropriate device
    model = Conv_AVE_Controller(args.adapter_hidden_dim, valid_mods=args.valid_mods, total_layers=args.total_layers)

    # We have similar variables between AVE_Early Model and Conv_AVE_Controller, this will help us initialize the backbones and the fusion layers
    print(model.load_state_dict(torch.load('./logs/Stage_1_Model/last.pt'), strict=False))
    ae_weights = torch.load('./logs/AE_Model/last.pt')
    #import pdb; pdb.set_trace()
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
    
    # Freeze all the parameters except for the controller
    for param in model.parameters():
        param.requires_grad=False

    for param in model.controller.output_head.parameters():
        param.requires_grad = True

    # for param in model.controller.encoder_dict.parameters():
    #     param.requires_grad = False
   

    # This was an artifact of when I was using different learning rates for each model component, no longer the case
    # params = [
    #     {"params": [p for name, p in model.controller.named_parameters() if "output_head" not in name], "lr": args.learning_rate},
    #     {"params": model.controller.output_head.parameters(), "lr": args.learning_rate},
    # ]
    optimizer = Adam(model.parameters(), lr = args.learning_rate)

    # We actually use a linear scheduler instead of Cosine
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=args.num_epochs)
    writer = SummaryWriter(log_dir='./logs/' + dt_string) # Implement tensorboard
    loss_fn = torch.nn.CrossEntropyLoss()
    # Training loop
    train_start = time.time()
    for epoch in trange(args.num_epochs, desc="Training"):
        
        batch_num = 0
        epoch_train_loss = 0
        ad_train_loss = 0
        model.train()
        
        # Change the temperature with the CosineAnnealer if we are doing progressive gumbel softmax with decreasing temperature
        #print("Changed temperature to ", annealer.forward(epoch))
 
        for batch in train_dataloader:
            batch_num += 1
            batch, noise_label = batch # Format is data, label for noise (0-3)
            for i in range(len(batch)):
                batch[i] = batch[i].cuda()
            
            

            _, _, label = batch
            # Forward pass gives us the loc results and the predicted noise of each modality
            batch_results, pred_noise = model(batch, controller_temperature = args.temp, discretization_method=args.discretization_method) #Dictionary

            train_loss = loss_fn(batch_results, label)
            with torch.no_grad():
                epoch_train_loss += train_loss
                print("Batch Num: ", batch_num, 'Train Loss', train_loss.detach().cpu().item(), 'Pred Noise', pred_noise[0], 'Noise Label', noise_label[0])
                print('\n')

            train_loss.backward()
            optimizer.step() 
            optimizer.zero_grad()           
            
        
        print('TRAIN LOSS', epoch_train_loss / batch_num)
        scheduler.step()
        print(scheduler.get_last_lr()[0])
        ad_train_loss /= batch_num
        writer.add_scalar("Training loss", epoch_train_loss / batch_num, epoch)

        batch_num = 0
        epoch_val_loss = 0
        with torch.no_grad():
            gt_labels = []
            pred_labels = []
            log_file = open('./logs/' + dt_string + '/validation.txt', "w")
            # Redirect output to another file
            temp_std_out = sys.stdout
            sys.stdout = Tee(sys.stdout, log_file)
            model.eval()
            for batch in val_dataloader:
                batch_num += 1
                val_loss = 0.0
                # Each batch is a dictionary containing all the sensor data, and the ground truth positions
                batch, noise_label = batch
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda()
                
                _, _, label = batch
                gt_labels.append(label.cpu())
                batch_results, pred_noise = model(batch, controller_temperature=1,  discretization_method=args.discretization_method) #Dictionary
                pred_labels.append(torch.argmax(batch_results, dim=-1).cpu())
                print("Batch Num: ", batch_num, 'GT_Noise_Label ', noise_label[0], 'Pred Idx', torch.argmax(batch_results[0]), 'GT_Label', label[0])
            pred_labels = torch.cat(pred_labels).numpy()
            gt_labels = torch.cat(gt_labels).numpy()
            val_acc = accuracy_score(gt_labels, pred_labels)
            print("Validation Accuracy", val_acc)
            log_file.close()
            sys.stdout = temp_std_out
            
        with open( './logs/' + dt_string + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(epoch_train_loss) + ' | Val Accuracy ' + str(val_acc)
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + dt_string + '/last.pt')
                
    print(time.time() - train_start)
    
if __name__ == '__main__':
    args = get_args_parser()
    main(args)

