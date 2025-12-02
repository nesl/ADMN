import numpy as np
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from cache_datasets import cache_data
from models.MMFI_Model import MMFI_Early
import argparse
import random
import argparse
from PickleDataset import transform_noise, transform_mask
from sklearn.metrics import accuracy_score
import config

def mseloss(t1, t2):
    sum = 0
    for i in range(len(t1)):
        sum += (t1[i].item() - t2[i].item()) ** 2
    return sum ** 0.5

def get_args_parser():
    parser = argparse.ArgumentParser(description='Trains the Stage 1 Model')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default=config.base_root, help="Base directory for datasets")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache datasets")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--max_layerdrop', type=float, default=0.2, help="LayerDrop Rate for training")
    parser.add_argument('--vision_vit_layers', type=int, default=12)
    parser.add_argument('--depth_vit_layers', type=int, default=12)
    parser.add_argument("--valid_mods", type=str, nargs="+", default=['image', 'depth'], help="List of valid modalities")
    parser.add_argument('--dir_name', type=str, default='Stage_1_Model')
    parser.add_argument('--from_scratch', action='store_true', default=False)
    args = parser.parse_args()
    return args


    



def main(args):
    
    seedVal = int(args.seedVal)
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(seedVal)
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    np.random.seed(seedVal)
    # Get current date and time to create new training directory within ./logs/ to store model weights
    from pathlib import Path
    Path('./logs/' + args.dir_name).mkdir(parents=True, exist_ok=True)

    
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
    model = MMFI_Early(layerdrop=0.0, vision_vit_layers=args.vision_vit_layers, depth_vit_layers=args.depth_vit_layers, valid_mods=args.valid_mods, from_scratch=args.from_scratch)
    model.to(device)


    #Establish from training parameters

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-3, total_iters=200)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir='./logs/' + args.dir_name) # Implement tensorboard
   
   
    # Training loop

    for epoch in trange(args.num_epochs, desc="Training"):
        batch_num = 0
        epoch_train_loss = 0
        model.train()
        scheduler.step()
        # Gradually increase layerdrop rate to ensure good learning
        if epoch % 10 == 9:
            if 'image' in args.valid_mods:
                model.vision.layerdrop_rate = min(args.max_layerdrop, model.vision.layerdrop_rate + 0.1)
            if 'depth' in args.valid_mods:
                model.depth.layerdrop_rate = min(args.max_layerdrop, model.depth.layerdrop_rate + 0.1)
        train_gt_labels = []
        train_pred_labels = []
        for batch in train_dataloader:
            batch_num += 1
            if epoch < 5:
                data, _ = transform_noise(batch, args.batch_size, img_std_max = 0, depth_std_max = 0)
            else:
                data, _ = transform_noise(batch, args.batch_size, img_std_max=2, depth_std_max=3)
            # Perform forward pass
            batch_results = model(data) #Dictionary
            # key is still ('modality', 'node') with a distribution estimated by the model
            train_loss = loss_fn(batch_results, data['labels'])
            train_pred_labels.extend(torch.argmax(batch_results, dim=-1).cpu().tolist())
            train_gt_labels.extend(data['labels'].cpu().tolist())
            train_loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
           

            with torch.no_grad():
                # Print one sample from the batch to see prediction result and loss
                print('Batch Number', batch_num)
                print('\tGT', data['labels'][0], 'predicted', torch.argmax(batch_results[0]), 'with loss', train_loss)
                print('-------------------------------------------------------------------------------------------------------------------------------')
                epoch_train_loss += train_loss # Accumulate batch loss into overall epoch loss
            
        epoch_train_loss /= batch_num
        print('TRAIN LOSS', epoch_train_loss)
        writer.add_scalar("Training loss", epoch_train_loss, epoch)

        batch_num = 0
        epoch_val_loss = 0
        pred_labels = []
        gt_labels = []
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                batch_num += 1
                if epoch < 5:
                    data, _ = transform_noise(batch, args.batch_size, img_std_max = 0, depth_std_max = 0)
                else:
                    data, _ = transform_noise(batch, args.batch_size, img_std_max=2, depth_std_max=3)
                # Perform forward pass
                batch_results = model(data) #Dictionary
                # key is still ('modality', 'node') with a distribution estimated by the model
                val_loss = loss_fn(batch_results, data['labels'])
                pred_labels.extend(torch.argmax(batch_results, dim=-1).cpu().tolist())
                gt_labels.extend(data['labels'].cpu().tolist())
                epoch_val_loss += val_loss
            epoch_val_loss /= batch_num
            print("Validation loss", epoch_val_loss)
            print("Accuracy: ", accuracy_score(gt_labels, pred_labels))
        vision_layerdrop = 0.0 if 'image' not in args.valid_mods else model.vision.layerdrop_rate
        depth_layerdrop = 0.0 if 'depth' not in args.valid_mods else model.depth.layerdrop_rate
        with open( './logs/' + args.dir_name + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(epoch_train_loss) + 
                   ' | Train Accuracy ' + str(accuracy_score(train_gt_labels, train_pred_labels)) + 
                  ' | Val Loss ' + str(epoch_val_loss) + 
                  ' | Val Accuracy ' + str(accuracy_score(gt_labels, pred_labels)) + 
                  ' | Dropout ' + str(vision_layerdrop) + 
                  ' ' + str(depth_layerdrop) +
                  ' | LR ' + str(scheduler.get_lr())
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + args.dir_name + '/last.pt')
                


if __name__ == '__main__':
    args = get_args_parser()
    main(args)


