import numpy as np
import os
from tqdm import trange

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from models.GTDM_Model import GTDM_Early
from sklearn.metrics import accuracy_score
import random
import argparse
from cacher import cache_data



def get_args_parser():
    parser = argparse.ArgumentParser(description='GTDM Controller Training, load config file and override params')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default = '/mnt/ssd_8t/jason/AVE_Dataset/', help="Base dataset root")
    parser.add_argument("--valid_mods", type=str, nargs="+", default=['image', 'audio'], help="List of valid modalities")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=10, help="Save model every X epochs")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--train_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])
    parser.add_argument('--max_layerdrop', type=float, default=0.2, help="LayerDrop Rate for training")
    parser.add_argument('--vision_vit_layers', type=int, default=12)
    parser.add_argument('--audio_vit_layers', type=int, default=12)

    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
    
    return args



def main(args):
    # Enable reproducability with seed
    seedVal = int(args.seedVal)
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(seedVal)
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    np.random.seed(seedVal)

    # Get current date and time to create new training directory within ./logs/ to store model weights
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y %H_%M_%S")
    os.mkdir('./logs/' + dt_string)
    
    cache_data(cached_root='/mnt/ssd_8t/jason/AVE_Dataset_Cached')
    trainset = PickleDataset(data_root = '/mnt/ssd_8t/jason/AVE_Dataset_Cached/', type='train')
    valset = PickleDataset(data_root = '/mnt/ssd_8t/jason/AVE_Dataset_Cached/', type='val')
    batch_size = args.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    

    # Create the overall model and load on appropriate device
    # IN GTDM Early, we initialize with the MAE pretrained weights if we pass in 12 layers for image and depth and then freeze the majority of the layers
    model = GTDM_Early(args.adapter_hidden_dim, valid_mods=args.valid_mods, layerdrop=0.0, vision_vit_layers=args.vision_vit_layers, audio_vit_layers=args.audio_vit_layers)
    model.to(device)
    

    #Establish from training parameters

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir='./logs/' + dt_string) # Implement tensorboard
   
    loss_fn = torch.nn.CrossEntropyLoss()
    # Training loop
   
    for epoch in trange(args.num_epochs, desc="Training"):
        batch_num = 0
        epoch_train_loss = 0
        model.train()
        # Gradually increase layerdrop rate to ensure good learning
        if epoch % 2 == 1:
            if 'image' in args.valid_mods:
                model.vision.layerdrop_rate = min(args.max_layerdrop, model.vision.layerdrop_rate + 0.1)
            if 'audio' in args.valid_mods:
                model.audio.layerdrop_rate = min(args.max_layerdrop, model.audio.layerdrop_rate + 0.1)
         
        pred_labels = []
        gt_labels = []
        for batch in train_dataloader:
            batch, noise_label = batch
            batch_num += 1
            optimizer.zero_grad()
            for i in range(len(batch)):
                batch[i] = batch[i].cuda()

            _, _, label = batch
            gt_labels.append(label.cpu())
            # Forward pass gives us the loc results and the predicted noise of each modality
            batch_results = model(batch) #Dictionary
            train_loss = loss_fn(batch_results, label)
            
            with torch.no_grad():
                pred_labels.append(torch.argmax(batch_results, dim=-1).detach().cpu())
                epoch_train_loss += train_loss
                print("Batch Num: ", batch_num, 'Train Loss', train_loss.detach().cpu().item())

            # Backprop and update
            train_loss.backward()
            optimizer.step() 
        
        print('TRAIN LOSS', epoch_train_loss / batch_num)
        pred_labels = torch.cat(pred_labels).numpy()
        gt_labels = torch.cat(gt_labels).numpy()
        train_acc = accuracy_score(gt_labels, pred_labels)
        writer.add_scalar("Training loss", epoch_train_loss / batch_num, epoch)

        # Peform validation
        batch_num = 0
        pred_labels = []
        gt_labels = []
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                batch_num += 1
                batch, noise_label = batch
                # Each batch is a dictionary containing all the sensor data, and the ground truth positions
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda()
                _, _, label = batch
                gt_labels.append(label.cpu())
                batch_results = model(batch) #Dictionary
                pred_labels.append(torch.argmax(batch_results, dim=-1).cpu())
            pred_labels = torch.cat(pred_labels).numpy()
            gt_labels = torch.cat(gt_labels).numpy()
            val_acc = accuracy_score(gt_labels, pred_labels)
            print("Validation Accuracy", val_acc)
        audio_layerdrop = model.audio.layerdrop_rate if 'audio' in args.valid_mods else 0
        vision_layerdrop = model.vision.layerdrop_rate if 'image' in args.valid_mods else 0
        with open( './logs/' + dt_string + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(epoch_train_loss) + ' | Train Acc ' + str(train_acc)  + ' | Val Loss ' + str(val_acc) + ' | Dropout ' + str(vision_layerdrop) + ' ' + str(audio_layerdrop)
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + dt_string + '/last.pt')
        # if epoch % args.save_every_X_model == 0:
        #      torch.save(model.state_dict(), './logs/' + dt_string + '/epoch' + str(epoch) + '.pt')
                


if __name__ == '__main__':
    args = get_args_parser()
    main(args)


