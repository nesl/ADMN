import numpy as np
import os
from tqdm import trange

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from PickleDataset import PickleDataset, transform_data
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from cache_datasets import cache_data
from models.GTDM_Model import GTDM_Early
import random
import configargparse
'''

We are training on noisy, lowlight data from the /mnt/.../Lowlight_Actual/ directory

'''
def mseloss(t1, t2):
    sum = 0
    for i in range(len(t1)):
        sum += (t1[i] - t2[i]) ** 2 # Previously used .item() on each tensor, which decoupled from the computational graph and wasn't accounted for in loss
    return sum ** 0.5

def get_args_parser():
    parser = configargparse.ArgumentParser(description='GTDM Controller Training, load config file and override params',
                                           default_config_files=['./configs/configs.yaml'], config_file_parser_class=configargparse.YAMLConfigFileParser)
    # Define the parameters with their default values and types
    parser.add("--base_root", type=str, help="Base directory for datasets")
    parser.add("--cache_dir", type=str, help="Directory to cache datasets")
    parser.add("--valid_mods", type=str, nargs="+", help="List of valid modalities")
    parser.add("--valid_nodes", type=int, nargs="+", help="List of valid nodes")
    parser.add("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add('--seedVal', type=int, default=100, help="Seed for training")
    parser.add('--max_layerdrop', type=float, default=0.2, help="LayerDrop Rate for training")
    parser.add('--vision_vit_layers', type=int, default=12)
    parser.add('--depth_vit_layers', type=int, default=12)
    parser.add('--from_scratch', action='store_true', default=False)
    parser.add('--dir_name', type=str, default='Stage_1_Model')

    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
    data_root = args.base_root + '/train'
    args.trainset = [
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
    ]
    data_root = args.base_root + '/val'
    args.valset = [
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
    ]
    data_root = args.base_root + '/test'
    args.testset = [
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
    ]


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
    
    os.mkdir('./logs/' + args.dir_name)

    cache_data(args) # Runs cacher from the data_configs.py file, will convert hdf5 to pickle if not already done
    
    # Create train and val datasets, these are going to be clean datasets that we then corrupt with noise through transform_noise
    trainset = PickleDataset(args.cache_dir + 'train', args.valid_mods, args.valid_nodes)
    valset = PickleDataset(args.cache_dir + 'val', args.valid_mods, args.valid_nodes)
    batch_size = args.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    

    # Create the overall model and load on appropriate device
    # IN GTDM Early, we initialize with the MAE pretrained weights if we pass in 12 layers for image and depth and then freeze the majority of the layers
    model = GTDM_Early(args.adapter_hidden_dim, valid_mods=args.valid_mods, 
                       valid_nodes = args.valid_nodes, layerdrop=0.0, vision_vit_layers=args.vision_vit_layers, depth_vit_layers=args.depth_vit_layers, from_scratch=args.from_scratch)
    model.to(device)
    

    #Establish from training parameters

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir='./logs/' + args.dir_name) # Implement tensorboard
   
   
    # Training loop
    for epoch in trange(args.num_epochs, desc="Training"):
        batch_num = 0
        epoch_train_loss = 0
        ad_train_loss = 0
        model.train()
        # Gradually increase layerdrop rate to ensure good learning
        if epoch % 10 == 9:
            model.vision.layerdrop_rate = min(args.max_layerdrop, model.vision.layerdrop_rate + 0.1)
            model.depth.layerdrop_rate = min(args.max_layerdrop, model.depth.layerdrop_rate + 0.1)
        for batch in train_dataloader:
            batch_num += 1
            optimizer.zero_grad()
            train_loss = 0.0

            # Each batch is a dictionary containing all the sensor data, and the ground truth positions
            data, gt_pos = batch['data'], batch['gt_pos']
            # Data itself is a dictionary with keys ('modality', 'node') that points to data of dimension batch_size
            gt_pos = gt_pos.to(device)

            data = transform_data(data)

            # Perform forward pass
            batch_results = model(data) #Dictionary
            # key is still ('modality', 'node') with a distribution estimated by the model
            for key in batch_results.keys():
                for i in range(len(batch_results[key]['dist'])):
                    # Use both mse and nll to supervise training
                    loss_mse = mseloss(torch.squeeze(batch_results[key]['dist'][i].mean), torch.squeeze(gt_pos[i][:, [0, 2]]))
                    pos_neg_log_probs = -batch_results[key]['dist'][i].log_prob(torch.squeeze(gt_pos[i][:, [0, 2]])) # Computes NLL loss for each node/modality combo
                    train_loss += pos_neg_log_probs + 0.05 * loss_mse # Accumulate all the losses into the batch loss
                    with torch.no_grad(): # Get the average distance during train to log
                        ad_train_loss += loss_mse / (batch_size * len(batch_results.keys()))
            train_loss /= (batch_size * len(batch_results.keys())) # Normalize wrt batch size and number of modality node combinations
            
            with torch.no_grad():
                # Print one sample from the batch to see prediction result and loss
                print('Batch Number', batch_num)
                print('Gamma', data['gamma', 'gamma'][0])
                key = 'early_fusion'
                print('Estimate', batch_results[key]['dist'][0].mean.data, " with cov ",  batch_results[key]['dist'][0].covariance_matrix.data)
                sample_mse_loss =  mseloss(torch.squeeze(batch_results[key]['dist'][0].mean), torch.squeeze(gt_pos[0][:, [0, 2]]))
                sample_nll_loss =  -batch_results[key]['dist'][0].log_prob(torch.squeeze(gt_pos[0][:, [0, 2]]))
                print('\tGT', gt_pos[0], 'with loss', sample_nll_loss + 0.05 * sample_mse_loss)
                print('-------------------------------------------------------------------------------------------------------------------------------')
                epoch_train_loss += train_loss # Accumulate batch loss into overall epoch loss

            # Backprop and update
            train_loss.backward()
            optimizer.step() 
        
        print('TRAIN LOSS', epoch_train_loss / batch_num)
        ad_train_loss /= batch_num
        writer.add_scalar("Training loss", epoch_train_loss / batch_num, epoch)

        # Peform validation
        batch_num = 0
        epoch_val_loss = 0
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                batch_num += 1
                val_loss = 0.0
                # Each batch is a dictionary containing all the sensor data, and the ground truth positions
                data, gt_pos = batch['data'], batch['gt_pos']
                gt_pos = gt_pos.to(device)
                # Get the noise
                data = transform_data(data)
                
                # Perform forward pass
                batch_results = model(data) #Dictionary
                # key is still ('modality', 'node') with a distribution estimated by the model
                for key in batch_results.keys():
                    for i in range(len(batch_results[key]['dist'])):     
                        val_loss += mseloss(torch.squeeze(batch_results[key]['dist'][i].mean), torch.squeeze(gt_pos[i][:, [0, 2]]))
                val_loss /= (len(batch_results[key]['dist']) * len(batch_results.keys())) # Normalize wrt batch size and number of modality node combinations
                epoch_val_loss += val_loss
            epoch_val_loss /= batch_num
            print("Validation loss", epoch_val_loss)
        with open( './logs/' + args.dir_name + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(ad_train_loss) + ' | Val Loss ' + str(epoch_val_loss) + ' | Dropout ' + str(model.vision.layerdrop_rate) + ' ' + str(model.depth.layerdrop_rate)
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + args.dir_name + '/last.pt')
                


if __name__ == '__main__':
    args = get_args_parser()
    main(args)


