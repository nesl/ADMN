import torch
import torch.nn as nn
import os
import configargparse
import numpy as np
import time
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange
from datetime import datetime
from numpy import random
from torch.utils.tensorboard import SummaryWriter

from models.GTDM_Model import AdaMML_Model_All
from PickleDataset import PickleDataset, transform_data
from cache_datasets import cache_data
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

def mseloss(t1, t2):
    sum = 0
    for i in range(len(t1)):
        sum += (t1[i] - t2[i]) ** 2 # Removed item to ensure gradient flow
    return sum ** 0.5 

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

def get_args_parser():
    parser = configargparse.ArgumentParser(description='GTDM Controller Training, load config file and override params',
                                           default_config_files=['./configs/configs.yaml'], config_file_parser_class=configargparse.YAMLConfigFileParser)
    # Define the parameters with their default values and types
    parser.add("--base_root", type=str, help="Base directory for datasets")
    parser.add("--cache_dir", type=str, help="Directory to cache datasets")
    parser.add("--valid_mods", type=str, nargs="+", help="List of valid modalities")
    parser.add("--valid_nodes", type=int, nargs="+", help="List of valid nodes")
    parser.add("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add('--seedVal', type=int, default=100, help="Seed for training")
    parser.add('--discretization_method', type=str, default='admn', choices=['admn', 'straight_through', 'progressive'])
    parser.add("--temp", type=float, default=1, help="Learning rate for training")
    parser.add("--img_ckp_path", type=str, default='logs/AdaMML_Subnet_Img_Test/last.pt', help="path of the pretrained image recognition model")
    parser.add("--dep_ckp_path", type=str, default='logs/AdaMML_Subnet_Dep_Test/last.pt', help="path of the pretrained depth recognition model")
    parser.add("--fused_ckp_path", type=str, default='logs/AdaMML_Subnet_Fusion/last.pt', help="path of the pretrained fused recognition model")
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
    # Set seed
    print("Starting training with seed value", args.seedVal)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seedVal)
    torch.manual_seed(args.seedVal)
    torch.cuda.manual_seed(args.seedVal)
    np.random.seed(args.seedVal)
    # Create based on noise type and number of layers
    dt_string = "AdaMML_Selector" + '_Seed_' + str(args.seedVal) + "_Layers_" + str(args.total_layers)
    os.mkdir('./logs/' + dt_string)
    cache_data(args) # Runs cacher from the data_configs.py file, will convert hdf5 to pickle if not already done
    
    #PickleDataset inherits from a Pytorch Dataset, creates train and val datasets
    trainset = PickleDataset(args.cache_dir + 'train', args.valid_mods, args.valid_nodes)
    valset = PickleDataset(args.cache_dir + 'val', args.valid_mods, args.valid_nodes)
    batch_size = args.batch_size
    
    #Creates PyTorch dataloaders for train and val 
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # create the overall model
    model = AdaMML_Model_All(args.adapter_hidden_dim, valid_nodes=args.valid_nodes, total_layers=args.total_layers)

    img_weights = torch.load(args.img_ckp_path, weights_only=False)
    depth_weights = torch.load(args.dep_ckp_path, weights_only=False)
    fused_weights = torch.load(args.fused_ckp_path, weights_only=False)

    new_img_weights = {}
    for key in img_weights.keys():
        if 'depth' not in key:
            new_img_weights[key] = img_weights[key]
    new_dep_weights = {}
    for key in depth_weights.keys():
        if 'vision' not in key:
            new_dep_weights[key] = depth_weights[key]
    # load the pretrained weights
    print(model.vision.load_state_dict(new_img_weights, strict=True))
    print(model.depth.load_state_dict(new_dep_weights, strict=True))
    print(model.fused.load_state_dict(fused_weights, strict=True))

    model.to(device)

    # freeze all the parameters except fpr the selector
    for param in model.parameters():
        param.requires_grad = False
    for param in model.selector.parameters():
        param.requires_grad = True

    params = [
        {"params": [p for name, p in model.selector.named_parameters() if "mod_sel_head" not in name], "lr": args.learning_rate},
        {"params": model.selector.mod_sel_head.parameters(), "lr": args.learning_rate},
    ]

    optimizer = Adam(params)

    # We actually use a linear scheduler instead of Cosine
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=args.num_epochs)
    writer = SummaryWriter(log_dir='./logs/' + dt_string) # Implement tensorboard
    
    # Training loop
    train_start = time.time()
    ce_loss = torch.nn.CrossEntropyLoss()
    for epoch in trange(args.num_epochs, desc="Training"):
        
        batch_num = 0
        epoch_train_loss = 0
        ad_train_loss = 0
        model.train()
 
        for batch in train_dataloader:
            batch_num += 1
            train_loss = 0.0

            # Each batch is a dictionary containing all the sensor data, and the ground truth positions
            data, gt_pos = batch['data'], batch['gt_pos']
            # Data itself is a dictionary with keys ('modality', 'node') that points to data of dimension batch_size
            gt_pos = gt_pos.to(device)
            data = transform_data(data)

            # forward pass
            batch_results, seletor_out = model(data)  # batch_results is a dictionary with keys 'subnet_model' and 'dist'
            print("modality selection result:", seletor_out)
            
            # Accumulate train loss
            for key in batch_results.keys():
                for i in range(len(batch_results[key]['dist'])):
                    # TODO Currently 2D, also introduce hybrid training, use MSE to help convergence at start then use NLL
                    loss_mse = mseloss(torch.squeeze(batch_results[key]['dist'][i].mean), torch.squeeze(gt_pos[i][:, [0, 2]]))
                    pos_neg_log_probs =  -batch_results[key]['dist'][i].log_prob(torch.squeeze(gt_pos[i][:, [0, 2]])) # Computes NLL loss for each node/modality combo
                    train_loss += pos_neg_log_probs + 0.05 * loss_mse # Accumulate all the losses into the batch loss
                    with torch.no_grad():
                        ad_train_loss += loss_mse / (batch_size * len(batch_results.keys()))
            
            train_loss /= (batch_size * len(batch_results.keys())) # Normalize wrt batch size and number of modality node combinations
            with torch.no_grad():
                # Print one sample from the batch to see prediction result and loss
                print('Batch Number', batch_num)
                print('chosen modality', )
                key = 'subnet_model'
                print('Estimate', batch_results[key]['dist'][0].mean.data, " with cov ",  batch_results[key]['dist'][0].variance.data)
                sample_mse_loss =  mseloss(torch.squeeze(batch_results[key]['dist'][0].mean), torch.squeeze(gt_pos[0][:, [0, 2]]))
                sample_nll_loss =  -batch_results[key]['dist'][0].log_prob(torch.squeeze(gt_pos[0][:, [0, 2]]))
                print('\tGT', gt_pos[0], 'with loss', sample_nll_loss + 0.05 * sample_mse_loss)
                #print('\tGT', gt_pos[0], 'with loss', mseloss(batch_results['img', 'node_1']['dist'][0].mean, gt_pos[0][:, 0:2]))
                print('-------------------------------------------------------------------------------------------------------------------------------')
                epoch_train_loss += train_loss
        
                
            train_loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) # No need to clip grad norm so far, can check this is errors occur
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
            log_file = open('./logs/' + dt_string + '/validation.txt', "w")
            # Redirect output to another file
            temp_std_out = sys.stdout
            sys.stdout = Tee(sys.stdout, log_file)
            model.eval()
            for batch in val_dataloader:

                batch_num += 1
                val_loss = 0.0
                # Each batch is a dictionary containing all the sensor data, and the ground truth positions
                data, gt_pos = batch['data'], batch['gt_pos']
                gt_pos = gt_pos.to(device)
                # val dataset is also clean, we add noise here
                data = transform_data(data)

                batch_results, selector_result = model(data)
                print("modality selection result:", selector_result)
                
                for key in batch_results.keys():
                    for i in range(len(batch_results[key]['dist'])):     
                        loss_mse = mseloss(torch.squeeze(batch_results[key]['dist'][i].mean), torch.squeeze(gt_pos[i][:, [0, 2]]))
                        pos_neg_log_probs =  -batch_results[key]['dist'][i].log_prob(torch.squeeze(gt_pos[i][:, [0, 2]])) # Computes NLL loss for each node/modality combo
                        val_loss += loss_mse # Accumulate all the losses into the batch loss
                val_loss /= (len(batch_results[key]['dist']) * len(batch_results.keys())) # Normalize wrt batch size and number of modality node combinations
                epoch_val_loss += val_loss
            epoch_val_loss /= batch_num
            print("Validation loss", epoch_val_loss)
            log_file.close()
            sys.stdout = temp_std_out
            
        with open( './logs/' + dt_string + '/log.txt', 'a') as handle:
            print('Epoch ' + str(epoch) + ' | Train loss ' + str(ad_train_loss) + ' | Val Loss ' + str(epoch_val_loss)
                  , file=handle)
        torch.save(model.state_dict(), './logs/' + dt_string + '/last.pt')
                
    print(time.time() - train_start)
    
if __name__ == '__main__':
    args = get_args_parser()
    main(args)