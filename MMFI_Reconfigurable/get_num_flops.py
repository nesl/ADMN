import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.MMFI_Model import Conv_MMFI_Controller
from PickleDataset import PickleDataset
from tracker import TorchMultiObsKalmanFilter
from video_generator import VideoGenerator
from cache_datasets import cache_data
import argparse
from PickleDataset import TestDataset
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table


def get_args_parser():
    parser = argparse.ArgumentParser(description='GTDM Controller Testing, load config file and override params')
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default='/mnt/ssd_8t/jason/MMFI_Pickles_Img_DepthColorized', help="Base directory for datasets")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--folder', type=str, default='./logs', help='Folder containing the model')
    parser.add_argument('--checkpoint', type=str, default='last.pt', help="ckpt nane")
    parser.add_argument('--test_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])
    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
    return args

def main(args):

    folder = str(args.folder)
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = folder + '/'
    # Create model architecture
    model = Conv_MMFI_Controller(total_layers=args.total_layers) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    model.load_state_dict(torch.load(dir_path + str(args.checkpoint)), strict=True)
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = TestDataset('./test_datasets/' + args.test_type + '/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False, num_workers=20)


    model.eval()
    total_model_time = 0.0
    for batch in tqdm(test_dataloader, desc = 'Computing test loss', leave=False):
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].cuda()   
            flops = FlopCountAnalysis(model, batch)
            num_flops = flops.total() / (1000 ** 3)
            print(flop_count_table(flops))
            print(num_flops)
            import pdb; pdb.set_trace()
            break
            
 

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

