import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.GTDM_Model import GTDM_Controller, Conv_GTDM_Controller_Test_FLOPS
from PickleDataset import PickleDataset
from tracker import TorchMultiObsKalmanFilter
from video_generator import VideoGenerator
from cache_datasets import cache_data
import configargparse
from PickleDataset import transform_noise, transform_set_noise, transform_mask, transform_finite_noise, transform_discrete_noise
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table

def computeDist(tensor1, tensor2):
    tensor1 = torch.squeeze(tensor1)
    tensor2 = torch.squeeze(tensor2)
    distance = 0.0
    for i in range(len(tensor1)):
        distance += (tensor1[i] - tensor2[i]) ** 2
    return distance ** 0.5


def get_args_parser():
    parser = configargparse.ArgumentParser(description='GTDM Controller Testing, load config file and override params',
                                           default_config_files=['./configs/configs.yaml'], config_file_parser_class=configargparse.YAMLConfigFileParser)
    # Define the parameters with their default values and types
    parser.add("--base_root", type=str, help="Base directory for datasets")
    parser.add("--cache_dir", type=str, help="Directory to cache datasets")
    parser.add("--valid_mods", type=str, nargs="+", help="List of valid modalities")
    parser.add("--valid_nodes", type=int, nargs="+", help="List of valid nodes")
    parser.add("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add('--seedVal', type=int, default=100, help="Seed for training")
    parser.add('--folder', type=str, default='./logs', help='Folder containing the model')
    parser.add('--checkpoint', type=str, default='last.pt', help="ckpt nane")
    parser.add('--test_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])
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

    folder = str(args.folder)
    cache_data(args)
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = folder + '/'
    # Create model architecture
    model = Conv_GTDM_Controller_Test_FLOPS(args.adapter_hidden_dim, valid_mods=args.valid_mods, valid_nodes = args.valid_nodes, total_layers=args.total_layers) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    model.load_state_dict(torch.load(dir_path + str(args.checkpoint), weights_only=False), strict=False)
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = PickleDataset(args.cache_dir + 'test', args.valid_mods, args.valid_nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False, num_workers=20)
    # Initialize the kalman filter
    kf = TorchMultiObsKalmanFilter(dt=1, std_acc=1)
    outputs = {}
    outputs['det_means'] = []
    outputs['det_covs'] = []
    outputs['track_means'] = []
    outputs['track_covs'] = []
    total_nll_loss = 0.0
    total_mse_loss = 0.0
    average_dist = 0.0
    mseloss = nn.MSELoss()
    mse_arr = []
    gt_pos_arr = []
    avg_distance_KF = 0.0


    model.eval()
    total_model_time = 0.0
    for batch in tqdm(test_dataloader, desc = 'Computing test loss', leave=False):
        with torch.no_grad():
            data, gt_pos = batch['data'], batch['gt_pos']
            gt_pos = gt_pos.to(device)[:, 0]
            if args.test_type == 'continuous':
                data, _ = transform_noise(data, args.batch_size, img_std_max=3, depth_std_max=0.75)
            elif args.test_type == 'finite':
                data, _ = transform_finite_noise(data, args.batch_size, img_std_max=3, depth_std_max=0.75)
            elif args.test_type == 'discrete':
                data, _ = transform_discrete_noise(data, args.batch_size, img_std_candidates=[0, 1, 2, 3], depth_std_candidates=[0, 0.25, 0.5, 0.75])
            else:
                raise Exception('Invalid test type specified')
            flops = FlopCountAnalysis(model, data)
            
            num_flops = flops.total() / (1000 ** 3)
            print(flop_count_table(flops))
            print(num_flops)
            import pdb; pdb.set_trace()
            break
            
 

if __name__ == '__main__':
    args = get_args_parser()
    main(args)

