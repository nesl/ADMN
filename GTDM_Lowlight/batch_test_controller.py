from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.GTDM_Model import Conv_GTDM_Controller
from PickleDataset import PickleDataset, transform_data
from tracker import TorchMultiObsKalmanFilter
from cache_datasets import cache_data
import configargparse
import time

'''

Performing testing on the ADMN controller
We have a separate, fixed noisy dataset stored under test_datasets to ensure consistent performance

'''
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
    parser.add("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add('--seedVal', type=int, default=100, help="Seed for training")
    parser.add('--folder', type=str, default='./logs', help='Folder containing the model')
    parser.add('--checkpoint', type=str, default='last.pt', help="ckpt nane")
    parser.add('--test_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])
    parser.add('--discretization_method', type=str, default='admn', choices=['admn', 'straight_through', 'progressive'])
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
    model = Conv_GTDM_Controller(args.adapter_hidden_dim, valid_mods=args.valid_mods, valid_nodes = args.valid_nodes, total_layers=args.total_layers) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    print(model.load_state_dict(torch.load(dir_path + str(args.checkpoint), weights_only=False), strict=True))
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = PickleDataset(args.cache_dir + 'test', args.valid_mods, args.valid_nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.batch_size != 1:
        raise Exception("Batch size not one, are you sure")
    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False)
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
            data = transform_data(data)
            gt_pos = gt_pos.to(device)[:, 0]
            start = time.time()
            results, _ = model(data, discretization_method=args.discretization_method) # Evaluate on test data
            total_model_time += time.time() - start
            all_pred_means = []
            all_pred_covs = []
            # Each modality and node combo has a predicted mean and covariance
            # Even if 3D, we only plot 2D so we take only x and y
            for result in results.values(): # This only runs once even with > 1 batch size
                for i in range(len(result['pred_mean'])):
                    # import pdb; pdb.set_trace()
                    predicted_means = result['pred_mean'][i][0:2] # Extract only x and y
                    predicted_covs = result['pred_cov'][i][0:2, 0:2]
                    predicted_means = predicted_means.cpu().detach()
                    predicted_covs = predicted_covs.cpu().detach()
                    gt_pos_arr.append(gt_pos[i, [0, 2]])
                    # Append to output
                    outputs['det_means'].append(predicted_means)
                    outputs['det_covs'].append(predicted_covs)
                    # Calculate loss
                    sample_mse_loss = mseloss(result['dist'][i].mean, gt_pos[i, [0,2]])
                    total_mse_loss += sample_mse_loss

                    average_dist += computeDist(predicted_means, gt_pos[i, [0, 2]])

                    mse_arr.append(sample_mse_loss)
                    total_nll_loss += -result['dist'][i].log_prob(gt_pos[i, [0,2]]) # May be 3D
                    # Perform Kalman Filters
                    kf.update(torch.unsqueeze(predicted_means, 1), [predicted_covs])
                    kf_mean = kf.predict() 
                    kf_cov = kf.P[0:2, 0:2]
                    outputs['track_means'].append(torch.squeeze(kf_mean))
                    outputs['track_covs'].append(kf_cov)
                    avg_distance_KF += computeDist(torch.squeeze(kf_mean).detach().cpu(), gt_pos[i, [0,2]].detach().cpu())
                   


    print("Finished running model inference, generating video with total test loss", total_nll_loss / (len(test_dataloader)))
    f = open(dir_path + "test_loss.txt", "w")
    f.write("\nAverage Distance " + str(average_dist / len(mse_arr)))
    f.write("\nLatency " + str(total_model_time))
    f.close()
    # Write the predictions and the gts
    f = open(dir_path + "predictions.txt", "a")
    f.write('Dets\t\t GT\n')
    for i in range(len(outputs['det_means'])):
        outputs['det_means'][i] = torch.unsqueeze(outputs['det_means'][i], 0)
        outputs['det_covs'][i] = torch.unsqueeze(outputs['det_covs'][i], 0)
        f.write(str(outputs['det_means'][i].cpu().numpy()))
        f.write("\t ")
        f.write(str(gt_pos_arr[i].cpu().numpy()))
        f.write('\n')

    print(total_model_time)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)

