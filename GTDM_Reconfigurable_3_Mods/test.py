import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.GTDM_Model import GTDM, GTDM_Early
from PickleDataset import PickleDataset
from tracker import TorchMultiObsKalmanFilter
from video_generator import VideoGenerator
import configs.data_configs as data_configs
import configs.train_configs as train_configs
from cache_datasets import cache_data


def main():
    cache_data()
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = './logs/Early_Fusion_Base/'
    # Create model architecture
    model = GTDM_Early(train_configs.adapter_hidden_dim, valid_mods=data_configs.valid_mods, valid_nodes = data_configs.valid_nodes) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    model.load_state_dict(torch.load(dir_path + 'best_val.pt'))
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = PickleDataset(data_configs.cache_dir + 'test', data_configs.valid_mods, data_configs.valid_nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = DataLoader(testset, batch_size = 1)
    # Initialize the kalman filter
    kf = TorchMultiObsKalmanFilter(dt=1, std_acc=1)
    outputs = {}
    outputs['det_means'] = []
    outputs['det_covs'] = []
    outputs['track_means'] = []
    outputs['track_covs'] = []
    total_test_loss = 0.0
    total_mse_loss = 0.0
    average_dist = 0.0
    mseloss = nn.MSELoss()
    mse_arr = []
    batch_num = 0
    avg_distance_KF = 0.0
    for batch in tqdm(test_dataloader, desc = 'Computing test loss', leave=False):
        with torch.no_grad():
            batch_loss = 0
            data, gt_pos = batch['data'], batch['gt_pos']
            gt_pos = gt_pos.to(device)
            for key in data.keys():
                if ('mocap' in key):
                    continue
                data[key] = data[key].to(device)

            results = model(data) # Evaluate on test data
   
            all_pred_means = []
            all_pred_covs = []
            # Each modality and node combo has a predicted mean and covariance
            # Even if 3D, we only plot 2D so we take only x and y
            for result in results.values():
                all_pred_means.append(torch.squeeze(result['pred_mean'])[0:2]) # Extract only x and y
                all_pred_covs.append(torch.squeeze(result['pred_cov'])[0:2, 0:2])
                sample_mse_loss = mseloss(result['dist'][0].mean, gt_pos[..., [0,2]])
                average_dist += sample_mse_loss ** 0.5
                mse_arr.append(sample_mse_loss)
                batch_loss += -result['dist'][0].log_prob(gt_pos[..., [0,2]]) # May be 3D
            total_test_loss += batch_loss
            total_mse_loss += sample_mse_loss
            print(all_pred_means)
            print("Ground truth", gt_pos)
            print(batch_loss / len(results.values()))
            print('----------------------------------------------------------------------------------------')
            # all_pred_means = np.array(all_pred_means)
            # all_pred_covs = np.array(all_pred_covs)
            all_pred_means = torch.stack(all_pred_means).cpu().detach()
            all_pred_covs = torch.stack(all_pred_covs).cpu().detach()
            outputs['det_means'].append(all_pred_means)
            outputs['det_covs'].append(all_pred_covs)
            # Update the kalman filter all the predictions for current data point
            kf.update(torch.transpose(all_pred_means, 0, 1), all_pred_covs) #param z: shape (2,K) tensor of observations :param zcov: list of K 2x2 observation covariances
            # Predict one unified mean and cov
            kf_mean = kf.predict() 
            kf_cov = kf.P[0:2, 0:2] # Extract cov matrix, not sure if this is right
            # Place into the tracked means and cov
            outputs['track_means'].append(torch.squeeze(kf_mean))
            avg_distance_KF += mseloss(torch.squeeze(kf_mean).detach().cpu(), gt_pos[..., [0,2]].detach().cpu()) ** 0.5
            outputs['track_covs'].append(kf_cov)
        #if batch_num % 899 == 0 and batch_num != 0:
        print("Finished running model inference, generating video with total test loss", total_test_loss / (len(test_dataloader)))
        f = open(dir_path + "test_loss.txt", "a")
        f.write("\n\n\nComputed NLL Test Loss " + str(total_test_loss / len(mse_arr)))
        f.write("\nComputed MSE Test Loss " + str(total_mse_loss / len(mse_arr)))
        f.write("\nMedian MSE Loss " + str(sorted(mse_arr)[int(len(mse_arr)/2)]))
        f.write("\nAverage Distance " + str(average_dist / len(mse_arr)))
        f.write("\nAverage KF distance " + str(avg_distance_KF / len(mse_arr)))
        f.close()
        total_test_loss = 0
        total_mse_loss = 0
        average_dist = 0
        avg_distance_KF = 0.0
        mse_arr = []
        batch_num += 1

      
    #import pdb; pdb.set_trace()
    # Generate the video in the same logs folder as train
    v = VideoGenerator(dir_path, data_configs.valid_mods, data_configs.valid_nodes, testset) # Pass valid mods, nodes, and also hidden layer size
    v.write_video(outputs)


if __name__ == '__main__':
    main()



