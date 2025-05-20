from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.GTDM_Model import AdaMML_Model_All
from PickleDataset import PickleDataset
from cacher import cache_data
from sklearn.metrics import accuracy_score
import time
import argparse
'''

Performing testing on the ADMN controller
We have a separate, fixed noisy dataset stored under test_datasets to ensure consistent performance

'''


def get_args_parser():
    parser = argparse.ArgumentParser("Arguments for batch_test_controller")
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default = '/mnt/ssd_8t/jason/AVE_Dataset/', help="Base dataset root")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--valid_mods", type=str, nargs="+", default=['image', 'audio'], help="List of valid modalities")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument('--total_layers', type=int, default=8, help="How many layers to reduce to")
    parser.add_argument('--folder', type=str, default='./logs', help='Folder containing the model')
    parser.add_argument('--checkpoint', type=str, default='last.pt', help="ckpt nane")
    parser.add_argument('--test_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])
    parser.add_argument('--discretization_method', type=str, default='admn', choices=['admn', 'straight_through', 'progressive'])
    # Parse arguments from the configuration file and command-line
    args = parser.parse_args()
    return args

def main(args):

    folder = str(args.folder)
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = folder + '/'
    # Create model architecture
    model = AdaMML_Model_All(args.adapter_hidden_dim, total_layers=args.total_layers)
    # Load model weights
    print(model.load_state_dict(torch.load(dir_path + str(args.checkpoint)), strict=False))
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = PickleDataset(data_root = '/mnt/ssd2/data/AVE_Dataset_Cached/', type='test')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.batch_size != 1:
        raise Exception("Batch size not one, are you sure")
    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False)

    pred_labels = []
    gt_labels = []

    model.eval()
    total_model_time = 0.0
    for batch in tqdm(test_dataloader, desc = 'Computing test loss', leave=False):
        with torch.no_grad():
            batch, noise_label = batch
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            _, _, label = batch
            start = time.time()
            logits, _ = model(batch, discretization_method=args.discretization_method) # Evaluate on test data
            total_model_time += time.time() - start
            # Each modality and node combo has a predicted mean and covariance
            # Even if 3D, we only plot 2D so we take only x and y
            pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            gt_labels.append(label.cpu())
    pred_labels = torch.cat(pred_labels).numpy()
    gt_labels = torch.cat(gt_labels).numpy()


    print("Finished running model inference with accuracy", accuracy_score(gt_labels, pred_labels))
    f = open(dir_path + "test_loss.txt", "w")
    f.write("\nAccuracy " + str(accuracy_score(gt_labels, pred_labels)))
    f.write("\nLatency " + str(total_model_time))
    f.close()
 

    print(total_model_time)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)

