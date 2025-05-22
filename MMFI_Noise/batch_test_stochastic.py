import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.MMFI_Model import MMFI_Early
from PickleDataset import PickleDataset, transform_noise, transform_finite_noise, transform_discrete_noise, transform_set_noise
from tracker import TorchMultiObsKalmanFilter
from video_generator import VideoGenerator
from sklearn.metrics import accuracy_score, f1_score
import argparse
import time

def get_args_parser():
    parser = argparse.ArgumentParser(description='GTDM Controller Testing')
                                     
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default='/mnt/ssd_8t/jason/MMFI_Pickles_Img_DepthColorized', help="Base directory for datasets")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache datasets")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--folder', type=str, default='./logs', help='Folder containing the model')
    parser.add_argument('--checkpoint', type=str, default='last.pt', help="ckpt nane")
    parser.add_argument('--test_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite', 'no_noise'])
    parser.add_argument('--drop_layers_img', type=int, nargs="+", help='List of layers to DROP')
    parser.add_argument('--drop_layers_depth', type=int, nargs="+", help='List of layers to DROP')
    args = parser.parse_args()

    return args


def main(args):
    folder = str(args.folder)
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = folder + '/'
    # Create model architecture
    model = MMFI_Early(drop_layers_img=args.drop_layers_img, drop_layers_depth=args.drop_layers_depth) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    print(model.load_state_dict(torch.load(dir_path + str(args.checkpoint)), strict=False))
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = PickleDataset(args.base_root, dataset_type='test')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    pred_list = []
    gt_list = []

    model.eval()
    total_model_time = 0.0
    test_loss = 0
    for batch in tqdm(test_dataloader, desc = 'Computing test loss', leave=False):
        with torch.no_grad():
            start = time.time()
            batch, _ = transform_set_noise(batch, img_std=2, depth_std=0)
            
            results = model(batch) # Evaluate on test data
            # if args.train_type == 'continuous':
            #     batch, gt_noise = transform_noise(batch, args.batch_size, img_std_max=4, depth_std_max=0.75)
            # elif args.train_type == 'finite':
            #     batch, gt_noise = transform_finite_noise(batch, args.batch_size, img_std_max=3, depth_std_max=0.75)
            # elif args.train_type == 'discrete':
            #     batch, gt_noise = transform_discrete_noise(batch, args.batch_size, img_std_candidates=[0, 1, 2, 3], depth_std_candidates=[0, 0.25, 0.5, 0.75])
            # else:
            #     raise Exception('Invalid test type specified')å
            
            total_model_time += time.time() - start
            test_loss += loss_fn(results, batch['labels'].to(device))
            pred_list.extend(torch.argmax(results, dim=-1).cpu().detach().tolist())
            gt_list.extend(batch['labels'].cpu().detach().tolist())
                   

    acc = accuracy_score(gt_list, pred_list)
    f1_micro = f1_score(gt_list, pred_list, average='micro')
    f1_macro = f1_score(gt_list, pred_list, average='macro')
    
    print("Finished running model inference, generating video with total test loss", test_loss / (len(test_dataloader)))
    f = open(dir_path + "test_loss.txt", "a")
    f.write('\nAccuracy: ' + str(acc))
    f.write('\nF1 Micro: ' + str(f1_micro))
    f.write('\nF1 Macro: ' + str(f1_macro))
    f.write('\nLatency: ' + str(total_model_time))
    f.close()

    print(total_model_time)



if __name__ == '__main__':
    args = get_args_parser()
    main(args)

