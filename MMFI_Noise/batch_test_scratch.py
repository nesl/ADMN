import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.MMFI_Model import MMFI_Early
from PickleDataset import TestDataset
from tracker import TorchMultiObsKalmanFilter
from video_generator import VideoGenerator
from cache_datasets import cache_data
import argparse
import time
from sklearn.metrics import accuracy_score, f1_score

def get_args_parser():
    parser = argparse.ArgumentParser(description='GTDM Controller Testing')
                                     
    # Define the parameters with their default values and types
    parser.add_argument("--base_root", type=str, default='/mnt/ssd_8t/jason/MMFI_Pickles_Img_DepthColorized', help="Base directory for datasets")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache datasets")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512, help="Dimension of adapter hidden layers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model")
    parser.add_argument("--save_every_X_model", type=int, default=5, help="Save model every X epochs")
    parser.add_argument('--seedVal', type=int, default=100, help="Seed for training")
    parser.add_argument('--folder', type=str, default='./logs', help='Folder containing the model')
    parser.add_argument('--checkpoint', type=str, default='last.pt', help="ckpt nane")
    parser.add_argument('--test_type', type=str, default='continuous', choices=['continuous', 'discrete', 'finite'])
    parser.add_argument('--vit_layers_img', type=int, help='Number of layers in the image ViT')
    parser.add_argument('--vit_layers_depth', type=int, help='Number of layers in depth ViT')
    parser.add_argument("--valid_mods", type=str, nargs="+", default=['image', 'depth'], help="List of valid modalities")
    args = parser.parse_args()

    return args


def main(args):
    folder = str(args.folder)
    #import pdb; pdb.set_trace()
    # Point test.py to appropriate log folder containing the saved model weights
    dir_path = folder + '/'
    # Create model architecture
    model = MMFI_Early(vision_vit_layers=args.vit_layers_img, depth_vit_layers=args.vit_layers_depth, valid_mods=args.valid_mods) # Pass valid mods, nodes, and also hidden layer size
    # Load model weights
    model.load_state_dict(torch.load(dir_path + str(args.checkpoint)), strict=True)
    model.eval() # Set model to eval mode for dropout
    # Create dataset and dataloader for test
    testset = TestDataset('./test_datasets/' + args.test_type + '/')
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
            for key in batch:
                batch[key] = batch[key].cuda()
            start = time.time()
            results = model(batch) # Evaluate on test data
            total_model_time += time.time() - start
            test_loss += loss_fn(results, batch['labels'])
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
    f.write('\nLatency: ' + str(total_model_time) + '\n')
    f.close()

    print(total_model_time)



if __name__ == '__main__':
    args = get_args_parser()
    main(args)

