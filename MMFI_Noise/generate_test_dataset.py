from PickleDataset import PickleDataset, transform_noise, transform_set_noise, transform_mask, transform_finite_noise, transform_discrete_noise
import argparse
import os
from torch.utils.data import DataLoader
import pickle
import numpy as np
def get_args_parser():
    parser = argparse.ArgumentParser(description='Generating the dataset for consistent testing')
    parser.add_argument("--base_root", type=str, default='/mnt/ssd_8t/redacted/MMFI_Pickles_Img_DepthColorized', help="Base directory for datasets")
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
    testset = PickleDataset(args.base_root, dataset_type='test')
    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False, num_workers=20)
    os.makedirs('./test_datasets/' + args.test_type, exist_ok=True)
    for idx, sample in enumerate(test_dataloader):
        #sample['data'], _ = transform_noise(sample['data'], args.batch_size, img_std_max=4, depth_std_max=0.75)
        if args.test_type == 'continuous':
            sample, _ = transform_noise(sample, args.batch_size, img_std_max=2, depth_std_max=4)
        elif args.test_type == 'finite':
            sample, _ = transform_finite_noise(sample, args.batch_size, img_std_max=2, depth_std_max=3)
        elif args.test_type == 'discrete':
            sample, _ = transform_discrete_noise(sample, args.batch_size, img_std_candidates=[0, 0.75, 1.5, 2], depth_std_candidates=[0, 2, 3, 4])
        for key in sample:
            sample[key] = sample[key][0].cpu().numpy()
        with open('./test_datasets/' + args.test_type + '/' + str(idx) + '.pickle', 'wb') as handle:
            pickle.dump(sample, handle)
        



if __name__ == '__main__':
    args = get_args_parser()
    main(args)
