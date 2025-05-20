from PickleDataset import PickleDataset, transform_noise, transform_set_noise, transform_mask, transform_finite_noise, transform_discrete_noise
import configargparse
import os
from torch.utils.data import DataLoader
import pickle
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
    testset = PickleDataset(args.cache_dir + 'test', args.valid_mods, args.valid_nodes)
    test_dataloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False, num_workers=20)

    os.mkdir('./test_datasets/' + args.test_type)
    for idx, sample in enumerate(test_dataloader):
        #sample['data'], _ = transform_noise(sample['data'], args.batch_size, img_std_max=4, depth_std_max=0.75)
        if args.test_type == 'continuous':
            sample['data'], _ = transform_noise(sample['data'], args.batch_size, img_std_max=4, depth_std_max=0.75)
        elif args.test_type == 'finite':
            sample['data'], _ = transform_finite_noise(sample['data'], args.batch_size, img_std_max=3, depth_std_max=0.75)
        elif args.test_type == 'discrete':
            sample['data'], _ = transform_discrete_noise(sample['data'], args.batch_size, img_std_candidates=[0, 1, 2, 3], depth_std_candidates=[0, 0.25, 0.5, 0.75])
        for key in sample['data']:
            if 'mocap' not in key:
                sample['data'][key] = sample['data'][key][0]
        sample['gt_pos'] = sample['gt_pos'][0]
        with open('./test_datasets/' + args.test_type + '/' + str(idx) + '.pickle', 'wb') as handle:
            pickle.dump(sample, handle)
        



if __name__ == '__main__':
    args = get_args_parser()
    main(args)
