from cacher import DataCacher



def cache_data(args):
    cache_train = DataCacher(hdf5_fnames = args.trainset, cache_dir= args.cache_dir + 'train', 
               valid_mods=args.valid_mods, 
               valid_nodes=args.valid_nodes)
    cache_test = DataCacher(hdf5_fnames = args.testset, cache_dir=args.cache_dir + 'test', 
               valid_mods=args.valid_mods, 
               valid_nodes=args.valid_nodes)
    cache_val = DataCacher(hdf5_fnames = args.valset, cache_dir=args.cache_dir + 'val', 
               valid_mods=args.valid_mods, 
               valid_nodes=args.valid_nodes)
    cache_train.cache()
    cache_test.cache()
    cache_val.cache()

# def main():

#     # assert args.out or args.eval or args.format_only or args.show \
#         # or args.show_dir, \
#         # ('Please specify at least one operation (save/eval/format/show the '
#          # 'results / save the results) with the argument "--out", "--eval"'
#          # ', "--format-only", "--show" or "--show-dir"')

#     # if args.eval and args.format_only:
#         # raise ValueError('--eval and --format_only cannot be both specified')

#     # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
#         # raise ValueError('The output file must be a pkl file.')
        
#     #building the datasets runs the cacher
#     #future calls (ie during training) will skip the caching step

#     cache_data()
#     data = PickleDataset('/mnt/nfs/vol2/redacted/train')
#     for item in data:
#         import pdb; pdb.set_trace()
#     train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
#     print("Success")
#     #build_dataset(cfg.testset)
    

# if __name__ == '__main__':
#     main()
