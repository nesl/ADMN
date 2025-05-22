import os
import pickle
from PickleDataset import AVE_Dataset
from tqdm import tqdm

def cache_data(base_root, cached_root):
    if not os.path.isdir(cached_root + '/train/'):
        print("Caching Train")
        os.mkdir(cached_root + '/train/')
        trainset = AVE_Dataset(dataset_root = base_root, dataset_txt_file='trainSet.txt')
        ctr = 0
        for item in tqdm(trainset):
            with open(cached_root + '/train/' + str(ctr) + '.pickle', 'wb') as handle:
                pickle.dump(item, handle)
            ctr += 1
    if not os.path.isdir(cached_root + '/val/'):
        print("Caching Val")
        os.mkdir(cached_root + '/val/')
        trainset = AVE_Dataset(dataset_root = base_root, dataset_txt_file='valSet.txt')
        ctr = 0
        for item in tqdm(trainset):
            with open(cached_root + '/val/' + str(ctr) + '.pickle', 'wb') as handle:
                pickle.dump(item, handle)
            ctr += 1
    if not os.path.isdir(cached_root + '/test/'):
        print("Caching Test")
        os.mkdir(cached_root + '/test/')
        trainset = AVE_Dataset(dataset_root = base_root, dataset_txt_file='testSet.txt')
        ctr = 0
        for item in tqdm(trainset):
            with open(cached_root + '/test/' + str(ctr) + '.pickle', 'wb') as handle:
                pickle.dump(item, handle)
            ctr += 1