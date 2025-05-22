import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from torchvision import transforms

# Used to expand one channel image into three channels
class ExpandChannels:
    def __call__(self, input):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, dim=1)
        return input.repeat((1, 3, 1, 1))
# Clamps values between 0 and 1
class ClampImg:
    def __call__(self, input):
        return torch.clamp(input, min=0, max=1)

transform_dict = {
    # Input images will have noise, first clamp and then resize/mean
    'zed_camera_left': 
        transforms.Compose([
            ClampImg(),
            transforms.Resize((224, 224)),       
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]),
    'realsense_camera_depth': 
        transforms.Compose([
            ClampImg(),
            ExpandChannels(),
            transforms.Resize((224, 224)),
        ])
}
def transform_data(data):
    for key in data.keys():
        if ('mocap' in key):
            continue
        data[key] = data[key].cuda()               
        if key[0] in transform_dict:
            data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
      
    return data


# Loads the items into RAM for fast training
class PickleDataset(Dataset):
    def __init__(self, file_path, valid_mods, valid_nodes):
        self.data = []
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes
        for file_name in tqdm(sorted(os.listdir(file_path)), desc="Loading pickle files into dataset"):
            curr_pickle = pickle.load(open(file_path + '/' + file_name,  'rb'))
            if curr_pickle['gamma', 'gamma'] == 10:
                continue
            # if curr_pickle['gamma', 'gamma'] != 100:
            #     continue
            for key in curr_pickle:
                if isinstance(curr_pickle[key], np.ndarray):
                    curr_pickle[key] = curr_pickle[key].astype(np.float32)
                if 'realsense_camera_depth' in key:
                    curr_pickle[key] = curr_pickle[key][:120, :160]
        
            self.data.append(curr_pickle)
      
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if 'gt_pos' in self.data[idx].keys():
            return self.data[idx]
        return {'data': self.data[idx], 'gt_pos': self.data[idx][('mocap', 'mocap')]['gt_positions']}