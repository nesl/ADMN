import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from torchvision import transforms

# Used to expand one channel image into three channels
# Clamps values between 0 and 1
class ClampImg:
    def __call__(self, input):
        return torch.clamp(input, min=0, max=1)
class ExpandDims:
    def __call__(self, input):
        if input.shape[1] == 1:
            return input.repeat(1, 3, 1, 1)
        return input

transform_dict = {
    # Input images will have noise, first clamp and then resize/mean
    'rgb': 
        transforms.Compose([
            ClampImg(),
            transforms.Resize((224, 224)),       
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]),
    'depth': 
        transforms.Compose([
            ClampImg(),
            transforms.Resize((224, 224)),
        ])
}

# Mask out modality (replace with 0s) with some given probability
# Ensures that at least one modality will be present
def transform_mask(data, b_size, img_mask=0.5, depth_mask=0.5):
    # Perform the random masking: should be replaced
    img_keep = (torch.rand(b_size) > img_mask).int().cuda()
    depth_keep = (torch.rand(b_size) > depth_mask).int().cuda()
    for i in range(b_size):
        if depth_keep[i] == 0 and img_keep[i] == 0:
            depth_keep[i] = 1
    print("Image[0]:", img_keep[0].item(), '\t', 'Depth[0]', depth_keep[0].item())
    for key in data.keys():
        data[key] = data[key].cuda()
        if 'rgb' in key:
            data[key] = data[key] * img_keep[:data[key].shape[0], None, None, None, None]
        if 'depth' in key:
            data[key] = data[key] * depth_keep[:data[key].shape[0], None, None, None, None]
        if key in transform_dict:
            batch_size, num_frames, channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (-1, channels, height, width))
            data[key] = transform_dict[key](data[key]) # Apply the relevant transformation
            _,  channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (batch_size, num_frames, channels, height, width))
    return data, None

def transform_discrete_noise(data, b_size, img_std_candidates = [0, 3], depth_std_candidates = [0, 0.75]):
    img_std_candidates = torch.tensor(img_std_candidates).cuda()
    depth_std_candidates = torch.tensor(depth_std_candidates).cuda()
    img_noise_idx = torch.trunc(torch.rand(b_size) * len(img_std_candidates)).int().cuda()
    depth_noise_idx = torch.trunc(torch.rand(b_size) * len(depth_std_candidates)).int().cuda()
    img_noise_scale = img_std_candidates[img_noise_idx]
    depth_noise_scale = depth_std_candidates[depth_noise_idx]
    print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item())
    for key in data.keys():
        data[key] = data[key].cuda()
        # Add the noise, and then clamp and resize
        if 'rgb' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:data[key].shape[0], None, None, None, None]
        if 'depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:data[key].shape[0], None, None, None, None]
        if key in transform_dict:
            batch_size, num_frames, channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (-1, channels, height, width))
            data[key] = transform_dict[key](data[key]) # Apply the relevant transformation
            _,  channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (batch_size, num_frames, channels, height, width))
      
    # Return both the noisy data, as well as the noise vectors to use as GT for noise-supervised training 
    return data, torch.stack((img_noise_scale[:data[key].shape[0]], depth_noise_scale[:data[key].shape[0]]), dim=-1)

# Image noise either 0 or img_std_max, depth noise either 0 or depth_std_max
# Ensure that both arent noisy at the same time
def transform_finite_noise(data, b_size, img_std_max = 3, depth_std_max=0.75):
    img_noise_scale = torch.round(torch.rand(b_size)).cuda() * img_std_max
    depth_noise_scale = torch.round(torch.rand(b_size)).cuda() * depth_std_max
    depth_noise_scale[torch.where(img_noise_scale==img_std_max)] = 0
    print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item())
    for key in data.keys():
        data[key] = data[key].cuda()
        # Add the noise, and then clamp and resize
        if 'rgb' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:data[key].shape[0], None, None, None, None]
        if 'depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:data[key].shape[0], None, None, None, None]
        if key in transform_dict:
            batch_size, num_frames, channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (-1, channels, height, width))
            data[key] = transform_dict[key](data[key]) # Apply the relevant transformation
            _,  channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (batch_size, num_frames, channels, height, width))
      
    # Return both the noisy data, as well as the noise vectors to use as GT for noise-supervised training 
    return data, torch.stack((img_noise_scale[:data[key].shape[0]], depth_noise_scale[:data[key].shape[0]]), dim=-1)

# Add a fixed amount of noise, no stochasticity 
def transform_set_noise(data, img_std = 3, depth_std = 0):
    
    print("Img Noise", img_std, "Depth Noise", depth_std)
    for key in data.keys():
        data[key] = data[key].cuda()
        b_size = data[key].shape[0]
        # Add the noise before so we can see the effect
        if 'rgb' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_std
        if 'depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_std
        if key in transform_dict:
            batch_size, num_frames, channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (-1, channels, height, width))
            data[key] = transform_dict[key](data[key]) # Apply the relevant transformation
            _,  channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (batch_size, num_frames, channels, height, width))
    # return transformed data and the noise distribution   
    return data, torch.cat([torch.full((b_size, 1), img_std), torch.full((b_size, 1), depth_std)], dim=-1).cuda()

# Returns any arrangement of image noise between 0->img_std_max and depth noise between 0->depth_std_max
# Also returns the noise for GT supervision, when training the base model ignore this
def transform_noise(data, b_size, img_std_max = 3, depth_std_max=0):
    
    img_noise_scale = img_std_max * torch.rand(b_size).cuda()
    depth_noise_scale = depth_std_max * torch.rand(b_size).cuda()
    print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item())
    for key in data.keys():
        data[key] = data[key].cuda()
        # Add the noise before so we can see the effect
        if 'rgb' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:data[key].shape[0], None, None, None, None]
        if 'depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:data[key].shape[0], None, None, None, None]
        if key in transform_dict:
            batch_size, num_frames, channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (-1, channels, height, width))
            data[key] = transform_dict[key](data[key]) # Apply the relevant transformation
            _,  channels, height, width = data[key].shape
            data[key] = torch.reshape(data[key], (batch_size, num_frames, channels, height, width))
      
    return data, torch.stack((img_noise_scale[:data[key].shape[0]], depth_noise_scale[:data[key].shape[0]]), dim=-1)


##  Depreciated
# def transform_noise_old(data, b_size, img_std_max = 3, depth_std_max=0.25):
    
#     img_noise_scale = img_std_max * torch.rand(b_size).cuda()
#     depth_noise_scale = depth_std_max * torch.rand(b_size).cuda()
#     print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item())
#     for key in data.keys():
#         if ('mocap' in key):
#             continue
#         data[key] = data[key].cuda()
#         if 'rgb' in key:
#             data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:data[key].shape[0], None, None, None]
#         if 'depth' in key:
#             data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:data[key].shape[0], None, None]
#         if key[0] in transform_dict:
#             data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
      
#     return data

## Depreciated
# def transform_inputs(data):
#     for key in data.keys():
#         if ('mocap' in key):
#             continue
#         data[key] = data[key].cuda()
#         # Add the noise before so we can see the effect
        
#         if key[0] in transform_dict:
#             data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
#     return data

# Point to the appropriate directory
class TestDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        for file in sorted(os.listdir(file_path)):
            with open(file_path + file, 'rb') as handle:
                self.data.append(pickle.load(handle))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    


# Loads the items into RAM for fast training
class PickleDataset(Dataset):
    def __init__(self, file_path, dataset_type):
        train_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26',
                          'S31', 'S32', 'S33', 'S34', 'S35', 'S36']
        val_subjects = ['S07', 'S17', 'S27', 'S37']
        test_subjects = ['S08', 'S09', 'S10', 'S18', 'S19', 'S20', 'S28', 'S29', 'S30', 'S38', 'S39', 'S40']
        if dataset_type == 'train':
            valid_subjects = train_subjects
        elif dataset_type == 'val':
            valid_subjects = val_subjects
        elif dataset_type == 'test':
            valid_subjects = test_subjects
        else:
            raise Exception("Must be train, val or test")
        self.data = []
        self.dataset_type = dataset_type
        for environment in tqdm(os.listdir(file_path)):
            for subject in sorted(os.listdir(file_path + '/' + environment)):
                if subject not in valid_subjects:
                    continue
                for action in sorted(os.listdir(file_path + '/' + environment + '/' + subject)):
                    curr_pickle = pickle.load(open(file_path + '/' + environment + '/' + subject + '/' + action + '/data.pickle',  'rb'))
                    if self.dataset_type != 'train':
                        curr_pickle['rgb'] = curr_pickle['rgb'][:30:2]
                        curr_pickle['depth'] = curr_pickle['depth'][:30:2]

                    curr_pickle['labels'] = int(action.split('A')[1]) - 1
                    self.data.append(curr_pickle)
                    if len(self.data) == 100:
                        return
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        curr_pickle = self.data[idx]
        if self.dataset_type == 'train':
            start = (torch.rand(1) * (296 - 30)).int().item()
            curr_pickle['rgb'] = curr_pickle['rgb'][start:start + 30 : 2]
            curr_pickle['depth'] = curr_pickle['depth'][start:start + 30 : 2]
        for key in curr_pickle.keys():
            if key == 'rgb':
                curr_pickle['rgb'] = curr_pickle['rgb'] / 255
            if key == 'depth':
                curr_pickle['depth'] = curr_pickle['depth'] / 255
            if isinstance(curr_pickle[key], np.ndarray):
                curr_pickle[key] = curr_pickle[key].astype(np.float32)
        
        return curr_pickle

