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
        ]),
    'range_doppler':
        transforms.Compose([
            ClampImg(),
            ExpandChannels()
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
        if ('mocap' in key):
            continue
        data[key] = data[key].cuda()
        
        if 'zed_camera_left' in key:
            data[key] = data[key] * img_keep[:data[key].shape[0], None, None, None]
        if 'realsense_camera_depth' in key:
            data[key] = data[key] * depth_keep[:data[key].shape[0], None, None]
        if key[0] in transform_dict:
            data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
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
        if ('mocap' in key):
            continue
        data[key] = data[key].cuda()
        # Add the noise, and then clamp and resize
        if 'zed_camera_left' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:data[key].shape[0], None, None, None]
        if 'realsense_camera_depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:data[key].shape[0], None, None]
        if key[0] in transform_dict:
            data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
      
    # Return both the noisy data, as well as the noise vectors to use as GT for noise-supervised training 
    return data, torch.stack((img_noise_scale[:data[key].shape[0]], depth_noise_scale[:data[key].shape[0]]), dim=-1)

# Image noise either 0 or img_std_max, depth noise either 0 or depth_std_max
# Ensure that both arent noisy at the same time
def transform_finite_noise(data, b_size, img_std_max = 3, depth_std_max=0.75, mmWave_std_max = 0.8):
    actual_b_size = data[('mocap', 'mocap')]['gt_positions'].shape[0]
    img_noise_scale = torch.round(torch.rand(actual_b_size)).cuda() * img_std_max
    depth_noise_scale = torch.round(torch.rand(actual_b_size)).cuda() * depth_std_max
    mmWave_noise_scale = torch.round(torch.rand(actual_b_size)).cuda() * mmWave_std_max
    depth_noise_scale[torch.where(img_noise_scale + mmWave_noise_scale == 3.8)] = 0
    print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item(), 'mmWave Noise', mmWave_noise_scale[0].item())
    for key in data.keys():
        if ('mocap' in key):
            continue
        data[key] = data[key].cuda()
        # Add the noise, and then clamp and resize
        if 'zed_camera_left' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:, None, None, None]
        if 'realsense_camera_depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:, None, None]
        if 'range_doppler' in key:
            mins = torch.min(data[key].view(actual_b_size, -1), dim=-1)[0]
            maxs = torch.max(data[key].view(actual_b_size, -1), dim=-1)[0]
            data[key] = (data[key] - mins[:, None, None])  / maxs[:, None, None]
            data[key] = data[key] + torch.randn_like(data[key]) * mmWave_noise_scale[:, None, None]
        if key[0] in transform_dict:
            data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
      
    # Return both the noisy data, as well as the noise vectors to use as GT for noise-supervised training 
    return data, torch.stack((img_noise_scale, depth_noise_scale, mmWave_noise_scale), dim=-1)

# Add a fixed amount of noise, no stochasticity 
def transform_set_noise(data, img_std = 3, depth_std = 0):
    
    print("Img Noise", img_std, "Depth Noise", depth_std)
    for key in data.keys():
        if ('mocap' in key):
            continue
        data[key] = data[key].cuda()
        b_size = data[key].shape[0]
        # Add the noise before so we can see the effect
        if 'zed_camera_left' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_std
        if 'realsense_camera_depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_std
        
        if key[0] in transform_dict:
            data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
    # return transformed data and the noise distribution   
    return data, torch.cat([torch.full((b_size, 1), img_std), torch.full((b_size, 1), depth_std)], dim=-1).cuda()

# Returns any arrangement of image noise between 0->img_std_max and depth noise between 0->depth_std_max
# Also returns the noise for GT supervision, when training the base model ignore this
def transform_noise(data, b_size, img_std_max = 3, depth_std_max=0, mmWave_std_max = 2):

    actual_b_size = data[('mocap', 'mocap')]['gt_positions'].shape[0]
    img_noise_scale = img_std_max * torch.rand(actual_b_size).cuda()
    depth_noise_scale = depth_std_max * torch.rand(actual_b_size).cuda()
    mmWave_noise_scale = mmWave_std_max * torch.rand(actual_b_size).cuda() # TODO CHANGE THIS BACK TO RAND
    print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item(), "mmWave Noise", mmWave_noise_scale[0])
    for key in data.keys():
        if ('mocap' in key):
            continue
        data[key] = data[key].cuda()
        # Add the noise before so we can see the effect
        if 'zed_camera_left' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:, None, None, None]
        if 'realsense_camera_depth' in key:
            data[key] = data[key] + torch.randn_like(data[key]) * depth_noise_scale[:, None, None]
        if 'range_doppler' in key:
            mins = torch.min(data[key].view(actual_b_size, -1), dim=-1)[0]
            maxs = torch.max(data[key].view(actual_b_size, -1), dim=-1)[0]
            data[key] = (data[key] - mins[:, None, None])  / maxs[:, None, None]
            data[key] = data[key] + torch.randn_like(data[key]) * mmWave_noise_scale[:, None, None]
        if key[0] in transform_dict:
            data[key] = transform_dict[key[0]](data[key]) # Apply the relevant transformation
      
    return data, torch.stack((img_noise_scale, depth_noise_scale, mmWave_noise_scale), dim=-1)


##  Depreciated
# def transform_noise_old(data, b_size, img_std_max = 3, depth_std_max=0.25):
    
#     img_noise_scale = img_std_max * torch.rand(b_size).cuda()
#     depth_noise_scale = depth_std_max * torch.rand(b_size).cuda()
#     print("Img Noise", img_noise_scale[0].item(), "Depth Noise", depth_noise_scale[0].item())
#     for key in data.keys():
#         if ('mocap' in key):
#             continue
#         data[key] = data[key].cuda()
#         if 'zed_camera_left' in key:
#             data[key] = data[key] + torch.randn_like(data[key]) * img_noise_scale[:data[key].shape[0], None, None, None]
#         if 'realsense_camera_depth' in key:
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


# Loads the items into RAM for fast training
class PickleDataset(Dataset):
    def __init__(self, file_path, valid_mods, valid_nodes):
        self.data = []
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes
        for file_name in tqdm(sorted(os.listdir(file_path)), desc="Loading pickle files into dataset"):
            curr_pickle = pickle.load(open(file_path + '/' + file_name,  'rb'))
            for key in curr_pickle:
                if isinstance(curr_pickle[key], np.ndarray):
                    curr_pickle[key] = curr_pickle[key].astype(np.float32)
            self.data.append(curr_pickle)
            # if len(self.data) == 100:
            #     break
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if 'gt_pos' in self.data[idx].keys():
            return self.data[idx]
        return {'data': self.data[idx], 'gt_pos': self.data[idx][('mocap', 'mocap')]['gt_positions']}

