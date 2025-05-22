import os
import pickle
import numpy as np
from pathlib import Path
root_dir = '/mnt/ssd_8t/jason/MotiviationalStudy/cached_dataset/'
destination_dir = '/mnt/ssd_8t/jason/Noisy_Dataset_Cache/AWGN/'

if 'AWGN' in destination_dir:
    for split in ['/train/', '/test/', '/val/']:
        Path(destination_dir + split).mkdir(parents=True, exist_ok=True)
        for sample in os.listdir(root_dir + split):
            print(split, sample)
            with open(root_dir + split + sample, 'rb') as handle:
                data = pickle.load(handle)
            
            img_scale = np.random.rand() * 2.5
            depth_scale = np.random.rand() * 0.25
            new_data = {('img_std', 'img_std'):img_scale, ('depth_std', 'depth_std'):depth_scale}
            img_noise = np.random.randn(*data[('zed_camera_left', 'node_1')].shape) * img_scale
            depth_noise = np.random.randn(*data[('realsense_camera_depth', 'node_1')].shape) * depth_scale

            for key in data:
                if 'mocap' in key:
                    new_data[key] = data[key]
                if 'zed_camera_left' in key:
                    new_data[key] = (data[key] + img_noise).astype(np.float16)
                if 'realsense_camera_depth' in key:
                    new_data[key] = (data[key] + depth_noise).astype(np.float16)
            with open(destination_dir + split + sample, 'wb') as handle:
                pickle.dump(new_data, handle)
        
if 'Saturation' in destination_dir:
    for split in ['/train/', '/test/', '/val/']:
        Path(destination_dir + split).mkdir(parents=True, exist_ok=True)
        for sample in os.listdir(root_dir + split):
            print(split, sample)
            with open(root_dir + split + sample, 'rb') as handle:
                data = pickle.load(handle)
            
            img_sat = np.random.rand() * 0.75
            depth_sat = np.random.rand() * 0.75
            new_data = {('img_sat', 'img_sat'):img_sat, ('depth_sat', 'depth_sat'):depth_sat}

            for key in data:
                if 'mocap' in key:
                    new_data[key] = data[key]
                if 'zed_camera_left' in key:
                    new_data[key] = (np.clip(data[key] + img_sat, 0, 1)).astype(np.float16)
                if 'realsense_camera_depth' in key:
                    new_data[key] = (np.clip(data[key] + depth_sat, 0, 1)).astype(np.float16)
            with open(destination_dir + split + sample, 'wb') as handle:
                pickle.dump(new_data, handle)

            

