import os
import pickle
import numpy as np
from pathlib import Path
import cv2

root_dir = '/mnt/ssd_8t/redacted/MotiviationalStudy/cached_dataset/'
destination_dir = '/mnt/ssd_8t/redacted/Noisy_Dataset_Cache/Lowlight/'




def simulate_low_light(image, gamma=2.5, noise_std=25, contrast_alpha=0.5, contrast_beta=10, color_shift=30):
    """Simulates a low-light environment by applying gamma correction, noise, contrast reduction, and color shift."""
    
    # 1. Apply Gamma Correction (Darkening)
    inv_gamma = gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    darkened = cv2.LUT(image, table)

    # 2. Add Gaussian Noise
    noise = np.random.normal(0, noise_std, darkened.shape).astype(np.int16)
    noisy = np.clip(darkened.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 3. Reduce Contrast
    contrast_adjusted = cv2.convertScaleAbs(noisy, alpha=contrast_alpha, beta=contrast_beta)

    # 4. Apply Color Shift (Increase Blue Channel)
    b, g, r = cv2.split(contrast_adjusted)
    b = np.clip(b + color_shift, 0, 255)  # Shift blue channel
    lowlight_image = cv2.merge([b, g, r])

    return lowlight_image

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


if 'Lowlight' in destination_dir:
    # for split in ['/train/', '/test/', '/val/']:
    #     Path(destination_dir + split).mkdir(parents=True, exist_ok=True)
    #     for sample in os.listdir(root_dir + split):
    #         print(split, sample)
    #         with open(root_dir + split + sample, 'rb') as handle:
    #             data = pickle.load(handle)
            
    #         img_sat = np.random.rand() * 0.75
    #         depth_sat = np.random.rand() * 0.75
    #         new_data = {('img_sat', 'img_sat'):img_sat, ('depth_sat', 'depth_sat'):depth_sat}

    #         for key in data:
    #             if 'mocap' in key:
    #                 new_data[key] = data[key]
    #             if 'zed_camera_left' in key:
    #                 new_data[key] = (np.clip(data[key] + img_sat, 0, 1)).astype(np.float16)
    #             if 'realsense_camera_depth' in key:
    #                 new_data[key] = (np.clip(data[key] + depth_sat, 0, 1)).astype(np.float16)
    #         with open(destination_dir + split + sample, 'wb') as handle:
    #             pickle.dump(new_data, handle)
    split = '/train/'
    for sample in sorted(os.listdir(root_dir + split)):
        print(split, sample)
        with open(root_dir + split + sample, 'rb') as handle:
            data = pickle.load(handle)
        img = (data['zed_camera_left', 'node_1'] * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lowlight_img = simulate_low_light(img, gamma=5)
        
        new_data = {('img_sat', 'img_sat'):img_sat, ('depth_sat', 'depth_sat'):depth_sat}

        for key in data:
            if 'mocap' in key:
                new_data[key] = data[key]
            if 'zed_camera_left' in key:
                new_data[key] = (np.clip(data[key] + img_sat, 0, 1)).astype(np.float16)
            if 'realsense_camera_depth' in key:
                new_data[key] = (np.clip(data[key] + depth_sat, 0, 1)).astype(np.float16)