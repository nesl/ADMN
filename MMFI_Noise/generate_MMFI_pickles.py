import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
img_root = '/mnt/ssd_8t/redacted/all_images/MMFi_Defaced_RGB/'
depth_root = '/home/redacted/Documents/Backups/MMFI_Dataset_backup/'
environments = ['E01', 'E02', 'E03', 'E04']
destination = '/mnt/ssd_8t/redacted/MMFI_Pickles_Img_DepthColorized/'

os.makedirs(destination, exist_ok=True)

for environment in tqdm(environments):
    for subject in sorted(os.listdir(img_root + '/' + environment)):
        for action in sorted(os.listdir(img_root + '/' + environment + '/' + subject)):
            print(environment, subject, action)
            img_path = img_root + '/' + environment + '/' + subject + '/' + action + '/rgb/'
            os.makedirs(destination + '/' + environment + '/' + subject + '/' + action, exist_ok=True)
            image_list = []
            img_files = sorted(os.listdir(img_path))
            for img in img_files:
                img_read = Image.open(img_path + img)
                img_resized = img_read.resize((224, 224))
                img_resized = np.array(img_resized)
                img_resized = np.transpose(img_resized, (2, 0, 1))
                image_list.append(img_resized)

            depth_path = depth_root + '/' + environment + '/' + subject + '/' + action + '/depthColorized/'
            depth_list = []
            depth_files = sorted(os.listdir(depth_path))
            for img in depth_files:
                img_read = Image.open(depth_path + img)
                img_resized = img_read.resize((224, 224))
                img_resized = np.array(img_resized)
                img_resized = np.transpose(img_resized, (2, 0, 1))
                depth_list.append(img_resized)
            img_data = np.stack(image_list, axis=0)
            depth_data = np.stack(depth_list, axis=0)
            final_dict = {
                'rgb': img_data,
                'depth': depth_data
            }
            with open(destination + '/' + environment + '/' + subject + '/' + action + '/data.pickle', 'wb') as handle:
                pickle.dump(final_dict, handle)



                
