import os
import pickle
import numpy as np
import torch
from pathlib import Path
root_dir = '/mnt/ssd_8t/redacted/MotiviationalStudy/cached_dataset/'
destination_dir = '/mnt/ssd_8t/redacted/Noisy_Dataset_Cache/Original/train_node1_errors/'

split = 'train/'
Path(destination_dir + split).mkdir(parents=True, exist_ok=True)
for sample in os.listdir(root_dir + split):
    print(split, sample)
    with open(root_dir + split + sample, 'rb') as handle:
        data = pickle.load(handle)
    gt_pos = torch.squeeze(data[('mocap', 'mocap')]['gt_positions'])
    if (gt_pos[0] > 130 and gt_pos[2] < 64) and (gt_pos[0] < 0 and gt_pos[2] < 0):
        if gt_pos[2] > -30:
            with open(destination_dir + split + sample, 'wb') as handle:
                pickle.dump(data, handle)
    