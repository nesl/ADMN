import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# OBSOLETE FILE SIMPLY USED FOR DEBUGGING

# Vision backbone with resnet
class VisionBackbone(nn.Module):
    def __init__(self):
        super(VisionBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 2) # 270 x 480 x 3 -> 133 x 238 x 16
        self.pool = nn.MaxPool2d(2, 2) # 67 x 119 x 16 I think they pad
        self.conv2 = nn.Conv2d(16, 32, 5, 2) # 32 x 57 x 32 after maxpool 15 x 29 x 32
        self.conv3 = nn.Conv2d(32, 64, 5, 2) # 6 x 13 x 64
        self.fc1 = nn.Linear(6 * 13 * 64, 1000)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(1000, 500)
        self.layerNorm1 = nn.LayerNorm((16, 66, 119))
        self.layerNorm2 = nn.LayerNorm((32, 15, 29))
        self.layerNorm3 = nn.LayerNorm((64, 6, 13))
        

    # Return 500 elements
    def forward(self, images):
        out = self.pool(self.conv1(images))
        out = self.layerNorm1(out)
        out = self.pool(self.conv2(out))
        out = self.layerNorm2(out)
        out = self.conv3(out)
        out = self.layerNorm3(out)
        out = self.GELU(self.fc1(torch.flatten(out, 1)))
        return self.fc2(out)
    
class DepthBackbone(nn.Module):
    def __init__(self):
        #120 x 160
        super(DepthBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 2) # 120 x 160 x 3 -> 58 x 78 x 16
        self.pool = nn.MaxPool2d(2, 2) # 29 x 39 x 16
        self.conv2 = nn.Conv2d(16, 32, 6, 2) # 12 x 17 x 32 after maxpool 6 x 9 x 32
        self.fc1 = nn.Linear(1536, 1000)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(1000, 500)
        self.layerNorm1 = nn.LayerNorm((16, 29, 39))
        self.layerNorm2 = nn.LayerNorm((32, 6, 8))

    def forward(self, depth):
        out = self.pool(self.conv1(depth))
        out = self.layerNorm1(out)
        out = self.pool(self.conv2(out))
        out = self.layerNorm2(out)
        out = self.GELU(self.fc1(torch.flatten(out, 1)))
        return self.fc2(out)

class AudioBackbone(nn.Module):
    def __init__(self):
        super(AudioBackbone, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, 5) # 1056 x 4 -> 1052 x 16
        self.pool4 = nn.MaxPool1d(4) #263 x 16
        self.pool2 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 6) #258 x 32 after maxpool2 129 x 32
        self.fc1 = nn.Linear(129 * 32, 1000)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(1000, 500)
        self.layerNorm1 = nn.LayerNorm((16, 263))
        self.layerNorm2 = nn.LayerNorm((32, 129))


    def forward(self, audio):
        out = self.pool4(self.conv1(audio))
        out = self.layerNorm1(out)
        out = self.pool2(self.conv2(out))
        out = self.layerNorm2(out)
        out = self.GELU(self.fc1(torch.flatten(out, 1)))
        return self.fc2(out)

class mmWaveBackbone(nn.Module):
    def __init__(self):
        super(mmWaveBackbone, self).__init__()
        #256, 16
        self.conv1 = nn.Conv2d(1, 16, 5) # 252 x 12 x 16
        self.pool = nn.MaxPool2d((2, 1), stride=(2, 1)) # 126 x 12 x 16
        self.conv2 = nn.Conv2d(16, 32, 5) # 122 x 8 x 32
        # After pool 61 x 8 x 32
        self.fc1 = nn.Linear(61 * 8 * 32, 1000)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(1000, 500)
        self.layerNorm1 = nn.LayerNorm((16, 126, 12))
        self.layerNorm2 = nn.LayerNorm((32, 61, 8))
 
    def forward(self, wave):
        out = self.pool(self.conv1(wave))
        out = self.layerNorm1(out)
        out = self.pool(self.conv2(out))
        out = self.layerNorm2(out)
        out = self.GELU(self.fc1(torch.flatten(out, 1)))
        return self.fc2(out)


