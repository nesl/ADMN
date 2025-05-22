import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from torchvision import transforms, io
import re
import torchaudio
import pickle

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


image_transform = transforms.Compose([
        ClampImg(),
        transforms.Resize((224, 224)),       
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])




class AVE_Dataset(Dataset):
    def __init__(self, dataset_root, dataset_txt_file, dataset_mean=-4.2677393, dataset_std=4.5689974, noise=False):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_txt_file
        self.dataset_root = dataset_root
        with open(dataset_root + dataset_txt_file, 'r') as fp:
            self.data_info = fp.readlines()
        self.data_info = self.data_info[1:]
        self.class_to_idx = {'Church bell': 0, 'Male speech, man speaking': 1, 'Bark': 2, 
                             'Fixed-wing aircraft, airplane': 3, 'Race car, auto racing': 4, ''
                             'Female speech, woman speaking': 5, 'Helicopter': 6, 'Violin, fiddle': 7, 
                             'Flute': 8, 'Ukulele': 9, 'Frying (food)': 10, 'Truck': 11, 'Shofar': 12, 
                             'Motorcycle': 13, 'Acoustic guitar': 14, 'Train horn': 15, 'Clock': 16, 
                             'Banjo': 17, 'Goat': 18, 'Baby cry, infant cry': 19, 'Bus': 20, 
                             'Chainsaw': 21, 'Cat': 22, 'Horse': 23, 'Toilet flush': 24, 
                             'Rodents, rats, mice': 25, 'Accordion': 26, 'Mandolin': 27}
        self.data_mean = dataset_mean
        self.data_std = dataset_std


        self.new_data = []
        for i in trange(len(self.data_info)):
            value = self.data_info[i]
            file_name = value.split('&')[1]
            self.new_data.append( (self.process(value, 
                                                audio_path=self.dataset_root + '/New_Data/' + file_name + '.wav',
                                                image_path=self.dataset_root + '/New_Data/' + file_name + '/'), 0) )
            self.new_data.append( (self.process(value, 
                                                audio_path=self.dataset_root + '/Noised_Audio/' + file_name + '_Noised.wav',
                                                image_path=self.dataset_root + '/New_Data/' + file_name + '/'), 1))
            self.new_data.append( (self.process(value, 
                                                audio_path=self.dataset_root + '/New_Data/' + file_name + '.wav',
                                                image_path=self.dataset_root + '/Noised_Images/' + file_name + '/'), 2))
            self.new_data.append( (self.process(value, 
                                                audio_path=self.dataset_root + '/Noised_Audio/' + file_name + '_Noised.wav',
                                                image_path=self.dataset_root + '/Noised_Images/' + file_name + '/'), 3))
        
       
        

    # No mixup used
    def _wav2fbank(self, filename):
        
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

       
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        # 1024 for AS
        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]


        return fbank
  

    def _fbank(self, filename):

        fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
        fbank = np.load(fn1)
        return torch.from_numpy(fbank)
      

    def process(self, datum, audio_path, image_path):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        
        # Split to get the data info
        str_label, file_name, data_qual, t_start, t_end = datum.split('&')
        label_idx = self.class_to_idx[str_label]
        

        # Get the fbank audio
        fbank = self._wav2fbank(audio_path)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        fbank = (fbank - self.data_mean) / (self.data_std * 2)

        # Get the sequence of images
        image_data = []
        for idx in range(1, 9): # Png files are labeled from 1-8
            curr_img = io.read_image(image_path + str(idx) + '.png') / 255
            curr_img = image_transform(curr_img)
            image_data.append(torch.unsqueeze(curr_img, dim=0))
        image_data = torch.cat(image_data, dim=0) # tensor of 8 x 3 x 224 x 224

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0), image_data, label_idx
    
    def __getitem__(self, idx):
        return self.new_data[idx]

    def __len__(self):
        return len(self.new_data)
    

class PickleDataset:
    def __init__(self, data_root, type='train', valid_noise_types=[0, 1, 2, 3]):
        folder = data_root + '/' + type
        self.data = []
        folder_files = os.listdir(folder)
        for i in trange(len(folder_files)): # Temp fix, iterate through based on index and then 
            with open(folder + '/' + folder_files[i], 'rb') as handle:
                curr_data = pickle.load(handle)
                if curr_data[1] in valid_noise_types:
                    self.data.append(curr_data)
            # if len(self.data) == 200:
            #     break

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        



if __name__ == '__main__':
    dataset = AVE_Dataset('/mnt/ssd_8t/redacted/AVE_Dataset/', dataset_txt_file = 'Annotations.txt', dataset_mean=-4.2677393, dataset_std=4.5689974)
    import pdb; pdb.set_trace()
