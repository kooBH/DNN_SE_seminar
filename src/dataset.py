"""
    On-Fly Dataset
"""

import os, glob
import torch
import librosa
import numpy as np

class DatasetMix(torch.utils.data.Dataset):
    def __init__(self, dir_clean,path_noise,sec=1.0):
        self.list_clean = glob.glob(os.path.join(dir_clean,"*.wav"))

        self.noise,_ = librosa.load(path_noise,sr=16000)

        self.len_item = int(sec*16000)

        print("Dataset:: {} clean data from {} | noise : {}  ".format(len(self.list_clean),dir_clean,self.noise.shape))

    def __getitem__(self, idx):

        # clean matching
        path_clean = self.list_clean[idx]
        tmp_clean,_ = librosa.load(path_clean,sr=16000)

        if len(tmp_clean) > self.len_item :
            idx_clean = np.random.randint(len(tmp_clean)-self.len_item)
            tmp_clean = tmp_clean[idx_clean:idx_clean + self.len_item]
        elif len(tmp_clean) < self.len_item :
            short = self.len_item - len(tmp_clean)
            tmp_clean =  np.pad(tmp_clean, (0,short))            
        else :
            pass
        
        # noise sampling
        idx_noise = np.random.randint(len(self.noise)-self.len_item)
        tmp_noise = self.noise[idx_noise:idx_noise+self.len_item]


        # SNR
        SNR = np.random.rand()*10
        energy_clean = np.sum(np.power(tmp_clean,2))
        energy_noise = np.sum(np.power(tmp_noise,2))

        normal = np.sqrt(energy_clean)/np.sqrt(energy_noise)
        weight = normal/np.sqrt(np.power(10,SNR/10))

        tmp_noise *=weight

        # Mixing
        noisy = tmp_clean + tmp_noise

        # Normalization
        scaling_factor = np.max(np.abs(noisy))

        noisy = noisy/scaling_factor
        clean = tmp_clean/scaling_factor

        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        noisy_spec =  torch.stft(noisy,n_fft=512,return_complex=True,center=True)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)

        #noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
        #noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

        data = {}

        data["clean_wav"] = clean
        data["noisy_mag"] = noisy_mag
        data["noisy_phase"] = noisy_phase

        return data

    def __len__(self):
        return len(self.list_clean)

class DatasetFix(torch.utils.data.Dataset):
    def __init__(self, dir_dataset):
        self.list_clean = glob.glob(os.path.join(dir_dataset,"clean","*.wav"))

        self.dir_dataset = dir_dataset

        print("Dataset:: {} clean data from {} | noise : {}  ".format(len(self.list_clean),dir_dataset))

    def __getitem__(self, idx):

        # clean matching
        path_clean = self.list_clean[idx]
        clean,_ = librosa.load(path_clean,sr=16000)

        name_clean = path_clean.split("/")[-1]

        noise,_ = librosa.load(os.path.join(self.dir_dataset,"noisy",name_clean),sr=16000)
        
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        noisy_spec =  torch.stft(noisy,n_fft=512,return_complex=True,center=True)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)

        #noisy_mag = torch.unsqueeze(noisy_mag,dim=0)
        #noisy_phase = torch.unsqueeze(noisy_phase,dim=0)

        data = {}

        data["clean_wav"] = clean
        data["noisy_mag"] = noisy_mag
        data["noisy_phase"] = noisy_phase

        return data

    def __len__(self):
        return len(self.list_clean)



