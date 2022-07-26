import torch
import numpy as np
import librosa
import soundfile as sf
import os

from glob import glob
from tqdm import tqdm


sec = 1.0
len_item = int(sec*16000)

def gen(idx):
    # clean matching

    path_clean = list_clean[idx]
    tmp_clean,_ = librosa.load(path_clean,sr=16000)

    if len(tmp_clean) > len_item :
        idx_clean = np.random.randint(len(tmp_clean)-len_item)
        tmp_clean = tmp_clean[idx_clean:idx_clean + len_item]
    elif len(tmp_clean) < len_item :
        short = len_item - len(tmp_clean)
        tmp_clean =  np.pad(tmp_clean, (0,short))            
    else :
        pass
    
    # noise sampling
    idx_noise = np.random.randint(len(noise)-len_item)
    tmp_noise = noise[idx_noise:idx_noise+len_item]

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

    sf.write(os.path.join(dir_out,"noisy","{}.wav".format(idx)),noisy,16000)
    sf.write(os.path.join(dir_out,"clean","{}.wav".format(idx)),clean,16000)


if __name__ == "__main__": 

    dir_out = "dev"
    os.makedirs(os.path.join(dir_out,"noisy"),exist_ok=True)
    os.makedirs(os.path.join(dir_out,"clean"),exist_ok=True)

    list_clean = [x for x in glob(os.path.join("clean_dev","*.wav"))]
    noise,_ = librosa.load("noise_dev.wav",16000)

    for i in tqdm(range(len(list_clean))) : 
        gen(i)

    dir_out = "eval"
    os.makedirs(os.path.join(dir_out,"noisy"),exist_ok=True)
    os.makedirs(os.path.join(dir_out,"clean"),exist_ok=True)

    list_clean = [x for x in glob(os.path.join("clean_eval","*.wav"))]
    noise,_ = librosa.load("noise_eval.wav",16000)

    for i in tqdm(range(len(list_clean))) : 
        gen(i)
