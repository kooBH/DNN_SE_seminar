
import os
from glob import glob
import numpy as np
import soundfile as sf
from tqdm import tqdm


# train 
#dir_in = "/home/nas/DB/AI_HUB_speech/KsponSpeech_01/KsponSpeech_0001"
#dir_out = "/home/data2/kbh/LG_seminar/clean_train"

# dev
#dir_in = "/home/nas/DB/AI_HUB_speech/KsponSpeech_01/KsponSpeech_0002"
#dir_out = "/home/data2/kbh/LG_seminar/clean_dev"

# eval
dir_in = "/home/nas/DB/AI_HUB_speech/KsponSpeech_01/KsponSpeech_0003"
dir_out = "/home/data2/kbh/LG_seminar/clean_eval"

list_target = glob(os.path.join(dir_in,"*.pcm"))

def convert(idx):
    path = list_target[idx]
    x = np.fromfile(path, dtype=np.int16)
    sf.write(os.path.join(dir_out,str(idx)+".wav"),x,16000)

os.makedirs(dir_out,exist_ok=True)

#for i in tqdm(range(len(list_target))):
#    convert(i)
for i in tqdm(range(100)):
    convert(i)
 

