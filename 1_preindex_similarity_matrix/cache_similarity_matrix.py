import os 
import sys
import json 
import copy
import scipy
import numpy as np
import torch
import soundfile as sf
import h5py
from tqdm import tqdm
from make_similarity_matrix import make_beatwise_similarity_matrix_and_beat_to_sample_map

def cache_similarity_matrix(split_file, h5_output):
    """
    split_file: json file with filename(osujson): [bar_length beat_length] pairs
    """
    with open(split_file) as file:
        d = json.load(file)
    dest_path = os.path.dirname(split_file) + '/'
    print(dest_path)
    with h5py.File(h5_output,'w') as h5f:
        key_list = list(d.keys())
        pbar = tqdm(key_list)
        for k in pbar:
            h5_key = k.replace(dest_path, '')
            sim_matrix, _ = make_beatwise_similarity_matrix_and_beat_to_sample_map(k)
            h5f.create_dataset(h5_key, data=sim_matrix)
            pbar.set_description(str(sim_matrix.shape))
    
if __name__=="__main__":
    cache_similarity_matrix('OSUFOLDER/train.json', 'train_m.h5')
    cache_similarity_matrix('OSUFOLDER/valid.json', 'valid_m.h5')
    cache_similarity_matrix('OSUFOLDER/test.json', 'test_m.h5')
