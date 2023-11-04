"""
the argument to feed into torch.util.data.DataLoader
"""

import os
import sys
import json
import time
import numpy as np 
import random
import h5py
from tqdm import tqdm
from .beatencoder import NBeatEncoder, TwoTwoEncoder

def round_to_nearest_48th(num):
    return round(num*48)/48

class AudioChartFineBeatDataset():
    def __init__(self, split_json_path="train.json", split_hdf5_path="train.h5", encoder="twotwo", 
                 length_in_beats=4, token_length=100, beat_upper_limit=20000, drop_first_two_rate=0.0, 
                 logspec=False, center=False, truncate=None, use_notecount_as_finediff=False,
                 debug=False) -> None :
        self.split_json_path = split_json_path
        self.split_hdf5_path = split_hdf5_path
        with open(split_json_path) as file:
            self.split_dict = json.loads(file.read())
        self.hdf5 = h5py.File(split_hdf5_path,'r') 
        self.dir_of_dirs = os.path.dirname(split_json_path)

        self.total_length_in_beats = sum([min(beat_len,beat_upper_limit) for bar_len, beat_len in self.split_dict.values()])

        self.index = []
        for osujson, (bar_len,beat_len) in self.split_dict.items():
            self.index.extend([(osujson,beat) for beat in range(min(beat_len,beat_upper_limit))])

        assert len(self.index) == self.total_length_in_beats

        self.beat_encoder = TwoTwoEncoder()
        self.pad = 177
        assert length_in_beats == 4
            
        self.length_in_beats = length_in_beats
        self.token_length = token_length
        self.drop_first_two_rate = drop_first_two_rate
        self.logspec = logspec
        self.center = center
        self.truncate = truncate
        self.use_notecount_as_finediff = use_notecount_as_finediff
        self.debug = debug

        if self.center is False and truncate is not None:
            raise ValueError("truncate is only usable if self.center is set.")
        if truncate is None:
            self.truncate = 0
        if self.truncate > self.token_length//2 or self.truncate < 0:
            raise ValueError("self.truncate must be a non-negative smaller than half of token length")

    def __len__(self) -> int:
        return self.total_length_in_beats
    
    def __getitem__(self,idx,benchmark=False) -> dict:
        """
        List of stuff to return:
            - the music for this bar, made into a melspectrogram with hop size = (beat) / 48 (loaded from a h5)
            - the encoded tokens for this _ beats
            - fine difficulty
        """
        if benchmark: tp = time.time()

        dropmode = (random.random() < self.drop_first_two_rate)

        if idx > self.total_length_in_beats:
            raise ValueError("idx over total length")
        osujson_fn, beat = self.index[idx]
        with open(osujson_fn) as file: 
            osujson = json.load(file)

        if benchmark: print(f"json read: {time.time() - tp}")
        if self.debug: print(self.index[idx])
        # fine diff
        try:
            if self.use_notecount_as_finediff:
                fine_difficulty = len(osujson['charts'][0]['notes'])
            else:
                fine_difficulty = float(osujson['charts'][0]['difficulty_fine'])    
                
        except Exception as e:
            print(osujson_fn)
            raise e

        assert beat == int(beat) # see directory 0_filtration
        nextbeat = beat + self.length_in_beats

        # load melspec from hdf5
        hdf5_dset_name = osujson_fn.replace(self.dir_of_dirs+'/','')
        mel = self.hdf5[hdf5_dset_name]
        mel_to_return = mel[48*round(beat):48*round(nextbeat),:].T # (80, 48*beat)
        if self.logspec:
            mel_to_return = np.log10(mel_to_return+1e-10)
        if mel_to_return.shape[1] < 192:
            mel_to_return = np.pad(mel_to_return, ((0,0),(0, 192 - mel_to_return.shape[1])), 'constant', constant_values=0)
        # load audio and account for offset

        if benchmark: print(f"reading melspectrogram: {time.time() - tp}")

        #make tokens
        notes = osujson['charts'][0]['notes']
        if not dropmode:
            notes = list(filter(lambda x:(x[1] >= beat and x[1] < nextbeat), notes))
        else:
            notes = list(filter(lambda x:(x[1] >= beat+2 and x[1] < nextbeat), notes)) # TODO: make this configurable?
        encoded_notes = self.beat_encoder.encode(notes,beat)
        where_is_96 = encoded_notes[1:].index(96) + 1
        where_should_96_be = self.token_length // 2
        encoded_notes = np.array(encoded_notes)
        if not self.center:
            where_should_96_be = where_is_96

        try:
            left_pad = where_should_96_be - where_is_96 
            right_pad = self.token_length - len(encoded_notes) - left_pad
            encoded_notes = np.pad(encoded_notes, (left_pad, right_pad), 'constant', constant_values=(self.pad,self.pad))
        except ValueError as e:
            self.bar_encoder.pretty_print(encoded_notes)
            print(e)
            raise ValueError(e)
        if benchmark: print(f"tokenmaking: {time.time() - tp}")

        beat_token_index = int(np.where(encoded_notes == 96)[0][1].item())
        mask = [0. for i in range(beat_token_index) ] + [1.for i in range(self.token_length - beat_token_index)]
        mask = np.array(mask)

        # truncation for centered
        if self.center:
            mask = mask[self.truncate:]
            encoded_notes = encoded_notes[self.truncate:]

        #print(mel_to_return.shape, encoded_notes.shape, fine_difficulty, mask.shape)
        return mel_to_return, encoded_notes, mask, fine_difficulty

def stress_test(split="train", **params): 
    print(f"=============================== split = {split}")
    split_json_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.json"
    split_hdf5_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.h5"
    dset = AudioChartFineBeatDataset(split_json_path=split_json_path, split_hdf5_path=split_hdf5_path, **params)
    print(len(dset))
    random_indices = random.sample(list(range(len(dset))),5000)
    start_time = time.time()
    #for i in tqdm(random_indices):
        #dset[i]
    print(dset[30]) # wait, why is there a minus mixed in?
    end_time = time.time()
    print(end_time-start_time) #five hours. wow, that's ridiculous

def size_test(split="train",**params):
    print(f"=============================== split = {split}")
    split_json_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.json"
    split_hdf5_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.h5"

    dset = AudioChartFineBeatDataset(split_json_path=split_json_path, split_hdf5_path=split_hdf5_path, **params)
    sz = len(dset)
    for i in tqdm(range(sz)):
        dset[i]
        
if __name__=="__main__":
    # for split in ['valid']:
    #    stress_test(split
    size_test(center=True, token_length=200, truncate=94)
    # stress_test(center=True)

""" XXX. 
benchmarks.

## mels: (80, T), windows

random, 5000 accesses:
- train (542966 samples): 148.2
- valid (73915 samples): 102.4
- test (78460 samples): 107.3

sequential, 20000 accesses:
- train (542966 samples): 343.1
- valid (73915 samples): 334.4
- test (78460 samples): 353.2

## mels: (T, 80), windows

random: 5000 accesses:
- train_transpose (560928 samples): 21.9
- valid_transpose (63999 samples): 16.4
- test_transpose (63595 samples): 16.7

## mels: (T, 80), wsl home dir

random: 5000 accesses:
- train_transpose (560928 samples): too scared to try
- valid_transpose (63999 samples): 14.7 
- test_transpose (63595 samples): 15.2

"""