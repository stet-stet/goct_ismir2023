"""
This is the one used to supply data to evaluation codes.
"""

import os
import sys
import json
import time
import numpy as np 
import random
import h5py
from tqdm import tqdm
from .beatencoder import NBeatEncoder, TwoTwoEncoder, WholeSongBeatEncoder

def round_to_nearest_48th(num):
    return round(num*48)/48

class AudioChartTwoBeatSongDataset():
    """
    usage of bs > 1 with this dataset will yield an error.
    """
    def __init__(self, split_json_path="train.json", split_hdf5_path="train.h5", 
                 logspec=True, use_notecount_as_finediff=False) -> None :
        self.split_json_path = split_json_path
        self.split_hdf5_path = split_hdf5_path
        with open(split_json_path) as file:
            self.split_dict = json.loads(file.read())
        self.hdf5 = h5py.File(split_hdf5_path,'r') 
        self.dir_of_dirs = os.path.dirname(split_json_path)

        self.total_length_in_songs = len(self.split_dict.values())

        self.index = []
        for osujson, (bar_len,beat_len) in self.split_dict.items():
            self.index.extend([(osujson,0,beat_len)])

        assert len(self.index) == self.total_length_in_songs

        self.beat_encoder = TwoTwoEncoder()
        self.pad = 177
        self.length_in_beats = 4
        self.token_length = 200
        self.truncate = 92
        self.logspec = logspec
        self.use_notecount_as_finediff = use_notecount_as_finediff

    def __len__(self) -> int:
        return self.total_length_in_songs
    
    def __getitem__(self,idx,benchmark=False) -> dict:
        """
        List of stuff to return:
            - the music for this song, made into a melspectrogram with hop size = (beat) / 48 (loaded from a h5)
            - the encoded tokens for the whole song, with a beat_token every two beats
            - fine difficulty
            - the encoded token for the first two beats
        """
        if benchmark: tp = time.time()

        if idx > self.total_length_in_songs:
            raise ValueError("idx over total length")
        osujson_fn, beat, length_in_beats = self.index[idx] # beat = 0
        beatmapset_path = os.path.dirname(osujson_fn)
        with open(osujson_fn) as file: 
            osujson = json.load(file)
        songname = f'{osujson["artist"]} - {osujson["title"]}, {osujson["charts"][0]["difficulty_fine"]}'

        if benchmark: print(f"json read: {time.time() - tp}")

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

        # load melspec from hdf5
        hdf5_dset_name = osujson_fn.replace(self.dir_of_dirs+'/','')
        mel = self.hdf5[hdf5_dset_name]
        mel_to_return = mel[:,:].T # (80, LENGTH)
        if self.logspec:
            mel_to_return = np.log10(mel_to_return+1e-10)
        if mel_to_return.shape[1] % 96 != 0:
            how_many_to_pad = 96 - mel_to_return.shape[1] % 96
            mel_to_return = np.pad(mel_to_return, ((0,0),(0,how_many_to_pad)), 'constant', constant_values=0)

        if benchmark: print(f"reading melspectrogram: {time.time() - tp}")

        #make tokens for all
        notes = osujson['charts'][0]['notes']
        # make charts for whole chart for evalutaion.
        whole_chart_iterable = []
        for beat in range(0,length_in_beats-2,2):
            nextbeat = beat + self.length_in_beats
            notes = list(filter(lambda x:(x[1] >= beat and x[1] < nextbeat), osujson['charts'][0]['notes'])) # TODO: make this configurable?
            temp_enc_notes = self.beat_encoder.encode(notes,beat)
            where_is_96 = temp_enc_notes[1:].index(96) + 1
            chart_for_this_section = np.array(temp_enc_notes[where_is_96+1:])
            right_pad = self.token_length // 2 - len(chart_for_this_section)
            chart_for_this_section = np.pad(chart_for_this_section, (0, right_pad), 'constant', constant_values=(self.pad, self.pad))
            whole_chart_iterable.append(chart_for_this_section)
        whole_chart_iterable = np.array(whole_chart_iterable)
        if benchmark: print(f"tokenmaking: {time.time() - tp}")

        return mel_to_return, whole_chart_iterable, fine_difficulty, songname

def stress_test(split="train"): 
    print(f"=============================== split = {split}")
    split_json_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.json"
    split_hdf5_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.h5"
    dset = AudioChartTwoBeatSongDataset(split_json_path=split_json_path, split_hdf5_path=split_hdf5_path)
    print(len(dset))
    random_indices = random.sample(list(range(len(dset))),5000)
    start_time = time.time()
    #for i in tqdm(random_indices):
        #dset[i]
    print(dset[0][0].shape)
    print(dset[30]) # wait, why is there a minus mixed in?
    end_time = time.time()
    print(end_time-start_time) #five hours. wow, that's ridiculous

def size_test(split="train"):
    print(f"=============================== split = {split}")
    split_json_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.json"
    split_hdf5_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.h5"
    dset = AudioChartTwoBeatSongDataset(split_json_path=split_json_path, split_hdf5_path=split_hdf5_path)
    print(len(dset))
    sz = len(dset)
    for i in tqdm(range(sz)):
        assert dset[i][0].shape == (80,192)

def print_one(split="train",index=0, **params):
    print(f"=============================== split = {split}")
    split_json_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.json"
    split_hdf5_path = f"/mnt/c/Users/manym/Desktop/SNU2023appendix/beatmap/2026_ddc_rere/{split}.h5"

    dset =AudioChartTwoBeatSongDataset(split_json_path=split_json_path, split_hdf5_path=split_hdf5_path, **params)
    print(dset[index])

if __name__=="__main__":
    # for split in ['valid']:
    #    stress_test(split)
    print_one()

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