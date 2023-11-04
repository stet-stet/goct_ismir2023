
import os
import sys
import json
import time
import numpy as np 
import random
import h5py
from tqdm import tqdm

def round_to_nearest_48th(num):
    return round(num*48)/48

class DDCDataset():
    def __init__(self, split_json_path="train.json", onset_hdf5_path="all.h5", music_hdf5_path="all_onset.h5",
                 min_diff = 0.0, max_diff = 100.0, overlaps=True) -> None:
        self.split_json_path = split_json_path
        self.onset_hdf5_path = onset_hdf5_path
        self.music_hdf5_path = music_hdf5_path
        self.biggest_dir = os.path.dirname(split_json_path)
        with open(split_json_path) as file:
            self.split_dict = json.loads(file.read())
        self.music_hdf5 = h5py.File(music_hdf5_path, 'r') 
        self.onset_hdf5 = h5py.File(onset_hdf5_path, 'r')
        self.dir_of_dirs = os.path.dirname(split_json_path)

        self.index = []

        self.frames = 112 # non-configurable.
        for osujson, _ in self.split_dict.items():
            diff = self._get_diff(osujson)
            if min_diff > diff or max_diff < diff: continue
            d = self.onset_hdf5[self._dset_name_for_onset(osujson)][:]
            start_ind = np.argmax(d == True)
            start_ind = max(start_ind - self.frames, 0) 
            end_ind = len(d) - np.argmax(d[::-1] == True) - 1
            end_ind = min(end_ind, len(d) - self.frames)
            if overlaps:
                self.index.extend([(osujson,meter) for meter in range(start_ind, end_ind+1)])
            else:
                offset = random.randint(0, self.frames)
                self.index.extend([(osujson,meter) for meter in range(start_ind + offset, end_ind+1, self.frames)])

        self.total_length = len(self.index)

    def _get_diff(self,osujson_fn):
        with open(osujson_fn) as file:
            a = json.load(file)
        return float(a['charts'][0]["difficulty_fine"])

    def __len__(self):
        return self.total_length

    def _dset_name_for_audio(self, audiopath):
        # "foldername".
        ret = str(os.path.dirname(audiopath))
        ret = ret.replace(str(self.biggest_dir), "")
        if ret[0] == '/': ret= ret[1:]
        return ret
    
    def _dset_name_for_onset(self, osujsonpath):
        ret = str(osujsonpath).replace(self.biggest_dir,'')
        if ret[0] == '/':
            ret= ret[1:]
        return ret    

    def __getitem__(self,idx):
        """
        what to return:
            - the music for this beats
            - the onsets i.e. the goals
            - fine diff
        """
        # TODO randomize
        if idx > self.total_length:
            raise ValueError("idx over total length")
        osujson_fn, frame = self.index[idx]
        with open(osujson_fn) as file: 
            osujson = json.load(file)

        # fine diff
        try:
            fine_difficulty = float(osujson['charts'][0]['difficulty_fine'])    
        except Exception as e:
            print(osujson_fn)
            raise e

        music_dset_name = self._dset_name_for_audio(osujson_fn) # nasty, but marginally acceptable.
        mel = self.music_hdf5[music_dset_name][frame:frame+self.frames, :, :] # (112, 80, 3)
        mel_to_return = mel
        #mel_to_return = mel[:].transpose((2,0,1)) # (3, 80, 112)

        # import onsets

        onset_dset_name = self._dset_name_for_onset(osujson_fn)
        onsets = self.onset_hdf5[onset_dset_name][frame:frame+self.frames]

        return mel_to_return, onsets, fine_difficulty

def size_test(split="train",**params):
    print(f"=============================== split = {split}")
    split_json_path = f"OSUFOLDER/{split}.json"
    music_hdf5_path = f"OSUFOLDER/all_ddc.h5"
    onset_hdf5_path = f"OSUFOLDER/all_onset.h5"

    dset = DDCDataset(split_json_path=split_json_path, music_hdf5_path=music_hdf5_path, onset_hdf5_path=onset_hdf5_path, **params)
    sz = len(dset)
    print(dset[0])
    for i in tqdm(range(sz)):
        dset[i]


if __name__=="__main__":
    size_test()