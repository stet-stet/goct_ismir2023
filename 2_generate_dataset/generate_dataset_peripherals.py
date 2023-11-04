"""
given dataset folder, generates the following:
- the list of all files with an ".osu.json.beat.json" extension, a.k.a. ones to be used for training
- the length (in beats) of each file
- the total number of existing files

This information will be used to init the DataLoader.

all information is saved in -> $dir / summary.json

"""
import os 
import sys
import json
from tqdm import tqdm
import random

def get_length_in_bars(osujson_filename):
    with open(osujson_filename,'r') as file:
        d = json.loads(file.read())
    charts = d['charts'][0]['notes']
    length_in_bars = charts[-1][0][0]
    return length_in_bars

def get_length_in_beats(osujsonbeatjson_filename):
    with open(osujsonbeatjson_filename,'r') as file:
        d = json.loads(file.read())
    final_beat = d['beat_to_sample_map'][-1][0]
    length_in_beats = round(final_beat)
    return length_in_beats

def do(dir_of_dirs, make_splits=True):
    random.seed(10101010)
    output_file = os.path.join(dir_of_dirs,'summary.json')
    all_files = {}
    # use first 1500 for training, all else for validation.
    for root, dirs, files in tqdm(os.walk(dir_of_dirs)):
        for file in files:
            if file.endswith(".osu.json"):
                file_fullpath = os.path.join(root,file)
                length_in_bars = get_length_in_bars(file_fullpath)
                length_in_beats = get_length_in_beats(f"{file_fullpath}.beat.json")
                all_files.update({file_fullpath: (length_in_bars, length_in_beats)})
    with open(output_file,'w') as file:
        file.write(json.dumps(all_files,indent=4)) 
    # split train-valid-test 8:1:1 (approx)
    all_paths = list( set( [os.path.dirname(e) for e in all_files.keys()] ) )
    random.shuffle(all_paths)

    if not make_splits: 
        return 

    split_names = ['train','test','valid']
    split_slices = [(0,1600),(1600,-200),(-200,len(all_paths))]
    for split_name, (a,b) in zip(split_names,split_slices):
        split_output = os.path.join(dir_of_dirs, f'{split_name}.json')
        split = {k:v for k,v in all_files.items() if os.path.dirname(k) in all_paths[a:b]}
        with open(split_output,'w') as file:
            file.write(json.dumps(split, indent=4))


if __name__=="__main__":
    do(sys.argv[1])