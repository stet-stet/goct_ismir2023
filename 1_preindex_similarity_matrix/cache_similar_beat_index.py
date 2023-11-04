import os 
import sys
import argparse
import json
from tqdm import tqdm 
import numpy as np
from make_similarity_matrix import make_beatwise_similarity_matrix_and_beat_to_sample_map

def folder_is_sane(folder):
    audiofiles = [e for e in os.listdir(folder) if not (e.endswith(".osu") or e.endswith(".json"))]
    return len(audiofiles)==1

def append_similar_beat_index(folder):
    # assert folder_is_sane(folder) <- not necessarily true for DDC.
    #we make index for all charts there is.
    contents = os.listdir(folder)
    ddc_jsons = [os.path.join(folder,e) for e in contents if e.endswith('.osu.json')]
    output_filenames = [f"{e}.beat.json" for e in ddc_jsons]

    for ddc_json,output_file in zip(ddc_jsons, output_filenames):
        sim_matrix, beat_to_sample_map = make_beatwise_similarity_matrix_and_beat_to_sample_map(ddc_json)
        length = sim_matrix.shape[0] # == sim_matrix.shape[1]
        ret = {"beat_to_sample_map":beat_to_sample_map, "use_bars":None}
        the_index = {}
        for row_id in range(length):
            top5_ind = [int(e) for e in np.argpartition(sim_matrix[row_id], -5)[-5:]]
            if row_id not in top5_ind:
                top5_ind.append(row_id)
            top5_ind = [e for e in top5_ind if e<=row_id]
            the_index.update({int(row_id): top5_ind})
        ret['use_bars']=the_index
        with open(output_file,'w') as file:
            file.write(json.dumps(ret,indent=4))


def do(folder_of_folders):
    assert os.path.isdir(folder_of_folders)
    a = os.listdir(folder_of_folders)
    folders = [os.path.join(folder_of_folders,e) for e in a if os.path.isdir(os.path.join(folder_of_folders,e))]
    pbar = tqdm(folders, position=0, leave=True)
    for folder in pbar:
        pbar.set_description(str(folder))
        try:
            append_similar_beat_index(folder)
        except Exception as e:
            print('')
            print(e)
            print(f"{folder} is not a typical osu folder, or I ran out of RAM.")
    
if __name__=="__main__":
    do(sys.argv[1])