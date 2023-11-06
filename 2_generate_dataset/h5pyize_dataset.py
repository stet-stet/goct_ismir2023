"""
make every song into a melspectrogram with 80 bins, and then packs them into one .h5 file
care is taken so that access time can be optimized.
(turns out transposing the mel before saving is about 10x faster, so we're sticking with it (see below).)

written by Jayeon Yi (stet-stet)
"""
import h5py
import os
import sys
import scipy
import json
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa


def make_mel_for_whole_file(osujson, fft_size=512, n_mels=80):
    beatjson = osujson + ".beat.json"
    osujson = beatjson.replace(".osu.json.beat.json",'.osu.json')
    beatmapset_path = os.path.dirname(osujson)
    with open(osujson) as file:
        osujson = json.load(file)
    with open(beatjson) as file:
        beatjson = json.load(file)
        beat_to_sample_map = beatjson['beat_to_sample_map']
    if osujson['music_fp'].startswith("/") or osujson['music_fp'][1] == "C:":
        music_fp = osujson['music_fp']
    else:
        music_fp = os.path.join(beatmapset_path, osujson['music_fp'])
    y, sr = sf.read(music_fp)
    try:
        y = y.mean(axis=1)
    except np.AxisError as e:
        pass # already in 1-dim
    # OFFSET!
    if osujson['offset'] < 0:
        left_pad = abs(round(osujson['offset']*sr/1000))
        left_deduct = 0
    else:
        left_pad = 0
        left_deduct = abs(round(osujson['offset']*sr/1000))
    right_pad = round( fft_size + sr*4*(60 / osujson['bpms'][-1][-1]))
    y = np.pad(y, (left_pad, right_pad))
    y = y[left_deduct:]

    to_take = [list(range(sample, sample+fft_size)) for _,sample in beat_to_sample_map]
    to_fft = np.take(y, to_take)
    fftd = np.power(np.absolute(scipy.fft.rfft(to_fft)),2)
    melspec = librosa.feature.melspectrogram(S=fftd.T,n_mels=n_mels)
    return melspec


def h5py_test(filename,attr):
    dset = h5py.File(filename,'r')
    print(dset[attr])

def pack_dset_into_h5py(split_json, transpose=False):
    with open(split_json) as file:
        split = json.load(file)
    split_name = os.path.splitext(split_json)[0].split('/')[-1]
    dir_of_dirs = os.path.dirname(split_json)
    h5f = h5py.File(os.path.join(dir_of_dirs,f'{split_name}.h5'),'w')

    pbar = tqdm(list(split.keys()))
    for osujson in pbar:
        pbar.set_description(os.path.basename(osujson))
        dataset_name = osujson.replace(dir_of_dirs+'/','')
        mel = np.float32(make_mel_for_whole_file(osujson))
        pbar.set_description(dataset_name)
        if transpose:
            h5f.create_dataset(dataset_name,data=mel.T)
        else:
            h5f.create_dataset(dataset_name,data=mel)
        # h5f.create_dataset(dataset_name,data=mel.T) <- [:, a:b] are training samples; we want this segment to be consecutive
    h5f.close()

if __name__=="__main__":
    pack_dset_into_h5py(sys.argv[1], transpose=True)
    #h5py_test('valid.h5',"153853/Trident - Blue Field (arronchu1207) [Shana's HD].osu.json")

