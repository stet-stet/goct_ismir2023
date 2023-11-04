import os 
import sys
import json 
import copy
import scipy
import numpy as np
import torch
import soundfile as sf

def read_ddc_file(filename):
    if not filename.endswith(".json"):
        raise ValueError("specified file is not ddc file")
    with open(filename,'r',errors='replace') as file:
        d = json.loads(file.read())
    return d

def mspb(bpm):
    return 60000 / bpm

def ms_per_48th_beat(bpm):
    return 60000 / bpm / 48

def round_to_nearest_48th(number):
    return round(number * 48)/48

def increment_by_48th(number):
    return round_to_nearest_48th(number+1/48)

def time_to_sample(time_in_ms, sr, time_mode="sample"):
    if time_mode == "sample":
        return round(time_in_ms * sr / 1000)
    elif time_mode == "time":
        return time_in_ms
    else:
        raise ValueError("invalid mode")

def get_beat_to_sample_map(bpms=None, sr=44100, make_until=1000, time_mode="sample"):
    """
    bpms is given as a List of two-element lists: [beat, bpm]

    returns: a list of tuples: (beat(48th), sample num)
    """
    if bpms is None:
        raise ValueError("param 'bpms' cannot be empty")
    
    ret = [(0.,0)]
    current_beat = 0
    current_bpm = bpms[0][-1]
    current_time = 0.

    bpms = copy.deepcopy(bpms[1:]) + [[float(make_until)-1/48,bpms[-1][-1]]]
    bpms = [e for e in bpms if e[0] < make_until] # For cases when chart is shorter than the bpm changes denoted.
    for bpm in bpms:
        delimiter_beat, next_bpm = bpm
        delimiter_beat = round_to_nearest_48th(delimiter_beat)
        while current_beat < delimiter_beat:
            if increment_by_48th(current_beat) < delimiter_beat:
                current_beat = increment_by_48th(current_beat)
                current_time += ms_per_48th_beat(current_bpm)
                ret.append((current_beat, time_to_sample(current_time, sr, time_mode=time_mode)))
            else:
                current_percentage = ((delimiter_beat) - (current_beat)) / (1/48)
                next_percentage = 1 - current_percentage
                current_beat = increment_by_48th(current_beat)
                current_time += current_percentage * ms_per_48th_beat(current_bpm) + \
                                next_percentage * ms_per_48th_beat(next_bpm)
                ret.append((current_beat, time_to_sample(current_time, sr,time_mode=time_mode)))
                
                current_bpm = next_bpm
        # TODO: catch corner cases where two BPM changes are b2b less than 1/48th beat away
        #       (probably nonexistent in ranked maps, but still)
    return ret

def make_indexes_from_beat_to_sample_map(beat_to_sample_map, fft_size=512):
    return [list(range(e,e+fft_size)) for _,e in beat_to_sample_map]

def make_beatwise_similarity_matrix_and_beat_to_sample_map(json_filepath, fft_size=512, no_matrix=False):
    d = read_ddc_file(json_filepath)
    json_path = os.path.dirname(json_filepath)
    if d["music_fp"].startswith("/") or d["music_fp"].startswith("C:") :
        audio_filepath = d["music_fp"]
    else:
        audio_filepath = os.path.join(json_path, d["music_fp"])
    y, sr = sf.read(audio_filepath)
    try:
        y = y.mean(axis=1)
    except np.AxisError as e:
        pass # this is already 1-dim.
    # account for offset 
    left_pad = abs(round(d['offset']*sr/1000))
    right_pad = round(fft_size + sr * 4 * mspb(d['bpms'][-1][-1]) /1000)
    y = np.pad(y, (left_pad,right_pad))

    max_beat = max([a for _,a,_,_ in d['charts'][0]['notes']])
    length_in_beats = int(max_beat) + 1
    beat_to_sample_map = get_beat_to_sample_map(bpms=d['bpms'], sr=sr, make_until=length_in_beats)

    try: 
        to_fft = np.take(y,make_indexes_from_beat_to_sample_map(beat_to_sample_map,fft_size=fft_size))
        fftd = scipy.fft.rfft(to_fft)
        fftd = np.abs(fftd)**2
        fftd_r = np.reshape(fftd, (length_in_beats, -1))
        fftd_r /= (np.linalg.norm(fftd_r, axis=1, keepdims=True)+1e-12)   
        fftd_r = torch.Tensor(fftd_r).to('cuda:0')
        
        similarity_matrix = torch.matmul(fftd_r, fftd_r.transpose(1,0))
        # return similarity_matrix.cpu().numpy(), beat_to_sample_map
    except Exception as e:
        print('')
        print(f"problematic/simmatrix defaults to zero for : {json_filepath}")
        print(e)
        print('')
        return np.zeros([length_in_beats,length_in_beats]), beat_to_sample_map

    return similarity_matrix.cpu().numpy(), beat_to_sample_map
    

############################################## FOR TESTING ############################################## 
def test_bpms_to_time(json_filepath, fft_size=512):
    d = read_ddc_file(json_filepath)
    json_path = os.path.dirname(json_filepath)
    audio_filepath = os.path.join(json_path, d["music_fp"])
    y, sr = sf.read(audio_filepath)
    y = y.mean(axis=1)
    # account for offset 
    left_pad = abs(round(d['offset']*sr/1000))
    right_pad = round(fft_size + sr * 4 * mspb(d['bpms'][-1][-1]) /1000)
    y = np.pad(y, (left_pad,right_pad))

    max_beat = max([a for _,a,_,_ in d['charts'][0]['notes']])
    length_in_beats = int(max_beat) + 1
    beat_to_sample_map = get_beat_to_sample_map(bpms=d['bpms'], sr=sr, make_until=length_in_beats, time_mode="time")
    beat_to_sample_map = [e for e in beat_to_sample_map if e[0] == float(int(e[0]))]
    beat_to_sample_increments = []
    for past,future in zip(beat_to_sample_map[:-1],beat_to_sample_map[1:]):
        beat_to_sample_increments.append((future[0], future[1]-past[1]))
    from pprint import pprint
    pprint(beat_to_sample_increments)
    # examined soflanchan output. no problems anywhere...


if __name__=="__main__":
    # make_beatwise_similarity_matrix(sys.argv[1])
    test_bpms_to_time(sys.argv[1])
     