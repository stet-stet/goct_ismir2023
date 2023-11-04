import os
import json
import copy 
import sys
from collections import defaultdict

from osuparser import parse_osu_file

# open library made from parsing nerinyan API responses.
# TODO eliminate hard-coded library path.
def fetch_ranked_map_library():
    with open("/mnt/c/Users/manym/Desktop/home_root/osu-to-ddc/osu_mania_ranked_library.json") as file:
        lib = file.read()
    return json.loads(lib)

ranked_map_library = fetch_ranked_map_library()

# generators for each output key.

def fill_title(osu_dict):
    return osu_dict['[Metadata]']['Title']

def fill_artist(osu_dict):
    return osu_dict['[Metadata]']['Artist']

def fill_chart_author(osu_dict):
    return osu_dict['[Metadata]']['Creator']

def fill_chart_type(osu_dict):
    return "dance-single"

def fill_difficulty_coarse(osu_dict):
    version = osu_dict['[Metadata]']['Version']
    if 'bg' in version.lower() or 'beginner' in version.lower():
        return 'Beginner'
    elif 'ez' in version.lower() or 'easy' in version.lower():
        return "Easy"
    elif 'nm' in version.lower() or 'normal' in version.lower():
        return "Normal"
    elif 'HD' in version.lower() or 'hard' in version.lower():
        return "Hard"
    else: 
        return "Challenging"

def fill_difficulty_fine(osu_dict):
    try:
        beatmapsetid = osu_dict['[Metadata]']['BeatmapSetID']
        beatmapid = osu_dict['[Metadata]']['BeatmapID']
        beatmaps = ranked_map_library[str(beatmapsetid)]['beatmaps']
        for beatmap in beatmaps:
            if beatmap['id'] == beatmapid:
                return beatmap["difficulty_rating"]
    except KeyError:
        return 2.00
    

def fill_music_fp(osu_dict):
    return osu_dict['[General]']['AudioFilename']


############
############ bpms and offset
############
"""
The rules:

1. if first timing point > 0, subtract by bar durations until <=0. This newly made timing point is now defined as t=0.
every beat and time shall be written w.r.t to this point.

2. A bpm change implies bar change, even if the previous bar is not "whole".

Return structure: List of [Beat, bpm]
"""

def mspb_to_bpm(mspb):
    primitive_bpm = 60000 / mspb
    return round(primitive_bpm,3) # 128.999999998 -> 129.000

def roundup(bar):
    if abs(int(bar)-bar) < 1e-3:
        return int(bar)
    else:
        return int(bar) + 1

def drag_timingpoint_to_before_0(osu_dict):
    while osu_dict['[TimingPoints]'][0]['time'] > 0:
        meter = osu_dict['[TimingPoints]'][0]['meter']
        mspb = osu_dict['[TimingPoints]'][0]['beatLength']
        osu_dict['[TimingPoints]'][0]['time'] -= meter * mspb 

    return osu_dict['[TimingPoints]'][0]['time']

def fill_bpms_and_offset(osu_dict, offset):
    timing_points = [e for e in osu_dict['[TimingPoints]'] if e['uninherited'] is True]

    assert offset == timing_points[0]['time'] 

    bpms = [ [0.0,mspb_to_bpm(timing_points[0]['beatLength'])] ]
    bars = [0]
    
    previous_mspb = timing_points[0]['beatLength']
    previous_ms = offset
    previous_beat = bpms[-1][0]
    previous_meter = timing_points[0]['meter']
    previous_bar = bars[-1]

    for timing_point in timing_points[1:]:
        now_ms = timing_point['time']
        now_beat = round( round_to_nearest_48th(previous_beat + (now_ms - previous_ms) / previous_mspb), 3)
        now_bpm = mspb_to_bpm(timing_point['beatLength'])
        now_bar = roundup(previous_bar + (now_beat - previous_beat) / previous_meter)

        bpms.append([now_beat, now_bpm])
        bars.append(now_bar)

        previous_ms = now_ms 
        previous_mspb = timing_point['beatLength']
        previous_beat = bpms[-1][0]
        previous_meter = timing_point['meter']
        previous_bar = bars[-1]

    return bpms, bars

def fill_bar_to_beat(osu_dict, offset, max_bar):
    timing_points = [e for e in osu_dict['[TimingPoints]'] if e['uninherited'] is True]
    assert offset == timing_points[0]['time'] 
    # dirty trick to ensure the entire length is considered
    timing_points += [
        {'time':1000*60*20,'beatLength':100.0,'meter':4,'sampleSet':None,'sampleIndex':None,'volume':None,'uninherited':True,'effects':None}
    ]

    current_beat = 0
    current_bar = 0
    current_time = offset
    current_timing_point = 0
    current_meter = timing_points[current_timing_point]['meter']
    current_mspb = timing_points[current_timing_point]['beatLength']

    bar_to_beat = [(0,0)]

    for next_timing_point in timing_points[1:]:
        while current_time + current_meter * current_mspb <= next_timing_point['time']:
            current_bar += 1
            current_beat += current_meter
            current_time += (current_meter * current_mspb)
            bar_to_beat.append((current_bar, round(current_beat,3)))
            if current_bar == max_bar: return bar_to_beat
        # a new bar begins with each new timing point:
        additional_beats = round_to_nearest_48th( (next_timing_point['time'] - current_time)/current_mspb )
        if additional_beats > 0.:
            current_bar += 1
            current_beat += additional_beats
            current_beat = round_to_nearest_48th(current_beat)
            bar_to_beat.append((current_bar, round(current_beat,3)))
            if current_bar == max_bar: return bar_to_beat
        
        current_time = next_timing_point['time']
        current_timing_point += 1
        current_meter = timing_points[current_timing_point]['meter']
        current_mspb = timing_points[current_timing_point]['beatLength']

    return bar_to_beat 

############
############ Notes
############
"""
List of [[bar, quant(possible placements in bar), beat phase], beat_from_beginning, time_from_beginning, notes]

quant will be fixed to 48 per beat, i.e. 192 per 4-meter bar.

regarding the last entry, tap = 1, hold start = 2, hold end = 3.
eg. "1002" "0103" etc
"""
def round_to_nearest_48th(number):
    return round(number * 48)/48

def round_time_to_48th_beat(time_in_ms, starting_time, mspb):
    beats = (time_in_ms - starting_time) / mspb
    rounded_beats = round(beats * 48)/48
    time = starting_time + rounded_beats * mspb
    return time

def fill_charts(osu_dict, bpm_list, bar_list, offset):
    tp_beats = [a for a,b in bpm_list]
    tp_bpms = [b for a,b in bpm_list]
    tp_bars = copy.deepcopy(bar_list)
    notes = copy.deepcopy(osu_dict['[HitObjects]'])
    timing_points = [copy.deepcopy(e) for e in osu_dict['[TimingPoints]'] if e['uninherited'] is True]

    # drag all times front by offset
    # offset will be a negative value, so in effect all times are dragged back(added) by abs(offset)
    for n in range(len(notes)):
        notes[n]['time'] -= offset
        notes[n]['holdEnd'] -= offset
    for n in range(len(timing_points)):
        timing_points[n]['time'] -= offset
    
    mspbs = [e['beatLength'] for e in timing_points if e['uninherited'] is True]
    transition_times = [e['time'] for e in timing_points if e['uninherited'] is True] # REALLY?


    # separate hold ends from hold beginnings
    hold_ends = []
    for note in notes:
        if note['type']['hold'] is True:
            new_note = copy.deepcopy(note)
            new_note['time'] = note['holdEnd']
            new_note['holdEnd'] = -1
            hold_ends.append(new_note)
    notes.extend(hold_ends)
    # print(hold_ends, file=sys.stderr)

    notes_by_time = defaultdict(list)
    for note in notes:
        notes_by_time[note['time']].append(note)
    
    sorted_time = sorted(list(notes_by_time.keys()),key=lambda x:int(x))

    ########################################

    ret = []

    current_bpm_index = 0
    for time in sorted_time:
        while current_bpm_index+1 < len(timing_points) and time >= transition_times[current_bpm_index+1]:
            current_bpm_index += 1
        current_starting_bar = tp_bars[current_bpm_index]
        current_starting_beat = tp_beats[current_bpm_index]
        current_mspb = timing_points[current_bpm_index]['beatLength']
        current_starting_time = transition_times[current_bpm_index]
        current_meter = timing_points[current_bpm_index]['meter']

        to_push = [[0,0,0],0,0,"0000"]

        current_delta_beat = round_to_nearest_48th((time - current_starting_time) / current_mspb )
        current_beat = current_starting_beat + current_delta_beat
        current_bar = int(round_to_nearest_48th(current_delta_beat) / current_meter) + current_starting_bar
        to_push[0][0] = current_bar

        if current_bpm_index == len(timing_points)-1:
            to_push[0][1] = current_meter*48
        else:
            if current_bar < tp_bars[current_bpm_index+1]-1:
                to_push[0][1] = current_meter * 48
            else:
                beats_in_between = round_to_nearest_48th(tp_beats[current_bpm_index+1] - tp_beats[current_bpm_index])
                while beats_in_between >= current_meter:
                    beats_in_between -= current_meter
                to_push[0][1] = beats_in_between * 48
        
        micro_beat = (current_beat - current_starting_beat)
        while micro_beat >= current_meter:
            micro_beat -= current_meter
        micro_beat = round_to_nearest_48th(micro_beat)
        micro_beat = round(micro_beat*48)
        to_push[0][2] = micro_beat
        
        to_push[1] = round_to_nearest_48th(current_beat)

        time_in_ms = round_time_to_48th_beat(time, current_starting_time, current_mspb)
        to_push[2] = time_in_ms / 1000

        current_code = [0,0,0,0] 
        for note in notes_by_time[time]:
            if note['type']['hit']:
                current_code[note['x']] = 1
            elif note['type']['hold'] and note['holdEnd'] != -1:
                current_code[note['x']] = 2
            elif note['type']['hold']:
                current_code[note['x']] = 3
            else:
                raise NotManiaFileException(f"Invalid note type: {note['type']}")
        to_push[3] = "".join([str(x) for x in current_code])

        ret.append(to_push)

    return ret
          

# main
def osu_to_ddc(filename):
    ret = {}
    osu_dict = parse_osu_file(filename)

    ret['title'] = fill_title(osu_dict)
    ret['artist'] = fill_artist(osu_dict)
    ret['stops'] = []
    ret['sm_fp'] = filename #point to any one of them
    ret['music_fp'] = fill_music_fp(osu_dict)
    ret['pack'] = "osumania"

    ret['offset'] = drag_timingpoint_to_before_0(osu_dict)
    ret['bpms'], bars = fill_bpms_and_offset(osu_dict, ret['offset'])
    ret['charts'] = [{
        "notes": fill_charts(osu_dict, ret['bpms'], bars, ret['offset']),
        "difficulty_coarse": fill_difficulty_coarse(osu_dict),
        "difficulty_fine": fill_difficulty_fine(osu_dict),
        "type": "dance-single",
        "desc_or_author": fill_chart_author(osu_dict)
    }]
    max_bar = ret['charts'][0]['notes'][-1][0][0] # ew!
    ret['bar_to_beat'] = fill_bar_to_beat(osu_dict, ret['offset'],max_bar)
    return ret

if __name__ == "__main__":
    from pprint import pprint
    import sys 
    import json 
    a = json.dumps(osu_to_ddc(sys.argv[1]),indent=4)
    with open(sys.argv[2],'w') as file:
        file.write(a)