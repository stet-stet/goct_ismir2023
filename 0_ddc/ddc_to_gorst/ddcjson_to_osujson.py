"""
Converts DDC-type json files to our osu-type json files.

Due to some inconveniences that arose in beat-aligned training, 
jsons from DDC and our framework has minute differences that need to be resolved before training can be carried out.

Code written by stet-stet (Jayeon Yi)
"""

import json
import copy

def get_48th_index(num):
    return round(num*48)%48
    
def ddcjson_to_osujson(ddcjson_file_base):
    """
    Differences:
    - .ddc.json has multiple charts in one file; .osu.json has each chart in a separate file
    - offset sign is backwards
    - four beats must equal 192. each beat[0][2] entry should be in multiples of 48.
    - .osu.json has an extra "bar_to_beat" field that translates bars to beats 
        - this information will not be needed as of now, so this is skipped.
        - If we need this in the future, be sure to start from the original .sm files.
    """
    ddcjson_file = f"{ddcjson_file_base}.json"
    with open(ddcjson_file) as file:
        a = json.load(file)

    # inspect each beat and determine which of 48 this has to be.

    all_except_charts = copy.deepcopy(a)
    del all_except_charts['charts']
    all_except_charts["offset"] = - (all_except_charts["offset"] * 1000)

    for n, chart in enumerate(a['charts']):
        to_output = copy.deepcopy(all_except_charts)
        to_output.update({"charts": [chart]})

        for note in to_output['charts'][0]['notes']:
            note[0][2] = get_48th_index(note[1])

        output_filename = f"{ddcjson_file_base}_{n}.osu.json"
        with open(output_filename,'w') as file:
            file.write(json.dumps(to_output,indent=4))
    
    
if __name__ == "__main__":
    ddcjson_to_osujson("examples/example/example_ddc")