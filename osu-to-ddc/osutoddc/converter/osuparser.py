import os
import sys 
import shutil
from parser_exceptions import NotManiaFileException, NotV14Exception, MissingAttributeException, Not4KException

# https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29

def parserFactory(action):
    def parser(string):
        string = string.strip()
        if len(string.strip()) == 0 : return None
        else: return action(string)
    return parser

int_parser = parserFactory(int)
str_parser = parserFactory(str)
float_parser = parserFactory(float)
bool_parser = parserFactory(lambda x: x.strip() == "1")

def dictParserFactory(fields, parsers):
    if len(fields)!=len(parsers): 
        raise ValueError("fields and parsers have unequal lengths")
    def parser(string):
        string = string.strip()
        if len(string)==0: return None
        strings = string.split(',')
        if len(strings) != len(fields):
            print(string)
            raise ValueError(f"invalid string {string}")
        ret = {}
        for s,f,p in zip(strings, fields, parsers):
            ret[f] = p(s)
        return ret
    return parser 


###############
############### General, Editor, Metadata, Difficulty.
###############

fields_of_note = {
    "AudioFilename":str_parser,
    "AudioLeadIn":int_parser,
    "Title":str_parser,
    "Artist":str_parser,
    "Mode":int_parser,
    "Version":str_parser,
    "BeatmapID":int_parser,
    "BeatmapSetID": int_parser,
    "CircleSize": float_parser # if this is not 4, raise Not4KException
}

def linebyline_section_parser(lines):
    ret = {}
    for line in lines:
        line = line.strip()
        if len(line)==0: 
            continue

        entries = line.split(':')
        attribute = entries[0]
        value = ":".join(entries[1:])

        if attribute in fields_of_note:
            ret[attribute] = fields_of_note[attribute]( value )
        else:
            ret[attribute] = str_parser( value )
    return ret

###############
############### Events, Colours
###############

def no_parser(str):
    return None

###############
############### TimingPoints
###############

timing_points_parser = dictParserFactory(
    ["time"     ,"beatLength","meter"   ,"sampleSet","sampleIndex"  ,"volume"   ,"uninherited"  ,"effects"  ],
    [float_parser ,float_parser,int_parser,int_parser ,int_parser     ,int_parser ,bool_parser    ,int_parser ]
)

def timing_points_section_parser(lines):
    lines = [e for e in lines if len(e.strip())>0]
    return [timing_points_parser(e) for e in lines]

###############
############### HitObjects
###############

def take_first_entry_from_semicolon_separated_list(sslist):
    return int(sslist.split(':')[0])

def note_type_parser(number):
    number = int(number)
    return {
        'hit': bool(number&1),
        'slider': bool(number&2),
        'spinner': bool(number&8),
        'hold': bool(number&128)
    }


x_parser = parserFactory(lambda x: min(3,max(0,int(x)*4//512)))
hold_endpoint_parser = parserFactory(take_first_entry_from_semicolon_separated_list)

hit_objects_parser = dictParserFactory(
    ["x"        ,"y"        ,"time"     ,"type"             ,"hitSound","holdEnd"         ],
    [x_parser   ,int_parser ,int_parser ,note_type_parser   ,int_parser,hold_endpoint_parser]
) 

def hit_objects_section_parser(lines):
    lines = [e for e in lines if len(e.strip())>0]
    return [hit_objects_parser(e) for e in lines]

###############
############### FIELDS
###############

fields = {"[General]": linebyline_section_parser,
          "[Editor]": no_parser,
          "[Metadata]": linebyline_section_parser,
          "[Difficulty]": linebyline_section_parser,
          "[Events]": no_parser,
          "[TimingPoints]": timing_points_section_parser,
          "[Colours]": no_parser,
          "[HitObjects]": hit_objects_section_parser } # These Sections appear in order.

mandatory_fields = ["[General]",
                  #  "[Editor]",
                    "[Metadata]",
                    "[Difficulty]",
                  #  "[Events]",
                    "[TimingPoints]",
                  #  "[Colours]",
                    "[HitObjects]"]


###############
############### all together...
###############

def parse_osu_file(filename):

    ret = {}

    with open(filename,'r',errors='replace') as file:
        lines = [line.strip() for line in file.readlines()]
    
    # correctness check
    if "osu file format v14" not in lines[0]:
        raise NotV14Exception("This file is not in osu file format v14.")

    extant_fields = []
    fields_index = []
    
    for n,line in enumerate(lines):
        for field in fields.keys():
            if line.startswith(field):
                extant_fields.append(field)
                fields_index.append(n)
    
    for field in mandatory_fields:
        if field not in extant_fields:
            raise MissingAttributeException(f"{filename} missing mandatory field {field}.")

    field_contents = [lines[a+1:b] for a,b in zip(fields_index[:-1],fields_index[1:])] + [lines[fields_index[-1]+1:]]

    for field, field_content in zip(extant_fields,field_contents):
        ret[field] = fields[field](field_content)

    # Correctness Check
    if ret["[Difficulty]"]["CircleSize"] != 4:
        raise Not4KException("This beatmap is not a 4K")
    if ret["[General]"]["Mode"] != 3:
        raise NotManiaFileException("This beatmap is not an Osu! Mania Beatmap") 
    
    return ret

if __name__=="__main__":
    from pprint import pprint
    pprint(parse_osu_file(sys.argv[1]))
