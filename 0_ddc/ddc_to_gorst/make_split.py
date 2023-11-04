import json
import os
import sys
from generate_dataset_peripherals import do

def make_summary(dir_of_dirs):
    print("summarizing...")
    do(dir_of_dirs, make_splits=False)

def translate_splits(dir_of_dirs):
    print("splitting...")
    # gather all txt files with the name "*_train.txt", "*_valid.txt", "*_test.txt"
    l = [e for e in os.listdir(dir_of_dirs) if e.endswith(".txt")]
    train, valid, test = [],[],[]
    trainjson, validjson, testjson = {}, {}, {}
    for textfile in l:
        with open(os.path.join(dir_of_dirs, textfile)) as file:
            b = [e.strip() for e in file.readlines()]
        if "_train.txt" in textfile: train.extend(b)
        elif "_valid.txt" in textfile: valid.extend(b)
        elif "_test.txt" in textfile: test.extend(b)
    
    with open(os.path.join(dir_of_dirs,"summary.json")) as file:
        summary = json.load(file)
    
    for key in summary: # this is slow and I know that, sorry
        to_look_for = "_".join(key.split('_')[:-1]) + ".json"
        for split, jsonfile in [(train, trainjson),(valid, validjson),(test,testjson)]:
            if to_look_for in split:
                jsonfile.update({key: summary[key]})
                break
    
    for name, jsonfile in [("train.json",trainjson),("valid.json",validjson),("test.json",testjson)]:
        with open(os.path.join(dir_of_dirs, name),'w') as file:
            file.write(json.dumps(jsonfile,indent=4))


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("==HELP==")
        print("Supply me with two or more arguments.")
        print(f"python {sys.argv[1]} (clean / do) (path to json_filt/(itg or fraxtil))")
        exit(0)

    elif sys.argv[1] == "summarize":
        the_path = sys.argv[2]
        make_summary(the_path)

    elif sys.argv[1] == "do":
        the_path = sys.argv[2]
        make_summary(the_path)
        translate_splits(the_path)
