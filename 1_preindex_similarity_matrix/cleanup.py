"""
cleans up after a cache_similar_beat_index.py run.

given a folder of folders, deletes folders that do not have equal numbers of ".osu.json" files and ".osu.json.beat.json" files.
"""
import os 
import sys
import shutil

def one_folder(dir):
    b = os.listdir(dir)
    one = len([e for e in b if e.endswith(".osu.json")])
    two = len([e for e in b if e.endswith(".osu.json.beat.json")])
    if one != two:
        print(f"delete {dir}!")
        shutil.rmtree(dir)


def do(dir_of_dirs):
    dirs = [os.path.join(dir_of_dirs,e) for e in os.listdir(dir_of_dirs) if os.path.join(dir_of_dirs, e)]
    for dir in dirs:
        one_folder(dir)

if __name__=="__main__":
    do(sys.argv[1])
