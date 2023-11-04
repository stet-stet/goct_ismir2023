from ddcjson_to_osujson import ddcjson_to_osujson
from cache_similar_beat_index import do
import os 
import sys
from tqdm import tqdm

def cleanup(biggest_folder):
    for root, dirs, files in os.walk(biggest_folder):
        for file in files:
            if file.endswith(".osu.json") or file.endswith(".osu.json.beat.json"):
                os.remove(os.path.join(root,file))

def run_on_all(biggest_folder):
    for root, dirs, files in os.walk(biggest_folder):
        print(root)
        for file in tqdm(files):
            if file.endswith(".json") and not file.endswith(".osu.json"):
                file_base = os.path.join(root,file[:-5])
                ddcjson_to_osujson(file_base)

def run_matrix_on_all(list_of_folder_of_folders):
    for folder_of_folders in list_of_folder_of_folders:
        do(folder_of_folders)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("==HELP==")
        print("Supply me with two or more arguments.")
        print("python main.py (clean / do) (path to json_filt)")
        exit(0)

    if sys.argv[1] == "clean":
        cleanup(sys.argv[2])
    elif sys.argv[1] == "do":
        the_path = sys.argv[2]
        print("converting ddc.json to osu.jsons...")
        run_on_all(the_path)
        print("fraxtil")
        do(os.path.join(the_path,'fraxtil'))
        print("itg")
        #do(os.path.join(the_path,'itg'))
    elif sys.argv[1] == "matrix":
        the_path = sys.argv[2]
        print("fraxtil")
        do(os.path.join(the_path,'fraxtil'))
        print("itg")
        do(os.path.join(the_path,'itg'))