import os
import sys

def for_one_file(inthisfile,replacethis,withthis):
    with open(inthisfile) as f:
        contents = f.read()
    contents = contents.replace(replacethis,withthis)
    with open(inthisfile,'w') as f:
        f.write(contents)

def iterate_folders(replacethis,withthis):
    for root,dirs,files in os.walk('.'):
        for file in files:
            if file.endswith(".json") or file.endswith(".sh") or file.endswith(".yaml") or file.endswith(".py"):
                for_one_file(os.path.join(root,file),replacethis,withthis)

if __name__=="__main__":
    iterate_folders(sys.argv[1],sys.argv[2])

