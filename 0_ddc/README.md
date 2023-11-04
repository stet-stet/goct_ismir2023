# Data preprocessing code from DDC

## Building Datsaet (part 1)

1. Install Python 2.7.
```
# for conda or mamba
conda create -n ddc python=2.7 
mamba create -n ddc python=2.7
```
2. Make a directory wherever you'd like, and change `scripts/var.sh`. The variable `SMDATA_DIR` must point to the directory you just made.
3. Under this directory, make directories `raw`, `json_raw` and `json_filt`
4. Under `data/raw` make directories `fraxtil` and `itg`
5. Under `data/raw/fraxtil`, download and unzip:
    - [Tsunamix III](https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20%5BSM5%5D.zip)
    - [Fraxtil's Arrow Arrangements](https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's%20Arrow%20Arrangements%20%5BSM5%5D.zip)
    - [Fraxtil's Beast Beats](https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's%20Beast%20Beats%20%5BSM5%5D.zip)
6. Under `data/raw/itg`, download and unzip:
    - [In the Groove](https://search.stepmaniaonline.net/link/In%20The%20Groove%201.zip)
    - [In the Groove 2](https://search.stepmaniaonline.net/link/In%20The%20Groove%202.zip)
7. Navigate to `scripts/`
8. Run the following.
```
./all.sh ./smd_1_extract.sh # parsing .sm files to .json
./all.sh ./smd_2_filter.sh # filter (removing mines, etc.)
./all.sh ./smd_3_dataset.sh # split dataset 80/10/10
./smd_4_analyze.sh fraxtil # analyzes dataset
```

## Building Dataset (part 2)

The above used the code included with [DDC](https://github.com/chrisdonahue/ddc). What follows afterwards are the procedures to make the dataset compatible to our system.

1. Now switch to an environment with python 3. (Our codes have been tested in version 3.9.16.)
2. navigate to `ddc_to_gorst`
3. Locate the folder where you had put the data. Locate the `json_filt`, where all the filtered jsons should be placed. Let this directory be `DIR/json_filt`. Run:
```
python main.py do "DIR/json_filt" # makes dataset peripherals needed to generate h5py files.
python make_split.py do "DIR/json_filt/itg" # translates the splits made by DDC into our json-like format
python h5pyize_dataset.py "DIR/json_filt/itg/test.json"
python h5pyize_dataset.py "DIR/json_filt/itg/valid.json"
python h5pyize_dataset.py "DIR/json_filt/itg/train.json"
```
4. And now you have the h5 files in `DIR/json_filt/itg/`! now navigate to the top, go to `conf/tr_dataset/` and modify `beatfine_itg` accordingly.

