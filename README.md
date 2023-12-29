# GOCT_ISMIR2023

This repo contains code for the extended abstract, "[BEAT-ALIGNED SPECTROGRAM-TO-SEQUENCE GENERATION OF RHYTHM-GAME CHARTS](https://drive.google.com/file/d/1XOT9zD6yIoSQS7bwQpI9pOaO0DHL0Bec/view?usp=sharing)", accepted to [ISMIR2023 LBD](https://ismir2023.ismir.net/cflbd/).

**This repo is also under construction**; this is a public version of my bigger, private repo, and I am yet to make sure that everything works without some of the parts that I had. This will definitely be done before Thursday. 

You may see the demo page [here](https://stet-stet.github.io/goct/), but this page is currently under construction as well. **Alternatively**, [here](https://drive.google.com/drive/folders/1vPYGO5TuGdGRiGkI9SsNWtvw1TZjcZPW?usp=sharing) is a folder that has videos of generated charts for a wide variety of genres and difficulties. 

# Pretrained models

We bundled all of the models used to generate the tables in the paper, into one .zip file of size `~300MB`. You can download it [here](https://drive.google.com/file/d/1d6J7Mtgdvx3ecqrK6K3Eakj136QR2S_9/view?usp=sharing). If you'd like to replicate the tables or generate your own charts, unzip it to `goct/ckpts`, as many scripts are written with this arrangement implied.

# Instructions

## dependencies

```
pip install -U h5py scipy tqdm numpy soundfile librosa hydra-core wandb
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

wandb is only used to log, so it's optional (but you'd have to remove relevant code)

## Downloading Data

Enclosed within the `data` directory are the list of songs and charts used for training/validation/testing. An entry from the list may look like this:
```
"BIGFOLDER/1003217/nekodex - circles! (FAMoss) [easy!].osu.json": [
        105,
        421
    ],
```
That means it's from beatmapset number 1003217, with the file name `nekodex - circles! (FAMoss) [easy!].osu`.

Due to the copyright laws of my country, I most probably can't distribute the raw data directly.
Moreover, the means to download data either:
- requires you to log in to `osu.ppy.sh`, the official website of osu!, or
- are "mirrors", run by individuals and rarely continued for years, sometimes switching their APIs.

Therefore, it would be meaningless to provide a script for downloading the script. Please somehow obtain the beatmaps yourself.

Some tips on the file format:
- `.osz` files are actually `.zip` files. Just rename them and you will be able to unzip them. even after unzipping, it is recommended to keep the contents of the folder in the same folder.
- Likewise, `.osu` files are plaintext. There's even an [wiki article](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29) for this online, which is fairly detailed.

## Preprocessing Data for osu!mania 4k dataset

~~If you cannot go through the steps below, I can provide you with the .h5 files. However, I honestly cannot think of a stable way to transport three files that sum up to `~40GB`, so you'll have to provide me with a good way to do so.~~ (EDIT: on second thought, I am not entirely sure if this is even legal to do in my nation. I do not expect Korean laws to exhibit any sort of decency nor leniency, not anymore. Do we even have "fair use"?)

**Prerequisite**. We strongly recommend that the data be organized like this.
```
BIGFOLDER
ㄴ136881
ㄴ153199
    ㄴaudio.mp3
    ㄴSHK - Couple Breaking (Sky_Demon) [MX].osu
    ㄴSHK - Couple Breaking (Sky_Demon) [NM].osu
    ㄴSHK - Couple Breaking (Sky_Demon) [Sakura's HD].osu
    ㄴ...
```
remember which directory the folders are located in. We will call this `(BIGFOLDER)`. This is not to be confused with `BIGFOLDER`, which is literally the length-9 text "BIGFOLDER".

We strongly recommend that the steps below be taken in a non-Windows operating system.

1. Wipe all data from the beatmapsets(`.osz`s) except for the .osu files referenced by `summary.json`, which is train, valid, and test json files combined.
2. Navigate to `osu-to-ddc/osutoddc/converter` and run `python converter.py (BIGFOLDER) (BIGFOLDER)`. 
3. Navigate to `1_preindex_similarity_matrix`. Run `python cache_similar_beat_index.py (BIGFOLDER) && python cleanup.py (BIGFOLDER)`. This may take very, very long. Please return tomorrow.
4. Navigate to the top directory and run `python replace_text.py BIGFOLDER (BIGFOLDER)`. Then, `cp data/* (BIGFOLDER)`. (This was primarily generated with `2_generate_dataset/generate_dataset_peripherals.py`, but the dataset splits were modified to filter out some beatmaps later, to combat a upstream problem that arose much later into the process. As a result the script does not yield the same outputs anymore.)
4. Navigate to `2_generate_dataset` and run `python h5pyize_dataset.py (BIGFOLDER)/test.json`. Do the same for valid.json and train.json. This will take less time than the previous step, but will take extremely long nonetheless (~30 hours). The generated .h5 files will be `~40GB` total.
5. Now the folder should look like this.
```
BIGFOLDER
ㄴ136881
ㄴ153199
    ㄴaudio.mp3
    ㄴSHK - Couple Breaking (Sky_Demon) [MX].osu
    ㄴSHK - Couple Breaking (Sky_Demon) [MX].osu.json
    ㄴSHK - Couple Breaking (Sky_Demon) [MX].osu.json.beat.json
    ㄴ...
...
ㄴtrain.h5
ㄴtrain.json
ㄴvalid.h5
ㄴvalid.json
ㄴtest.h5
ㄴtest.json
ㄴsummary.json
```

## Preprocessing data for Fraxtil and ITG datasets.

Navigate to `0_ddc` and follow instructions there. 

Henceforth we will call the big folder for the beatmania data, where dirs such as `json_filt` are, as `(SMALLFOLDER)`.

## Preprocessing data for DDC model training.

Clone [my fork of ddc_onset](https://github.com/stet-stet/ddc_onset), and and please run
```
python h5pyize_tree.py do (BIGFOLDER) (BIGFOLDER)/all_ddc.h5
python h5pyize_tree.py onset (BIGFOLDER) (BIGFOLDER)/all_ddc.h5 (BIGFOLDER)/all_onset.h5
```
This will also generate a `~40GB` file.

## Running Training

navigate to the top directory, and run 
```
python text_replacer.py OSUFOLDER (BIGFOLDER)
python text_replacer.py STEPMANIAFOLDER (SMALLFOLDER)
```
This is a script that replaces all certain instance of a text within the folder to another, for files with certain extensions. The files modified with this are mainly in the folder `conf/`.

now, run any script in `scripts`, from the top directory. I hope the titles are intuitive enough: if not, please raise an issue (or head there to see if I've already answered the question).

On my system, with a RTX 3060, one epoch of `mel.sh` takes two hours. With a RTX 3090, it should take one for each epoch.

Metrics in Table 2 can be genererated with `metrics_timing_AR.py` and `ddc_eval.py`. Please see that file for notes.

## (WIP) Generating charts

unzip `generated.zip` to `generate` on the highest directory of this repo, if you need this feature or would like to try replicating Table 3.

`metrics_cond_centered_AR.py` is used to generate notes for the models trained on osu!. For the commands and the numbers please see that file. Moreover, `generated/millin_and_anmillin.ipynb` is used to generate stats for Table 3. Also in the `generated` folder are the files output from `metrics_cond_centered_AR.py` from the beat-aligned and non-beat-aligned models.

The generated intermediate is output both on the console, and on the `outputs` folder that will be generated. Either route the output to a folder of your liking or copy the logged output from `outputs` folder, as well as the ground truth, generated with `generate_ref.py`.


## (WIP) Generating charts for the test set from the notes generated

`gen_to_beatmap_all.ipynb` has the code, but I would not dare make this into a script, since I the code is very unrefined. The scripts here read in the log files generated, parse them, and translate them into the `.osu` format with the help of preexisting .osu files.

I have not implemented making stepmania charts at the moment, nor I have implemented making osu!mania charts from scratch. hopefully I will have the time to work on it in the future and release it to both communities.
