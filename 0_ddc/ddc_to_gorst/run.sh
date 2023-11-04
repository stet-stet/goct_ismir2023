DIR=STEPMANIAFOLDER/json_filt

python main.py clean ${DIR}
python main.py do ${DIR}
python make_split.py do ${DIR}/itg
python h5pyize_dataset.py ${DIR}/itg/test.json
python h5pyize_dataset.py ${DIR}/itg/valid.json
python h5pyize_dataset.py ${DIR}/itg/train.json

