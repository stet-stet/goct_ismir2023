
python train_cond_centered.py \
  +tr_dataset=beatfine_8_100 \
  +cv_dataset=beatfine_8_100 \
  +experiment=test \
  +model=mel \
  +loss=crossentropy \
  +optimizer=adam \
  +ckpt_path=ckpts/mel/
