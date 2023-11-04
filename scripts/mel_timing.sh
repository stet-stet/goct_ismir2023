
python train_cond_centered.py \
  +tr_dataset=beatfine_timingonly \
  +cv_dataset=beatfine_timingonly \
  +experiment=test \
  +model=mel \
  +loss=crossentropy \
  +optimizer=adam \
  +ckpt_path=ckpts/mel_timingonly/
