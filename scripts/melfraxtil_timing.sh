
python train_cond_centered.py \
  +tr_dataset=beatfine_fraxtil_timingonly \
  +cv_dataset=beatfine_fraxtil_timingonly \
  +experiment=test \
  +model=mel \
  +loss=crossentropy \
  +optimizer=adam \
  +ckpt_path=ckpts/fraxtil_mel_timingonly/
