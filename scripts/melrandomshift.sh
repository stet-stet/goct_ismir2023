python train_cond_centered.py \
  +tr_dataset=melrandomshift \
  +cv_dataset=melrandomshift \
  +experiment=test \
  +model=mel \
  +loss=crossentropy \
  +optimizer=adam \
  +ckpt_path=ckpts/mel_shift/
