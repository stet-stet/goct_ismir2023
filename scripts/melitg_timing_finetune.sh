
python train_cond_centered.py \
  +tr_dataset=beatfine_itg_timingonly \
  +cv_dataset=beatfine_itg_timingonly \
  +experiment=test \
  +model=mel \
  +loss=crossentropy \
  +optimizer=adam_mel_finetune \
  +ckpt_path=ckpts/itg_mel_timingonly_finetune \
  +load_from=ckpts/mel_timingonly/0.0002/ckpt_epoch_9.pth
