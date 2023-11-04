
python train_cond_centered.py \
  +tr_dataset=beatfine_itg \
  +cv_dataset=beatfine_itg \
  +experiment=test \
  +model=mel \
  +loss=crossentropy \
  +optimizer=adam
