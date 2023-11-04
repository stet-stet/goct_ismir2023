if [ $# -eq 0 ]; then
  echo "no args"
  exit 1
fi
diff=$1
python tune_thresholds.py \
	+cv_dataset=ddc_${diff} \
	+model=ddc_cnn \
	+ckpt_path=/mnt/c/Users/manym/Desktop/gorst/gorst/ckpts/ddc_cnn/2e4_and_64_try2/ckpt_epoch_1.pth \
	+experiment=ddc
