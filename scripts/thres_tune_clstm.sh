if [ $# -eq 0 ]; then
  echo "no args"
  exit 1
fi
diff=$1
python tune_thresholds.py \
	+cv_dataset=ddc_${diff} \
	+model=ddc_clstm \
	+ckpt_path="/mnt/c/Users/manym/Desktop/gorst/gorst/ckpts/ddc_clstm/0.0002_and_64/ckpt_epoch_0.pth" \
	+experiment=ddc
