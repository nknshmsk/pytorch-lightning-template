DATE=`date "+%Y%m%d"`

python train.py \
--model_name xxxxx \
--train_json_path ./data.json \
--output_dir "./result/$DATE" \
--log_dir "./logs/$DATE" \
--input_image_size 256 \
--max_epochs 2000 \
--train_batch_size 32 \
--validation_batch_size 1 \
--auto_select_gpus 1 \
--learning_rate 1e-3 \
--check_val_every_n_epoch 30 \
--discription discription \
--gpus -1 \
--auto_select_gpus True \
--benchmark True 