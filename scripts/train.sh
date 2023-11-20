python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py \
--model convnext_base --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path ./datasets \
--output_dir ./results/base \
--data_set CIFAR \
--enable_wandb true \
--project template \
--use_amp true \
