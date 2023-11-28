python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py \
--model odls3d \
--drop_path 0.1 \
--batch_size 3 \
--lr 4e-3 \
--update_freq 4 \
--model_ema true \
--model_ema_eval true \
--data_path /data/disk1/datasets/T1map_dataset/train \
--output_dir ./results/base \
--dataset t1recon \
--enable_wandb true \
--project t1map \
--use_amp true \
--eval_data_path /data/disk1/datasets/T1map_dataset/val \
--auto_resume false \
--ncoil 10 \
--kspace_path kspace_multicoil_cropped_singleslice \
--model_ema false \
--use_amp true \
