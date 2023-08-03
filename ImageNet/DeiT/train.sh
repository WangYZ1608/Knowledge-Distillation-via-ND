OMP_NUM_THREADS=1 \
python3 -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr='' \
        --master_port=23456 \
        train_launch_distilled.py \
        --model_name deit_base_patch16 \
        --teacher_weights "./ckpt/regnet160/regnety_160-a5fe301d.pth" \
        --epoch 300 \
        --batch_size 2048 \
        --blr 1e-4 \
        --drop_path 0.1 \
        --kd_loss_factor 0.5 \
        --nd_loss_factor 1.5 \
        --save_dir "./run/DeiT-distilled/kd0.5-nd1.5"