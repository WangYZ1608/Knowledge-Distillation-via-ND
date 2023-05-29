# ----------------------------------------------------------------------------------------------
# KD++
# --------------------------------------------------------
# 1、ViT-S -  resnet18     1.0*cls + 1.5*kd + 1.0*nd
# 2、ViT-B -  resnet18     1.0*cls + 2.0*kd + 1.0*nd
# --------------------------------------------------------
python3 train_kd.py \
        --model_name resnet18 \
        --teacher vitb \
        --dist_url 'tcp://localhost:' \
        --multiprocessing-distributed \
        --world_size 1 \
        --rank 0 \
        --batch_size 512 \
        --lr 0.2 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 2.0 \
        --norm_loss_factor 1.0 \
        --save_dir "./run/KD++/vitb-res18"
# ----------------------------------------------------------------------------------------------
# DKD++
# --------------------------------------------------------
# 1、ViT-S -  resnet18     1.0*cls + 1.0*alpha + 1.0*beta + 2.0*nd
# 2、ViT-B -  resnet18     1.0*cls + 1.4*alpha + 0.5*beta + 2.0*nd
# --------------------------------------------------------
python3 train_dkd.py \
        --model_name resnet18 \
        --teacher vits \
        --dist_url 'tcp://localhost:' \
        --multiprocessing-distributed \
        --world_size 1 \
        --rank 0 \
        --batch_size 512 \
        --lr 0.2 \
        --cls_loss_factor 1.0 \
        --dkd_alpha 1.4 \
        --dkd_beta 0.5 \
        --norm_loss_factor 2.0 \
        --save_dir "./run/DKD++/vits-res18"