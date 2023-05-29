# ----------------------------------------------------------------------------------------------
# KD++
# --------------------------------------------------------
# 1、resnet34 -  resnet18     1.0*cls + 2.5*kd + 1.0*nd
# 2、resnet50 -  resnet18     1.0*cls + 4.0*kd + 1.0*nd
# 3、resnet101 - resnet18     1.0*cls + 3.5*kd + 4.0*nd
# 4、resnet152 - resnet18     1.0*cls + 4.0*kd + 2.0*nd
# 5、resnet50 - mobilenetv1   1.0*cls + 4.0*kd + 1.0*nd
# --------------------------------------------------------
python3 train_kd.py \
        --model_name resnet18 \
        --teacher resnet152 \
        --teacher_weights 'Teacher WEIGHT PATH' \
        --dist_url 'tcp://localhost:' \
        --multiprocessing-distributed \
        --world_size 1 \
        --rank 0 \
        --batch_size 512 \
        --lr 0.2 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 4.0 \
        --nd_loss_factor 2.0 \
        --save_dir "./run/ImageNet/KD++/res152-res18"
# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
# DKD++
# --------------------------------------------------------
# 1、resnet34 -  resnet18     1.0*cls + 0.8*alpha + 0.5*beta + 1.0*nd
# 2、resnet50 -  resnet18     1.0*cls + 0.6*alpha + 0.5*beta + 2.0*nd
# 3、resnet101 - resnet18     1.0*cls + 0.8*alpha + 0.5*beta + 1.0*nd
# 4、resnet152 - resnet18     1.0*cls + 1.0*alpha + 1.0*beta + 1.0*nd
# 5、resnet50 - mobilenetv1   1.0*cls + 0.8*alpha + 0.5*beta + 1.0*nd
# --------------------------------------------------------
python3 train_dkd.py \
        --model_name resnet18 \
        --teacher resnet101 \
        --teacher_weights 'Teacher WEIGHT PATH' \
        --dist_url 'tcp://localhost:' \
        --multiprocessing-distributed \
        --world_size 1 \
        --rank 0 \
        --batch_size 512 \
        --lr 0.2 \
        --cls_loss_factor 1.0 \
        --dkd_alpha 0.8 \
        --dkd_beta 0.5 \
        --nd_loss_factor 2.0 \
        --save_dir "./run/ImageNet/dKD++/res101-res18"