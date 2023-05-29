# DIST++
# --------------------------------------------------------
# 1、resnet56 - resnet20         1.0*cls + 2.0*kd + 1.0*nd
# 2、resnet32x4 - resnet8x4      1.0*cls + 1.5*kd + 4.0*nd
# 3、wrn40_2 - wrn40_1           1.0*cls + 2.5*kd + 3.0*nd
# 4、resnet50 - mobilenetv2      1.0*cls + 7.0*kd + 1.0*nd
# 5、resnet32x4 - shufflenetv1   1.0*cls + 2.5*kd + 3.0*nd
# 6、resnet32x4 - shufflenetv2   1.0*cls + 3.0*kd + 4.0*nd
# --------------------------------------------------------
python3 train_dist.py \
        --model_name shufflenetv2 \
        --teacher resnet32x4 \
        --teacher_weights 'Teacher WEIGHT PATH' \
        --dataset 'cifar100' \
        --epoch 240 \
        --batch_size 64 \
        --lr 0.02 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 3.0 \
        --nd_loss_factor 4.0 \
        --save_dir "./run/CIFAR100/DIST++/res32X4-SV2"