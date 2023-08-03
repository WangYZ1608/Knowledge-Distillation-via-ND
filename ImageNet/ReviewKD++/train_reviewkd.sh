# ----------------------------------------------------------------------------------------------
# ReviewKD++
# --------------------------------------------------------
# 1、resnet34 -  resnet18     1.0*cls + 1.0*kd + 3.0*nd
# 2、resnet50 -  resnet18     1.0*cls + 3.0*kd + 2.0*nd
# 3、resnet101 - resnet18     1.0*cls + 3.0*kd + 1.0*nd
# 4、resnet152 - resnet18     1.0*cls + 3.0*kd + 1.0*nd
# 5、resnet50 - mobilenetv1   1.0*cls + 8.0*kd + 4.0*nd
# --------------------------------------------------------
python3 train_reviewkd.py \
        --model_name mobilenetv1 \
        --teacher resnet50 \
        --teacher_weights 'Teacher WEIGHT PATH' \
        --dist_url 'tcp://localhost:' \
        --multiprocessing-distributed \
        --world_size 1 \
        --rank 0 \
        --batch_size 512 \
        --lr 0.2 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 8.0 \
        --nd_loss_factor 4.0 \
        --save_dir "./run/ImageNet/ReviewKDKD++/res50-mv1"