# ReviewKD++
# ----------------------------------------------------------------------------------------------
# 1、resnet34 - resnet18         1.0*cls + 3.0*nd + 1.0*kd
# 2、resnet50 - resnet18         1.0*cls + 2.0*nd + 3.0*kd
# 3、resnet101 - resnet18        1.0*cls + 1.0*nd + 3.0*kd
# 4、resnet152 - resnet18        1.0*cls + 1.0*nd + 3.0*kd
# 5、resnet50 - mobilenet        1.0*cls + 4.0*nd + 8.0*kd
# ----------------------------------------------------------------------------------------------
python3 train_dkd.py \
        --model_name mobilenetv1 \
        --teacher resnet50 \
        --teacher_weights './ckpt/teacher/resnet50/resnet50-19c8e357.pth' \
        --dist_url 'tcp://localhost:10152' \
        --multiprocessing-distributed \
        --world_size 1 \
        --rank 0 \
        --batch_size 512 \
        --lr 0.2 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 6.0 \
        --nd_loss_factor 1.0 \
        --save_dir "./run/student_ckpt/ReviewKD++/res50-mv1"