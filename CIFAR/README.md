## KD++/DKD++/DIST++/ReviewKD++ on CIFAR-100.
### 0. Environments

- Python 3.6.9
- PyTorch 1.10.0
- torchvision 0.11.0
- Please put the CIFAR100 dataset in Dataset/.

### 1. train naive teacher and student baselines
  ```bash
  # for instance, train ResNet-56 on CIFAR-100.
  python3 train_cifar_baseline.py \
        --model_name resnet56_cifar \
        --dataset 'cifar100' \
        --epoch 240 \
        --batch_size 64 \
        --lr 0.1 \
        --save_dir "./run/resnet56"
  ```
### 2. compute the class-mean of teachers on training set
  ```bash
  python3 emb_fea_distribution.py \
        --model_name resnet56_cifar \
        --model_weights 'MODEL WEIGHT PATH' \
        --emb_size 64 \
        --dataset 'cifar100' \
        --batch_size 128
  # The class-mean results, json file, be put in ckpt/teacher/.
  # e.g., ckpt/teacher/resnet56/center_emb_train.json
  ```

### 3. train student by KD++
  ```bash
  # for instance, distillation from resnet-56 to resnet-20 by KD++.
  python3 train_cifar_kd.py \
        --model_name resnet20_cifar \
        --teacher resnet56_cifar \
        --teacher_weights 'Teacher WEIGHT PATH' \
        --dataset 'cifar100' \
        --epoch 240 \
        --batch_size 64 \
        --lr 0.1 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 1.0 \
        --nd_loss_factor 2.0 \
        --save_dir "./run/CIFAR100/KD++/res56-res20"

  # other models, please refer to train_cifar.sh.
  ```

### 4. DKD++/DIST++/ReviewKD++
 - DKD++
    ```bash
    python3 train_cifar_dkd.py \
            --model_name shufflev1_cifar \
            --teacher resnet32x4_cifar \
            --teacher_weights 'Teacher WEIGHT PATH' \
            --dataset 'cifar100' \
            --epoch 240 \
            --batch_size 64 \
            --lr 0.02 \
            --cls_loss_factor 1.0 \
            --dkd_alpha 0.5 \
            --dkd_beta 8.0 \
            --nd_loss_factor 4.0 \
            --save_dir "./run/CIFAR100/DKD++/res32x4-sv1"

    # other models, please refer to train_cifar.sh.
    ```
 - DIST++

   cd ./DIST++
    ```bash
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

    # other models, please refer to train_dist.sh.
    ```
 - ReviewKD++

   cd ./ReviewKD++
    ```bash
    python3 train_reviewkd.py \
        --model_name wrn40_1_cifar \
        --teacher wrn40_2_cifar \
        --teacher_weights 'Teacher WEIGHT PATH' \
        --dataset 'cifar100' \
        --epoch 240 \
        --batch_size 64 \
        --lr 0.1 \
        --cls_loss_factor 1.0 \
        --kd_loss_factor 5.0 \
        --nd_loss_factor 4.0 \
        --save_dir "./run/CIFAR100/ReviewKD++/res32X4-SV2"

    # other models, please refer to train_reviewkd.sh.
    ```