## KD++/DKD++/ReviewKD++ on ImageNet.
### 0. Environments

- Python 3.6.9
- PyTorch 1.10.0
- torchvision 0.11.0
- Download the ImageNet dataset, and write the path to the `./Dataset/ImageNet.py` file.

### 1. Download Pretrained weights for teachers

  - please put the teacher weights in `./ckpt/`
  
### 2. compute the class-mean of teachers on training set
  ```bash
  python3 emb_fea_distribution.py \
        --model_name resnet152 \
        --model_weights 'MODEL WEIGHT PATH' \
        --emb_size 2048 \
        --batch_size 512
  # The class-mean results, json file, be put in ./ckpt/teacher/.
  # e.g., ckpt/teacher/resnet152/center_emb_train.json
  ```

### 3. train student by KD++
  ```bash
  # for instance, distillation from resnet-152 to resnet-18 by KD++.
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

  # other models, please refer to train.sh
  ```

### 4. DKD++/ReviewKD++
 - DKD++
    ```bash
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
        --save_dir "./run/ImageNet/dkd++/res101-res18"

    # other models, please refer to train.sh
    ```

 - ReviewKD++

   cd `./ReviewKD++`
    ```bash
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
        --save_dir "./run/ImageNet/ReviewKD++/res50-mv1"

    # other models, please refer to train_reviewkd.sh.
    ```

### 5. ViT $\rightarrow$ ResNet
 - Install `timm==0.6.12`, and modify ViT to extract embedding features.
 
 - cd `./ViT`
 - KD++
    ```bash
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

    # other models, please refer to train.sh
   ```
 
 - DKD++
    ```bash
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

    # other models, please refer to train.sh
    ```
### 6. CNN $\rightarrow$ ViT
 - cd `./DeiT`
 - We only experimented with KD++.
    ```bash
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
    ```
