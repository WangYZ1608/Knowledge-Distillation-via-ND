## KD++/DKD++/ReviewKD++ on ImageNet.
### 0. Environments

- Python 3.6.9
- PyTorch 1.10.0
- torchvision 0.11.0
- Download the ImageNet dataset, and write the path to the ./Dataset/ImageNet.py file.

### 1. Download Pytorch weights for teachers

  - please put the teacher weights in ./ckpt/
  
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
- resnet-152  --> resnet-18
  ```bash
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
        --save_dir "./run/ImageNet/dKD++/res101-res18"

    # other models, please refer to train.sh
    ```

 - ReviewKD++

   cd ./ReviewKD++
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
        --save_dir "./run/ImageNet/ReviewKDKD++/res50-mv1"

    # other models, please refer to train_reviewkd.sh.
    ```

### 5. ViT distillate to CNN
 - Install timm==0.6.12, and modify ViT to extract embedding features.
 cd ./ViT
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