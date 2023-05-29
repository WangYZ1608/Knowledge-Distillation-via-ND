## KD++/ReviewKD++ on COCO object detection.

### 0. Environments
Our code is based on Detectron2, please install Detectron2 refer to https://github.com/facebookresearch/detectron2.
- Python 3.9.13
- PyTorch 1.9.0
- cuda 10.2
- torchvision 0.10.0
- detectron2  0.6+cu102
- Please put the [COCO](https://cocodataset.org/#download) dataset in datasets/.
- Please put the pretrained weights for teacher and student in pretrained/. The teacher's weights come from Detectron2's pretrained detector. The student's weights are ImageNet pretrained weights.

### 1. train student by KD++
- the class-mean put in configs/Center/

```
# Tea: R-101, Stu: R-18
python3 train_net.py \
        --config-file configs/KD++/R18-R101.yaml \
        --num-gpus 4

# Tea: R-101, Stu: R-50
python3 train_net.py \
        --config-file configs/KD++/R50-R101.yaml \
        --num-gpus 4

# Tea: R-50, Stu: MV2
python3 train_net.py \
        --config-file configs/KD++/MV2-R50.yaml \
        --num-gpus 4
```

### 2. train student by ReviewKD++

```
# Tea: R-101, Stu: R-18
python3 train_net.py \
        --config-file configs/ReviewKD++/R18-R101.yaml \
        --num-gpus 4

# Tea: R-101, Stu: R-50
python3 train_net.py \
        --config-file configs/ReviewKD++/R50-R101.yaml \
        --num-gpus 4

# Tea: R-50, Stu: MV2
python3 train_net.py \
        --config-file configs/ReviewKD++/MV2-R50.yaml \
        --num-gpus 4
```