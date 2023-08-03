import torchvision
import torchvision.transforms as T
from timm.data import create_transform

def ImageNet(args, scale_size=256, target_size=224):
    train_data = torchvision.datasets.ImageFolder(root='/root/DataSets/ImageNet2012/ILSVRC2012_img_train')
    test_data = torchvision.datasets.ImageFolder(root='/root/DataSets/ImageNet2012/ILSVRC2012_img_val')
    num_class = 1000

    # base augmentation
    # train_data.transform = T.Compose([
    #     T.RandomResizedCrop(target_size),
    #     T.RandomHorizontalFlip(0.5),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # strong augmentation
    train_data.transform = create_transform(
        input_size=target_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation='bicubic',
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    test_data.transform = T.Compose([
        T.Resize(scale_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_data, test_data, num_class