import torch
from torchvision import datasets
import torchvision.transforms as T


def CIFAR(name='cifar10', valid_size=None):

    assert name in ["cifar10", "cifar100"]
    if name == "cifar10":
        train_data = datasets.CIFAR10(root="Dataset/data", train=True)
        test_data = datasets.CIFAR10(root="Dataset/data", train=False)
        num_class = 10

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]

    elif name == "cifar100":
        train_data = datasets.CIFAR100(root="Dataset/data", train=True)
        test_data = datasets.CIFAR100(root="Dataset/data", train=False)
        num_class = 100

        mean = [0.5071, 0.4867, 0.4408]
        std=[0.2675, 0.2565, 0.2761]

    train_data.transform = T.Compose([
        T.Pad(4),
        T.RandomCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    test_data.transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    

    if valid_size:

        if name == 'cifar10':
            valid_data = datasets.CIFAR10(root="Dataset/data", train=True)
        elif name == 'cifar100':
            valid_data = datasets.CIFAR100(root="Dataset/data", train=True)

        valid_data.transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
        ])
        
        indices = torch.randperm(len(train_data))
        train_indices = indices[:len(indices) - int(valid_size * len(indices))]
        valid_indices = indices[len(indices) - int(valid_size * len(indices)):]
        train_data = torch.utils.data.Subset(train_data, train_indices)
        valid_data = torch.utils.data.Subset(valid_data, valid_indices)
    else:
        valid_data = None
    
    return train_data, test_data, num_class