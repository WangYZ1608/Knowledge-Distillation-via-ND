import torchvision
import torchvision.transforms as T

def ImageNet(scale_size=256, target_size=224):
    train_data = torchvision.datasets.ImageFolder(root='ImageNet Train PATH')
    test_data = torchvision.datasets.ImageFolder(root='ImageNet Val PATH')
    num_class = 1000

    train_data.transform = T.Compose([
        T.RandomResizedCrop(target_size),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data.transform = T.Compose([
        T.Resize(scale_size),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_data, test_data, num_class