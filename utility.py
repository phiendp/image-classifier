import torch
from torchvision import datasets, transforms, models


MEANS = [0.485, 0.456, 0.406]
STANDARD_DEVIATIONS = [0.229, 0.224, 0.225]


def load_img(input_dir):
    train_dir = input_dir + '/train'
    test_dir = input_dir + '/test'
    valid_dir = input_dir + '/valid'

    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MEANS, STANDARD_DEVIATIONS)]),
        'testing': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEANS, STANDARD_DEVIATIONS)]),
        'validation': transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MEANS, STANDARD_DEVIATIONS)])
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=32),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32)
    }

    class_labels = image_datasets['training'].classes
    return class_labels, dataloaders['training'], dataloaders['testing'], dataloaders['validation']
