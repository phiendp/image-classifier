import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict


MEANS = [0.485, 0.456, 0.406]
STANDARD_DEVIATIONS = [0.229, 0.224, 0.225]


def load_img(input_dir):
    '''
    Load the training, testing and validation data using torchvision
    '''
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


def load_json(filename):
    '''
    Load in a mapping from category label to category name
    '''
    with open(filename, 'r') as f:
        cat_to_name = json.load(f, object_pairs_hook=OrderedDict)
    return cat_to_name
