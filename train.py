import argparse
import utility
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default='flowers', type = str,
        help = 'Path to the folder flowers.')
    parser.add_argument('--save_dir', type = str,
        help = 'Path to save directory.')
    parser.add_argument('--arch',type=str, default='densenet', choices=('vgg13', 'densenet'),
        help = 'CNN model architecture to use.')
    parser.add_argument('--learning_rate', default = 0.0001,
        help = 'Learn rate used to train network.')
    parser.add_argument('--hidden_units', type = int, default = 2509,
        help = 'Number of hidden unit layers in network.')
    parser.add_argument('--epochs', type = int, default=5,
        help = 'Number of epochs used to train network.')
    parser.add_argument('--gpu', action="store_true", default = False,
        help = 'Turn GPU on to use for testing.')

    return parser.parse_args()


def main():
    user_args = get_args()

    class_labels, train_data, test_data, valid_data = utility.load_img(user_args.data_dir)
    model = utility.load_pretrained_model(user_args.arch, user_args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=user_args.learning_rate)
    utility.train(model, user_args.learning_rate, criterion, train_data, valid_data, user_args.epochs, user_args.gpu)
    utility.test(model, test_data, user_args.gpu)
    model.to('cpu')

    # Save Checkpoint for predection
    utility.save_checkpoint({
                    'arch': user_args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'hidden_units': user_args.hidden_units,
                    'class_labels': class_labels
                }, user_args.save_dir)
    print('Saved checkpoint!')


if __name__ == "__main__":
    main()
