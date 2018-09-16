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


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='Path to the input image file.')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File that contains categories mapping.')
    parser.add_argument('--gpu', action="store_true", default=False, help='Turn GPU on.')

    return parser.parse_args()


def main():
    user_args = get_input_args()

    model, class_labels = utility.load_saved_model()
    cat_to_name = utility.load_json(user_args.category_names)
    probs, labels, _ = utility.predict(user_args.input, model, user_args.top_k, cat_to_name, class_labels, user_args.gpu)

    print("------------------Processing------------------")
    for i in range(len(probs)):
        result_label = labels[i]
        result_prob = probs[i] * 100
        print("The probability of the {} is {:.2f} %.".format(result_label, result_prob))


if __name__ == "__main__":
    main()
