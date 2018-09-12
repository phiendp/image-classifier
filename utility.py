import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import json
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt


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


def save_checkpoint(state, save_dir=None):
    if save_dir:
        torch.save(state, save_dir + '/checkpoint.pth')
    else:
        torch.save(state, 'checkpoint.pth')


def load_saved_model(filename='checkpoint.pth'):
    print("Loading '{}'".format(filename))
    checkpoint = torch.load(filename)
    model = load_pretrained_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['class_labels']


def load_pretrained_model(model_name, hidden_units):
    '''
    Define a new, untrained/pretrained feed-forward network as a classifier.
    Using ReLU activations and dropout.
    '''
    if model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    else:
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p = 0.3)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p = 0.2)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    return model


def validation(model, valid_data, criterion, gpu=False):
    '''
    Track the loss and accuracy on the validation set to determine the best hyperparameters.
    '''
    test_loss = 0
    accuracy = 0
    total = 0
    correct = 0

    for images, labels in valid_data:
        if gpu is True:
            images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()
    accuracy = 100 * correct / total
    return test_loss, accuracy


def train(model, learning_rate, criterion, train_data, valid_data, epochs, gpu=False):
    '''
    Train the classifier layers using backpropagation using the pre-trained network to get the features.
    '''
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    epochs = epochs
    print_every = 40
    steps = 0

    if gpu is True:
        model.to('cuda')

    if epochs == 0:
        return

    for e in range(epochs):
        total_loss = 0
        running_loss = 0
        for inputs, labels in train_data:
            steps += 1
            if gpu is True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                    "Loss every 40 steps: {:.4f}".format(running_loss / print_every))
                running_loss = 0

        model.eval()
        with torch.no_grad():
            train_loss, train_accuracy = validation(model, train_data, criterion)
            valid_loss, accuracy = validation(model, valid_data, criterion, gpu)

            print("Epoch: {}/{}... ".format(e + 1, epochs),
                "Training Loss: {:.3f}.. ".format(train_loss),
                "Training Accuracy: {:.3f}.. ".format(train_accuracy),
                "Validation Loss: {:.3f}.. ".format(valid_loss),
                "Validation Accuracy: {:.3f}".format(accuracy))
        total_loss = 0
        model.train()


def test(model, test_data, gpu=False):
    '''
    Run the test images through the network and measure the accuracy.
    '''
    correct = 0
    total = 0
    if gpu is True:
        model.to('cuda')
    model.eval()
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            if gpu is True:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))


def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    '''
    img_loader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(MEANS, STANDARD_DEVIATIONS)])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)

    return np_image


def predict(image_path, model, topk, cat_to_name, class_labels, gpu=False):
    '''
    Predict the class (or classes) of an image using the trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor.resize_([1, 3, 224, 224])
    model.to('cpu')

    if gpu is True:
        print("Using GPU")
        model.to('cuda')
        image_tensor = image_tensor.to('cuda')

    result = torch.exp(model(image_tensor))

    probs, index = result.topk(topk)
    probs, index = probs.detach(), index.detach()
    probs.resize_([topk])
    index.resize_([topk])
    probs, index = probs.tolist(), index.tolist()

    label_index = []
    for i in index:
        label_index.append(int(class_labels[int(i)]))

    labels = []
    for i in label_index:
        labels.append(cat_to_name[str(i)])
    return probs, labels, label_index
