import torch
from torch import nn, optim
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
    Track the loss and accuracy on the validation set to determine the best hyperparameters
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
    Train the classifier layers using backpropagation using the pre-trained network to get the features
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
        for inputs, labels in enumerate(train_data):
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
            valid_loss, accuracy = validation(model, valid_data, criterion, gpu)
            print("Epoch: {}/{}... ".format(e + 1, epochs),
                "Training Loss: {:.3f}.. ".format(total_loss),
                "Validation Loss: {:.3f}.. ".format(valid_loss),
                "Validation Accuracy: {:.3f}".format(accuracy))
        total_loss = 0
        model.train()
