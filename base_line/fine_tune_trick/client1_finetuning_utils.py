import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
from tqdm import tqdm
import numpy as np

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf_train = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trf_test = Compose([ToTensor(), ColorJitter(contrast=0.5, brightness=1.0),
                        Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf_train)
    testset = CIFAR10("./data", train=False, download=True, transform=trf_train)
    trainset_ood = CIFAR10("./data", train=True, download=True, transform=trf_test)
    testset_ood = CIFAR10("./data", train=False, download=True, transform=trf_test)
    return DataLoader(trainset, batch_size=256, shuffle=True, drop_last=True)\
        , DataLoader(testset, batch_size=256, shuffle=True, drop_last=True)\
        ,DataLoader(trainset_ood, batch_size=256, shuffle=True, drop_last=True)\
        ,DataLoader(testset_ood, batch_size=256, shuffle=True, drop_last=True)
def average_models(net: torch.nn.Module, orig_pretrained_net: torch.nn.Module):
    """
    Average the weights of the current network with the original pretrained network.

    Parameters:
    net (torch.nn.Module): The neural network model to be averaged.
    orig_pretrained_net (torch.nn.Module): The original pretrained neural network.

    Returns:
    None
    """
    for p, op in zip(net.parameters(), orig_pretrained_net.parameters()):
        p.data += op.data
        p.data /= 2.0
def train(net, trainloader, epochs, DEVICE):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()
def test(net, testloader, DEVICE):
    """Validate the model on the test set."""
    acc_out = []
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    acc_all = []
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            len_images = len(testloader.dataset)
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            acc_all.append(correct / len_images)
    accuracy = correct / len(testloader.dataset)

    return loss, accuracy, acc_all

def load_pretrained_resnet_cifar10(model_name):
    # Check if the model name is valid
    if model_name not in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        raise ValueError("Invalid model name. Please choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'.")

    # Load the pre-trained ResNet model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        model = torchvision.models.resnet152(pretrained=True)

    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)

    return model
def evaluate_on_loaders(net, testloader, DEVICE):
    _, accuracy, acc_all = test(net, testloader, DEVICE)
    return accuracy, acc_all
def save_iteration_results(native_file_name, aug_file_name,accuracy_list_on_ood_data, accuracy_list_on_standard_data):

    with open(aug_file_name, 'wb') as f:
        np.save(f, np.array(accuracy_list_on_ood_data))

    with open(native_file_name, 'wb') as f:
        np.save(f, np.array(accuracy_list_on_standard_data))

