import warnings
from collections import OrderedDict
import torch
import torchvision
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data_ood", train=True, download=True, transform=trf)
    testset = CIFAR10("./data_ood", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True, drop_last=True), DataLoader(testset, batch_size=32, shuffle=True, drop_last=True)


# model = torch.load('resnet56.pt')
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
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



# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
#net = Net().to(DEVICE)


# Step 1: Initialize model with the best available weights

# Define a function to load the pre-trained ResNet model on CIFAR-10
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

# Usage

model_name = "resnet50"  # Choose the ResNet model you want to load
net = load_pretrained_resnet_cifar10(model_name).to(DEVICE)
print(f"\nLoaded pre-trained {model_name} model on CIFAR-10.\n")

trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy,_ = test(net, testloader)

        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)


def evaluate_on_loaders(net, testloader):
    _, accuracy, acc_all = test(net, testloader)
    print('\nAll accuracies: ', acc_all)
    return accuracy, acc_all
_,npy_test = evaluate_on_loaders(net, testloader)
acc_npy = np.array(npy_test)
np.save('test_accuracies_ood.npy', acc_npy)
