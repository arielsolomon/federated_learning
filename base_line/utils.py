import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device("cpu")#torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')#  # Try "cuda" to train on GPU

def load_datasets(BATCH_SIZE):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    transform_ood = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.75, 0.25, 1.0), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("/home/user1/ariel/federated_learning/data/cifar", train=True, download=True, transform=transform)
    testset = CIFAR10("/home/user1/ariel/federated_learning/data/cifar", train=False, download=True, transform=transform)
    testset_ood = CIFAR10("/home/user1/ariel/federated_learning/data/cifar", train=False, download=True, transform=transform_ood)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    testloader_ood = DataLoader(testset_ood, batch_size=BATCH_SIZE)
    return trainloader, testloader, testloader_ood

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in tqdm(range(epochs)):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    accuracies = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracies.append(correct / total)
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracies

def evaluate_on_loaders(net: torch.nn.Module, loaders):
    """
    Evaluate the network on multiple data loaders.

    Parameters:
    net: The neural network model to be evaluated.
    loaders (List[torch.utils.data.DataLoader]): List of DataLoaders for evaluation.

    Returns:
    List[float]: List of accuracies on the respective data loaders.
    """

    accuracies = [test(net=net, testloader=loader)[1] for loader in loaders]
    return accuracies

