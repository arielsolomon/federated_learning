
print('\nLoading dataset')
import os.path

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def load_data(root: str, batch_size=32, ood=False):
    """Load CIFAR-10 (training and test set)."""

    assert os.path.exists(root), f'{root} does not exist'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) if not ood else transforms.Compose(
        [transforms.ToTensor(), transforms.ColorJitter(contrast=0.5, brightness=1.0),
         transforms.Normalize((1.0, 0.25, 0.1),
                              (0.5, 0.5, 0.5))]
    )

    print(f'load_data to {root}, ood? {ood}')
    trainset = CIFAR10(root, train=True, download=True, transform=transform)
    testset = CIFAR10(root, train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    print('\nLoaded dataset')
    return trainloader, testloader
    

