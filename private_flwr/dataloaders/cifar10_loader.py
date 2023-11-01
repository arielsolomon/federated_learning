import os.path
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset


def load_data(root: str, batch_size=32, ood=False):
    """
    Load CIFAR-10 dataset.

    Parameters:
    root (str): Root directory where the dataset will be stored.
    batch_size (int, optional): Batch size for data loaders. Defaults to 32.
    ood (bool, optional): Flag indicating if the data is out-of-distribution (OOD). Defaults to False.

    Returns:
    Tuple[DataLoader, DataLoader]: DataLoader for the training set and DataLoader for the test set.
    """
    assert os.path.exists(root), f'{root} does not exist'

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) if not ood else transforms.Compose(
        [transforms.ToTensor(), transforms.ColorJitter(contrast=0.5, brightness=1.0),
         transforms.Normalize((1.0, 0.25, 0.1), (0.5, 0.5, 0.5))]
    )

    print(f'load_data to {root}, ood? {ood}')
    train_set = CIFAR10(root, train=True, download=True, transform=transform)
    test_set = CIFAR10(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, test_loader


def load_raw_data(root):
    train_set = CIFAR10(root, train=True, download=True)
    test_set = CIFAR10(root, train=False, download=True)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return train_set, test_set, classes
