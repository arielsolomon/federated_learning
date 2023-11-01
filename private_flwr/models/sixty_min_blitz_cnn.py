import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Convolutional Neural Network (CNN) model for image classification.

    This network architecture consists of two convolutional layers followed by
    max-pooling and three fully connected layers.

    Attributes:
    - conv1: First convolutional layer
    - pool: Max-pooling layer
    - conv2: Second convolutional layer
    - fc1: First fully connected layer
    - fc2: Second fully connected layer
    - fc3: Third fully connected layer
    """

    def __init__(self) -> None:
        """
        Initialize the neural network layers.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor

        Returns:
        - torch.Tensor: Output tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
