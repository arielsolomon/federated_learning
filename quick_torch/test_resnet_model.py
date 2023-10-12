import torch
import torchvision
from torchvision import transforms

# Define a function to load the pre-trained ResNet model on CIFAR-10
def load_pretrained_resnet_cifar10(model_name):
    # Check if the model name is valid
    if model_name not in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        raise ValueError("Invalid model name. Please choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152.")

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
if __name__ == "__main__":
    model_name = "resnet18"  # Choose the ResNet model you want to load
    model = load_pretrained_resnet_cifar10(model_name)

    # Create a batch of input data with the expected shape
    dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image size

    # Pass the batch of input data through the model
    output = model(dummy_input)
    print(f"Loaded pre-trained {model_name} model on CIFAR-10. Output shape: {output.shape}")
