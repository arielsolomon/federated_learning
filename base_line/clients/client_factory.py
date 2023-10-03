from clients.simple_numpy_client import SimpleNumpyClient


def get_client(net, train_fn, test_fn, train_loader, test_loader):
    """
    Get a SimpleNumpyClient instance.

    Parameters:
    net: The neural network model.
    train_fn: The training function for the model.
    test_fn: The testing function for the model.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    test_loader (torch.utils.data.DataLoader): DataLoader for testing data.

    Returns:
    SimpleNumpyClient: An instance of the SimpleNumpyClient.
    """
    return SimpleNumpyClient(net=net, train_fn=train_fn, test_fn=test_fn, train_loader=train_loader,
                             test_loader=test_loader)
