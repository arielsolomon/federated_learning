from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from clients.client_factory import get_client
from multi_process_federated.node_launchers import start_node
from trainers.simple_trainer import train, test
import torch.multiprocessing as mp


def federated_train(net: torch.nn.Module, test_loader: DataLoader, test_loader_ood: DataLoader,
                    train_loader: DataLoader, num_rounds: int = 1,
                    num_standard_clients: int = 1, add_ood_client: bool = True):
    """
    Perform federated training with multiple clients.

    Parameters:
    net (torch.nn.Module): The neural network model.
    test_loader (torch.utils.data.DataLoader): DataLoader for standard testing data.
    test_loader_ood (torch.utils.data.DataLoader): DataLoader for out-of-distribution (OOD) testing data.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    num_rounds (int, optional): Number of clients sampling rounds. Defaults to 1.
    num_standard_clients (int, optional): Number of standard clients. Defaults to 1.
    add_ood_client (bool, optional): Whether to add an OOD client. Defaults to True.

    Returns:
    None
    """
    clients = [get_client(net=net, train_fn=train, test_fn=test, train_loader=train_loader, test_loader=test_loader)
               for _ in range(num_standard_clients)]

    if add_ood_client:
        client_ood = get_client(net=net, train_fn=train, test_fn=test, train_loader=train_loader,
                                test_loader=test_loader_ood)
        clients += [client_ood]

    nodes = [num_rounds] + clients

    mp.spawn(start_node, args=(nodes,), nprocs=len(nodes))

    load_state_from_client(net=net, client=clients[-1])


def load_state_from_client(client, net):
    """
    Load the state from a client to the given neural network.

    Parameters:
    client: Client from which to load the parameters.
    net (torch.nn.Module): The neural network model.

    Returns:
    None
    """
    parameters = client.get_parameters(config={})
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
