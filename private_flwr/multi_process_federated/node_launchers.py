from time import sleep
import flwr as fl
from clients.simple_numpy_client import SimpleNumpyClient


def start_client(client):
    """
    Start a client using NumPy.

    Parameters:
    client: The client to be started.

    Returns:
    None
    """
    print('start_client')
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)
    print('after start client')


def start_server(num_rounds=1):
    """
    Start the federated learning server.

    Returns:
    None
    """
    print('start_server')
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds))
    print('after start server')


def start_node(i, arg):
    """
    Start a node, either a client or a server.

    Parameters:
    arg: An instance of SimpleNumpyClient for client, or anything else for the server.

    Returns:
    None
    """
    arg=arg[i]
    if isinstance(arg, SimpleNumpyClient):
        print('Sleep ...')
        sleep(1)
        print('Launch Client')
        start_client(arg)
        print('Exit Client')
    else:
        assert isinstance(arg, int), f'server excepts integer argument. Got {arg}'
        assert arg > 0, f'server excepts a positive integer argument for num_rounds. Got {arg}'
        print('Launch Server')
        start_server(arg)
        print('Exit Server')
