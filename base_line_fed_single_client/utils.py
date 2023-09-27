import flwr as fl
from time import sleep
from trainer import train, test
from simple_numpy_client import SimpleNumpyClient

# training and testing the model via trainer.py
def trainer1(trainloaders, valloaders, testloader, net):
    net = net
    trainloader = trainloaders[0]
    valloader = valloaders[0]
    testloader = testloader
    for epoch in range(5):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloaders)
        print(f"Epoch {epoch + 1}: validation loss {loss}, accuracy {accuracy}")

    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

# simulating client in form of SimpleNumpyClient
def get_client(net, train_fn, test_fn, trainloader, testloader):
    return SimpleNumpyClient(net=net, train_fn=train_fn, test_fn=test_fn, trainloader=trainloader,
                             testloader=testloader)

# starting communication from client
def start_client(client):
    print('start_client')
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)
    print('after start client')

# starting communication from server
def start_server():
    print('start_server')
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=2))
    print('after start server')

# simulating connection between client and server
def start_node(arg):
    if isinstance(arg, SimpleNumpyClient):
        print('Sleep ...')
        sleep(1)
        print('Launch Client')
        start_client(arg)
        print('Exit Client')
    else:
        print('Launch Server')
        start_server()
        print('Exit Server')
