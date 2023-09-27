from collections import OrderedDict
import flwr as fl
import torch


class SimpleNumpyClient(fl.client.NumPyClient):
    client_counter = 0
    def __init__(self, net, train_fn, test_fn, trainloader, testloader):
        super(SimpleNumpyClient, self).__init__()
        SimpleNumpyClient.client_counter += 1
        self._net = net
        self._train = train_fn
        self._test = test_fn
        self._trainloader = trainloader
        self._testloader = testloader
        self._id = SimpleNumpyClient.client_counter
        print('SimpleNumpyClient.__init__', self._id)

    def get_parameters(self, config):
        print()
        print('******************************************')
        print(f'SimpleNumpyClient.get_parameters id {self._id} config {config}')
        print('******************************************')
        return [val.cpu().numpy() for _, val in self._net.state_dict().items()]

    def set_parameters(self, parameters):
        print()
        print('******************************************')
        print(f'SimpleNumpyClient.set_parameters id {self._id} parameters num {len(parameters)}')
        print('******************************************')
        params_dict = zip(self._net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self._net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print()
        print('******************************************')
        print('SimpleNumpyClient.fit', self._id)
        self.set_parameters(parameters)
        self._train(self._net, self._trainloader, epochs=1)
        return self.get_parameters(config={}), len(self._trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print()
        print('******************************************')
        print('SimpleNumpyClient.evaluate', self._id)
        print('******************************************')
        self.set_parameters(parameters)
        loss, accuracy = self._test(self._net, self._testloader)
        return float(loss), len(self._testloader.dataset), {"accuracy": float(accuracy)}