import torch
from federated_trainer import federated_train
import flwr as fl
import numpy as np
from utils import load_datasets, Net, evaluate_on_loaders
DEVICE = torch.device("cpu") #torch.device('cuda' if torch.cuda.is_available() else 'cpu')#torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
NUM_CLIENTS = 1
num_rounds = 1

BATCH_SIZE = 32


trainloader, testloader, testloader_ood = load_datasets(BATCH_SIZE)
net = Net().to(DEVICE)
federated_train(num_standard_clients=NUM_CLIENTS, net=net, test_loader=testloader,
                        test_loader_ood=testloader_ood, train_loader=trainloader, num_rounds=num_rounds)

                #regular training:
# training function

# for epoch in range(5):
#     train(net, trainloader, 1)
#     loss, accuracy = test(net, testloader)
#     print(f"Epoch {epoch+1}: test loss {loss}, accuracy {accuracy}")
#loss, accuracy = test(net, testloader)
#print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

# we convert the parameters and send via flwr to central server
[base_accuracy_on_regular_data, base_accuracy_on_ood_data] = evaluate_on_loaders(net=net, loaders=[testloader, testloader_ood])
np.save('acc_reg_data.npy',np.array(base_accuracy_on_regular_data)), np.save('acc_ood_data.npy', np.array(base_accuracy_on_ood_data))



