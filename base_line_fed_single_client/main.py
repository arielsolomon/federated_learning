# Imports
from collections import OrderedDict
import numpy as np
import torch
import pathlib
from utils import trainer1, get_client, start_client, start_server, start_node
from load_dataset import load_data
from load_model import Net
from trainer import train, test
import flwr as fl
import multiprocessing
from tqdm import tqdm



def main():
    DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

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
    # Loading model

    net = Net().to(DEVICE)
    net1 = Net().to(DEVICE)
    net_freez = net
    net_freez1 = net1

    #freez all model layers

    for param in net_freez.parameters():
        param.requires_grad = False
    for param in net_freez1.parameters():
        param.requires_grad = False

    # unfreez last model layer:

    for param in net_freez.fc3.parameters():
        param.requires_grad = True
    for param in net_freez1.fc3.parameters():
        param.requires_grad = True
    print('\nFreezing suceed')

    # loading datasets into trainloader and testloader

    NUM_CLIENTS = 1
    BATCH_SIZE = 32
    root = pathlib.Path().absolute()

    trainloader, testloader = load_data(root,BATCH_SIZE,ood=False)
    trainloader_ood, testloader_ood = load_data(root,BATCH_SIZE,ood=True)

    # Simulating clients
    print('\nGetting clients')
    clients = [get_client(net=net, train_fn=train, test_fn=test, trainloader=trainloader, testloader=testloader)
               for _ in range(NUM_CLIENTS)]
    print('\nGetting clients_ood')
    client_ood = get_client(net=net1, train_fn=train, test_fn=test, trainloader=trainloader, testloader=testloader_ood)
    # What is the nodes? connectivity to server?
    nodes = [1] + clients + [client_ood]

    # my guess is connecting nodes to server
    with multiprocessing.Pool() as pool:
        pool.map(start_node, nodes)

    print()
    print('******************')
    print('Few shot training')
    print('******************')
    parameters = client_ood.get_parameters(config={})
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    print()
    print('Verify weights')
    loss, acc = test(net=net, testloader=testloader)
    loss_ood, acc_ood = test(net=net, testloader=testloader_ood)
    losses_on_original, accs_on_original, losses_on_aug, accs_on_aug = [loss_ood], [acc], [loss_ood], [acc_ood]
    print(f'Retrain')
    for i in tqdm(range(len(trainloader_ood))):
        # num_few_shots = (i+1) * args.batch_size
        # print(f'train using {num_few_shots} images. ({i+1} batches of {args.batch_size})')
        train(net=net, trainloader=trainloader_ood, epochs=1, iterations=1)
        # print('Original test set. Expect degraded acc')
        loss, acc = test(net=net, testloader=testloader)
        losses_on_original.append(loss)
        accs_on_original.append(acc)
        # print('OOD test set. Expect better acc than earlier')
        loss, acc = test(net=net, testloader=testloader_ood)
        losses_on_aug.append(loss)
        accs_on_aug.append(acc)
        with open('accs_on_aug.npy', 'wb') as f:
            np.save(f, np.array(accs_on_aug))
        with open('losses_on_aug.npy', 'wb') as f:
            np.save(f, np.array(losses_on_aug))
        with open('accs_on_original.npy', 'wb') as f:
            np.save(f, np.array(accs_on_original))
        with open('losses_on_original.npy', 'wb') as f:
            np.save(f, np.array(losses_on_original))

if __name__ == "__main__":
    main()