import warnings
import numpy as np
import torch
from datetime import datetime
from client1_finetuning_utils import load_data, \
    train, test, load_pretrained_resnet_cifar10,\
    evaluate_on_loaders, average_models,save_iteration_results
from tqdm import tqdm
import copy

# big pic:
#     load model, load data both in distribution and ood
#     train single epoch with ood, take original and ood models and average it
#     run evaluation with averaged model

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "resnet34"
#model_name = model_name
# Choose the ResNet model you want to load
net = load_pretrained_resnet_cifar10(model_name).to(DEVICE)
print(f"\nLoaded pre-trained {model_name} model on CIFAR-10.\n")

trainloader, testloader, trainloader_ood, testloader_ood = load_data()
_,accuracy_list_on_original = evaluate_on_loaders(net, testloader, DEVICE)
_,accuracy_list_on_ood = evaluate_on_loaders(net, testloader_ood, DEVICE)
current_datetime = datetime.now()
current_time = current_datetime.strftime("%d_%m_%y")
native_file_name, aug_file_name = current_time+'_'+model_name+'_'+'_native_fine_tune_train.npy', current_time+'_'+model_name+'_'+'_aug_fine_tune_train.npy'

orig_pretrained_net = copy.deepcopy(net)
for _ in tqdm(range(len(trainloader))):
    train(net, testloader_ood, 1, DEVICE)

    average_models(net, orig_pretrained_net)
    _,_, acc = test(net, testloader, DEVICE)
    for i in acc:
        #appending in list1
        accuracy_list_on_original.append(i)
        #accuracy_list_on_original.append(acc)
    _,_, acc = test(net,testloader_ood, DEVICE)
    for i in acc:
        #appending in list1
        accuracy_list_on_ood.append(i)

    #accuracy_list_on_ood.append(acc)
    save_iteration_results(native_file_name, aug_file_name,accuracy_list_on_ood, accuracy_list_on_original)
#np.save('test_accuracies_native_'+model_name+'_'+current_time+'.npy', acc_npy)

