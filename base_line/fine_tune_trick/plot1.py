import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
root = os.getcwd()+'/'
np_list = os.listdir(root)


def npy_paths(np_list):
    npy_list_paths = []
    for path in np_list:
        if '19_10' in path:
            npy_list_paths.append(root+path)
    return npy_list_paths
npy_list_paths = npy_paths(np_list)
def plot(npy_list_paths):
    arrs = []
    labels = []
    for path in npy_list_paths:
        arr = np.load(path, allow_pickle=True)
        label = str(path.split('/')[-1].split('.npy')[0])
        arrs.append(arr)
        labels.append(label)
    for ar, lab in zip(arrs, labels):
        plt.plot(ar, label=lab)
    plt.legend(loc='upper left', prop={'size': 6})
    plt.savefig('all_fed_resnet')
    plt.show()
plot(npy_list_paths)

