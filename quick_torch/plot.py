import matplotlib.pyplot as plt
import os
import numpy as np
root = os.getcwd()+'/'
np_list = os.listdir(root)

for path in np_list:
    if 'npy' in path:
        if 'ood' in path:
            aug_path = root+path
        else:
            native_path = root+path
def plot(native_path,aug_path):
    arr_native, arr_aug  = (np.load(native_path,allow_pickle=True),
                            np.load(aug_path,allow_pickle=True))
    plt.plot(arr_native, label='Native_data')
    plt.plot(arr_aug, label='Ood_data')
    plt.legend(loc='upper left')
    plt.show()
def plot1(aug_path, native_path):
    arr_aug, arr_native  = np.load(aug_path), np.load(native_path)
    plt.plot(arr_aug, label='Augmented data')
    plt.plot(arr_native, label='Native_data')
    plt.legend(loc='upper left')
    plt.show()

plot(native_path, aug_path)

