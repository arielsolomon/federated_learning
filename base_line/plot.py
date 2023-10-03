import matplotlib.pyplot as plt
import numpy as np

root = '/home/user1/ariel/federated_learning/base_based_on_tutorial/'
path1 = root + 'acc_arr.npy'
path2 = root + 'acc_ood_data.npy'
def plot(path1, path2):
    reg_data = np.load(path1)
    ood_data = np.load(path2)
    plt.plot(reg_data, label='reg_data')
    plt.plot(ood_data, label='ood_data')
    plt.legend(loc='upper left')
    plt.show()

plot(path1, path2)
