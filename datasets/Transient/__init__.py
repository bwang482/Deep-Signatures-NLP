import numpy as np
import os
import torch
import torch.utils.data as torchdata

from . import generate_data


def get(train=True, num_peaks=5, **kwargs):
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), str(num_peaks))
    if not os.path.isdir(base_path):
        generate_data.gen_and_save_data(num_peaks=num_peaks, loc=base_path, **kwargs)
    join = lambda filename: os.path.join(base_path, filename)
    if train:
        x, y = np.load(join('x_train.npy')), np.load(join('y_train.npy'))
    else:
        x, y = np.load(join('x_test.npy')), np.load(join('y_test.npy'))
    return torchdata.TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float))
