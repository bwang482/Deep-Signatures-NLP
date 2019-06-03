import numpy as np
import os
import torch
import torch.utils.data as torchdata

from . import generate_data


def get_signal(num_samples=1000, **kwargs):

    paths = np.array([generate_data.gen_data(**kwargs) for _ in range(num_samples)])
     
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float))


def get_noise(num_samples=1000, **kwargs):

    paths = np.array([generate_data.gen_noise(**kwargs) for _ in range(num_samples)])
    y = np.zeros_like(paths[:, 0, :-1])
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float), torch.tensor(y, dtype=torch.float))