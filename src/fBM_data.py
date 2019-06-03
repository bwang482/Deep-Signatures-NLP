from fbm import FBM
import numpy as np
import pandas as pd
import random
import iisignature
from paths_transformers import *
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
from torch.utils import data

def generate_fBM(n_paths, n_samples, hurst_exponents):
    """Generate FBM paths"""
    X = []
    y = []
    for j in range(n_paths):
        hurst = random.choice(hurst_exponents)
        X.append(FBM(n=n_samples, hurst=hurst, length=1, method='daviesharte').fbm())
        y.append(hurst)
    return np.array(X), np.array(y)

def generate_data(n_paths_train, n_paths_test, n_samples, hurst_exponents, flag=None, sig=True, depth_lin=5):
    """Generate train and test datasets"""
   
    # generate dataset
    x_train, y_train = generate_fBM(n_paths_train, n_samples, hurst_exponents)
    x_test, y_test = generate_fBM(n_paths_test, n_samples, hurst_exponents)

    # reshape targets
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    if flag == 'lstm':
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)
    
    elif flag == 'time-transform': # this is for linear signatures
        path_transform = AddTime()
        x_train = np.array([iisignature.sig(x, depth_lin) for x in path_transform.fit_transform(x_train)])
        x_test = np.array([iisignature.sig(x, depth_lin) for x in path_transform.fit_transform(x_test)])

    elif flag == 'lead-lag-transform': # this is for linear signatures
        path_transform = LeadLag()
        if sig:
            x_train = np.array([iisignature.sig(x, depth_lin) for x in path_transform.fit_transform(x_train)])
            x_test = np.array([iisignature.sig(x, depth_lin) for x in path_transform.fit_transform(x_test)])
        else:
            x_train = np.swapaxes(np.array(path_transform.fit_transform(x_train)), 1, 2)
            x_test = np.swapaxes(np.array(path_transform.fit_transform(x_test)), 1, 2)
    
    else: # this is for ReluNet, RNN, Signet & Deepsignet
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)

    return x_train, y_train, x_test, y_test


def generate_torch_batched_data(x_train, y_train, x_test, y_test, train_batch_size, test_batch_size):
    """Generate torch dataloaders"""

    # make torch dataset
    train_dataset = data.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    test_dataset = data.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

    # process with torch dataloader
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    test_dataloader = torchdata.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)

    example_batch_x, example_batch_y = next(iter(train_dataloader))

    return train_dataloader, test_dataloader, example_batch_x, example_batch_y


def hurst_fn1(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags] 
    # calculate Hurst as slope of log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2.0
    return hurst