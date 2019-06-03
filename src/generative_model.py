import functools as ft
import torch
from siglayer import modules, backend
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from esig import tosig
import ast

def get_keys(dim, order):
    """Returns a basis of the truncated tensor algebra over a vector space of a certain dimension."""

    s = tosig.sigkeys(dim, order)
    tuples = []
    
    for t in s.split():
        if len(t) > 2:
            t = t.replace(")", ",)")
        tuples.append(ast.literal_eval(t))
        
    return tuples

def psi(x, M=4, a=1):
    """Psi function, as defined in the following paper:

    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """

    if x <= M:
        return x
    
    return M + M ** (1 + a) * (M ** (-a) - x ** (-a)) / a
  
    
def normalise_instance(x, order):
    """Normalise signature, following the paper

    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """

    x = torch.cat([torch.tensor([1.]), x])
    keys = get_keys(2, order)

    a = x ** 2
    a[0] -= psi(torch.norm(x))
    
    
    # Newton-Raphson
    x0 = 1.
    
    moments = torch.tensor([x0 ** (2 * m) for m in range(len(x))])
    polx0 = torch.dot(a, moments)
    
    d_moments = torch.tensor([2 * m * x0 ** (2 * m - 1) for m in range(len(x))])
    d_polx0 = torch.dot(a, d_moments)
    x1 = x0 - polx0 / d_polx0

    if x1 < 0.2:
        x1 = 1.

    keys = get_keys(2, order)
    
    Lambda = torch.tensor([x1 ** len(t) for t in keys])

    
    return Lambda * x


def normalise(x, order):
    """Normalise signature."""

    return torch.stack([normalise_instance(sig, order) for sig in x])


def loss(orig_paths, sig_depth=2, normalise_sigs=True):
    """Loss function is the T statistic defined in
    
    Chevyrev, I. and Oberhauser, H., 2018. Signature moments to
    characterize laws of stochastic processes. arXiv preprint arXiv:1810.10971.

    """
    
    sig = backend.SigLayer(sig_depth)
    orig_signatures = sig(orig_paths)
    if normalise_sigs:
        orig_signatures = normalise(orig_signatures, sig_depth)

    T1 = torch.mean(torch.mm(orig_signatures, orig_signatures.t()))

    def loss_fn(output, *args):
        timeline = torch.tensor(np.linspace(0, 1, output.shape[1] + 1), dtype=torch.float32)
        paths = torch.stack([torch.stack([timeline, torch.cat([torch.tensor([0.]), path])]) for path in output])

        generated_sigs = sig(paths)

        if normalise_sigs:
            generated_sigs = normalise(generated_sigs, sig_depth)



        T2 = torch.mean(torch.mm(orig_signatures, generated_sigs.t()))
        T3 = torch.mean(torch.mm(generated_sigs, generated_sigs.t()))

        return torch.log(T1 - 2 * T2 + T3)

    return loss_fn

def print_results(history):
    NAMES = {"train log-loss": "train loss",
             "val log-loss": "val loss"}
    fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': 0.6, 'hspace': 0.6}, figsize=(12, 4))
    axs = axs.flatten()
    for i, metric_name in enumerate(NAMES.keys()):
        ax = axs[i]
        for model in history:
            metric = history[model][metric_name]
            
            # Moving average
            metric = np.convolve(metric, np.ones(10), 'valid') / 10.
            ax.semilogy(np.exp(metric))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(NAMES[metric_name])

    plt.show()



class GenerativeModel(nn.Module):
    """Generative model for paths.

    Given a dataset of paths, learns to transform the Wiener measure
    to the measure on path space that the dataset comes from.

    """
    
    def __init__(self, path_dim, **kwargs):
        super(GenerativeModel, self).__init__(**kwargs)

        self.path_dim = path_dim
        
        order = 3
        sig_dim = backend.sig_dim(2, order)
        
        self.layer = nn.Linear(sig_dim, 1, bias=False)
        self.layers = nn.ModuleList([self.layer])
        self.sigs = [modules.ViewSigLayer(2, i + 1, order) for i in range(1, path_dim)]
        self.augmention = modules.Augment((8, 2,), 1, include_original=False, include_time=False)

    def forward(self, x):
        x = self.augmention(x)
        x = torch.stack([self.layer(sig(x[:, :, :i + 2])).flatten() for i, sig in enumerate(self.sigs)]).t()
        
        return x
