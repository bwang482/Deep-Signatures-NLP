import numpy as np
from esig import tosig
import torch.nn as nn
import torch
from siglayer import modules, backend
import iisignature
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def plot(points, label=""):
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Interpolation for different methods:
    alpha = np.linspace(0, 1, 75)

    interpolator =  interp1d(distance, points, kind="quadratic", axis=0)
    interpolated_points = interpolator(alpha)

    plt.plot(*interpolated_points.T, '-', label=label)


def _get_tree_reduced_steps(X, order=4, steps=4, tol=0.1):
    if len(X) < steps:
        return X
    
    dim = X.shape[1]

    for i in range(steps - 1, len(X)):
        new_path = X[i - steps + 1 : i + 1]
        new_path2 = np.r_[X[i - steps + 1].reshape(-1, dim),
                          X[i].reshape(-1, dim)]

        new_path_sig = iisignature.sig(new_path, order)
        new_path2_sig = iisignature.sig(new_path2, order)

        norm = np.linalg.norm(new_path_sig - new_path2_sig)
        if norm < tol:
            return _get_tree_reduced_steps(np.r_[X[:i - steps + 2],
                                                 X[i:]])
        
    return X

def get_tree_reduced(X, order=4, tol=0.1):
    """Removes tree-like pieces of the path."""

    X = np.r_[X, [X[-1]]]

    for step in range(3, len(X) + 1):
        X = _get_tree_reduced_steps(X, order, step, tol)
        
    if (X[-1] == X[-2]).all():
        return X[:-1]

    return X
        

def loss_fn(order):
    lengths = torch.tensor([np.floor(np.log(i + 1) / np.log(2)) for i in range(1, 2 ** (order + 1) - 1)], dtype=torch.float)
    def loss(output, target):
        normalisation = lengths
        output *= normalisation
        target *= normalisation
        
        return torch.log(((output - target)**2).mean())

    return loss


class Invert(nn.Module):
    """Given a signature, we build a NN that learns the inverse of that signature."""
    def __init__(self, n_steps, order, derivatives=False, **kwargs):
        super(Invert, self).__init__(**kwargs)

        self.n_steps = n_steps
        self.order = order
        self.derivatives = derivatives

        self.path = nn.Linear(1, 2 * n_steps, bias=False)
        self.layers = nn.ModuleList([self.path])
        self.sig = modules.ViewSigLayer(2, self.n_steps, self.order)
        
    def forward(self, x):
        x = self.path(x)
        if self.derivatives:
            x = torch.cumsum(x, 1)
            
        x.view(-1, 2, self.n_steps)

        return self.sig(x)

    def get_path(self):
        x = torch.ones(1, 1, 1)
        x = self.path(x)
        if self.derivatives:
            x = torch.cumsum(x, 1)

        x = x.view(-1, 2, self.n_steps)
        return np.array(x.detach().numpy()[0].T, dtype=float)