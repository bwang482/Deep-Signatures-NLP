import numpy as np
import os
import sdepy

def gen_data(n_points=100):
    """Generate a stochastic process."""

    sde = sdepy.ornstein_uhlenbeck_process()
    timeline = np.linspace(0, 1, n_points)
    values = sde(timeline).flatten()
    path = np.c_[timeline, values.tolist()]

    return path.T
    


def gen_noise(n_points=100):
    """Generate a stochastic process."""
    
    dt = 1 / np.sqrt(n_points)
    bm = dt * np.r_[0., np.random.randn(n_points - 1).cumsum()]
    timeline = np.linspace(0, 1, n_points)


    return np.c_[timeline, bm].T    
    #return bm.reshape(1, -1)
