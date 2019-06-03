import numpy as np
import os


def gen_data(num_peaks=4, num_samples=100, scale=10):
    """Generate peak data. Peaks are placed in [0, 1], discretised into :num_samples: many points.
    There are :num_peaks: number of peaks, and they decay with a tail whose weight is proportional to :scale:.
    """
    
    peak_locations = []
    peak_region_length = 0.8 / num_peaks
    for i in range(num_peaks):
        start = 0.1 + i * peak_region_length
        peak_locations.append(np.random.uniform(start, start + peak_region_length))
    peak_signs = np.random.binomial(1, 0.5, size=num_peaks)
    results = np.zeros(num_samples)
    x = np.linspace(0, 1, num_samples)
    for peak_location, peak_sign in zip(peak_locations, peak_signs):
        results += (2 * peak_sign - 1) * np.exp(scale * (x - peak_location)) * np.heaviside(peak_location - x, 0)
    results = np.expand_dims(results, axis=0)  # add channel dimension
    return results, peak_signs  # features, labels
    

def gen_and_save_data(num_train=50000, num_test=10000, num_peaks=4, num_samples=100, scale=10, loc='.'):
    """Generates and saves peaks data."""
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    for _ in range(num_train):
        x, y = gen_data(num_peaks, num_samples, scale)
        x_train.append(x)
        y_train.append(y)
        
    for _ in range(num_test):
        x, y = gen_data(num_peaks, num_samples, scale)
        x_test.append(x)
        y_test.append(y)
        
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)
    
    os.makedirs(loc, exist_ok=True)
    
    np.save(os.path.join(loc, 'x_train'), x_train)
    np.save(os.path.join(loc, 'y_train'), y_train)
    np.save(os.path.join(loc, 'x_test'), x_test)
    np.save(os.path.join(loc, 'y_test'), y_test)
