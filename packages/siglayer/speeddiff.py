import sys
sys.path.append('..')

import backend as siglayer
import timeit
import torch

ref = []

def setup():
    path = torch.randn(10, 3, dtype=torch.float32, requires_grad=True, 
                       device='cpu')
    ref[:] = [path]

def stmt():
    x = siglayer.path_sig(path=ref[0], depth=4)
    x.backward(torch.ones_like(x))

print('Using iisignature')
siglayer.select_backend('iisignature')
iisignature_time = timeit.timeit(setup=setup, stmt=stmt, number=1000)
print('iisignature time: {}'.format(iisignature_time))
print('Using PyTorch')
siglayer.select_backend('pytorch')
pytorch_time = timeit.timeit(setup=setup, stmt=stmt, number=1000)
print('PyTorch time: {}'.format(pytorch_time))
print('Ratio: {}'.format(pytorch_time / iisignature_time))

