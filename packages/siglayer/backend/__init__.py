# We'll use the iisignature-based module if we can, as it's much faster (written directly in C++).  But if need be we
# can use the pure-PyTorch version; the functionality of the two modules is identical.

import candle
import torch.nn as nn
import warnings

from .utils import sig_dim

try:
    import iisignature
except ImportError:
    from .pytorch_implementation import path_sig_base
    selected_backend = 'pytorch'
else:
    from .iisignature_implementation import path_sig_base
    selected_backend = 'iisignature'
    
    
def select_backend(backend):
    global selected_backend, path_sig_base

    if backend == 'pytorch':
        from .pytorch_implementation import path_sig_base
        selected_backend = 'pytorch'
    elif backend == 'iisignature':
        from .iisignature_implementation import path_sig_base
        selected_backend = 'iisignature'
    else:
        raise ValueError(f'Backend {backend} not understood. Valid values are "pytorch" or "iisignature".')
    
    
def path_sig(path, depth):
    return path_sig_base(path, depth)
    
    
batch_path_sig = candle.batch_fn(path_sig)


class Signature(nn.Module):
    """Given some path mapping from, say, [0, 1] into \reals^n, we may define the 'signature' of the path as a
    particular sigtensor with respect to an alphabet of n letters. (Note how n is the target dimension of the path.)
    That is, the signature is a map from the space of paths to the tensor algebra. Up to certain mathematical niceties,
    this map may be inverted; the signature is sufficient to define the path. (Technically speaking, it defines the path
    up to 'tree-like equivalence': this means that the signature does not pick up on back-tracking, or changing the
    speed at which one moves along the path. This is actually very useful: if one wishes to be invariant to such things,
    for example when doing character recognition, when the speed at which a character is drawn should not affect its
    classification, then building in this invariance in this way is desirable.)

    Thus the signature is a natural way to characterise a path; in the language of machine learning is an excellent
    feature map.

    Given a tensor of shape (x, y), then one may interpret this a piecewise constant path from [0, x] into \reals^y,
    changing its value at each integer. Whether this is a natural interpretation depends on the data that the tensor
    represents, of course, but this allows for taking the signature of a tensor, which is precisely what this Module
    does.
    """

    def __init__(self, depth, **kwargs):
        if not isinstance(depth, candle.Integer) or depth < 1:
            raise ValueError(f'Depth must be an integer greater than or equal to one. Given {depth} of type '
                             f'{type(depth)}')
        super(Signature, self).__init__(**kwargs)
        self.depth = depth

    def forward(self, path):
        if path.size(1) == 1:
            warnings.warn(f'{self.__class__.__name__} called on path with only one channel; the signature is now just '
                          f'the moments of the path, so there is no interesting information from cross terms.')
        # path is expected to be a 3-dimensional tensor, with batch, channel and length axes respectively, say of shape
        # (b, c, l). Each batch element is treated separately. Then values are interpreted as l sample points from a
        # path in \reals^c
        return batch_path_sig(path, depth=self.depth)

    def extra_repr(self):
        return f'depth={self.depth}'


# Deprecated
class SigLayer(nn.Module):
    """Given some path mapping from, say, [0, 1] into \reals^n, we may define the 'signature' of the path as a
    particular sigtensor with respect to an alphabet of n letters. (Note how n is the target dimension of the path.)
    That is, the signature is a map from the space of paths to the tensor algebra. Up to certain mathematical niceties,
    this map may be inverted; the signature is sufficient to define the path. (Technically speaking, it defines the path
    up to 'tree-like equivalence': this means that the signature does not pick up on back-tracking, or changing the
    speed at which one moves along the path. This is actually very useful: if one wishes to be invariant to such things,
    for example when doing character recognition, when the speed at which a character is drawn should not affect its
    classification, then building in this invariance in this way is desirable.)

    Thus the signature is a natural way to characterise a path; in the language of machine learning is an excellent
    feature map.

    Given a tensor of shape (x, y), then one may interpret this a piecewise constant path from [0, x] into \reals^y,
    changing its value at each integer. Whether this is a natural interpretation depends on the data that the tensor
    represents, of course, but this allows for taking the signature of a tensor, which is precisely what this Module
    does.
    """

    def __init__(self, sig_depth, **kwargs):
        import warnings
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('SigLayer is deprecated. It has been renamed to Signature.', DeprecationWarning)

        if not isinstance(sig_depth, candle.Integer) or sig_depth < 1:
            raise ValueError(f'Depth must be an integer greater than or equal to one. Given {sig_depth} of type '
                             f'{type(sig_depth)}')
        super(SigLayer, self).__init__(**kwargs)
        self.depth = sig_depth

    def forward(self, path):
        if path.size(1) == 1:
            warnings.warn(f'{self.__class__.__name__} called on path with only one channel; the signature is now just '
                          f'the moments of the path, so there is no interesting information from cross terms.')
        # path is expected to be a 3-dimensional tensor, with batch, channel and length axes respectively, say of shape
        # (b, c, l). Each batch element is treated separately. Then values are interpreted as l sample points from a
        # path in \reals^c
        return batch_path_sig(path, depth=self.depth)
    
    def extra_repr(self):
        return f'(depth) {self.depth}'
