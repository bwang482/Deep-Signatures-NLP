import candle
import torch
import torch.nn.functional as F

from . import backend
from . import modules


def create_feedforward(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x, 
                       layer_sizes=(32, 32, 32)):
    """This simple model uses a few hidden layers with signature nonlinearities between them.
    If :sig: is falsy then the signature layers will be replaced with ReLU instead.
    It expects input tensors of two dimensions: (batch, features).

    Note that whilst this is a simple example, this is fundamentally quite a strange idea: there's no natural path-like
    structure here, we just reshape things so that it's present.
    """

    if sig:
        nonlinearity = lambda: modules.ViewSignature(channels=2, length=16, sig_depth=sig_depth)
    else:
        nonlinearity = lambda: F.relu

    layers = []
    for layer_size in layer_sizes:
        layers.append(layer_size)
        layers.append(nonlinearity())
    return candle.CannedNet((candle.Flatten(),
                             *layers,
                             torch.Size(output_shape).numel(),
                             candle.View(output_shape),
                             final_nonlinearity))


def create_simple(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x, 
                  augment_layer_sizes=(8, 8, 2), augment_kernel_size=1, augment_include_original=True, 
                  augment_include_time=True, layer_sizes=(32, 32)):
    """This model uses a single signature layer:
        - Augment the features with something learnable
        - Apply signature
        - Small ReLU network afterwards.
    If :sig: is falsy then the signature layers will be replaced with flatten-and-ReLU instead.
    It expects input tensors of three dimensions: (batch, channels, length).
    """

    if sig:
        siglayer = (backend.Signature(sig_depth),)
    else:
        siglayer = (candle.Flatten(), F.relu)

    layers = []
    for layer_size in layer_sizes:
        layers.append(layer_size)
        layers.append(F.relu)

    return candle.CannedNet((modules.Augment(layer_sizes=augment_layer_sizes,
                                             kernel_size=augment_kernel_size,
                                             include_original=augment_include_original,
                                             include_time=augment_include_time),
                             *siglayer,
                             *layers,
                             torch.Size(output_shape).numel(),
                             candle.View(output_shape),
                             final_nonlinearity))


def create_windowed_simpler(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x):
    """As create_windowed, but is less general, to provide a simpler example of how these things fit together.
    
    Basically applies a couple of RNNs to the input data with signatures in between.
    """
    
    if sig:
        transformation = lambda: backend.Signature(depth=sig_depth)
    else:
        transformation = lambda: candle.batch_flatten
        
    output_size = torch.Size(output_shape).numel()

    return candle.CannedNet((modules.Augment(layer_sizes=(16, 16, 2), kernel_size=4),
                             candle.Window(length=5, stride=1, transformation=transformation()),
                             # We could equally well have an Augment here instead of a Recur; both are path-preserving
                             # neural networks.
                             candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                   32, F.relu,
                                                                   16,  # memory size + output size
                                                                   candle.Split((8, 8)))),  # memory size, output size
                                          memory_shape=(8,)),  # memory size
                             candle.Window(length=10, stride=5, transformation=transformation()),
                             candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                   32, F.relu, 16, F.relu,
                                                                   8 + output_size,  # memory size + output size
                                                                   candle.Split((8, output_size)))),  # memory size, output size
                                          memory_shape=(8,),  # memory size
                                          intermediate_outputs=False),
                             candle.View(output_shape),
                             final_nonlinearity))


# Now THIS is deep signatures!
def create_windowed(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x,
                    augment_layer_sizes=(32, 32, 2), augment_kernel_size=8, augment_include_original=True,
                    augment_include_time=True,
                    lengths=(5, 5, 10), strides=(1, 1, 5), adjust_lengths=(0, 0, 0), memory_sizes=(8, 8, 8),
                    layer_sizes_s=((32,), (32,), (32, 16)), hidden_output_sizes=(8, 8)):
    """This model stacks multiple layers of signatures on top of one another in a natural way, using the RecurrentSig
    Module:
    - Augment the features with something learnable
    - Slide a window across the augmented features
    - Take the signature of each window
    - Put this list of signatures back together to recover the path dimension
    - Apply a RNN across the path dimension, preserving the intermediate outputs, so the path dimension is preserved
    - Slide another window
    - Take another signature
    - Reassemble signatures along path dimension
    - Another RNN
    - ... 
    - etc. for some number of times
    - ...
    - Slide another window
    - Take another signature
    - Reassemble signatures along path dimension
    - Another RNN; this time throw away intermediate outputs and just present the final output as the overall output.
    If :sig: is falsy then the signature layers will be replaced with flattening instead.
    It expects input tensors of three dimensions: (batch, channels, length).
    
    For a simpler example in the same vein, see create_windowed_simpler.

    Arguments:
        output_shape: The final output shape from the network.
        sig: Optional, whether to use signatures in the network. If True a signature will be applied between each 
            window. If False then the output is simply flattened. Defaults to True.
        sig_depth: Optional. If signatures are used, then this specifies how deep they should be truncated to.
        final_nonlinearity: Optional. What final nonlinearity to feed the final tensors of the network through, e.g. a
            sigmoid when desiring output between 0 and 1. Defaults to the identity.
        augment_layer_sizes: Optional. A tuple of integers specifying the size of the hidden layers of the feedforward
            network that is swept across the input stream to augment it. May be set to the empty tuple to do no 
            augmentation.
        augment_kernel_size: Optional. How far into the past the swept feedforward network (that is doing augmenting) 
            should take inputs from. For example if this is 1 then it will just take data from a single 'time', making
            it operate in a 'pointwise' manner. If this is 2 then it will take the present and the most recent piece of
            past information, and so on.
        augment_include_original: Optional. Whether to include the original path in the augmentation.
        augment_include_time: Optional. Whether to include an increasing 'time' parameter in the augmentation.
        lengths, strides, adjust_lengths, memory_sizes: Optional. Should each be a tuple of integers, all of the same
            length as one another. The length of these arguments determines the number of windows; this length must be
            at least one. The ith values determine the length, stride and adjust_length arguments of the ith Window,
            and the size of the memory of the ith RNN.
        layer_sizes_s: Optional. Should be a tuple of the same length as lengths, strides, adjust_lengths, 
            memory_sizes. Each element of the tuple should itself be a tuple of integers specifying the sizes of the
            hidden layers of each RNN.
        hidden_output_sizes: Optional. Should be a tuple of integers one shorter than the length of lengths, strides,
            adjust_lengths, memory_sizes. It determines the output size of each RNN. It is of a slightly shorter length
            because the final output size is actually already determined by the output_shape argument!
    """
    # TODO: Explain all of this a bit better... it's a bit convoluted! (Pun intended.)
    
    num_windows = len(lengths)
    assert num_windows >= 1
    assert len(strides) == num_windows
    assert len(adjust_lengths) == num_windows
    assert len(layer_sizes_s) == num_windows
    assert len(memory_sizes) == num_windows
    assert len(hidden_output_sizes) == num_windows - 1

    if sig:
        transformation = lambda: backend.Signature(depth=sig_depth)
    else:
        transformation = lambda: candle.batch_flatten
        
    final_output_size = torch.Size(output_shape).numel()
    output_sizes = (*hidden_output_sizes, final_output_size)
    
    recurrent_layers = []
    for (i, length, stride, adjust_length, layer_sizes, memory_size, output_size
         ) in zip(range(num_windows), lengths, strides, adjust_lengths, layer_sizes_s, memory_sizes, output_sizes):
        
        window_layers = []
        for layer_size in layer_sizes:
            window_layers.append(layer_size)
            window_layers.append(F.relu)
            
        intermediate_outputs = (num_windows - 1 != i)

        recurrent_layers.append(candle.Window(length=length, stride=stride, adjust_length=adjust_length,
                                              transformation=transformation()))
        recurrent_layers.append(candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                      *window_layers,
                                                                      memory_size + output_size,
                                                                      candle.Split((memory_size, output_size)))),
                                             memory_shape=(memory_size,),
                                             intermediate_outputs=intermediate_outputs))

    return candle.CannedNet((modules.Augment(layer_sizes=augment_layer_sizes, 
                                             kernel_size=augment_kernel_size,
                                             include_original=augment_include_original, 
                                             include_time=augment_include_time),
                             *recurrent_layers,
                             candle.View(output_shape),
                             final_nonlinearity))
