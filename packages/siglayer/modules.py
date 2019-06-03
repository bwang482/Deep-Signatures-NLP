import candle
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backend


class Augment(nn.Module):
    """Augmenting the path before feeding it into a signature is crucial to obtain higher-order information. A way to do
    this is to apply a feedforward neural network to sections of the path, to obtain a nonlinear function of the path
    before the signature is applied.
    
    Both the original path and time can be specifically included in the augmentation.
    """

    def __init__(self, layer_sizes, kernel_size, activation=F.relu, include_original=True, include_time=True, **kwargs):
        """See Augment.__doc__.

        The assumption is that the input tensor is three dimensional, with dimensions (batch, channel, path). A
        feedforward neural network is applied to subtensors of shape (channel, :kernel_size:); the input tensor is split
        up along its batch dimension, and the subtensors are successive overlapping slices along the path dimension. The
        result of this feedforward neural network provides the augmentation of the path, thus giving an output tensor of
        shape (batch, new_channel, path - :kernel_size: + 1), where the size of new_channel depends on the
        :include_original: and :include_time: arguments. In pseudocode:
        
        new_channel = :layer_sizes:[-1]
        if :include_original::
            new_channel += channel
        if :include_time::
            new_channel += 1

        Arguments:
            layer_sizes: tuple of int. Specifies the size of the feedforward neural network to apply to the path. The
                final value of this tuple specifies the number of channels in the augmented path.
            kernel_size: int specifying the size of the kernel to slide over the path.
            activation: Optional, defaults to ReLU. The activation function to use in the feedforward neural network.
            include_original: Optional, defaults to True. Whether or not to include the original path (pre-augmentation)
                in the augmented path.
            include_time: Optional, defaults to True. Whether or not to also augment the path with a 'time' value. These
                are values in [0, 1] corresponding to how far along the path dimension the element is.
        """

        super(Augment, self).__init__(**kwargs)

        if isinstance(layer_sizes, int):
            layer_sizes = (layer_sizes,)

        self.layer_sizes = layer_sizes
        self.kernel_size = kernel_size
        self.activation = activation
        self.include_original = include_original
        self.include_time = include_time

        self.convs = nn.ModuleList()
        if layer_sizes:
            self.convs.append(candle.NoInputSpec(nn.Conv1d, out_channels=layer_sizes[0], kernel_size=kernel_size))
            last_layer_channels = layer_sizes[0]
            for augment_channel in layer_sizes[1:]:
                # These pointwise convolutions correspond to sliding a standard feedforward network across the input.
                self.convs.append(nn.Conv1d(in_channels=last_layer_channels, out_channels=augment_channel,
                                            kernel_size=1))
                last_layer_channels = augment_channel

    def extra_repr(self):
        return f'include_original={self.include_original}, include_time={self.include_time}'

    def forward(self, x):
        if len(x.shape) != 3:
            raise RuntimeError(f'Argument x should have three dimensions, batch, channnel, path. Given shape'
                               f'{x.shape} dimensions with {x}.')
        pieces = []
        if self.include_original:
            truncated_x = x.narrow(2, self.kernel_size - 1, x.size(2) - self.kernel_size + 1)
            pieces.append(truncated_x)
            
        if self.include_time:
            time = torch.linspace(0, 1, x.size(2) - self.kernel_size + 1, dtype=torch.float, device=x.device)
            time = time.expand(x.size(0), 1, -1)
            pieces.append(time)
        
        if self.layer_sizes:
            augmented_x = self.convs[0](x)
            for conv in self.convs[1:]:
                augmented_x = self.activation(augmented_x)
                augmented_x = conv(augmented_x)
            pieces.append(augmented_x)
        return candle.cat(pieces, dim=1)  # concatenate along channel axis


class ViewSignature(nn.Module):
    """Applies a signature Module in a manner akin to any other nonlinearity. As the signature requires a
    three-dimensional input of shape (batch, channel, path), this Module reshapes the input automatically before passing
    it to the signature Module.
    """

    def __init__(self, channels, length, sig_depth, **kwargs):
        """See ViewSigLayer.__doc__.

        Arguments:
            channels: int, specifying the number of channels in the reshaped tensor.
            length: int, specifying the length of the path in the reshaped tensor.
            sig_depth: int, specifying the depth at which to truncate the signature.
        """

        super(ViewSignature, self).__init__(**kwargs)

        self.channels = channels
        self.length = length

        self.sig = backend.Signature(sig_depth)

    def forward(self, x):
        x = x.view(x.size(0), self.channels, self.length)
        return self.sig(x)

    def extra_repr(self):
        return f'channels={self.channels}, length={self.length}'


# Deprecated
class ViewSigLayer(nn.Module):
    """Applies a signature layer in a manner akin to any other nonlinearity. As the signature requires a
    three-dimensional input of shape (batch, channel, path), this Module reshapes the input automatically before passing
    it to the signature layer.
    """

    def __init__(self, channels, length, sig_depth, **kwargs):
        """See ViewSigLayer.__doc__.

        Arguments:
            channels: int, specifying the number of channels in the reshaped tensor.
            length: int, specifying the length of the path in the reshaped tensor.
            sig_depth: int, specifying the depth at which to truncate the signature.
        """
        import warnings
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('ViewSigLayer is deprecated. It has been renamed to ViewSignature', DeprecationWarning)

        super(ViewSigLayer, self).__init__(**kwargs)

        self.channels = channels
        self.length = length
        self.sig_depth = sig_depth

        self.sig = backend.SigLayer(sig_depth)

    def forward(self, x):
        x = x.view(x.size(0), self.channels, self.length)
        return self.sig(x)


# Deprecated
class RecurrentSig(candle.Recurrent):
    """Applies a sliding window across data before applying the signature to each window, and then feeding the result
    through a specified Module in a recurrent manner. The input shape should be (batch, channel, path).

    As the signature is applied to each window, rather than the whole path, then the results may be concatenated along
    the path dimension to have a three-dimensional output shape of (batch, channel_out, path_out).
    """

    def __init__(self, window, module, memory_shape, sig_depth, intermediate_outputs=True, keep_memory=False, **kwargs):
        """See RecurrentSig.__doc__.

        Arguments:
            window: A sliding window to apply to the data. Expected to be an instance of candle.Window.
            module: A module to apply to the result of the signature. It should take two inputs (memory, sig), and
                return two outputs (memory, out).
            sig_depth: The depth at which to truncate the signature.
            memory_shape: The shape of the memory, without the batch dimension, used with :module:.
            intermediate_outputs: Optional, defaults to True. Whether to output all of the intermediate values from the
                recurrent structure, or just the final value. If True then just the final value will be outputted, and
                the resulting Tensor will be only two-dimensional.
            keep_memory: Optional, defaults to False. Whether to keep or delete the internal memory for both the
                recurrent neural network and the Window once the forward pass is complete. By default the memory will be
                cleared, so that each call to forward(...) is independent of the previous one.
        """
        import warnings
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('RecurrentSig is deprecated.', DeprecationWarning)

        self.sig_depth = sig_depth
        transformation = backend.SigLayer(sig_depth)
        super(RecurrentSig, self).__init__(window, module, memory_shape, transformation=transformation,
                                           intermediate_outputs=intermediate_outputs, keep_memory=keep_memory, **kwargs)
