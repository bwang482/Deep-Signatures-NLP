import functools as ft
import queue
import torch
import torch.nn as nn
import warnings

from . import utils


def identity(x):
    return x


class Window(nn.Module):
    """Creates a sliding window along a Tensor, yielding slices of the Tensor as it goes along. It is given Tensors to
    store in memory, and then when requested, will yield slices from them as if those Tensors were one long Tensor,
    concatenated along a specified axis. This is useful for time series data, for example, when data may be arriving
    continuously, at variable rates and lengths.

    Can iterate on a Window instance to get all of the possible slices from its memory. For particular choices of length
    and stride, given a collection of input Tensors of some total length (along the specified dimension), this may mean
    that the final slice is smaller than requested. Instead of being yielded, it will be retained in the Window's memory
    and used to start off from later, once new Tensors have been added to its memory.

    May also be used as a PyTorch Module. in this case the single input is pushed into the Window, and the results
    returned as a single tensor, stacked along a new dimension. (Which will be the last dimension in the shape.) This
    usage is dependent on the results all having the same shape. That is, if adjust_length != 0 then a suitable
    transformation should be applied to ensure that the results remain the same length.

    Note that if you just want a sliding window with no transformations then torch.Tensor.unfold is going to be much
    quicker.
    """

    def __init__(self, length, stride, adjust_length=0, dim=-1, clone=True, transformation=identity, items=(),**kwargs):
        """See Window.__doc__.

        Arguments:
            length: The length of the slice taken from the input Tensor
            stride: How much to move the start point of the slice after the previous slice has been yielded. Interesting
                choices include 1 (to yield overlapping slices, each one just offset from the previous one), or
                :length:, which will yield completely nonoverlapping slices, or 0, which will yield slices starting from
                the same start point every time (for example, to use alongside a nonzero value for :adjust_length:).
            adjust_length: Optional, integer or callable, defaults to 0. How much the length is changed when a slice has
                been yielded. If an integer it will be added on to the length. If a callable then the current length
                will be passed as an input, and the new length should be returned as an output. For example, setting
                :stride:=0 and :adjust_length:=1 will give an expanding window.
            dim: Optional, defaults to -1. The dimension of the input Tensor to move along whilst yielding slices.
            clone: Optional, defaults to True. Whether to clone the output before yielding it. Otherwise later in-place
                operations could affect the Window's memory. If you're sure this isn't going to happen, for example
                because a copy is made somewhere within the transformation argument, then setting this this to False
                will give a speed-up.
            transformation: Optional, defaults to no transformation. A transformation to apply to the output before
                yielding it.
            items: Optional, iterable, defaults to (). An iterable of Tensors to prepopulate the Window's memory with.
        """

        super(Window, self).__init__(**kwargs)

        self.length = length
        self._original_length = length
        self.stride = stride
        self.adjust_length = adjust_length
        self.dim = dim
        self.clone = clone
        self.transformation = transformation if transformation is not None else identity

        self.last = torch.zeros(0)
        self.queue = queue.Queue()
        for item in items:
            self.queue.put_nowait(item)
        
        self._device = None

    def extra_repr(self):
        msg = f'length={self.length}, stride={self.stride}, adjust_length={self.adjust_length}, dim={self.dim}'
        if self.transformation is not identity and not isinstance(self.transformation, nn.Module):
            if hasattr(self.transformation, '__name__'):
                msg += f', transformation={self.transformation.__name__}'
            elif isinstance(self.transformation, ft.partial):
                fn = self.transformation
                msg += f', transformation=partial({fn.func.__name__}, args={fn.args}, keywords={fn.keywords})'
        return msg

    def push(self, item):
        """Add a Tensor to the Window's memory."""
        if self._device is None:
            self._device = item.device
            self.last = self.last.to(device=item.device)
        if self._device != item.device:
            raise RuntimeError(f'{self.__class__.__name__} previously had tensors of backend {self._device} pushed, but'
                               f' have now had tensor of backend {item.device} pushed.')
        self.queue.put_nowait(item)

    def pull(self):
        """Take a slice from the Tensors in the Window's memory."""

        size_so_far = self.last.size(self.dim)
        items = [self.last]

        while True:
            if size_so_far < self.length:
                try:
                    last = self.queue.get_nowait()
                except queue.Empty:
                    self.last = utils.cat(items, dim=self.dim)
                    raise
                size_so_far += last.size(self.dim)
                items.append(last)
            else:
                break

        out = utils.cat(items, dim=self.dim)
        rem = out.size(self.dim) - self.stride
        out, self.last = out.narrow(self.dim, 0, self.length), out.narrow(self.dim, self.stride, rem)

        if callable(self.adjust_length):
            self.length = self.adjust_length(self.length)
        else:
            self.length = self.length + self.adjust_length

        out = self.transformation(out)
        if self.clone:
            out = out.clone()
        return out

    def clear(self):
        """Clear the Window's memory."""

        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self.last = torch.zeros(0)
        self._device = None
        self.length = self._original_length

    def __iter__(self):
        not_iterated = True
        try:
            while True:
                yield self.pull()
                not_iterated = False
        except queue.Empty:
            if not_iterated:
                warnings.simplefilter('always', RuntimeWarning)
                warnings.warn(f'{self.__class__.__name__} did not iterate over any windows. This means there was not '
                              f'enough input data to create one entire window: either increase the length of the input '
                              f'data or decrease the size of the window.', RuntimeWarning)

    def forward(self, x):
        self.push(x)
        try:
            out = utils.stack(list(self), dim=-1)
        finally:
            self.clear()
        return out


class Recur(nn.Module):
    """Takes a tensor of shape (..., channels, path), splits it up into individual tensors along the last (path)
    dimension, and applies the specified network to them in a recurrent manner.
    """

    def __init__(self, module, memory_shape, intermediate_outputs=True, **kwargs):
        super(Recur, self).__init__(**kwargs)

        self.module = module
        self.memory_shape = memory_shape
        self.intermediate_outputs = intermediate_outputs

    def extra_repr(self):
        return f'memory_shape={self.memory_shape}, intermediate_outputs={self.intermediate_outputs}'

    def forward(self, x):
        outs = []
        memory = torch.zeros(x.size(0), *self.memory_shape, device=x.device)
        xs = x.unbind(dim=-1)
        for inp in xs:
            memory, out = self.module((memory, inp))
            memory = memory.view(x.size(0), *self.memory_shape)
            if self.intermediate_outputs:
                outs.append(out)
        if self.intermediate_outputs:
            return utils.stack(outs, dim=-1)
        else:
            return out


# Deprecated
class Recurrent(nn.Module):
    """Applies a sliding window across data, applies a transformation if desired, and then feeds the result through a
    Module in a recurrent manner. The input shape should be (batch, ..., path), where '...' represents some nonnegative
    integer number of channels.
    """

    def __init__(self, window, module, memory_shape, transformation=lambda x: x, intermediate_outputs=True,
                 keep_memory=False, **kwargs):
        """See Recurrent.__doc__.

        Arguments:
            window: A sliding window to apply to the data. Expected to be an instance of Window.
            module: A module to apply recurrently to the results of the transformation. It should take two inputs
                (memory, sig), and return two outputs (memory, out).
            memory_shape: The shape of the memory, without the batch dimension, used with :module:.
            transformation: Optional, defaults to the identity. A map applied to each window before it is given to
                :module:. (In principle this functionality is already available by changing the supplied module, but
                this is a sufficienly common use-case that it makes sense to make it easier this way.)
            intermediate_outputs: Optional, defaults to True. Whether to output all of the intermediate values from the
                recurrent structure, or just the final value. If True then just the final value will be outputted, and
                the resulting Tensor will be only two-dimensional.
            keep_memory: Optional, defaults to False. Whether to keep or delete the internal memory for both the
                recurrent neural network and the Window once the forward pass is complete. By default the memory will be
                cleared, so that each call to forward(...) is independent of the previous one.
        """
        import warnings
        warnings.warn('Recurrent is deprecated. Please switch to using Window and Recur separately.',
                      DeprecationWarning)

        super(Recurrent, self).__init__(**kwargs)

        self.window = window
        self.module = module
        self.memory_shape = memory_shape
        self.transformation = transformation
        self.intermediate_outputs = intermediate_outputs
        self.keep_memory = keep_memory

        self.memory = None

    def forward(self, x):
        outs = []
        if self.memory is None:
            self.memory = torch.zeros(x.size(0), *self.memory_shape, device=x.device)
        self.window.push(x)
        not_iterated = True
        for windowed in self.window:
            not_iterated = False
            transformed_windowed = self.transformation(windowed)
            memory, out = self.module((self.memory, transformed_windowed))
            self.memory = memory.view(x.size(0), *self.memory_shape)
            if self.intermediate_outputs:
                outs.append(out)
        if not self.keep_memory:
            self.window.clear()
            self.memory = None
        if not_iterated:
            raise RuntimeError(f'{self.__class__.__name__} did not iterate over any windows. This means there was not '
                               f'enough input data to create one entire window: either increase the length of the input '
                               f'data or decrease the size of the window. Length of input data is {x.size(-1)}. Length of '
                               f'window is {self.window.length}.')
        if self.intermediate_outputs:
            return utils.stack(outs, dim=-1)
        else:
            return out
