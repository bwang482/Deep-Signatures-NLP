"""Defines the necessary operations for taking the signature of a path. See SigTensor.__doc__ and SigLayer.__doc__ for
more explanation.
"""

import candle
import numbers
import torch
import torch.nn as nn

from . import utils


# TODO: rename to Signature
class SigTensor:
    r"""Instances of SigTensor represent what we shall refer to here as a 'sigtensor', but which are usually called just
    'tensors' in mathematics. As we shall use 'tensor' to refer to something else here, we need to disambiguate.

    Here, a 'tensor' has the usual meaning it has in computing, as a generalisation of the notion of a matrix.

    Meanwhile, we use 'sigtensor' to refer to a member of the set of formal power series in n noncommuting variables,
    with coefficients taken from the real numbers. (One can take coefficients from other spaces too, but real numbers
    suffice for our purpose.) Now a formal power series may be infinite but in practice our computer's memory is not, so
    instances of this class actually correspond to members of the set of polynomials in n noncommuting variables. The
    collection of sigtensors is referred to as the 'tensor algebra'; we may equip this algebra with natural notions of
    addition, multiplication, exponentiation and logarithms.

    In more concrete terms, the tensor algebra may thought of as

    \union_{d = 0}^\infty \reals \times \reals^n \times \reals^{n^2} \times \reals^{n^3} \times \cdots \times \reals^{n^d}

    where \union is the union usually denoted \bigcup
    and \reals are the real numbers, usually denoted \mathbb{R}
    and n is the number of noncommuting variables, often called the alphabet size.

    So given the above description of the tensor algebra, any sigtensor must belong to one of the parts of the union,
    which is indexed by some d. The d corresponding to this sigtensor is called the sigtensor's depth.

    The 'normal' tensors (if you're a computer scientist; they're the abnormal tensors if you're a certain kind of
    mathematician) are of course the elements of \reals^{n^i} for some i, so we see that a sigtensor is essentially a
    tuple of tensors, of shapes (), (n,), (n, n), (n, n, n), ..., (n, ... , n) respectively, where the final tuple has d
    elements. The length of this tuple is one more than the sigtensor's depth.

    Thus one needs three things to specify a sigtensor: the size of the alphabet that one is working over (above denoted
    n), the depth of the sigtensor (above denoted d), and the values of the tuple of tensors.

    Note that scalar values are a member of the tensor algebra: they have depth zero.

    This class is written to calculate the addition, multiplication and exponentiation of sigtensors. (One can also make
    sense of division with respect to a scalar value as multiplication with respect to the inverse of the scalar value.)
    Note that multiplication and exponentiation in particular quickly lead to very large depths, even when starting from
    sigtensors with small depths. As such for such operations a 'depth' parameter may also be provided; the resulting
    sigtensor is truncated to this depth (by throwing away the higher-depth tensors) to avoid speed or memory issues. By
    default this truncation will not be performed, but it is highly recommended to use it! (At a level appropriate for
    your needs.)
    """

    def __init__(self, alphabet_size, depth, device=None, tensors=None):
        """
        Creates a sigtensor with respect to the corresponding alphabet size and depth.

        Arguments:
            alphabet_size: A nonnegative integer specifying the size of the alphabet that this sigtensor is created with
                respect to.
            depth: A nonnegative integer specifying the depth of the sigtensor.
            device: The device to perform the operations on.
            tensors: Optional. A tuple of tensors specifying the value of this sigtensor. If not passed then it will
                default to the 'zero' sigtensor, in which all tensors have zero entries.

        See SigTensor.__doc__ for more information.
        """

        if not isinstance(alphabet_size, candle.Integer):
            raise ValueError(f'alphabet_size must be an integer. Given {alphabet_size} of type {type(alphabet_size)}.')
        if alphabet_size < 0:
            raise ValueError(f'alphabet_size must be a nonnegative integer. Given {alphabet_size}.')
        if not isinstance(depth, candle.Integer):
            raise ValueError(f'depth must be an integer. Given {depth} of type {type(depth)}.')
        if alphabet_size < -1:
            raise ValueError(f'depth must be a nonnegative integer or -1. Given {alphabet_size}.')

        if tensors is not None:
            if len(tensors) != depth + 1:
                raise ValueError(f'The length of the given tuple of tensors must equal depth + 1. Given depth {depth} '
                                 f'and tuple of tensors of length {len(tensors)}')
            for i, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f'Data must be a torch.Tensor. Given tensors of type {type(tensor)} at level {i}.')
                if tensor.shape != (alphabet_size,) * i:
                    raise ValueError(f'Data must be of the correct shape. Given tensors of shape {tensor.shape} at '
                                     f'level {i}.')
                if tensor.device != device:
                    raise ValueError(f'Tensors must be on the same device as the {self.__class__.__name__} is '
                                     f'registered to. Tensor is on device {tensor.device}, but '
                                     f'{self.__class__.__name__} is on device {device}.')

        self._alphabet_size = alphabet_size
        self.device = device
        if tensors is None:
            self.tensors = tuple(torch.zeros((alphabet_size,) * i, dtype=torch.float, device=device)
                                 for i in range(depth + 1))
        else:
            self.tensors = tuple(tensors)

    @property
    def alphabet_size(self):
        """The size of the alphabet that this sigtensor operates over. See SigTensor.__doc__ for more information."""
        return self._alphabet_size

    @property
    def depth(self):
        """The depth of this sigtensor. See SigTensor.__doc__ for more information."""
        return len(self.tensors) - 1

    @property
    def numel(self):
        """The number of values in this sigtensor. That is, the number of scalar values in all of its tensors combined.
        """
        # + 1 for the scalar (at depth zero), which isn't included in the signature
        return 1 + utils.sig_dim(self.alphabet_size, self.depth)

    def __repr__(self):
        if self.depth <= 2:
            tensors = self.tensors
        else:
            tensors = f'{str(self.tensors[:3])[:-1]}, ...)'
        return f'<{self.__class__.__name__}: alphabet_size={self.alphabet_size} depth={self.depth}, tensors={tensors}>'

    @classmethod
    def zero(cls, alphabet_size, depth=0, device=None):
        """Creates a zero sigtensor, in which all elements are zero. This element is the additive identity."""

        return cls(alphabet_size, depth, device)

    @classmethod
    def one(cls, alphabet_size, depth=0, device=None):
        """Creates a one sigtensor, in which the scalar element takes value one, and all other values are zero. This
        element is the multiplicative identity.
        """

        tensors = tuple(torch.zeros((alphabet_size,) * i, dtype=torch.float, device=device)
                        for i in range(1, depth + 1))
        tensors = (torch.ones((), dtype=torch.float, device=device), *tensors)
        self = cls(alphabet_size, depth, device, tensors)
        return self

    @classmethod
    def randn(cls, alphabet_size, depth, device=None):
        """Creates a sigtensor with random values."""

        tensors = tuple(torch.randn((alphabet_size,) * i, dtype=torch.float, device=device) for i in range(depth + 1))
        return cls(alphabet_size, depth, device, tensors)

    def clone(self):
        """Create a copy of this sigtensor."""

        tensors = tuple(tensor.clone() for tensor in self.tensors)
        return self.__class__(self.alphabet_size, self.depth, self.device, tensors)

    def _check_other(self, other):
        """Check that :other: is also a SigTensor or real number; in the latter case convert it into a SigTensor."""

        if isinstance(other, SigTensor):
            if self.alphabet_size != other.alphabet_size:
                raise ValueError('f{self} and {other} have different alphabet sizes.')
            if self.device != other.device:
                raise ValueError(f'{self} and {other} are not on the same device.')
            return other
        elif isinstance(other, numbers.Real):
            return self.__class__(self.alphabet_size, depth=0, device=self.device,
                                  tensors=(torch.tensor(other, dtype=torch.float, device=self.device),))
        else:
            raise ValueError(f'Type {type(other)} not understood.')

    def __eq__(self, other):
        if self.alphabet_size != other.alphabet_size:
            return False
        if self.depth != other.depth:
            # This does mean that zero extensions of a tensor don't equate equal to each other.
            # TODO: change this behaviour?
            return False
        for tensor1, tensor2 in zip(self.tensors, other.tensors):
            if not (tensor1 == tensor2).all():
                return False
        return True

    def __neg__(self):
        """Negation is defined in the obvious way: all values of the sigtensor's tensors are negated."""

        out = self.clone()
        for tensor in out.tensors:
            tensor *= -1
        return out

    def neg(self):
        """See SigTensor.__neg__.__doc__"""
        return self.__neg__()

    def __add__(self, other, out=None):
        """Addition is defined in the obvious way. If two sigtensors are of different depths then the shallower
        sigtensor is extended to the same depth as the deeper one by giving it zero tensors at larger depths.
        """

        other = self._check_other(other)
        if out is None:
            out = self.__class__.zero(self.alphabet_size, max(self.depth, other.depth), self.device)
        out += self
        out += other
        return out

    def add(self, other, out=None):
        """See SigTensor.__add__.__doc__"""
        return self.__add__(other, out)

    def __iadd__(self, other):
        """See SigTensor.__add__.__doc__. This operation acts in-place to modify the existing SigTensor."""

        other = self._check_other(other)
        for tensor1, tensor2 in zip(self.tensors, other.tensors):
            tensor1 += tensor2
        return self

    def iadd(self, other):
        """See SigTensor.__iadd__.__doc__"""
        return self.__iadd__(other)

    def __sub__(self, other, out=None):
        """Subtraction is defined in the obvious way. If two sigtensors are of different depths then the shallower
        sigtensor is extended to the same depth as the deeper one by giving it zero tensors at larger depths.
        """

        other = self._check_other(other)
        if out is None:
            out = self.__class__.zero(self.alphabet_size, max(self.depth, other.depth), self.device)
        out += self
        out -= other
        return out

    def sub(self, other, out=None):
        """See SigTensor.__sub__.__doc__"""
        return self.__sub__(other, out)

    def __isub__(self, other):
        """See SigTensor.__sub__.__doc__. This operation acts in-place to modify the existing SigTensor."""

        other = self._check_other(other)
        for tensor1, tensor2 in zip(self.tensors, other.tensors):
            tensor1 -= tensor2
        return self

    def isub(self, other):
        """See SigTensor.__isub__.__doc__"""
        return self.__isub__(other)

    # TODO: imul (bit of a faff, need to handle the edge cases where we don't +=)
    def __mul__(self, other, depth=None, out=None):
        """Mutiplication is defined in the obvious way when considering a sigtensor as a polynomial in n noncommuting
        variables. In terms of a sigtensor's component tensors, this means taking outer products between all possible
        pairs of tensors, and adding up tensors of the same order. This will in general result in a deeper tensor than
        either of the two that we started off with.

        Arguments:
            (in addition to the usual arguments)
            depth: A depth to truncate the result of the multiplication to, to avoid increasingly-large tensors.
            out: An instance SigTensor to use as the output. Should already be initialized to zero; if it is not then
                the result of this multiplication will be accumulated (added on) to whatever is already there. It should
                be safe to use either of the Tensors taking part in the multiplication as the :out: Tensor.
        """

        other = self._check_other(other)
        if depth is None:
            depth = self.depth + other.depth
        if out is None:
            out = self.__class__.zero(self.alphabet_size, depth, self.device)
        # Why this slightly strange way of iterating? It's because this works from computing the higher depths down to
        # the lower depths, so that in the event that :out: is one of :self: or :other:, then the additions (on to
        # out_tensor) will not affect the later computations. Thus this correctly computes out + (self * other) as
        # desired.
        for d in range(depth, -1, -1):
            out_tensor = out.tensors[d]
            for i in range(max(0, d - other.depth), min(self.depth, d) + 1):
                j = d - i
                tensor1 = self.tensors[i]
                tensor2 = other.tensors[j]
                out_tensor += candle.outer_product(tensor1, tensor2)
        return out

    def mul(self, other, depth=None, out=None):
        """See SigTensor.__mul__.__doc__"""
        return self.__mul__(other, depth, out)

    def __truediv__(self, other):
        """Division by a scalar may be made sense of in the natural way. (Division by a general sigtensor can also be
        made sense of, but this isn't implemented.) This will return the value in a new PyTorch Tensor; see also
        __itruediv__.
        """

        out = self.clone()
        out /= other
        return out

    def truediv(self, other):
        """See SigTensor.__truediv__.__doc__"""
        return self.__truediv__(other)

    def __itruediv__(self, other):
        """See SigTensor.__truediv__.__doc__. This will modify this SigTensor in-place."""
        if not isinstance(other, numbers.Real):
            raise ValueError('Can only divide by real numbers.')
        for tensor in self.tensors:
            tensor /= other
        return self

    def itruediv(self, other):
        """See SigTensor.__itruediv__.__doc__."""
        return self.__itruediv__(other)

    def __floordiv__(self, other):
        """See SigTensor.__truediv__.__doc__"""
        out = self.clone()
        out //= other
        return out

    def floordiv(self, other):
        """See SigTensor.__truediv__.__doc__"""
        return self.__floordiv__(other)

    def __ifloordiv__(self, other):
        """See SigTensor.__itruediv__.__doc__"""
        if not isinstance(other, numbers.Real):
            raise ValueError('Can only divide by real numbers.')
        for tensor in self.tensors:
            tensor //= other
        return self

    def ifloordiv(self, other):
        """See SigTensor.__itruediv__.__doc__"""
        return self.__ifloordiv__(other)

    def __pow__(self, power, depth=None):
        """Powers with respect to a nonnegative integer exponent may be defined in the natural way, as repetaed
        multiplication.

        Arguments:
            (in addition to the usual arguments)
            depth: A depth to truncate the result of the power to, to avoid increasingly-large tensors.
        """

        if not isinstance(power, candle.Integer):
            raise ValueError(f'Power must an integer. Given {power} of type {type(power)}.')
        if power < 0:
            raise ValueError(f'Power must a nonnegative integer. Given {power}.')
        out = self.__class__.one(self.alphabet_size, device=self.device)
        for _ in range(power):
            out = self.mul(out, depth)
        return out

    def pow(self, power, depth=None):
        """See SigTensor.__pow__.__doc__"""
        return self.__pow__(power, depth)

    # TODO: why should the depth also be the maximum power?
    def exp(self, depth, out=None):
        """Given that multiplication, and thus powers, exist, exponentiation may also be defined in the natural way, in
        terms of the usual Taylor series expansion of the exponential. (In practice we perform only a finite sum here.)

        Arguments:
            (in addition to the usual arguments)
            out: An instance of SigTensor to use as the output. Should already be initialized to zero; if it is not then
                the result of this exponentiation will be accumulated (added on) to whatever is already there.
        """

        if not isinstance(depth, candle.Integer):
            raise ValueError(f'Depth must an integer. Given {depth} of type {type(depth)}.')
        if depth < -1:
            raise ValueError(f'Depth must a nonnegative integer or -1. Given {depth}.')
        if out is self:
            raise ValueError(f'Exponentiation does not work with the same out SigTensor as the SigTensor that is being '
                             f'exponentiated.')

        if out is None:
            # If depth == -1 then the iteration below doesn't happen, and this will just give the empty sigtensor, as
            # desired.
            out = self.__class__.one(self.alphabet_size, depth, self.device)
        for power in range(depth, 0, -1):
            out /= power
            out = out.mul(self, depth=depth)  # No in-place multiplication yet, sadly
            out += 1
        return out

    @utils.getitemfnmethod
    def as_flat_tensor(self, item=None):
        """Returns the values of this SigTensor's Tensors as a flat single-dimensional PyTorch Tensor, with all their
        values concatenated together, ordered from shallower tensors to deeper tensors.

        This function may be called:
        SigTensor(...).as_flat_tensor()
        which will return the entirety of this SigTensor's Tensors.

        Alternatively, it supports __getitem__ notation:
        SigTensor(...).as_flat_tensor[1:]
        which will return only the Tensors of the depths specified by the slice or index. (This is not necessarily the
        same as just doing SigTensor(...).as_flat_tensor()[1:], for example: the latter may separate Tensor values of
        the same depth. Also it's slicing a PyTorch Tensor, which makes a (probably undesirable) copy.
        """

        if item is None:
            tensors = self.tensors
        elif isinstance(item, slice):
            tensors = self.tensors[item]
            if len(tensors) == 0:
                raise ValueError('Sliced tensor tuple is empty.')
        else:  # Indexing
            tensors = [self.tensors[item]]
        return torch.cat(tuple(candle.flatten(tensor) for tensor in tensors))


def increment_sig(increments, depth):
    """Takes the signature of the specified collection of :increments:. The signature is truncated to the given :depth:.

    It is computed using Chen's relation:

    S(X) = S(X_1) \otimes S(X_2) \otimes ... \otimes S(X_n) =
        S(X_1 concat ... concat X_n) = S(X_1 concat ... concat X_n-1) \otimes X_n

    where X is a path with increments X_1, ..., X_n, and S represents the signature map, and \otimes is multiplication
    in the tensor algebra, and 'concat' refers to concatenating the paths.
    """

    alphabet_size, number_of_increments = increments.shape
    device = increments.device
    out = SigTensor.one(alphabet_size, device=device)
    for increment in increments.t():
        tensors = (torch.tensor(0.0, dtype=torch.float, device=device),
                   candle.convert_to_tensor(increment, dtype=torch.float, device=device))
        sig_increment = SigTensor(alphabet_size, depth=1, tensors=tensors, device=device)
        out = out.mul(sig_increment.exp(depth=depth), depth=depth)
    return out.as_flat_tensor[1:]  # exclude the constant term


def path_sig_base(path, depth):
    """Takes the signature of the specified :path:. The signature is truncated to the given :depth:. See also
    increment_sig.__doc__.
    """

    increments = path[:, 1:] - path[:, :-1]
    return increment_sig(increments, depth)


# import functools as ft
# import warnings
#
#
# def SigLayerJit(depth, path_shape):
#     """Equivalent to SigLayer, but uses torch.jit.trace to try and achieve a speedup. Requires an extra initialisation
#     parameter, :path_shape:, specifying the shape of the tensor that represents the path.
#     """

#     if not isinstance(depth, candle.Integer):
#         raise ValueError(f'Depth must an integer. Given {depth} of type {type(depth)}.')
#     if depth < 1:
#         raise ValueError(f'Depth must an integer >= 1. (depth == 0 or -1 would just give the empty and unit tensors, '
#                          f'which don\'t give any information.) Given {depth}.')

#     example_path = torch.randn(path_shape, dtype=torch.float)
#     path_sig_fn = ft.partial(path_sig, depth=depth)
#     path_sig_fn.__name__ = path_sig.__name__
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         layer = torch.jit.trace(path_sig_fn, (example_path,))

#     def wrapper(path):
#         if path.shape != path_shape:
#             raise ValueError('Path specified does not fit the specified path_shape.')
#         return layer(path)
#     return wrapper
