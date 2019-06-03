def sig_dim(alphabet_size, depth):
    """Calculates the number of terms in a signature of depth :depth: over an alphabet of size :alphabet_size:."""
    return int(alphabet_size * (1 - alphabet_size ** depth) / (1 - alphabet_size))
    # == sum(alphabet_size ** i for i in range(1, depth + 1)) (geometric sum formula)


# Copied from tools.getitemfn
# https://github.com/patrick-kidger/tools
# To avoid a dependency for something so small
class getitemfn:
    """Converts a function to operate via __getitem__ syntax rather than the normal one.

    Note that this is not a 'dictionary' in any reasonable way: it cannot have reasonable methods for keys, values,
    items, for example.

    It still supports calling it in the normal way.

    This is probably most useful when wanting to call the function with a slice as an argument.

    Example:
    >>> def f(x):
    ...    return x ** 2
    >>> g = getitemfn(f)
    >>> g[4]
    16
    """

    def __init__(self, fn, **kwargs):
        self.fn = fn
        super(getitemfn, self).__init__(**kwargs)

    def __getitem__(self, item):
        return self.fn(item)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class getitemfnmethod:
    """As getitemfn, but for decorating instance methods with."""

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self._cache = {}
        super(getitemfnmethod, self).__init__(**kwargs)

    def __get__(self, instance, owner):
        try:
            return self._cache[(id(instance), id(owner))]
        except KeyError:
            out = getitemfn(self.fn.__get__(instance, owner))
            self._cache[(id(instance), id(owner))] = out
            return out
