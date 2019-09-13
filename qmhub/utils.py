import types
import functools
import weakref

import numpy as np
from numpy.lib.user_array import container


def cache_update(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self.update_cache()
        return method(self, *args, **kwargs)
    return wrapper


class DependArray(container):

    def __init__(self, data, name=None, func=None, dependencies=None, dependants=None, cache_valid=None):
        self.array = np.asarray(data)

        if dependencies is None:
            dependencies = []
        if dependants is None:
            dependants = []

        if func is None:
            cache_valid = True
        else:
            cache_valid = False

        for item in dependencies:
            item.add_dependant(self)

        self._name = name
        self._func = func
        self._dependencies = dependencies
        self._dependants = dependants
        self._cache_valid = cache_valid

    def __getitem__(self, index):
        darray = self._rc(self.array[index])
        if self._func is None:
            darray._dependants = self._dependants
        return darray

    def __setitem__(self, index, value):
        if self._func is not None:
            raise NameError("Cannot set the value of <" + self._name + "> directly")
        self.array[index] = np.asarray(value, self.dtype)
        self.invalidate_cache()

    def add_dependant(self, dependant):
        self._dependants.append(weakref.ref(dependant))

    def update_cache(self):
        if not self._cache_valid:
            self.array[:] = np.asarray(self._func(*self._dependencies))
            self._cache_valid = True

    def invalidate_cache(self):
        if self._func is not None:
            self._cache_valid = False

        for item in self._dependants:
            item().invalidate_cache()

    # Wrap methods from parent class
    for method_name in dir(container):
        if method_name not in ["_rc", "__array_wrap__", "__setattr__"]:
            attr = getattr(container, method_name)
            if isinstance(attr, types.FunctionType):
                setattr(container, method_name, cache_update(attr))

    # Add some missing methods
    __truediv__ = cache_update(container.__div__)

    __rtruediv__ = cache_update(container.__rdiv__)

    __itruediv__ = cache_update(container.__idiv__)

    @cache_update
    def __floordiv__(self, other):
        return self._rc(np.floor_divide(self.array, np.asarray(other)))

    @cache_update
    def __rfloordiv__(self, other):
        return self._rc(np.floor_divide(np.asarray(other), self.array))

    @cache_update
    def __ifloordiv__(self, other):
        np.floor_divide(self.array, other, self.array)
        return self

    @cache_update
    def __matmul__(self, other):
        return self._rc(np.matmul(self.array, other))

    @cache_update
    def __rmatmul__(self, other):
        return self._rc(np.matmul(other, self.array))

    @cache_update
    def __imatmul__(self, other):
        np.matmul(self.array, other, self.array)
        return self