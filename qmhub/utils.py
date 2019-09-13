import types
import functools
import weakref

import numpy as np
from numpy.lib.user_array import container


def cache_update(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        update_cache(self)
        return method(self, *args, **kwargs)
    return wrapper


def update_cache(darray):
    if not darray._cache_valid:
        if darray._func is update_cache:
            update_cache(*darray._dependencies)
        elif darray._func is not None:
            darray.array[:] = np.asarray(darray._func(*darray._dependencies))
        darray._cache_valid = True


def invalidate_cache(darray):
    if darray._func is not None:
        darray._cache_valid = False

    for item in darray._dependants:
        invalidate_cache(item())


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

    @cache_update
    def __getitem__(self, index):
        if self._func is not None:
            darray = self.__class__(
                self.array[index],
                func=update_cache,
                dependencies=[self],
            )
        else:
            darray = self._rc(self.array[index])
            darray._dependants = self._dependants
        return darray

    def __setitem__(self, index, value):
        if self._func is not None:
            raise NameError("Cannot set the value of <" + self._name + "> directly")
        self.array[index] = np.asarray(value, self.dtype)
        invalidate_cache(self)

    def add_dependant(self, dependant):
        self._dependants.append(weakref.ref(dependant))

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
