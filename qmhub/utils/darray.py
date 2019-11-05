import types

import numpy as np
from numpy.lib.user_array import container

from .dobject import DependObject, cache_update, invalidate_cache


class DependArray(DependObject, container):

    def __init__(self, data=None, *, name=None, func=None, kwargs=None, dependencies=None, dependants=None):
        if data is not None:
            self.array = np.asarray(data)
        else:
            self.array = None

        super().__init__(name=name, func=func, kwargs=kwargs, dependencies=dependencies, dependants=dependants)

    @cache_update
    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        if self._func is not None:
            raise NameError(f"Cannot set the value of <{self._name}> directly")
        self.array[index] = np.asarray(value, self.dtype)
        invalidate_cache(self)

    def subarray(self, index, depend=True):
        darray = self._rc(self.array[index])

        if depend and isinstance(darray, DependArray):
            if self._func is not None:
                darray.add_dependency(self)
            else:
                darray._dependants = self._dependants

        return darray

    def update_cache(self):
        if not self._cache_valid:
            if self._func is None:
                if self._dependencies:
                    for dobject in self._dependencies:
                        dobject.update_cache()
            elif self.array is not None:
                self.array[:] = np.asarray(self._func(*self._dependencies, **self._kwargs))
            else:
                self.array = np.ascontiguousarray(self._func(*self._dependencies, **self._kwargs))
            self._cache_valid = True

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
