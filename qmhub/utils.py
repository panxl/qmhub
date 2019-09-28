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
            if darray.array is not None:
                darray.array[:] = np.asarray(darray._func(*darray._dependencies, **darray._kwargs))
            else:
                darray.array = np.ascontiguousarray(darray._func(*darray._dependencies, **darray._kwargs))
        darray._cache_valid = True


def invalidate_cache(darray):
    if darray._func is not None:
        darray._cache_valid = False

    for item in darray._dependants:
        if item() is not None:
            invalidate_cache(item())


class DependArray(container):

    def __init__(self, data=None, name=None, func=None, kwargs=None, dependencies=None, dependants=None, cache_valid=None):
        if data is not None:
            self.array = np.asarray(data)
        else:
            self.array = None

        if dependencies is None:
            dependencies = []
        if dependants is None:
            dependants = []

        if func is None:
            cache_valid = True
        else:
            cache_valid = False

        if kwargs is None:
            kwargs = {}

        for item in dependencies:
            item.add_dependant(self)

        self._name = name
        self._func = func
        self._kwargs = kwargs
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
            if isinstance(darray, DependArray):
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


def get_numerical_gradient(system, property, gradient, mask=None, over='i', i=None, j=None, k=None):
    ndim = property.ndim
    if mask is not None:
        j2 = np.where(mask)[0][j]
    else:
        j2 = j

    if ndim == 2:
        if over == 'i':
            analytical_gradient = -np.asscalar(gradient[k, i, j])

            system.atoms.positions[k, i:i+1] += 0.001
            property_pos = np.copy(property)

            system.atoms.positions[k, i:i+1] -= 0.002
            property_neg = np.copy(property)

            system.atoms.positions[k, i:i+1] += 0.001
        elif over == 'j':
            analytical_gradient = np.asscalar(gradient[k, i, j])

            system.atoms.positions[k, j2:j2+1] += 0.001
            property_pos = np.copy(property)

            system.atoms.positions[k, j2:j2+1] -= 0.002
            property_neg = np.copy(property)

            system.atoms.positions[k, j2:j2+1] += 0.001

        numerical_gradient = (property_pos[i, j] - property_neg[i, j]) / 0.002

    if ndim == 1:
        if len(property) == len(system.qm.atoms):
            if over == 'i':
                analytical_gradient = np.asscalar(gradient[k, i])

                system.atoms.positions[k, i:i+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, i:i+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, i:i+1] += 0.001

                numerical_gradient = (property_pos[i] - property_neg[i]) / 0.002
            elif over == 'j':
                return
        else:
            if over == 'i':
                analytical_gradient = -np.asscalar(gradient[k, i, j])

                system.atoms.positions[k, i:i+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, i:i+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, i:i+1] += 0.001

                numerical_gradient = (property_pos[j] - property_neg[j]) / 0.002
            elif over == 'j':
                analytical_gradient = np.asscalar(gradient[k, :, j].sum())

                system.atoms.positions[k, j2:j2+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, j2:j2+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, j2:j2+1] += 0.001

                numerical_gradient = (property_pos[j] - property_neg[j]) / 0.002

    return analytical_gradient, numerical_gradient, analytical_gradient - numerical_gradient
