import weakref
from collections.abc import Sequence

from .darray import cache_update, invalidate_cache


class DependList(Sequence):
    def __init__(self, data=None, *, name=None, func=None, kwargs=None, dependencies=None, dependants=None):
        if data is not None:
            self._data = list(data)
        else:
            self._data = []

        if name is not None:
            self._name = name
        else:
            raise ValueError("'name' can not be 'None'.")

        if func is not None:
            self._func = func
        else:
            raise ValueError("'func' can not be 'None'.")

        if kwargs is None:
            kwargs = {}
        if dependencies is None:
            dependencies = []
        if dependants is None:
            dependants = []

        for item in dependencies:
            item.add_dependant(self)

        self._kwargs = kwargs
        self._dependencies = dependencies
        self._dependants = dependants
        self._cache_valid = False

    @cache_update
    def __len__(self):
        return len(self._data)

    @cache_update
    def __getitem__(self, index):
        return self._data[index]

    @property
    @cache_update
    def data(self):
        return self._data

    def add_dependency(self, dependency):
        self._dependencies.append(dependency)
        dependency.add_dependant(self)
        invalidate_cache(self)

    def add_dependant(self, dependant):
        self._dependants.append(weakref.ref(dependant))

    def update_cache(self):
        if not self._cache_valid:
            self._data[:] = list(self._func(*self._dependencies, **self._kwargs))
            self._cache_valid = True
