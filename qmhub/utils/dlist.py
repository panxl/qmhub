from collections.abc import MutableSequence

from .dobject import DependObject, cache_update, invalidate_cache


class DependList(DependObject, MutableSequence):
    def __init__(self, data=None, **kwargs):
        if data is not None:
            self._data = list(data)
        else:
            self._data = []

        super().__init__(**kwargs)

    @cache_update
    def __len__(self):
        return len(self._data)

    @cache_update
    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        if self._func is not None:
            raise NameError(f"Cannot set the value of <{self._name}> directly")
        self._data[index] = list(value)
        invalidate_cache(self)

    @cache_update
    def __delitem__(self, index):
        del self._data[index]

    @cache_update
    def insert(self, index, value):
        self._data.insert(index, value)

    @cache_update
    def __iter__(self):
        return iter(self._data)

    @cache_update
    def __reversed__(self):
        return reversed(self._data)

    def update_cache(self):
        if not self._cache_valid:
            self._data = list(self._func(*self._dependencies, **self._kwargs))
            self._cache_valid = True
