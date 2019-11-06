import functools
import weakref


def cache_update(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self.update_cache()
        return method(self, *args, **kwargs)
    return wrapper


def invalidate_cache(dobject):
    if dobject._func is not None:
        dobject._cache_valid = False
    elif dobject._dependencies:
        dobject._cache_valid = False

    for item in dobject._dependants:
        try:
            invalidate_cache(item)
        except ReferenceError:
            pass


class DependObject(object):
    def __init__(self, *, name=None, func=None, kwargs=None, dependencies=None, dependants=None):

        if kwargs is None:
            kwargs = {}
        if dependencies is None:
            dependencies = []
        if dependants is None:
            dependants = []

        if func is None:
            cache_valid = True
        else:
            cache_valid = False

        self._name = name
        self._func = func
        self._kwargs = kwargs
        self._dependencies = dependencies
        self._dependants = dependants
        self._cache_valid = cache_valid

        for item in dependencies:
            item.add_dependant(self)

    def add_dependency(self, dependency):
        self._dependencies.append(dependency)
        dependency.add_dependant(self)
        invalidate_cache(self)

    def add_dependant(self, dependant):
        self._dependants.append(weakref.proxy(dependant))
