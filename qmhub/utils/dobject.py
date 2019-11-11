import functools
import weakref


def cache_update(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self.update_cache()
        return method(self, *args, **kwargs)
    return wrapper


def invalidate_cache(dobject):
    if dobject._func is not None or dobject._dependencies:
        dobject._cache_valid = False

    for item in dobject._dependants:
        try:
            invalidate_cache(item)
        except ReferenceError:
            pass


class DependObject(object):
    def __init__(self, *, name=None, func=None, kwargs=None, dependencies=None, dependants=None):

        kwargs = kwargs or {}
        dependencies = dependencies or []
        dependants = dependants or []
        cache_valid = True if func is None else False

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
        proxy = weakref.proxy(dependant)
        if id(proxy) not in [id(p) for p in self._dependants]:
            self._dependants.append(proxy)
