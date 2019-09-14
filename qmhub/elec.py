import numpy as np
import functools

from .func import (
    get_rij,
    get_dij2,
    get_dij2_gradient,
    get_dij,
    get_dij_gradient,
    get_dij_inverse,
    get_dij_inverse_gradient,
    get_dij_min,
    get_dij_min_gradient,
)
from .utils import DependArray


class Elec(object):

    _darray = ['rij', 'dij2', 'dij2_gradient', 'dij', 'dij_gradient',
               'dij_inverse', 'dij_inverse_gradient', 'dij_min', 'dij_min_gradient']

    _func = {key: globals()["get_" + key] for key in _darray}

    def __init__(self, **kwargs):
        for key in Elec._darray:
            value = kwargs[key]
            setattr(self, key, value)

    @classmethod
    def new(cls, ri, rj, cell_basis):
        rij = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="rij",
            func=Elec._func['rij'],
            dependencies=[ri, rj],
        )
        dij2 = DependArray(
            np.zeros((ri.shape[1], rj.shape[1])),
            name="dij2",
            func=Elec._func['dij2'],
            dependencies=[rij],
        )
        dij2_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij2_gradient",
            func=Elec._func['dij2_gradient'],
            dependencies=[rij],
        )
        dij = DependArray(
            np.zeros((ri.shape[1], rj.shape[1])),
            name="dij",
            func=Elec._func['dij'],
            dependencies=[dij2],
        )
        dij_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij_gradient",
            func=Elec._func['dij_gradient'],
            dependencies=[dij, dij2_gradient],
        )
        dij_inverse = DependArray(
            np.zeros((ri.shape[1], rj.shape[1])),
            name="dij_inverse",
            func=Elec._func['dij_inverse'],
            dependencies=[dij],
        )
        dij_inverse_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij_inverse_gradient",
            func=Elec._func['dij_inverse_gradient'],
            dependencies=[dij_inverse, dij_gradient],
        )
        dij_min = DependArray(
            np.zeros((rj.shape[1])),
            name="dij_min",
            func=Elec._func['dij_min'],
            dependencies=[dij_inverse],
        )
        dij_min_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij_min_gradient",
            func=Elec._func['dij_min_gradient'],
            dependencies=[dij_min, dij_inverse, dij_inverse_gradient],
        )

        kwargs = {}
        for key in Elec._darray:
            kwargs[key] = locals()[key]

        return cls(**kwargs)

    @classmethod
    def from_elec(cls, elec, index=None):
        kwargs = {key: getattr(elec, key)[..., index] for key in Elec._darray}
        return cls(**kwargs)