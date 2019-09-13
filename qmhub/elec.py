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


class Elecs(object):
    def __init__(self, ri, rj, cell_basis, elecs=None):
        self.ri = ri
        self.rj = rj
        self.cell_basis = cell_basis

        self.rij = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="rij",
            func=get_rij,
            dependencies=[ri, rj],
        )
        self.dij2 = DependArray(
            np.zeros((ri.shape[1], rj.shape[1])),
            name="dij2",
            func=get_dij2,
            dependencies=[self.rij],
        )
        self.dij2_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij2_gradient",
            func=get_dij2_gradient,
            dependencies=[self.rij],
        )
        self.dij = DependArray(
            np.zeros((ri.shape[1], rj.shape[1])),
            name="dij",
            func=get_dij,
            dependencies=[self.dij2],
        )
        self.dij_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij_gradient",
            func=get_dij_gradient,
            dependencies=[self.dij, self.dij2_gradient],
        )
        self.dij_inverse = DependArray(
            np.zeros((ri.shape[1], rj.shape[1])),
            name="dij_inverse",
            func=get_dij_inverse,
            dependencies=[self.dij],
        )
        self.dij_inverse_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij_inverse_gradient",
            func=get_dij_inverse_gradient,
            dependencies=[self.dij_inverse, self.dij_gradient],
        )
        self.dij_min = DependArray(
            np.zeros((rj.shape[1])),
            name="dij_min",
            func=get_dij_min,
            dependencies=[self.dij_inverse],
        )
        self.dij_min_gradient = DependArray(
            np.zeros((3, ri.shape[1], rj.shape[1])),
            name="dij_min_gradient",
            func=get_dij_min_gradient,
            dependencies=[self.dij_min, self.dij_inverse, self.dij_inverse_gradient],
        )