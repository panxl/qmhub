import math
import numpy as np

from .functools import Ewald
from .functools.distance import *
from .utils import DependArray


class Elec(object):

    def __init__(self, ri, rj, charges, cell_basis, cutoff=None):
        self.rij = DependArray(
            name="rij",
            func=get_rij,
            dependencies=[ri, rj],
        )
        self.dij = DependArray(
            name="dij",
            func=get_dij,
            dependencies=[self.rij],
        )
        self.dij_gradient = DependArray(
            name="dij_gradient",
            func=get_dij_gradient,
            dependencies=[self.rij, self.dij],
        )
        self.dij_inverse = DependArray(
            name="dij_inverse",
            func=get_dij_inverse,
            dependencies=[self.dij],
        )
        self.dij_inverse_gradient = DependArray(
            name="dij_inverse_gradient",
            func=get_dij_inverse_gradient,
            dependencies=[self.dij_inverse, self.dij_gradient],
        )
        self.dij_min = DependArray(
            name="dij_min",
            func=get_dij_min,
            dependencies=[self.dij_inverse],
        )
        self.dij_min_gradient = DependArray(
            name="dij_min_gradient",
            func=get_dij_min_gradient,
            dependencies=[self.dij_min, self.dij_inverse, self.dij_inverse_gradient],
        )

        self.ewald = Ewald(ri, rj, charges, cell_basis, cutoff=cutoff, rij=self.rij)
