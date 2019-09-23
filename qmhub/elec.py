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
        self.coulomb_exclusion = DependArray(
            name="coulomb_exclusion",
            func=Elec.get_coulomb_exclusion,
            kwargs={'dij_min': self.dij_min},
        )
        self.qm_exclusion_esp = DependArray(
            name="qm_exclusion_esp",
            func=Elec.get_qm_exclusion_esp,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                charges,
                self.coulomb_exclusion,
            ],
        )

        self.ewald = Ewald(ri, rj, charges, cell_basis, cutoff=cutoff, rij=self.rij)

        self.qm_total_esp = DependArray(
            name="qm_total_esp",
            func=Elec.get_qm_total_esp,
            dependencies=[
                self.ewald.qm_ewald_esp,
                self.qm_exclusion_esp,
            ],
        )

    @staticmethod
    def get_coulomb_exclusion(dij_min):
        return np.where(dij_min < .8)[0]

    @staticmethod
    def get_qm_exclusion_esp(dij_inverse, dij_inverse_gradient, charges, coulomb_exclusion):
        coulomb_tensor = np.zeros((4, dij_inverse.shape[0], len(coulomb_exclusion)))
        coulomb_tensor[0] = dij_inverse[:, coulomb_exclusion]
        coulomb_tensor[1:] = -dij_inverse_gradient[:, :, coulomb_exclusion]
        coulomb_tensor[0, range(len(dij_inverse)), range(len(dij_inverse))] = 0.

        return coulomb_tensor @ charges[coulomb_exclusion,]

    @staticmethod
    def get_qm_total_esp(qm_ewald_esp, qm_exclusion_esp):
        return qm_ewald_esp - qm_exclusion_esp
