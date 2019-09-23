import math
import numpy as np

from ..utils import DependArray
from .distance import *
from .elec_near import ElecNear

try:
    from .pme import Ewald
except ImportError:
    from .ewald import Ewald


class Elec(object):

    def __init__(self, ri, rj, charges, cell_basis, switching_type=None, cutoff=None, swdist=None):
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
            func=Elec._get_coulomb_exclusion,
            kwargs={'dij_min': self.dij_min},
        )
        self.qm_exclusion_esp = DependArray(
            name="qm_exclusion_esp",
            func=Elec._get_qm_exclusion_esp,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                charges,
                self.coulomb_exclusion,
            ],
        )

        self.ewald = Ewald(
            ri=ri,
            rj=rj,
            charges=charges,
            cell_basis=cell_basis,
            cutoff=cutoff,
            rij=self.rij,
        )

        self.near_field = ElecNear(
            dij_min=self.dij_min,
            dij_min_gradient=self.dij_min_gradient, 
            dij_inverse=self.dij_inverse,
            dij_inverse_gradient=self.dij_inverse_gradient,
            charges=charges,
            switching_type=switching_type,
            cutoff=cutoff,
            swdist=swdist,
        )

        self.qm_total_esp = DependArray(
            name="qm_total_esp",
            func=Elec._get_qm_total_esp,
            dependencies=[
                self.ewald.qm_ewald_esp,
                self.qm_exclusion_esp,
            ],
        )
        self.qm_residual_esp = DependArray(
            name="qm_residual_esp",
            func=Elec._get_qm_residual_esp,
            dependencies=[
                self.qm_total_esp,
                self.near_field.qm_scaled_esp,
           ],
        )

    @staticmethod
    def _get_coulomb_exclusion(dij_min):
        return np.where(dij_min < .8)[0]

    @staticmethod
    def _get_qm_exclusion_esp(dij_inverse, dij_inverse_gradient, charges, coulomb_exclusion):
        coulomb_tensor = np.zeros((4, dij_inverse.shape[0], len(coulomb_exclusion)))
        coulomb_tensor[0] = dij_inverse[:, coulomb_exclusion]
        coulomb_tensor[1:] = -dij_inverse_gradient[:, :, coulomb_exclusion]
        coulomb_tensor[0, range(len(dij_inverse)), range(len(dij_inverse))] = 0.

        return coulomb_tensor @ charges[coulomb_exclusion,]

    @staticmethod
    def _get_qm_total_esp(qm_ewald_esp, qm_exclusion_esp):
        return qm_ewald_esp - qm_exclusion_esp

    @staticmethod
    def _get_qm_residual_esp(qm_total_esp, qm_scaled_esp):
        return qm_total_esp - qm_scaled_esp
