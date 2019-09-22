import numpy as np

from .elec import Elec
from .utils import DependArray
from .functools import get_scaling_factor, get_scaling_factor_gradient

class Model(object):
    def __init__(self, system, switching_type='shift', cutoff=None, swdist=None):

        self.switching_type = switching_type

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            raise ValueError("cutoff is not set")

        self.elec = Elec(
            system.qm.atoms.positions,
            system.atoms.positions,
            system.atoms.charges,
            system.cell_basis,
            cutoff=self.cutoff,
        )

        if swdist is None:
            self.swdist = cutoff * .75

        self.near_field_mask = DependArray(
            name="near_field_mask",
            func=Model.get_near_field_mask,
            kwargs={'cutoff': self.cutoff},
            dependencies=[self.elec.dij_min],
        )
        self.scaling_factor = DependArray(
            name="scaling_factor",
            func=get_scaling_factor(switching_type),
            kwargs={'cutoff': self.cutoff, 'swdist': self.swdist},
            dependencies=[self.elec.dij_min],
        )
        self.scaling_factor_gradient = DependArray(
            name="scaling_factor_gradient",
            func=get_scaling_factor_gradient(switching_type),
            kwargs={'cutoff': self.cutoff, 'swdist': self.swdist},
            dependencies=[self.elec.dij_min, self.elec.dij_min_gradient],
        )
        self.mm1_index = DependArray(
            name="mm1_index",
            func=Model.get_mm1_index,
            kwargs={'dij': self.elec.dij},
        )
        self.coulomb_tensor = DependArray(
            name="coulomb_tensor",
            func=Model.get_coulomb_tensor,
            dependencies=[
                self.elec.dij_inverse,
                self.elec.dij_inverse_gradient,
            ],
        )
        self.scaled_coulomb_tensor = DependArray(
            name="scaled_coulomb_tensor",
            func=Model.get_scaled_coulomb_tensor,
            kwargs={
                'qm_index': system.qm_index,
                'mm1_index': self.mm1_index,
            },
            dependencies=[
                self.coulomb_tensor,
                self.scaling_factor,
                self.scaling_factor_gradient,
                self.near_field_mask,
           ],
        )

    @staticmethod
    def get_near_field_mask(dij_min=None, cutoff=None):
        return (dij_min < cutoff) * (dij_min > .8)

    @staticmethod
    def get_mm1_index(dij):
        dij_min = dij.min(axis=0)
        return np.where((dij_min < .8) * (dij_min > 0.))[0]

    @staticmethod
    def get_coulomb_tensor(dij_inverse, dij_inverse_gradient):
        coulomb_tensor = np.zeros((4, dij_inverse.shape[0], dij_inverse.shape[1]))
        coulomb_tensor[0] = dij_inverse
        coulomb_tensor[1:] = -dij_inverse_gradient
        coulomb_tensor[0, range(len(dij_inverse)), range(len(dij_inverse))] = 0.
        return coulomb_tensor

    @staticmethod
    def get_scaled_coulomb_tensor(coulomb_tensor, scaling_factor, scaling_factor_gradient, near_field_mask, qm_index=None, mm1_index=None):
        scaled_coulomb_tensor = np.zeros_like(coulomb_tensor)
        scaled_coulomb_tensor[:, :, near_field_mask] = coulomb_tensor[:, :, near_field_mask] @ np.diag(scaling_factor[near_field_mask])
        scaled_coulomb_tensor[1:, :, near_field_mask] += coulomb_tensor[0, :, near_field_mask].T * scaling_factor_gradient[:, :, near_field_mask]

        scaled_coulomb_tensor[:, :, qm_index] = coulomb_tensor[:, :, qm_index]
        scaled_coulomb_tensor[:, :, mm1_index] = coulomb_tensor[:, :, mm1_index]

        return scaled_coulomb_tensor
