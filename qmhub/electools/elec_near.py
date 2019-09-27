import numpy as np

from ..utils import DependArray
from ..units import COULOMB_CONSTANT
from .switching import get_scaling_factor, get_scaling_factor_gradient


class ElecNear(object):
    def __init__(self, dij_min, dij_min_gradient,
                 dij_inverse, dij_inverse_gradient,
                 charges, switching_type=None,
                 cutoff=None, swdist=None):
        
        if switching_type is None:
            self.switching_type = 'shift'
        else:
            self.switching_type = switching_type

        self.cutoff = cutoff
        self.swdist = swdist

        self.near_field_mask = DependArray(
            name="near_field_mask",
            func=ElecNear._get_near_field_mask,
            kwargs={'cutoff': self.cutoff},
            dependencies=[dij_min],
        )
        self.dij_min = DependArray(
            name="dij_min",
            func=ElecNear._get_masked_array,
            dependencies=[dij_min, self.near_field_mask],
        )
        self.dij_min_gradient = DependArray(
            name="dij_min_gradient",
            func=ElecNear._get_masked_array,
            dependencies=[dij_min_gradient, self.near_field_mask],
        )
        self.dij_inverse = DependArray(
            name="dij_inverse",
            func=ElecNear._get_masked_array,
            dependencies=[dij_inverse, self.near_field_mask],
        )
        self.dij_inverse_gradient = DependArray(
            name="dij_inverse_gradient",
            func=ElecNear._get_masked_array,
            dependencies=[dij_inverse_gradient, self.near_field_mask],
        )
        self.charges = DependArray(
            name="charges",
            func=ElecNear._get_masked_array,
            dependencies=[charges, self.near_field_mask],
        )
        self.scaling_factor = DependArray(
            name="scaling_factor",
            func=get_scaling_factor(self.switching_type),
            kwargs={'cutoff': self.cutoff, 'swdist': self.swdist},
            dependencies=[self.dij_min],
        )
        self.scaling_factor_gradient = DependArray(
            name="scaling_factor_gradient",
            func=get_scaling_factor_gradient(self.switching_type),
            kwargs={'cutoff': self.cutoff, 'swdist': self.swdist},
            dependencies=[self.dij_min, self.dij_min_gradient],
        )
        self.qmmm_coulomb_tensor = DependArray(
            name="qmmm_coulomb_tensor",
            func=ElecNear._get_qmmm_coulomb_tensor,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                self.scaling_factor,
                self.scaling_factor_gradient,
           ],
        )
        self.qmmm_coulomb_tensor_inv = DependArray(
            name="qmmm_coulomb_tensor",
            func=ElecNear._get_qmmm_coulomb_tensor_inv,
            dependencies=[self.qmmm_coulomb_tensor],
        )
        self.qm_scaled_esp = DependArray(
            name="qm_scaled_esp",
            func=ElecNear._get_qm_scaled_esp,
            dependencies=[
                self.qmmm_coulomb_tensor,
                self.charges,
           ],
        )

    @staticmethod
    def _get_near_field_mask(dij_min=None, cutoff=None):
        return (dij_min < cutoff) * (dij_min > .8)

    @staticmethod
    def _get_masked_array(array, mask):
        return array[..., mask]

    @staticmethod
    def _get_qmmm_coulomb_tensor(dij_inverse, dij_inverse_gradient, scaling_factor, scaling_factor_gradient):
        coulomb_tensor = np.zeros((4, dij_inverse.shape[0], dij_inverse.shape[1]))
        coulomb_tensor[0] = dij_inverse * scaling_factor
        coulomb_tensor[1:] = dij_inverse_gradient * scaling_factor + dij_inverse * scaling_factor_gradient

        return coulomb_tensor * COULOMB_CONSTANT

    @staticmethod
    def _get_qmmm_coulomb_tensor_inv(qmmm_coulomb_tensor):
        return np.linalg.pinv(qmmm_coulomb_tensor[0], rcond=1e-5)

    @staticmethod
    def _get_qm_scaled_esp(qmmm_coulomb_tensor, charges):
        return qmmm_coulomb_tensor @ charges
