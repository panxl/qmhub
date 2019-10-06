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
            func=(lambda x: x * COULOMB_CONSTANT),
            dependencies=[
                self.dij_inverse,
            ],
        )
        self.qmmm_coulomb_tensor_gradient = DependArray(
            name="qmmm_coulomb_tensor_gradient",
            func=(lambda x: x * COULOMB_CONSTANT),
            dependencies=[
                self.dij_inverse_gradient,
            ],
        )
        self.weighted_qmmm_coulomb_tensor = DependArray(
            name="weighted_qmmm_coulomb_tensor",
            func=ElecNear._get_weighted_qmmm_coulomb_tensor,
            dependencies=[
                self.dij_inverse,
                self.scaling_factor,
            ],
        )
        self.weighted_qmmm_coulomb_tensor_inv = DependArray(
            name="weighted_qmmm_coulomb_tensor_inv",
            func=np.linalg.pinv,
            kwargs={"rcond": 1e-5},
            dependencies=[
                self.weighted_qmmm_coulomb_tensor,
            ],
        )
        self.qm_scaled_esp = DependArray(
            name="qm_scaled_esp",
            func=ElecNear._get_qm_scaled_esp,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                self.scaling_factor,
                self.scaling_factor_gradient,
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
    def _get_weighted_qmmm_coulomb_tensor(dij_inverse, weights):
        return COULOMB_CONSTANT * weights * dij_inverse

    @staticmethod
    def _get_tensor_inverse_gradient(t, t_grad, t_inv):
        """https://mathoverflow.net/q/29511"""

        t_inv_grad = (
            -(t_inv @ t_grad @ t_inv)
            + (t_inv @ t_inv.T) @ np.swapaxes(t_grad, -1, -2) @ (np.identity(t_grad.shape[-2]) - t @ t_inv)
            + (np.identity(t_grad.shape[-1]) - t_inv @ t) @ np.swapaxes(t_grad, -1, -2) @ (t_inv.T @ t_inv)
        )

        return t_inv_grad

    @staticmethod
    def _get_qm_scaled_esp(
        dij_inverse,
        dij_inverse_gradient,
        scaling_factor,
        scaling_factor_gradient,
        charges
        ):

        esp = np.zeros((4, len(dij_inverse)))

        esp[0] = dij_inverse @ (scaling_factor * charges)
        esp[1:] = -(dij_inverse_gradient @ (scaling_factor * charges) + (dij_inverse * scaling_factor_gradient) @ charges)

        return esp * COULOMB_CONSTANT
