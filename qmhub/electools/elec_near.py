import numpy as np

from ..utils.darray import DependArray
from .distance import *
from .switching import get_scaling_factor, get_scaling_factor_gradient


class ElecNear(object):
    def __init__(self, rij, dij,
                 charges, switching_type=None,
                 cutoff=None, swdist=None):

        self.switching_type = switching_type or 'shift'

        self.cutoff = cutoff
        self.swdist = swdist

        self.near_field_buffered_mask = DependArray(
            name="near_field_buffered_mask",
            func=ElecNear._get_near_field_buffered_mask,
            kwargs={'cutoff': self.cutoff},
            dependencies=[dij],
        )
        self.dij_min_buffered = DependArray(
            name="dij_min_buffered",
            func=ElecNear._get_dij_min_buffered,
            dependencies=[dij, self.near_field_buffered_mask],
        )
        self.near_field_mask = DependArray(
            name="near_field_mask",
            func=ElecNear._get_near_field_mask,
            kwargs={'cutoff': self.cutoff},
            dependencies=[self.dij_min_buffered, self.near_field_buffered_mask],
        )
        self.dij_min = DependArray(
            name="dij_min",
            func=ElecNear._get_dij_min,
            kwargs={'cutoff': self.cutoff},
            dependencies=[self.dij_min_buffered],
        )
        self.rij = DependArray(
            name="rij",
            func=ElecNear._get_masked_array,
            dependencies=[rij, self.near_field_mask],
        )
        self.dij = DependArray(
            name="dij",
            func=ElecNear._get_masked_array,
            dependencies=[dij, self.near_field_mask],
        )
        self.dij_gradient = DependArray(
            name="dij_gradient",
            func=get_dij_gradient,
            dependencies=[self.rij, self.dij],
        )
        self.qmmm_coulomb_tensor = DependArray(
            name="qmmm_coulomb_tensor",
            func=get_dij_inverse,
            dependencies=[self.dij],
        )
        self.qmmm_coulomb_tensor_gradient = DependArray(
            name="qmmm_coulomb_tensor_gradient",
            func=get_dij_inverse_gradient,
            dependencies=[self.qmmm_coulomb_tensor, self.dij_gradient],
        )
        self.dij_min_gradient = DependArray(
            name="dij_min_gradient",
            func=get_dij_min_gradient,
            dependencies=[self.dij_min, self.qmmm_coulomb_tensor, self.qmmm_coulomb_tensor_gradient],
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
        self.weighted_qmmm_coulomb_tensor = DependArray(
            name="weighted_qmmm_coulomb_tensor",
            func=(lambda x, y: x * y),
            dependencies=[
                self.scaling_factor,
                self.qmmm_coulomb_tensor,
            ],
        )
        self.weighted_qmmm_coulomb_tensor_inv = DependArray(
            name="weighted_qmmm_coulomb_tensor_inv",
            func=np.linalg.pinv,
            kwargs={"rcond": 1e-8},
            dependencies=[
                self.weighted_qmmm_coulomb_tensor,
            ],
        )
        self.qm_scaled_esp = DependArray(
            name="qm_scaled_esp",
            func=ElecNear._get_qm_scaled_esp,
            dependencies=[
                self.qmmm_coulomb_tensor,
                self.qmmm_coulomb_tensor_gradient,
                self.scaling_factor,
                self.scaling_factor_gradient,
                self.charges,
           ],
        )

    @staticmethod
    def _get_near_field_buffered_mask(dij, cutoff=None, buffer=1.0):
        dij_min = dij.min(axis=0)
        return (dij_min < (cutoff + buffer)) * (dij_min > .8)

    @staticmethod
    def _get_dij_min_buffered(dij, mask):
        dij_inverse = 1 / dij[:, mask]
        dij_min = get_dij_min(dij_inverse)
        return dij_min

    @staticmethod
    def _get_near_field_mask(dij_min, mask, cutoff=None):
        _mask = np.copy(mask)
        _mask[mask] = (dij_min < cutoff)
        return _mask

    @staticmethod
    def _get_dij_min(dij_min, cutoff):
        return dij_min[dij_min < cutoff]

    @staticmethod
    def _get_masked_array(array, mask):
        return array[..., mask]

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
    def _get_qm_scaled_esp(t, t_grad, w, w_grad, charges):

        esp = np.zeros((4, len(t)))

        esp[0] = t @ (w * charges)
        esp[1:] = -(t_grad @ (w * charges) + (t * w_grad) @ charges)

        return esp
