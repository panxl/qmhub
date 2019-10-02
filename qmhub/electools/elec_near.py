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
                self.scaling_factor,
            ],
        )
        self.qmmm_coulomb_tensor_gradient_qm = DependArray(
            name="qmmm_coulomb_tensor_gradient_qm",
            func=ElecNear._get_qmmm_coulomb_tensor_gradient_qm,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                self.scaling_factor,
                self.scaling_factor_gradient,
            ],
        )
        self.qmmm_coulomb_tensor_gradient_mm = DependArray(
            name="qmmm_coulomb_tensor_gradient_mm",
            func=ElecNear._get_qmmm_coulomb_tensor_gradient_mm,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                self.scaling_factor,
                self.scaling_factor_gradient,
            ],
        )
        self.qmmm_coulomb_tensor_inv = DependArray(
            name="qmmm_coulomb_tensor_inv",
            func=np.linalg.pinv,
            kwargs={"rcond": 1e-5},
            dependencies=[
                self.qmmm_coulomb_tensor,
            ],
        )
        self.qmmm_coulomb_tensor_inv_gradient_qm = DependArray(
            name="qmmm_coulomb_tensor_inv_gradient_qm",
            func=ElecNear._get_tensor_inverse_gradient,
            dependencies=[
                self.qmmm_coulomb_tensor,
                self.qmmm_coulomb_tensor_gradient_qm,
                self.qmmm_coulomb_tensor_inv,
            ],
        )
        self.qmmm_coulomb_tensor_inv_gradient_mm = DependArray(
            name="qmmm_coulomb_tensor_inv_gradient_mm",
            func=ElecNear._get_tensor_inverse_gradient,
            dependencies=[
                self.qmmm_coulomb_tensor,
                self.qmmm_coulomb_tensor_gradient_mm,
                self.qmmm_coulomb_tensor_inv,
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
    def _get_qmmm_coulomb_tensor(dij_inverse, scaling_factor):
        return dij_inverse * scaling_factor * COULOMB_CONSTANT

    @staticmethod
    def _get_qmmm_coulomb_tensor_gradient_qm(dij_inverse, dij_inverse_gradient, scaling_factor, scaling_factor_gradient):
        t_grad = np.zeros((3, dij_inverse.shape[0], dij_inverse.shape[0],  dij_inverse.shape[1]))
        for i in range(dij_inverse_gradient.shape[1]):
            for j in range(dij_inverse_gradient.shape[2]):
                t_grad[:, i, i, j] = -dij_inverse_gradient[:, i, j]
    
        tw_grad = dij_inverse[np.newaxis] * -scaling_factor_gradient[:, :, np.newaxis] + t_grad * scaling_factor

        return tw_grad * COULOMB_CONSTANT

    @staticmethod
    def _get_qmmm_coulomb_tensor_gradient_mm(dij_inverse, dij_inverse_gradient, scaling_factor, scaling_factor_gradient):
        t = dij_inverse
        t_grad = np.zeros((3, dij_inverse.shape[1], dij_inverse.shape[0],  dij_inverse.shape[1]))
        for i in range(dij_inverse_gradient.shape[1]):
            for j in range(dij_inverse_gradient.shape[2]):
                t_grad[:, j, i, j] = dij_inverse_gradient[:, i, j]

        w_grad = np.zeros((3, scaling_factor_gradient.shape[2], scaling_factor_gradient.shape[2],  scaling_factor_gradient.shape[2]))
        for i in range(dij_inverse_gradient.shape[1]):
            for j in range(dij_inverse_gradient.shape[2]):
                w_grad[:, j, j, j] = scaling_factor_gradient[:, :, j].sum(axis=1)

        tw_grad = t @ w_grad + t_grad * scaling_factor

        return tw_grad * COULOMB_CONSTANT

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
