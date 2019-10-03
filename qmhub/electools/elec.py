import math
import numpy as np

from ..utils import DependArray
from ..units import COULOMB_CONSTANT
from .distance import *
from .elec_near import ElecNear

try:
    from .pme import Ewald
except ImportError:
    from .ewald import Ewald


class Elec(object):

    def __init__(self, ri, rj, charges, cell_basis, switching_type=None, cutoff=None, swdist=None, pbc=False):
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

        if pbc:
            self.full = Ewald(
                ri=ri,
                rj=rj,
                charges=charges,
                cell_basis=cell_basis,
                cutoff=cutoff,
                rij=self.rij,
            )
        else:
            import importlib
            NonPBC = importlib.import_module(".nonpbc", package='qmhub.electools').__getattribute__('NonPBC')

            self.full = NonPBC(
                rij=self.rij,
                charges=charges,
                cell_basis=cell_basis,
                dij_inverse=self.dij_inverse,
                dij_inverse_gradient=self.dij_inverse_gradient,
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

        self.coulomb_exclusion = DependArray(
            name="coulomb_exclusion",
            func=Elec._get_coulomb_exclusion,
            kwargs={'dij_min': self.dij_min},
        )
        self.qm_exclusion_esp = DependArray(
            name="qm_exclusion_esp",
            func=Elec._get_qm_esp,
            dependencies=[
                self.dij_inverse,
                self.dij_inverse_gradient,
                charges,
                self.coulomb_exclusion,
            ],
        )
        self.qm_total_esp = DependArray(
            name="qm_total_esp",
            func=Elec._get_qm_total_esp,
            dependencies=[
                self.full.qm_full_esp,
                self.qm_exclusion_esp,
            ],
        )
        self.qm_residual_esp = DependArray(
            name="qm_residual_esp",
            func=Elec._get_qm_residual_esp,
            dependencies=[
                self.full.qm_full_esp,
                self.qm_exclusion_esp,
                self.near_field.qm_scaled_esp,
           ],
        )
        self.qm_residual_esp_gradient_qm = DependArray(
            name="qm_residual_esp_gradient_qm",
            func=Elec._get_qm_residual_esp_gradient_qm,
            dependencies=[
                self.qm_total_esp,
                self.near_field.qm_scaled_esp_gradient_qm,
           ],
        )
        self.scaled_mm_charges = DependArray(
            name="scaled_mm_charges",
            func=Elec._get_scaled_mm_charges,
            dependencies=[
                self.near_field.charges,
                self.near_field.scaling_factor,
            ],
        )
        self.projected_mm_charges = DependArray(
            name="projected_mm_charges",
            func=Elec._get_projected_mm_charges,
            dependencies=[
                self.near_field.qmmm_coulomb_tensor_inv,
                self.qm_residual_esp,
                self.near_field.scaling_factor,
            ],
        )
        self.embedding_mm_charges = DependArray(
            name="embedding_mm_charges",
            func=Elec._get_embedding_mm_charges,
            dependencies=[
                self.scaled_mm_charges,
                self.projected_mm_charges,
            ],
        )
        self.embedding_mm_positions = DependArray(
            name="embedding_mm_positions",
            func=Elec._get_embedding_mm_positions,
            dependencies=[
                rj,
                self.near_field.near_field_mask,
            ],
        )
        self.projected_mm_charges_gradient_qm = DependArray(
            name="projected_mm_charges_gradient_qm",
            func=Elec._get_projected_mm_charges_gradient_qm,
            dependencies=[
                self.near_field.qmmm_coulomb_tensor_inv,
                self.near_field.qmmm_coulomb_tensor_inv_gradient_qm,
                self.qm_residual_esp,
                self.qm_residual_esp_gradient_qm,
                self.near_field.scaling_factor,
                self.near_field.scaling_factor_gradient,
            ],
        )

    @staticmethod
    def _get_coulomb_exclusion(dij_min):
        return np.where(dij_min < .8)[0]

    @staticmethod
    def _get_qm_esp(dij_inverse, dij_inverse_gradient, charges, index=None):
        if index is None:
            coulomb_tensor = np.zeros((4, dij_inverse.shape[0], dij_inverse.shape[1]))
        else:
            coulomb_tensor = np.zeros((4, dij_inverse.shape[0], len(index)))
        coulomb_tensor[0] = dij_inverse[:, index]
        coulomb_tensor[1:] = -dij_inverse_gradient[:, :, index]
        coulomb_tensor[0][np.where(np.isinf(dij_inverse))] = 0.

        return (coulomb_tensor @ charges[index,]) * COULOMB_CONSTANT

    @staticmethod
    def _get_qm_total_esp(qm_full_esp, qm_exclusion_esp):
        return qm_full_esp - qm_exclusion_esp

    @staticmethod
    def _get_qm_residual_esp(qm_full_esp, qm_exclusion_esp, qm_scaled_esp):
        return qm_full_esp - qm_exclusion_esp - qm_scaled_esp

    @staticmethod
    def _get_qm_residual_esp_gradient_qm(qm_total_esp, qm_scaled_esp_gradient_qm):
        t_grad = np.zeros_like(qm_scaled_esp_gradient_qm)
        for i in range(qm_total_esp.shape[1]):
                t_grad[:, i, i] = qm_total_esp[1:, i]

        return t_grad - qm_scaled_esp_gradient_qm

    @staticmethod
    def _get_scaled_mm_charges(charges, scaling_factor):
        return charges * scaling_factor

    @staticmethod
    def _get_projected_mm_charges(qmmm_coulomb_tensor_inv, qm_residual_esp, scaling_factor):
        return (qmmm_coulomb_tensor_inv @ qm_residual_esp[0]) * scaling_factor

    @staticmethod
    def _get_embedding_mm_charges(scaled_mm_charges, projected_mm_charges):
        return scaled_mm_charges + projected_mm_charges

    @staticmethod
    def _get_embedding_mm_positions(mm_positions, near_field_mask):
        return mm_positions[:, near_field_mask]

    @staticmethod
    def _get_projected_mm_charges_gradient_qm(
        t_inv,
        t_inv_grad,
        qm_esp,
        qm_esp_grad,
        scaling_factor,
        scaling_factor_gradient
        ):

        w = np.diag(scaling_factor)
        w_grad = np.zeros((3, scaling_factor_gradient.shape[1], scaling_factor_gradient.shape[2],  scaling_factor_gradient.shape[2]))
        for i in range(scaling_factor_gradient.shape[1]):
            for j in range(scaling_factor_gradient.shape[2]):
                w_grad[:, i, j, j] = -scaling_factor_gradient[:, i, j]

        gradient = (
            t_inv @ qm_esp[0] @ w_grad + t_inv_grad @ qm_esp[0] @ w + qm_esp_grad @ t_inv.T @ w
        )
        return gradient
