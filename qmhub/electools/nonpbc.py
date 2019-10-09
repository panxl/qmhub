import numpy as np

from ..utils.darray import DependArray
from ..units import COULOMB_CONSTANT


class NonPBC(object):
    def __init__(self, rij, mm_charges, cell_basis=None, exclusion=None, *, dij_inverse=None, dij_inverse_gradient=None):

        self.mm_charges = mm_charges

        self.qmmm_coulomb_tensor = DependArray(
            name="qmmm_coulomb_tensor",
            func=NonPBC._get_qmmm_coulomb_tensor,
            dependencies=[dij_inverse, exclusion],
        )
        self.qmmm_coulomb_tensor_gradient = DependArray(
            name="qmmm_coulomb_tensor_gradient",
            func=NonPBC._get_qmmm_coulomb_tensor_gradient,
            dependencies=[dij_inverse_gradient, exclusion],
        )
        self.qm_total_esp = DependArray(
            name="qm_total_esp",
            func=NonPBC._get_qm_total_esp,
            dependencies=[
                self.qmmm_coulomb_tensor,
                self.qmmm_coulomb_tensor_gradient,
                mm_charges,
            ],
        )

    @staticmethod
    def _get_qmmm_coulomb_tensor(dij_inverse, exclusion=None):

        if exclusion is not None:
            dij_inverse = np.copy(dij_inverse)
            dij_inverse[:, np.asarray(exclusion)] = 0.

        return dij_inverse * COULOMB_CONSTANT

    @staticmethod
    def _get_qmmm_coulomb_tensor_gradient(dij_inverse_gradient, exclusion=None):

        if exclusion is not None:
            dij_inverse_gradient = np.copy(dij_inverse_gradient)
            dij_inverse_gradient[:, :, np.asarray(exclusion)] = 0.

        return dij_inverse_gradient * COULOMB_CONSTANT

    @staticmethod
    def _get_qm_total_esp(t, t_grad, mm_charges):
        coulomb_tensor = np.zeros((4, t.shape[0], t.shape[1]))

        coulomb_tensor[0] = t
        coulomb_tensor[1:] = -t_grad

        return coulomb_tensor @ mm_charges

    def _get_total_espc_gradient(self, qm_esp_charges):

        grad = np.zeros((3, len(qm_esp_charges) + len(self.mm_charges)))
        grad[:, len(qm_esp_charges):] = qm_esp_charges @ self.qmmm_coulomb_tensor_gradient * self.mm_charges
        return grad
