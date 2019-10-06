import numpy as np

from ..utils import DependArray
from ..units import COULOMB_CONSTANT


class NonPBC(object):
    def __init__(self, rij, charges, cell_basis=None, exclusion=None, *, dij_inverse=None, dij_inverse_gradient=None):

        self.exclusion = exclusion

        self.qm_full_esp = DependArray(
            name="qm_full_esp",
            func=NonPBC._get_qm_full_esp,
            dependencies=[dij_inverse, dij_inverse_gradient, charges],
        )

    @staticmethod
    def _get_qm_full_esp(dij_inverse, dij_inverse_gradient, charges):
        coulomb_tensor = np.zeros((4, dij_inverse.shape[0], dij_inverse.shape[1]))

        coulomb_tensor[0] = dij_inverse
        coulomb_tensor[1:] = -dij_inverse_gradient
        coulomb_tensor[0][np.where(np.isinf(dij_inverse))] = 0.

        return (coulomb_tensor @ charges) * COULOMB_CONSTANT

    def get_mm_total_espc_gradient(self, dij_inverse_gradient, qm_esp_charges, mm_charges):
        if self.exclusion is not None:
            mm_charges = np.copy(mm_charges)
            mm_charges[np.asarray(self.exclusion)] = 0.
        return COULOMB_CONSTANT * qm_esp_charges @ dij_inverse_gradient * mm_charges
