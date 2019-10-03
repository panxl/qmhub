import numpy as np

from ..utils import DependArray
from ..units import COULOMB_CONSTANT, HARTREE_IN_KCAL_PER_MOLE, FORCE_AU_IN_IU


class Result(object):

    def __init__(
        self,
        qm_energy,
        qm_energy_gradient,
        mm_esp,
        qm_esp,
        qm_esp_gradient_qm,
        mm_charges,
        near_field_mask,
        scaling_factor,
        scaling_factor_gradient,
        qmmm_coulomb_tensor,
        qmmm_coulomb_tensor_gradient_qm,
        qmmm_coulomb_tensor_inv,
        qmmm_coulomb_tensor_inv_gradient_qm,
        ):

        self.qm_energy = DependArray(
            name="qm_energy",
            func=(lambda x: x * HARTREE_IN_KCAL_PER_MOLE),
            dependencies=[qm_energy],
        )
        self.mm_esp = DependArray(
            name="mm_esp",
            func=Result._get_mm_esp,
            dependencies=[mm_esp],
        )
        self.qm_esp_charges = DependArray(
            name="qm_esp_charges",
            func=Result._get_qm_esp_charges,
            dependencies=[
                self.mm_esp,
                scaling_factor,
                qmmm_coulomb_tensor_inv,
            ],
        )

        # QM energy gradient
        self._qm_energy_gradient_term1 = DependArray(
            name="qm_energy_gradient_term1",
            func=(lambda x: x * FORCE_AU_IN_IU),
            dependencies=[
                qm_energy_gradient,
            ],
        )
        self._qm_energy_gradient_term2 = DependArray(
            name="qm_energy_gradient_term2",
            func=Result._get_qm_energy_gradient_term2,
            dependencies=[
                qmmm_coulomb_tensor_inv,
                qmmm_coulomb_tensor_inv_gradient_qm,
                qm_esp,
                self.mm_esp,
                scaling_factor,
                scaling_factor_gradient,
                mm_charges,
            ],
        )
        self._qm_energy_gradient_term3 = DependArray(
            name="qm_energy_gradient_term3",
            func=Result._get_qm_energy_gradient_term3,
            dependencies=[
                qm_esp_gradient_qm,
                self.qm_esp_charges,
            ],
        )
        self.qm_energy_gradient = DependArray(
            name="qm_energy_gradient",
            func=Result._get_energy_gradient,
            dependencies=[
                self._qm_energy_gradient_term1,
                self._qm_energy_gradient_term2,
                self._qm_energy_gradient_term3,
            ],
        )

        # MM energy gradient
        self._mm_energy_gradient_term1 = DependArray(
            name="mm_energy_gradient_term1",
            func=Result._get_mm_energy_gradient_term1,
            dependencies=[
                self.mm_esp,
                mm_charges,
            ],
        )
        self._mm_energy_gradient_term2 = DependArray(
            name="mm_energy_gradient_term2",
            func=Result._get_mm_energy_gradient_term2,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_inv,
                qm_esp,
                self.mm_esp,
            ],
        )
        # self._mm_energy_gradient_term3 = DependArray(
        #     name="mm_energy_gradient_term3",
        #     func=Result._get_mm_energy_gradient_term3,
        #     dependencies=[
        #         qmmm_coulomb_tensor,
        #         qmmm_coulomb_tensor_inv,
        #         qm_esp,
        #         self.mm_esp,
        #     ],
        # )
        self.mm_energy_gradient = DependArray(
            name="mm_energy_gradient",
            func=Result._get_energy_gradient,
            dependencies=[
                self._mm_energy_gradient_term1,
                self._mm_energy_gradient_term2,
                # self._mm_energy_gradient_term3,
            ],
        )

    @staticmethod
    def _get_mm_esp(mm_esp):
        _mm_esp = np.zeros_like(mm_esp)
        _mm_esp[0] = mm_esp[0] * HARTREE_IN_KCAL_PER_MOLE
        _mm_esp[1:] = mm_esp[1:] * FORCE_AU_IN_IU

        return _mm_esp

    @staticmethod
    def _get_qm_esp_charges(mm_esp, scaling_factor, qmmm_coulomb_tensor_inv):
        return (mm_esp[0] * scaling_factor) @ qmmm_coulomb_tensor_inv

    @staticmethod
    def _get_qm_energy_gradient_term2(t_inv, t_inv_grad, qm_esp, mm_esp, scaling_factor, scaling_factor_gradient, mm_charges):
        w = np.diag(scaling_factor)
        w_grad = np.zeros((3, scaling_factor_gradient.shape[1], scaling_factor_gradient.shape[2],  scaling_factor_gradient.shape[2]))
        for i in range(scaling_factor_gradient.shape[1]):
            for j in range(scaling_factor_gradient.shape[2]):
                w_grad[:, i, j, j] = -scaling_factor_gradient[:, i, j]

        gradient = mm_esp[0] @ w_grad @ (t_inv @ qm_esp[0] + mm_charges) + mm_esp[0] @ w @ t_inv_grad @ qm_esp[0]

        return gradient

    @staticmethod
    def _get_qm_energy_gradient_term3(qm_esp_gradient, qm_esp_charges):
        return qm_esp_gradient @ qm_esp_charges

    @staticmethod
    def _get_mm_energy_gradient_term1(mm_esp, mm_charges):
        return mm_esp[1:] * mm_charges

    @staticmethod
    def _get_mm_energy_gradient_term2(t, t_inv, qm_esp, mm_esp):
        mm_esp_t_inv = mm_esp[0] @ t_inv
        t_inv_qm_esp = t_inv @ qm_esp[0]

        gradient = (
            (qm_esp[0] - t @ t_inv_qm_esp) @ t[1:] * (mm_esp_t_inv @ t_inv.T)
            + (t_inv.T @ t_inv_qm_esp) @ t[1:] * (mm_esp[0] - mm_esp_t_inv @ t)
            - mm_esp_t_inv @ t[1:] * t_inv_qm_esp
        )

        return gradient

    # @staticmethod
    # def _get_mm_energy_gradient_term3(qm_esp, qm_esp_charges):
    #     return qm_esp[1:] * qm_esp_charges

    @staticmethod
    def _get_energy_gradient(gradient_term1, gradient_term2, gradient_term3):
        return gradient_term1 + gradient_term2 + gradient_term3
