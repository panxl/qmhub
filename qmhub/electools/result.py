import math
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
        scaling_factor,
        scaling_factor_gradient,
        qmmm_coulomb_tensor,
        qmmm_coulomb_tensor_inv,
        ):

        self.qm_energy = DependArray(
            name="qm_energy",
            func=(lambda x: x * HARTREE_IN_KCAL_PER_MOLE),
            dependencies=[qm_energy],
        )
        self.qm_energy_gradient = DependArray(
            name="qm_energy_gradient",
            func=(lambda x: x * FORCE_AU_IN_IU),
            dependencies=[qm_energy_gradient],
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
        self.qm_term1 = DependArray(
            name="qm_term1",
            func=Result._get_qm_term1,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_inv,
                qm_esp,
                self.mm_esp,
            ],
        )
        self.mm_term1 = DependArray(
            name="mm_term1",
            func=Result._get_mm_term1,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_inv,
                qm_esp,
                self.mm_esp,
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
    def _get_qm_term1(t, t_inv, qm_esp, mm_esp):
        mm_esp_t_inv = mm_esp[0] @ t_inv
        t_inv_qm_esp = t_inv @ qm_esp[0]

        qm_term1 = (
                t[1:] @ (mm_esp_t_inv @ t_inv.T) * (qm_esp[0] - t[0] @ t_inv_qm_esp)
                + t[1:] @ (mm_esp[0] - mm_esp_t_inv @ t[0]) * (t_inv.T @ t_inv_qm_esp)
                - t[1:] @ t_inv_qm_esp * mm_esp_t_inv
                )

        return qm_term1

    @staticmethod
    def _get_mm_term1(t, t_inv, qm_esp, mm_esp):
        mm_esp_t_inv = mm_esp[0] @ t_inv
        t_inv_qm_esp = t_inv @ qm_esp[0]

        mm_term1 = (
            (qm_esp[0] - t[0] @ t_inv_qm_esp) @ t[1:] * (mm_esp_t_inv @ t_inv.T)
            + (t_inv.T @ t_inv_qm_esp) @ t[1:] * (mm_esp[0] - mm_esp_t_inv @ t[0])
            - mm_esp_t_inv @ t[1:] * t_inv_qm_esp
        )

        return mm_term1
