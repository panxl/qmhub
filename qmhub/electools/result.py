import numpy as np

from ..utils import DependArray
from ..units import COULOMB_CONSTANT, HARTREE_IN_KCAL_PER_MOLE, FORCE_AU_IN_IU


class Result(object):

    def __init__(
        self,
        qm_energy,
        qm_energy_gradient,
        mm_esp,
        mm_charges,
        near_field_mask,
        scaling_factor,
        scaling_factor_gradient,
        qmmm_coulomb_tensor,
        qmmm_coulomb_tensor_gradient,
        weighted_qmmm_coulomb_tensor,
        weighted_qmmm_coulomb_tensor_inv,
        elec,
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
                weighted_qmmm_coulomb_tensor_inv,
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
                scaling_factor_gradient,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                self.mm_esp,
                mm_charges,
            ],
        )
        self._qm_energy_gradient_term3 = DependArray(
            name="qm_energy_gradient_term3",
            func=Result._get_qm_energy_gradient_term3,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_gradient,
                scaling_factor,
                scaling_factor_gradient,
                weighted_qmmm_coulomb_tensor,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                self.mm_esp,
            ],
        )
        self._qm_energy_gradient_term4 = DependArray(
            name="qm_energy_gradient_term4",
            func=Result._get_qm_energy_gradient_term4,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_gradient,
                scaling_factor,
                scaling_factor_gradient,
                mm_charges,
                elec.full.qm_total_esp,
                self.qm_esp_charges,
            ],
        )
        self.qm_energy_gradient = DependArray(
            name="qm_energy_gradient",
            func=Result._get_qm_energy_gradient,
            dependencies=[
                self._qm_energy_gradient_term1,
                self._qm_energy_gradient_term2,
                self._qm_energy_gradient_term3,
                self._qm_energy_gradient_term4,
            ],
        )

        # MM energy gradient
        self._mm_energy_gradient_term1 = DependArray(
            name="mm_energy_gradient_term1",
            func=Result._get_mm_energy_gradient_term1,
            dependencies=[
                scaling_factor,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                self.mm_esp,
                mm_charges,
            ],
        )
        self._mm_energy_gradient_term2 = DependArray(
            name="mm_energy_gradient_term2",
            func=Result._get_mm_energy_gradient_term2,
            dependencies=[
                scaling_factor_gradient,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                self.mm_esp,
                mm_charges,
            ],
        )
        self._mm_energy_gradient_term3 = DependArray(
            name="mm_energy_gradient_term3",
            func=Result._get_mm_energy_gradient_term3,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_gradient,
                scaling_factor,
                scaling_factor_gradient,
                weighted_qmmm_coulomb_tensor,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                self.mm_esp,
            ],
        )
        self._mm_energy_gradient_term4 = DependArray(
            name="mm_energy_gradient_term4",
            func=Result._get_mm_energy_gradient_term4,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_gradient,
                scaling_factor,
                scaling_factor_gradient,
                mm_charges,
                self.qm_esp_charges,
            ],
        )
        self._mm_energy_gradient_term5 = DependArray(
            name="mm_total_esp_gradient",
            func=Result._get_mm_energy_gradient_term5,
            dependencies=[
                elec.full.qm_total_esp_gradient,
                self.qm_esp_charges,
            ],
        )
        self.mm_energy_gradient = DependArray(
            name="mm_energy_gradient",
            func=Result._get_mm_energy_gradient,
            dependencies=[
                self._mm_energy_gradient_term1,
                self._mm_energy_gradient_term2,
                self._mm_energy_gradient_term3,
                self._mm_energy_gradient_term4,
                self._mm_energy_gradient_term5,
                near_field_mask,
            ],
        )

    @staticmethod
    def _get_mm_esp(mm_esp):
        _mm_esp = np.zeros_like(mm_esp)
        _mm_esp[0] = mm_esp[0] * HARTREE_IN_KCAL_PER_MOLE
        _mm_esp[1:] = mm_esp[1:] * FORCE_AU_IN_IU

        return _mm_esp

    @staticmethod
    def _get_qm_esp_charges(mm_esp, scaling_factor, weighted_qmmm_coulomb_tensor_inv):
        return (mm_esp[0] * scaling_factor) @ weighted_qmmm_coulomb_tensor_inv

    @staticmethod
    def _get_qm_energy_gradient_term2(w_grad, wt_inv, qm_esp, mm_esp, mm_charges):
        return mm_esp[0] * -w_grad @ (wt_inv @ qm_esp[0] + mm_charges)

    @staticmethod
    def _get_qm_energy_gradient_term3(t, t_grad, w, w_grad, wt, wt_inv, qm_esp, mm_esp):

        # w * mm_esp[0] @ t_inv_grad @ qm_esp[0]
        w_mm_esp_wt_inv =  w * mm_esp[0] @ wt_inv
        wt_inv_qm_esp = wt_inv @ qm_esp[0]

        grad = (
            (w_mm_esp_wt_inv @ t * w_grad) @ wt_inv_qm_esp
            + w_mm_esp_wt_inv * (t_grad * w @ wt_inv_qm_esp)
            - (qm_esp[0] - wt @ wt_inv_qm_esp) @ t * w_grad @ (w_mm_esp_wt_inv @ wt_inv.T)
            - (qm_esp[0] - wt @ wt_inv_qm_esp) * (t_grad * w @ (w_mm_esp_wt_inv @ wt_inv.T))
            - (wt_inv.T @ wt_inv_qm_esp) @ t * w_grad @ (w * mm_esp[0] - w_mm_esp_wt_inv @ wt)
            - (wt_inv.T @ wt_inv_qm_esp) * (t_grad * w @ (w * mm_esp[0] - w_mm_esp_wt_inv @ wt))
        )
        return grad

    @staticmethod
    def _get_qm_energy_gradient_term4(
        t,
        t_grad,
        w,
        w_grad,
        mm_charges,
        qm_total_esp,
        qm_esp_charges
        ):

        grad = (
            qm_total_esp[1:] * qm_esp_charges
            + (mm_charges * w_grad) @ (qm_esp_charges @ t)
            + (w * t_grad) @ mm_charges * qm_esp_charges
        )

        return grad

    @staticmethod
    def _get_mm_energy_gradient_term1(w, wt_inv, qm_esp, mm_esp, mm_charges):
        return  w * (wt_inv @ qm_esp[0] + mm_charges) * mm_esp[1:]

    @staticmethod
    def _get_mm_energy_gradient_term2(w_grad, wt_inv, qm_esp, mm_esp, mm_charges):
        return  mm_esp[0] * w_grad.sum(axis=1) * (wt_inv @ qm_esp[0] + mm_charges)

    @staticmethod
    def _get_mm_energy_gradient_term3(t, t_grad, w, w_grad, wt, wt_inv, qm_esp, mm_esp):

        w_mm_esp_wt_inv =  w * mm_esp[0] @ wt_inv
        wt_inv_qm_esp = wt_inv @ qm_esp[0]

        grad = (
            -(w_mm_esp_wt_inv @ t) * w_grad.sum(axis=1) * wt_inv_qm_esp
            - w_mm_esp_wt_inv @ t_grad * (w * wt_inv_qm_esp)
            + ((qm_esp[0] - wt @ wt_inv_qm_esp) @ t) * w_grad.sum(axis=1) * (w_mm_esp_wt_inv @ wt_inv.T)
            + (qm_esp[0] - wt @ wt_inv_qm_esp) @ t_grad * (w * (w_mm_esp_wt_inv @ wt_inv.T))
            + (wt_inv.T @ wt_inv_qm_esp @ t) * w_grad.sum(axis=1) * (w * mm_esp[0] - w_mm_esp_wt_inv @ wt)
            + (wt_inv.T @ wt_inv_qm_esp) @ t_grad * (w * (w * mm_esp[0] - w_mm_esp_wt_inv @ wt))
        )
        return grad

    @staticmethod
    def _get_mm_energy_gradient_term4(
        t,
        t_grad,
        w,
        w_grad,
        mm_charges,
        qm_esp_charges
        ):

        grad = (
            -qm_esp_charges @ t_grad * (w * mm_charges)
            - mm_charges * w_grad.sum(axis=1) * (qm_esp_charges @ t)
        )

        return grad

    @staticmethod
    def _get_mm_energy_gradient_term5(qm_esp_gradient, qm_esp_charges):

        return qm_esp_charges @ qm_esp_gradient

    @staticmethod
    def _get_qm_energy_gradient(*args):
        return sum(args)
        
    @staticmethod
    def _get_mm_energy_gradient(term1, term2, term3, term4, term5, near_field_mask):
        grad = np.copy(term5)
        grad[:, near_field_mask] += term1 + term2 + term3 + term4
        return grad
