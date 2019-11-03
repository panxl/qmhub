import numpy as np

from ..utils.darray import DependArray
from ..units import CODATA08_BOHR_TO_A


class Result(object):

    def __init__(
        self,
        qm_energy,
        qm_energy_gradient,
        mm_esp,
        qm_charges,
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

        self.mm_esp = DependArray(
            name="mm_esp",
            func=(lambda x: x / np.repeat([[CODATA08_BOHR_TO_A], [CODATA08_BOHR_TO_A**2]], [1, 3], axis=0)),
            dependencies=[mm_esp],
        )
        self._qm_energy_gradient = DependArray(
            name="qm_energy_gradient",
            func=(lambda x: x / CODATA08_BOHR_TO_A**2),
            dependencies=[
                qm_energy_gradient,
            ],
        )
        self.qm_esp_charges = DependArray(
            name="qm_esp_charges",
            func=Result._get_qm_esp_charges,
            dependencies=[
                scaling_factor,
                weighted_qmmm_coulomb_tensor_inv,
                self.mm_esp,
            ],
        )
        self._total_espc_gradient = DependArray(
            name="total_espc_gradient",
            func=elec.full._get_total_espc_gradient,
            dependencies=[
                self.qm_esp_charges,
            ],
        )

        # QM energy gradient
        self._energy_gradient_qm = DependArray(
            name="energy_gradient_qm",
            func=Result._get_energy_gradient_qm,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_gradient,
                scaling_factor,
                scaling_factor_gradient,
                weighted_qmmm_coulomb_tensor,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                elec.full.qm_total_esp,
                self._qm_energy_gradient,
                self.mm_esp,
                mm_charges,
            ],
        )

        # MM energy gradient
        self._energy_gradient_mm = DependArray(
            name="energy_gradient_mm",
            func=Result._get_energy_gradient_mm,
            dependencies=[
                qmmm_coulomb_tensor,
                qmmm_coulomb_tensor_gradient,
                scaling_factor,
                scaling_factor_gradient,
                weighted_qmmm_coulomb_tensor,
                weighted_qmmm_coulomb_tensor_inv,
                elec.qm_residual_esp,
                self.mm_esp,
                mm_charges,
            ],
        )

        # QM-QM energy and gradient correction
        if elec.qmqm is not None:
            self._qmqm_energy = DependArray(
                name="qmqm_energy",
                func=(lambda x, y: x[0] @ y / 2.),
                dependencies=[
                    elec.qmqm.qm_total_esp,
                    qm_charges,
                ],
            )
            self._qmqm_energy_gradient = DependArray(
                name="qmqm_energy_gradient",
                func=(lambda x, y: x[1:] * y),
                dependencies=[
                    elec.qmqm.qm_total_esp,
                    qm_charges,
                ],
            )
        else:
            self._qmqm_energy = DependArray(
                data=0,
                name="qmqm_energy",
            )
            self._qmqm_energy_gradient = DependArray(
                data=np.zeros((3, len(qm_charges))),
                name="qmqm_energy_gradient",
            )

        # Total QM/MM energy and gradient
        self.energy = DependArray(
            name="energy",
            func=(lambda x, y: x - y * CODATA08_BOHR_TO_A),
            dependencies=[
                qm_energy,
                self._qmqm_energy],
        )
        self.energy_gradient = DependArray(
            name="energy_gradient",
            func=Result._get_energy_gradient,
            kwargs={"scale": CODATA08_BOHR_TO_A**2},
            dependencies=[
                self._energy_gradient_qm,
                self._energy_gradient_mm,
                self._total_espc_gradient,
                near_field_mask,
                self._qmqm_energy_gradient,
            ],
        )

    @staticmethod
    def _get_qm_esp_charges(w, wt_inv, mm_esp):
        return w * mm_esp[0] @ wt_inv

    @staticmethod
    def _get_energy_gradient_qm(t, t_grad, w, w_grad, wt, wt_inv, qm_esp, qm_total_esp, qm_grad, mm_esp, mm_charges):

        qm_esp_charges =  w * mm_esp[0] @ wt_inv
        mm_usp_charges = wt_inv @ qm_esp[0] # Unscaled projected MM charges

        return (
            qm_grad + 
            w_grad @ (
                (mm_usp_charges + mm_charges) * (qm_esp_charges @ t - mm_esp[0]) -
                (qm_esp[0] - wt @ mm_usp_charges) @ t * (qm_esp_charges @ wt_inv.T) -
                (wt_inv.T @ mm_usp_charges) @ t * (w * mm_esp[0] - qm_esp_charges @ wt)
            ) +
            (t_grad @ ((mm_usp_charges + mm_charges) * w) + qm_total_esp[1:]) * qm_esp_charges -
            t_grad @ (qm_esp_charges @ wt_inv.T * w) * (qm_esp[0] - wt @ mm_usp_charges) -
            t_grad @ ((w * mm_esp[0] - qm_esp_charges @ wt) * w) * (wt_inv.T @ mm_usp_charges)
        )

    @staticmethod
    def _get_energy_gradient_mm(t, t_grad, w, w_grad, wt, wt_inv, qm_esp, mm_esp, mm_charges):

        qm_esp_charges =  w * mm_esp[0] @ wt_inv
        mm_usp_charges = wt_inv @ qm_esp[0] # Unscaled projected MM charges

        return (
            (mm_usp_charges + mm_charges) * (
                (mm_esp[1:] - qm_esp_charges @ t_grad) * w + 
                (mm_esp[0] - qm_esp_charges @ t) * w_grad.sum(axis=1)
            ) +
            (qm_esp_charges @ wt_inv.T) * (
                (qm_esp[0] - wt @ mm_usp_charges) @ t * w_grad.sum(axis=1) +
                (qm_esp[0] - wt @ mm_usp_charges) @ t_grad * w
            ) + 
            (w * mm_esp[0] - qm_esp_charges @ wt) * (
                (wt_inv.T @ mm_usp_charges @ t) * w_grad.sum(axis=1) +
                (wt_inv.T @ mm_usp_charges @ t_grad) * w
            )
        )

    @staticmethod
    def _get_energy_gradient(
        energy_gradient_qm,
        energy_gradient_mm,
        total_espc_gradient,
        near_field_mask,
        qmqm_energy_gradient,
        scale,
        ):

        grad = np.copy(total_espc_gradient)
        grad[:, :energy_gradient_qm.shape[1]] += energy_gradient_qm - qmqm_energy_gradient 
        grad[:, near_field_mask] += energy_gradient_mm
        return grad * scale
