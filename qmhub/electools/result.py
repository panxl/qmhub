import numpy as np

from ..utils.darray import DependArray


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
        units=None,
        ):

        if units is None:
            from ..units import CODATA18_HARTREE_TO_KCAL, CODATA18_BOHR_TO_A
            units = (CODATA18_HARTREE_TO_KCAL, CODATA18_BOHR_TO_A)

        HARTREE_TO_KCAL = units[0]
        BOHR_TO_ANGSTROM = units[1]
        COULOMB_CONSTANT = HARTREE_TO_KCAL * BOHR_TO_ANGSTROM

        self.mm_esp = DependArray(
            name="mm_esp",
            func=(lambda x: x / np.repeat([[BOHR_TO_ANGSTROM], [BOHR_TO_ANGSTROM**2]], [1, 3], axis=0)),
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
        self.total_espc_gradient = DependArray(
            name="total_espc_gradient",
            func=elec.full._get_total_espc_gradient,
            dependencies=[
                self.qm_esp_charges,
            ],
        )

        # QM energy gradient
        self._qm_energy_gradient_term1 = DependArray(
            name="qm_energy_gradient_term1",
            func=(lambda x: x / BOHR_TO_ANGSTROM**2),
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

        # MM energy gradient
        self._mm_energy_gradient_term1 = DependArray(
            name="mm_energy_gradient_term1",
            func=Result._get_mm_energy_gradient_term1,
            dependencies=[
                self.mm_esp,
                elec.embedding_mm_charges,
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

        # QM-QM energy and gradient correction
        if elec.qmqm is not None:
            self.qmqm_energy = DependArray(
                name="qmqm_energy",
                func=(lambda x, y: x[0] @ y / 2.),
                dependencies=[
                    elec.qmqm.qm_total_esp,
                    qm_charges,
                ],
            )
            self.qmqm_energy_gradient = DependArray(
                name="qmqm_energy_gradient",
                func=(lambda x, y: x[1:] * y),
                dependencies=[
                    elec.qmqm.qm_total_esp,
                    qm_charges,
                ],
            )
        else:
            self.qmqm_energy = DependArray(
                data=0,
                name="qmqm_energy",
            )
            self.qmqm_energy_gradient = DependArray(
                data=np.zeros((3, len(qm_charges))),
                name="qmqm_energy_gradient",
            )

        # Total QM/MM energy and gradient
        self.energy = DependArray(
            name="energy",
            func=(lambda x, y: x * HARTREE_TO_KCAL - y * COULOMB_CONSTANT),
            dependencies=[
                qm_energy,
                self.qmqm_energy],
        )
        self.energy_gradient = DependArray(
            name="energy_gradient",
            func=Result._get_energy_gradient,
            kwargs={"coulomb_constant": COULOMB_CONSTANT},
            dependencies=[
                self._qm_energy_gradient_term1,
                self._qm_energy_gradient_term2,
                self._qm_energy_gradient_term3,
                self._qm_energy_gradient_term4,
                self._mm_energy_gradient_term1,
                self._mm_energy_gradient_term2,
                self._mm_energy_gradient_term3,
                self._mm_energy_gradient_term4,
                self.total_espc_gradient,
                near_field_mask,
                self.qmqm_energy_gradient,
            ],
        )

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

        return (
            (w_mm_esp_wt_inv @ t * w_grad) @ wt_inv_qm_esp
            + w_mm_esp_wt_inv * (t_grad * w @ wt_inv_qm_esp)
            - (qm_esp[0] - wt @ wt_inv_qm_esp) @ t * w_grad @ (w_mm_esp_wt_inv @ wt_inv.T)
            - (qm_esp[0] - wt @ wt_inv_qm_esp) * (t_grad * w @ (w_mm_esp_wt_inv @ wt_inv.T))
            - (wt_inv.T @ wt_inv_qm_esp) @ t * w_grad @ (w * mm_esp[0] - w_mm_esp_wt_inv @ wt)
            - (wt_inv.T @ wt_inv_qm_esp) * (t_grad * w @ (w * mm_esp[0] - w_mm_esp_wt_inv @ wt))
        )

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

        return (
            qm_total_esp[1:] * qm_esp_charges
            + (mm_charges * w_grad) @ (qm_esp_charges @ t)
            + (w * t_grad) @ mm_charges * qm_esp_charges
        )

    @staticmethod
    def _get_mm_energy_gradient_term1(mm_esp, embedding_mm_charges):
        return embedding_mm_charges * mm_esp[1:]

    @staticmethod
    def _get_mm_energy_gradient_term2(w_grad, wt_inv, qm_esp, mm_esp, mm_charges):
        return  mm_esp[0] * w_grad.sum(axis=1) * (wt_inv @ qm_esp[0] + mm_charges)

    @staticmethod
    def _get_mm_energy_gradient_term3(t, t_grad, w, w_grad, wt, wt_inv, qm_esp, mm_esp):

        w_mm_esp_wt_inv =  w * mm_esp[0] @ wt_inv
        wt_inv_qm_esp = wt_inv @ qm_esp[0]

        return (
            -(w_mm_esp_wt_inv @ t) * w_grad.sum(axis=1) * wt_inv_qm_esp
            - w_mm_esp_wt_inv @ t_grad * (w * wt_inv_qm_esp)
            + ((qm_esp[0] - wt @ wt_inv_qm_esp) @ t) * w_grad.sum(axis=1) * (w_mm_esp_wt_inv @ wt_inv.T)
            + (qm_esp[0] - wt @ wt_inv_qm_esp) @ t_grad * (w * (w_mm_esp_wt_inv @ wt_inv.T))
            + (wt_inv.T @ wt_inv_qm_esp @ t) * w_grad.sum(axis=1) * (w * mm_esp[0] - w_mm_esp_wt_inv @ wt)
            + (wt_inv.T @ wt_inv_qm_esp) @ t_grad * (w * (w * mm_esp[0] - w_mm_esp_wt_inv @ wt))
        )

    @staticmethod
    def _get_mm_energy_gradient_term4(
        t,
        t_grad,
        w,
        w_grad,
        mm_charges,
        qm_esp_charges
        ):

        return (
            -qm_esp_charges @ t_grad * (w * mm_charges)
            - mm_charges * w_grad.sum(axis=1) * (qm_esp_charges @ t)
        )

    @staticmethod
    def _get_energy_gradient(
        qm_term1,
        qm_term2,
        qm_term3,
        qm_term4,
        mm_term1,
        mm_term2,
        mm_term3,
        mm_term4,
        total_espc_gradient,
        near_field_mask,
        qmqm_energy_gradient,
        coulomb_constant,
        ):

        grad = np.copy(total_espc_gradient)
        grad[:, :qm_term1.shape[1]] += qm_term1 + qm_term2 + qm_term3 + qm_term4 - qmqm_energy_gradient 
        grad[:, near_field_mask] += mm_term1 + mm_term2 + mm_term3 + mm_term4
        return grad * coulomb_constant
