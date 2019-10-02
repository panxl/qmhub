import numpy as np
from ..units import COULOMB_CONSTANT


def _get_qmmm_coulomb_tensor_gradient_qm(
    dij_inverse,
    dij_inverse_gradient,
    scaling_factor,
    scaling_factor_gradient,
    ):

    t = dij_inverse
    t_grad = np.zeros((3, dij_inverse.shape[0], dij_inverse.shape[0],  dij_inverse.shape[1]))
    for i in range(dij_inverse_gradient.shape[1]):
        for j in range(dij_inverse_gradient.shape[2]):
            t_grad[:, i, i, j] = -dij_inverse_gradient[:, i, j]

    w = np.diag(scaling_factor)
    w_grad = np.zeros((3, scaling_factor_gradient.shape[1], scaling_factor_gradient.shape[2],  scaling_factor_gradient.shape[2]))
    for i in range(dij_inverse_gradient.shape[1]):
        for j in range(dij_inverse_gradient.shape[2]):
            w_grad[:, i, j, j] = -scaling_factor_gradient[:, i, j]

    tw_grad = t @ w_grad + t_grad @ w

    return tw_grad * COULOMB_CONSTANT


def _get_qmmm_coulomb_tensor_gradient_mm(dij_inverse, dij_inverse_gradient, scaling_factor, scaling_factor_gradient):
    t = dij_inverse
    t_grad = np.zeros((3, dij_inverse.shape[1], dij_inverse.shape[0],  dij_inverse.shape[1]))
    for i in range(dij_inverse_gradient.shape[1]):
        for j in range(dij_inverse_gradient.shape[2]):
            t_grad[:, j, i, j] = dij_inverse_gradient[:, i, j]

    w = np.diag(scaling_factor)
    w_grad = np.zeros((3, scaling_factor_gradient.shape[2], scaling_factor_gradient.shape[2],  scaling_factor_gradient.shape[2]))
    for i in range(dij_inverse_gradient.shape[1]):
        for j in range(dij_inverse_gradient.shape[2]):
            w_grad[:, j, j, j] = scaling_factor_gradient[:, :, j].sum(axis=1)

    tw_grad = t @ w_grad + t_grad @ w

    return tw_grad * COULOMB_CONSTANT


@staticmethod
def _get_qm_scaled_esp(
    qmmm_coulomb_tensor,
    qmmm_coulomb_tensor_gradient_qm,
    charges,
    ):
    esp = np.zeros((4, qmmm_coulomb_tensor.shape[0]))
    esp[0] = qmmm_coulomb_tensor @ charges
    esp[1] = np.diag((qmmm_coulomb_tensor_gradient_qm @ charges)[0])
    esp[2] = np.diag((qmmm_coulomb_tensor_gradient_qm @ charges)[1])
    esp[3] = np.diag((qmmm_coulomb_tensor_gradient_qm @ charges)[2])
    return esp
