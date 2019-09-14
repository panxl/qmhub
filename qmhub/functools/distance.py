import numpy as np


def get_rij(ri, rj):
    return rj[:, np.newaxis, :] - ri[:, :, np.newaxis]


def get_dij2(rij):
    return np.sum(rij**2, axis=0)


def get_dij2_gradient(rij):
    return -2 * rij


def get_dij(dij2=None, *, rij=None):
    if dij2 is None:
        dij2 = get_dij2(rij=rij)

    return np.sqrt(dij2)


def get_dij_gradient(dij=None, dij2_gradient=None, *, rij=None):
    if dij is None:
        dij = get_dij(rij=rij)

    if dij2_gradient is None:
        dij2_gradient = get_dij2_gradient(rij)

    return dij2_gradient / dij / 2.


def get_dij_inverse(dij=None, *, rij=None):
    if dij is None:
        dij = get_dij(rij=rij)

    return 1 / dij


def get_dij_inverse_gradient(dij_inverse=None, dij_gradient=None, *, rij=None):
    if dij_inverse is None:
        dij_inverse = get_dij_inverse(rij=rij)

    if dij_gradient is None:
        dij_gradient = get_dij_gradient(rij=rij)

    return -1 * dij_inverse**2 * dij_gradient


def get_dij_min(dij_inverse=None, *, rij=None, beta=500.):
    if dij_inverse is None:
        dij_inverse = get_dij_inverse(rij=rij)

    a = beta * dij_inverse

    if np.all(np.isinf(np.diag(dij_inverse))):
        np.fill_diagonal(a, 0)

    a_max = a.max(axis=0)
    logsumexp = np.log(np.exp(a - a_max).sum(axis=0)) + a_max
    dij_min = beta / logsumexp

    return dij_min


def get_dij_min_gradient(dij_min=None, dij_inverse=None, dij_inverse_gradient=None, *, rij=None, beta=500.):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    if dij_inverse is None:
        dij_inverse = get_dij_inverse(rij=rij)

    if dij_inverse_gradient is None:
        dij_inverse_gradient = get_dij_inverse_gradient(rij=rij)

    dij_min_gradient = -1 * dij_min**2 * np.exp(beta * dij_inverse - beta / dij_min) * dij_inverse_gradient

    return dij_min_gradient
