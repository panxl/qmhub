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


def get_scaling_factor(cutoff, swdist, dij_min=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    ratio = dij_min / cutoff
    scaling_factor = (1 - ratio**2)**2
    scaling_factor *= (ratio < 1.)

    return scaling_factor


def get_scaling_factor_gradient(cutoff, swdist, dij_min=None, dij_min_gradient=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    if dij_min_gradient is None:
        dij_min_gradient = get_dij_min_gradient(rij=rij)

    ratio = dij_min / cutoff
    scaling_factor_gradient = -4 * (1 - ratio**2) * ratio / cutoff * dij_min_gradient
    scaling_factor_gradient *= (ratio < 1.)

    return scaling_factor_gradient


def get_numerical_gradient(func, grad):
    ri = np.random.rand(3, 5) * 10
    rj = np.random.rand(3, 10) * 10
    rij = rj[:, np.newaxis, :] - ri[:, :, np.newaxis]
    gradient = grad(cutoff=10., rij=rij)

    m = np.random.randint(0, 3)
    i = np.random.randint(0, 5)
    ri_pos = np.copy(ri)
    ri_pos[m, i] += 0.001
    rij_pos = rj[:, np.newaxis, :] - ri_pos[:, :, np.newaxis]
    ri_neg = np.copy(ri)
    ri_neg[m, i] -= 0.001
    rij_neg = rj[:, np.newaxis, :] - ri_neg[:, :, np.newaxis]

    numerical_gradient = (func(cutoff=10., rij=rij_pos) - func(cutoff=10., rij=rij_neg)) / 0.002
    return gradient[m, i], numerical_gradient

if __name__ == "__main__":
    print(get_numerical_gradient(get_scaling_factor, get_scaling_factor_gradient))