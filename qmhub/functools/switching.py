import numpy as np

from . import get_dij_min, get_dij_min_gradient

def get_scaling_factor(switching_type):
    if switching_type.lower() == "shift":
        return get_scaling_factor_shift
    else:
        raise ValueError("Switching function" +  switching_type + "not supported.")


def get_scaling_factor_gradient(switching_type):
    if switching_type.lower() == "shift":
        return get_scaling_factor_gradient_shift
    else:
        raise ValueError("Switching function" +  switching_type + "not supported.")


def get_scaling_factor_shift(cutoff, swdist, dij_min=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    ratio = dij_min / cutoff
    scaling_factor = (1 - ratio**2)**2
    scaling_factor *= (ratio < 1.)

    return scaling_factor


def get_scaling_factor_gradient_shift(cutoff, swdist, dij_min=None, dij_min_gradient=None, *, rij=None):
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
    print(get_numerical_gradient(get_scaling_factor_shift, get_scaling_factor_gradient_shift))
