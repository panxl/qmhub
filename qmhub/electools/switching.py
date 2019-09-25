import warnings
import numpy as np

from .distance import get_dij_min, get_dij_min_gradient


__all__ = ["get_scaling_factor", "get_scaling_factor_gradient"]


def get_scaling_factor(switching_type):
    if switching_type.lower() == "shift":
        return get_scaling_factor_shift
    elif switching_type.lower() == "switch":
        return get_scaling_factor_switch
    elif switching_type.lower() == "lrec":
        return get_scaling_factor_lrec
    else:
        raise ValueError("Switching function" +  switching_type + "not supported.")


def get_scaling_factor_gradient(switching_type):
    if switching_type.lower() == "shift":
        return get_scaling_factor_gradient_shift
    elif switching_type.lower() == "switch":
        return get_scaling_factor_gradient_switch
    elif switching_type.lower() == "lrec":
        return get_scaling_factor_gradient_lrec
    else:
        raise ValueError("Switching function" +  switching_type + "not supported.")


def get_scaling_factor_shift(dij_min=None, cutoff=None, swdist=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    ratio = dij_min / cutoff
    scaling_factor = (1 - ratio**2)**2
    scaling_factor *= (ratio < 1.)

    # MM1
    scaling_factor[np.where(dij_min < .8)] = 1.0

    return scaling_factor


def get_scaling_factor_gradient_shift(dij_min=None, dij_min_gradient=None, cutoff=None, swdist=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    if dij_min_gradient is None:
        dij_min_gradient = get_dij_min_gradient(rij=rij)

    ratio = dij_min / cutoff
    scaling_factor_gradient = -4 * (1 - ratio**2) * ratio / cutoff * dij_min_gradient
    scaling_factor_gradient *= (ratio < 1.)

    # MM1
    scaling_factor_gradient[:, :, np.where(dij_min < .8)] = 0.0

    return scaling_factor_gradient


def get_scaling_factor_switch(dij_min=None, cutoff=None, swdist=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    if swdist is None:
        swdist = 0.75 * cutoff
    if cutoff < swdist:
        raise ValueError("Cutoff should be greater than Swdist.")

    dratio2 = (dij_min / cutoff)**2
    sratio2 = (swdist / cutoff)**2

    if sratio2 < 1.:
        scaling_factor = ((1 - dratio2)**2
                        * (3 * sratio2 + 2 * dratio2 - 1)
                        / (1 - sratio2)**3
                        * (dratio2 >= sratio2)
                        * (dratio2 < 1.)
                        + (dratio2 < sratio2))
    elif sratio2 == 1.:
        warnings.warn("Not switching MM charges might cause discontinuity at the cutoff.")
        scaling_factor = (dratio2 < sratio2).astype(float)

    # MM1
    scaling_factor[np.where(dij_min < .8)] = 1.0

    return scaling_factor


def get_scaling_factor_gradient_switch(dij_min=None, dij_min_gradient=None, cutoff=None, swdist=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    if dij_min_gradient is None:
        dij_min_gradient = get_dij_min_gradient(rij=rij)

    dratio2 = (dij_min / cutoff)**2
    sratio2 = (swdist / cutoff)**2

    if sratio2 < 1.:
        scaling_factor_gradient = (12 * (dratio2 - sratio2)
                                * (1 - dratio2)
                                / (1 - sratio2)**3 * dij_min / cutoff**2 * dij_min_gradient)
        scaling_factor_gradient *= (dratio2 >= sratio2) * (dratio2 < 1.)
    elif sratio2 == 1.:
        scaling_factor_gradient = np.zeros_like(dij_min_gradient)

    # MM1
    scaling_factor_gradient[:, :, np.where(dij_min < .8)] = 0.0

    return scaling_factor_gradient


def get_scaling_factor_lrec(dij_min=None, cutoff=None, swdist=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    ratio = 1 - dij_min / cutoff
    scaling_factor = 1 - (2 * ratio**3 - 3* ratio**2 + 1)**2
    scaling_factor *= (ratio < 1.)

    # MM1
    scaling_factor[np.where(dij_min < .8)] = 1.0

    return scaling_factor


def get_scaling_factor_gradient_lrec(dij_min=None, dij_min_gradient=None, cutoff=None, swdist=None, *, rij=None):
    if dij_min is None:
        dij_min = get_dij_min(rij=rij)

    if dij_min_gradient is None:
        dij_min_gradient = get_dij_min_gradient(rij=rij)

    ratio = 1 - dij_min / cutoff
    scaling_factor_gradient = 12 * ratio * (2 * ratio**3 - 3 * ratio**2 + 1) * dij_min / cutoff**2 * dij_min_gradient
    scaling_factor_gradient *= (ratio < 1.)

    # MM1
    scaling_factor_gradient[:, :, np.where(dij_min < .8)] = 0.0

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
