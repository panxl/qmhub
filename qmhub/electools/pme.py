import math
import numpy as np
from scipy.special import erfc

from ..utils.darray import DependArray
from ..utils.dpme import DependPME


PI = math.pi
SQRTPI = math.sqrt(math.pi)


class Ewald(object):
    def __init__(
        self,
        qm_positions,
        positions,
        charges,
        cell_basis,
        exclusion=None,
        tol=1e-8,
        *,
        cutoff=None,
        order=6,
        **kwargs
        ):

        self.qm_positions = qm_positions
        self.positions = positions
        self.charges = charges
        self.cell_basis = cell_basis
        self.exclusion = exclusion
        self.tol = tol
        self.cutoff = cutoff
        self.order = order

        self.alpha = DependArray(
            name="alpha",
            func=Ewald._get_alpha,
            kwargs={
                'cutoff': cutoff,
                'tol': tol,
            },
        )
        self.nfft = DependArray(
            name="nfft",
            func=Ewald._get_nfft,
            kwargs={'cell_basis': cell_basis},
        )

        self.pme = DependPME(self.cell_basis, self.alpha, self.order, self.nfft)

        self.ewald_real_tensor = DependArray(
            name="ewald_real_tensor",
            func=Ewald._get_ewald_real_tensor,
            kwargs={
                'cutoff': cutoff,
                'alpha': self.alpha,
                'exclusion': exclusion,
            },
            dependencies=[
                qm_positions,
                positions,
            ],
        )
        self.ewald_real = DependArray(
            name="ewald_real",
            func=(lambda x, y: x @ y),
            dependencies=[
                self.ewald_real_tensor,
                charges,
            ],
        )
        self.ewald_recip_exclusion_tensor = DependArray(
            name="ewald_recip_exclusion_tensor",
            func=Ewald._get_ewald_recip_exclusion_tensor,
            kwargs={
                'alpha': self.alpha,
                'exclusion': exclusion,
            },
            dependencies=[
                qm_positions,
                positions,
            ],
        )
        self.ewald_recip = DependArray(
            name="ewald_recip",
            func=Ewald._get_ewald_recip,
            kwargs={
                'alpha': self.alpha,
                'exclusion': exclusion,
            },
            dependencies=[
                self.pme,
                qm_positions,
                positions,
                charges,
                cell_basis,
                self.ewald_recip_exclusion_tensor,
            ],
        )
        self.qm_total_esp = DependArray(
            name="qm_full_esp",
            func=(lambda x, y: x + y),
            dependencies=[self.ewald_real, self.ewald_recip],
        )

    @property
    def volume(self):
        return np.linalg.det(self.cell_basis)

    @staticmethod
    def _get_alpha(cutoff, tol):
        alpha = 1.
        while erfc(alpha * cutoff) / cutoff >= tol:
            alpha *= 2.
        alpha_lo = 0.
        alpha_hi = alpha
        for _ in range(100):
            alpha = .5 * (alpha_lo + alpha_hi)
            if erfc(alpha * cutoff) / cutoff >= tol:
                alpha_lo = alpha
            else:
                alpha_hi = alpha
        return alpha

    @staticmethod
    def _get_nfft(cell_basis):
        minimum = np.floor(np.diag(cell_basis)).astype(int)

        return [Ewald._find_fft_dimension(m) for m in minimum]

    @staticmethod
    def _find_fft_dimension(minimum):
        '''From from OpenMM'''
        if minimum < 1:
            return 1
        while True:
            unfactored = minimum
            for factor in range(2, 6):
                while unfactored > 1 and unfactored % factor == 0:
                    unfactored /= factor
            if unfactored == 1:
                return minimum
            minimum += 1

    @staticmethod
    def _get_ewald_real_tensor(
        qm_positions,
        positions,
        cutoff,
        alpha,
        exclusion=None
        ):

        t = np.zeros((4, qm_positions.shape[1], positions.shape[1]))

        rij = positions[:, np.newaxis, :] - qm_positions[:, :, np.newaxis]
        d = np.linalg.norm(rij, axis=0)
        d2 = np.power(d, 2)
        mask = (d < cutoff)
        mask[:, np.asarray(exclusion)] = False
        d = d[mask]
        d2 = d2[mask]

        prod = np.zeros((rij.shape[1], rij.shape[2]))
        prod[mask] = erfc(alpha * d) / d
        t[0] = prod

        prod2 = np.zeros((rij.shape[1], rij.shape[2]))
        prod2[mask] = prod[mask] / d2 + 2 * alpha * np.exp(-1 * alpha**2 * d2) / SQRTPI / d2
        t[1:] = (prod2 * rij)

        return t

    @staticmethod
    def _get_ewald_recip_exclusion_tensor(
        qm_positions,
        positions,
        alpha=None,
        exclusion=None,
        ):

        r = positions[:, np.newaxis, np.asarray(exclusion)] - qm_positions[:, :, np.newaxis]
        d = np.linalg.norm(r, axis=0)
        d2 = np.power(d, 2)
        prod = (1 - erfc(alpha * d)) / d
        prod2 = prod / d2 - 2 * alpha * np.exp(-1 * alpha**2 * d2) / SQRTPI / d2
        np.nan_to_num(prod, copy=False)
        np.nan_to_num(prod2, copy=False)

        return np.concatenate((prod[np.newaxis], prod2 * r))

    @staticmethod
    def _get_ewald_recip(
        pme,
        qm_positions,
        positions,
        charges,
        cell_basis,
        recip_exclusion_tensor,
        alpha=None,
        exclusion=None,
        ):

        recip_esp = pme.compute_recip_esp(qm_positions, positions, charges)

        recip_esp[0] -= recip_exclusion_tensor[0] @ charges[np.asarray(exclusion)]
        recip_esp[1:] -= recip_exclusion_tensor[1:] @ charges[np.asarray(exclusion)]

        # Self energy correction
        recip_esp[0] -= 2 * charges[:qm_positions.shape[1]] * alpha / SQRTPI

        # Net charge correction
        recip_esp[0] -= (PI / np.linalg.det(cell_basis) / alpha**2) * charges.sum()

        return recip_esp

    def _get_total_espc_gradient(self, qm_esp_charges):

        recip_grad = self.pme.compute_recip_esp(self.positions, self.qm_positions, qm_esp_charges)[1:]
        recip_grad[:, np.asarray(self.exclusion)] += qm_esp_charges @ self.ewald_recip_exclusion_tensor[1:]

        real_grad = qm_esp_charges @ -self.ewald_real_tensor[1:]

        return (real_grad + recip_grad) * self.charges
