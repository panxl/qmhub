import math
import numpy as np
from scipy.special import erfc

import qmhub.helpmelib as pme

from ..utils import DependArray


PI = math.pi
SQRTPI = math.sqrt(math.pi)


class Ewald(object):
    def __init__(self, ri, rj, charges, cell_basis, tol=1e-6, cutoff=None, *, rij=None):
        self.cell_basis = cell_basis
        self.tol = tol
        self.cutoff = cutoff

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
        self.ewald_real = DependArray(
            name="ewald_real",
            func=Ewald._get_ewald_real,
            kwargs={
                'cutoff': cutoff,
                'alpha': self.alpha,
            },
            dependencies=[rij, charges],
        )
        self.ewald_recip = DependArray(
            name="ewald_recip",
            func=Ewald._get_ewald_recip,
            kwargs={
                'nfft': self.nfft,
                'alpha': self.alpha
            },
            dependencies=[rij, ri, rj, charges, cell_basis],
        )
        self.qm_full_esp = DependArray(
            name="qm_full_esp",
            func=Ewald._get_qm_full_esp,
            dependencies=[self.ewald_real, self.ewald_recip],
        )

    def __getitem__(self, index):
        return None

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
    def _get_ewald_real(rij, charges, cutoff, alpha):
        t = np.zeros((4, rij.shape[1]))

        d = np.linalg.norm(rij, axis=0)
        d2 = np.power(d, 2)
        mask = (d < cutoff) * (d > 0.)

        prod = np.zeros((rij.shape[1], rij.shape[2]))
        prod[mask] = erfc(alpha * d[mask]) / d[mask]
        t[0] = prod @ charges

        prod2 = np.zeros((rij.shape[1], rij.shape[2]))
        prod2[mask] = prod[mask] / d2[mask] + 2 * alpha * np.exp(-1 * alpha**2 * d2[mask]) / SQRTPI / d2[mask]
        t[1:] = (prod2 * rij) @ charges

        return t

    @staticmethod
    def _get_ewald_recip(rij, ri, rj, charges, cell_basis, nfft=None, alpha=None, order=6, correction=True):
        t = np.zeros((ri.shape[1], 4))

        pmeD = pme.PMEInstanceD()
        pmeD.setup(
            1,np.asscalar(alpha),
            order,
            *nfft.tolist(),
            1.,
            1,
        )
        pmeD.set_lattice_vectors(
            *np.diag(cell_basis).tolist(),
            *[90., 90., 90.],
            pmeD.LatticeType.XAligned,
        )

        charge = charges.array[:, np.newaxis]
        coord1 = np.ascontiguousarray(rj.T)
        coord2 = np.ascontiguousarray(ri.T)
        mat = pme.MatrixD
        pmeD.compute_P_rec(
            0, 
            mat(charge),
            mat(coord1),
            mat(coord2),
            1,
            mat(t),
        )

        if correction:
            # Self energy correction
            t[:, 0] -= (np.all(rij == 0., axis=0) * 2 * alpha / SQRTPI) @ charges

            # Net charge correction
            t[:, 0] -= (PI / np.linalg.det(cell_basis) / alpha**2) * charges.sum()

        return t.T

    @staticmethod
    def _get_qm_full_esp(ewald_real, ewald_recip):
        return (ewald_real + ewald_recip)
