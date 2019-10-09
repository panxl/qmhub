import math
import numpy as np
from scipy.special import erfc

import qmhub.helpmelib as pme

from ..utils import DependArray
from ..units import COULOMB_CONSTANT


PI = math.pi
SQRTPI = math.sqrt(math.pi)


class Ewald(object):
    def __init__(
        self,
        qm_positions,
        mm_positions,
        qm_charges,
        mm_charges,
        cell_basis,
        exclusion=None,
        tol=1e-10,
        *,
        cutoff=None,
        order=6,
        **kwargs
        ):

        self.qm_positions = qm_positions
        self.mm_positions = mm_positions
        self.qm_charges = qm_charges
        self.mm_charges = mm_charges
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
                mm_positions,
            ],
        )
        self.ewald_real = DependArray(
            name="ewald_real",
            func=(lambda x, y: x @ y),
            dependencies=[
                self.ewald_real_tensor,
                mm_charges,
            ],
        )
        self.ewald_recip = DependArray(
            name="ewald_recip",
            func=Ewald._get_ewald_recip,
            kwargs={
                'nfft': self.nfft,
                'alpha': self.alpha,
                'exclusion': exclusion,
                'order': order,
            },
            dependencies=[
                qm_positions,
                mm_positions,
                qm_charges,
                mm_charges,
                cell_basis,
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
        mm_positions,
        cutoff,
        alpha,
        exclusion=None
        ):

        t = np.zeros((4, qm_positions.shape[1], mm_positions.shape[1]))

        rij = mm_positions[:, np.newaxis, :] - qm_positions[:, :, np.newaxis]
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

        return t * COULOMB_CONSTANT

    @staticmethod
    def _get_ewald_recip(
        qm_positions,
        mm_positions,
        qm_charges,
        mm_charges,
        cell_basis,
        nfft=None,
        alpha=None,
        exclusion=None,
        order=6,
        ):

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

        ri = qm_positions.T
        rj = mm_positions.T
        t = np.zeros((len(ri), 4))

        charges = np.concatenate((qm_charges, mm_charges))[:, np.newaxis]
        coord1 = np.ascontiguousarray(np.concatenate((ri, rj)))
        coord2 = np.ascontiguousarray(ri)
        mat = pme.MatrixD
        pmeD.compute_P_rec(
            0, 
            mat(charges),
            mat(coord1),
            mat(coord2),
            1,
            mat(t),
        )

        if exclusion is not None:
            r = np.concatenate((ri, rj[np.asarray(exclusion)]))[np.newaxis:, ] - ri[:, np.newaxis]
            d = np.linalg.norm(r, axis=-1)
            d2 = np.power(d, 2)
            prod = (1 - erfc(alpha * d)) / d
            prod2 = prod / d2 - 2 * alpha * np.exp(-1 * alpha**2 * d2) / SQRTPI / d2
            prod[np.nonzero(np.isnan(prod))] = 0.
            prod2[np.nonzero(np.isnan(prod2))] = 0.

            exclusion_charges = np.concatenate((qm_charges, mm_charges[np.asarray(exclusion)]))
            t[:, 0] -= prod @ exclusion_charges
            t[:, 1:] -= exclusion_charges @ (prod2[:, :, np.newaxis] * r)

        # Self energy correction
        t[:, 0] -= 2 * qm_charges * alpha / SQRTPI

        # Net charge correction
        t[:, 0] -= (PI / np.linalg.det(cell_basis) / alpha**2) * charges.sum()

        return t.T * COULOMB_CONSTANT

    def _get_total_espc_gradient(self, qm_esp_charges):

        pmeD = pme.PMEInstanceD()
        pmeD.setup(
            1,np.asscalar(self.alpha),
            self.order,
            *self.nfft.tolist(),
            1.,
            1,
        )
        pmeD.set_lattice_vectors(
            *np.diag(self.cell_basis).tolist(),
            *[90., 90., 90.],
            pmeD.LatticeType.XAligned,
        )

        ri = self.qm_positions.T
        rj = self.mm_positions.T
        t = np.zeros((len(ri) + len(rj), 4))

        charges = np.ascontiguousarray(qm_esp_charges)[:, np.newaxis]
        coord1 = np.ascontiguousarray(ri)
        coord2 = np.ascontiguousarray(np.concatenate((ri, rj)))
        mat = pme.MatrixD
        pmeD.compute_P_rec(
            0,
            mat(charges),
            mat(coord1),
            mat(coord2),
            1,
            mat(t),
        )

        grad = t.T[1:] * COULOMB_CONSTANT

        r = self.qm_positions[:, np.newaxis, :] - self.qm_positions[:, :, np.newaxis]
        d = np.linalg.norm(r, axis=0)
        d2 = np.power(d, 2)
        prod = (1 - erfc(self.alpha * d)) / d
        prod2 = prod / d2 - 2 * self.alpha * np.exp(-1 * self.alpha**2 * d2) / SQRTPI / d2
        prod[np.nonzero(np.isnan(prod))] = 0.
        prod2[np.nonzero(np.isnan(prod2))] = 0.
 
        grad[:, np.arange(len(ri))] += qm_esp_charges @(prod2[np.newaxis] * r) * COULOMB_CONSTANT

        if self.exclusion is not None:
            r = self.mm_positions[:, np.newaxis, np.asarray(self.exclusion)] - self.qm_positions[:, :, np.newaxis]
            d = np.linalg.norm(r, axis=0)
            d2 = np.power(d, 2)
            prod = (1 - erfc(self.alpha * d)) / d
            prod2 = prod / d2 - 2 * self.alpha * np.exp(-1 * self.alpha**2 * d2) / SQRTPI / d2

            grad[:, np.asarray(self.exclusion + len(ri))] += qm_esp_charges @(prod2[np.newaxis] * r) * COULOMB_CONSTANT

        grad[:, len(ri):] += qm_esp_charges @ -self.ewald_real_tensor[1:]

        charges = np.concatenate((self.qm_charges, self.mm_charges))

        return grad * charges
