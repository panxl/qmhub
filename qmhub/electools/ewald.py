import math
import numpy as np
from scipy.special import erfc

from ..utils import DependArray
from ..units import COULOMB_CONSTANT

PI = math.pi
SQRTPI = math.sqrt(math.pi)


class Ewald(object):
    def __init__(self, ri, rj, charges, cell_basis, tol=1e-6, *, order='spherical', rij=None, **kwargs):
        self.cell_basis = cell_basis
        self.tol = tol
        self.order = order

        self.threshold = DependArray(
            name="threshold",
            func=Ewald._get_threshold,
            kwargs={'tol': self.tol},
        )
        self.alpha = DependArray(
            name="alpha",
            func=Ewald._get_alpha,
            kwargs={'cell_basis': self.cell_basis},
        )

        # Real space
        self.nmax = DependArray(
            name="nmax",
            func=Ewald._get_nmax,
            kwargs={
                'threshold': self.threshold,
                'alpha': self.alpha,
                'cell_basis': self.cell_basis,
            },
        )
        self._real_vectors = DependArray(
            name="_real_vectors",
            func=Ewald._get_vectors,
            kwargs={
                'maxes': self.nmax,
                'space': 'real',
            },
        )
        self.real_lattice = DependArray(
            name="real_lattice",
            func=Ewald._get_lattice,
            kwargs={
                'vectors': self._real_vectors,
                'maxes': self.nmax,
                'order': self.order,
            },
            dependencies=[self.cell_basis],
        )

        # Reciprocal space
        self.recip_basis = DependArray(
            name="recip_basis",
            func=Ewald._get_recip_basis,
            dependencies=[self.cell_basis],
        )
        self.kmax = DependArray(
            name="kmax",
            func=Ewald._get_kmax,
            kwargs={
                'threshold': self.threshold,
                'alpha': self.alpha,
                'recip_basis': self.recip_basis,
            },
        )
        self._recip_vectors = DependArray(
            name="_recip_vectors",
            func=Ewald._get_vectors,
            kwargs={
                'maxes': self.kmax,
                'space': 'recip',
            }
        )
        self.recip_lattice = DependArray(
            name="recip_lattice",
            func=Ewald._get_lattice,
            kwargs={
                'vectors': self._recip_vectors,
                'maxes': self.kmax,
                'order': self.order,
            },
            dependencies=[self.recip_basis],
        )

        # Ewald
        self.ewald_real = DependArray(
            name="ewald_real",
            func=Ewald._get_ewald_real,
            kwargs={'alpha': self.alpha},
            dependencies=[rij, self.real_lattice],
        )
        self.ewald_recip = DependArray(
            name="ewald_recip",
            func=Ewald._get_ewald_recip,
            kwargs={'alpha': self.alpha},
            dependencies=[rij, self.recip_lattice, self.cell_basis],
        )
        self.qm_full_esp = DependArray(
            name="qm_full_esp",
            func=Ewald._get_qm_full_esp,
            dependencies=[self.ewald_real, self.ewald_recip, charges],
        )

    def __getitem__(self, index):
        return None

    @property
    def volume(self):
        return np.linalg.det(self.cell_basis)

    @staticmethod
    def _get_threshold(tol):
        return math.sqrt(-1 * math.log(tol))

    @staticmethod
    def _get_alpha(cell_basis):
        return SQRTPI / np.diag(cell_basis).max()

    @staticmethod
    def _get_nmax(threshold, alpha, cell_basis):
        return np.ceil(threshold / alpha / np.diag(cell_basis)).astype(int)

    @staticmethod
    def _get_kmax(threshold, alpha, recip_basis):
        return np.ceil(2 * threshold * alpha / np.diag(recip_basis)).astype(int)

    @staticmethod
    def _get_recip_basis(cell_basis):
        return 2 * PI * np.linalg.inv(cell_basis).T
 
    @staticmethod
    def _get_vectors(maxes, space):
        vectors = np.mgrid[-maxes[0]:maxes[0]+1, -maxes[1]:maxes[1]+1, -maxes[2]:maxes[2]+1]
        vectors = vectors.reshape(3, -1)

        if space.lower() == 'recip':
            vectors = vectors[:, ~np.all(vectors == 0, axis=0)]

        return vectors

    @staticmethod
    def _get_lattice(cell_basis, vectors, maxes, order):
        lattice = np.dot(cell_basis, vectors)
        if order.lower() == 'spherical':
            mask = np.linalg.norm(lattice, axis=0) <= np.max(maxes * np.diag(cell_basis))
            return lattice[:, mask]
        elif order.lower() == 'rectangular':
            return lattice

    @staticmethod    
    def _get_ewald_real(rij, lattice, alpha):
        t = np.zeros((4, rij.shape[1], rij.shape[2]))

        r = rij[:, np.newaxis] + lattice[:, :, np.newaxis, np.newaxis]
        d = np.linalg.norm(r, axis=0)
        d2 = np.power(d, 2)
        prod = erfc(alpha * d) / d
        prod[np.where(np.isinf(prod))] = 0.
        t[0] = prod.sum(axis=0)

        prod2 = prod / d2 + 2 * alpha * np.exp(-1 * alpha**2 * d2) / SQRTPI / d2
        prod2[np.where(np.isnan(prod2))] = 0.
        t[1:] = (prod2[np.newaxis] * r).sum(axis=1)

        return t * COULOMB_CONSTANT

    @staticmethod
    def _get_ewald_recip(rij, lattice, cell_basis, alpha, correction=True):
        t = np.zeros((4, rij.shape[1], rij.shape[2]))

        volume = np.linalg.det(cell_basis)
        k2 = np.linalg.norm(lattice, axis=0)**2
        prefac = (4 * PI / volume) * np.exp(-1 * k2 / (4 * alpha**2)) / k2

        kr = (rij.T @ lattice)
        t[0] = (np.cos(kr) @ prefac).T
        t[1:] = (np.sin(kr) @ (prefac * lattice).T).T

        if correction:
            # Self energy correction
            t[0] -= np.all(rij == 0., axis=0) * 2 * alpha / SQRTPI

            # Net charge correction
            t[0] -= PI / volume / alpha**2

        return t * COULOMB_CONSTANT

    @staticmethod
    def _get_qm_full_esp(ewald_real, ewald_recip, charges):
        return (ewald_real + ewald_recip) @ charges
