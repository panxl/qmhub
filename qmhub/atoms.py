from collections.abc import Sequence

import numpy as np

from .utils.darray import DependArray


class Atoms(Sequence):
    def __init__(self, positions=None, charges=None, elements=None):
        self.positions = positions
        self.charges = charges
        self.elements = elements

    @classmethod
    def new(cls, n_atoms):
        kwargs = {
            'positions': DependArray(np.zeros((3, n_atoms))),
            'charges': DependArray(np.zeros(n_atoms)),
            'elements': DependArray(np.zeros(n_atoms, dtype=int)),
        }

        return cls(**kwargs)

    @classmethod
    def from_atoms(cls, atoms, index=None):
        kwargs = {
            'positions': DependArray.from_darray(atoms.positions, np.s_[:, index]),
            'charges': DependArray.from_darray(atoms.charges, index),
            'elements': DependArray.from_darray(atoms.elements, index),
        }

        return cls(**kwargs)

    def __len__(self):
        return len(self.charges)

    def __getitem__(self, index):
        return self.from_atoms(self, index)
