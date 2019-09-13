from collections import namedtuple

import numpy as np

from .utils import DependArray


class Atoms(object):
    def __init__(self, n_atoms, prebind=None):
        self.n_atoms = n_atoms

        if prebind is None:
            self.positions = DependArray(np.zeros((3, n_atoms)))
            self.charges = DependArray(np.zeros(n_atoms))
            self.indices = DependArray(np.zeros(n_atoms, dtype=int))
            self.elements = DependArray(np.zeros(n_atoms, dtype=str))
        else:
            self.positions = prebind[0]
            self.charges = prebind[1]
            self.indices = prebind[2]
            self.elements = prebind[3]

    @classmethod
    def from_atoms(cls, atoms, index=None):
        n_atoms = len(atoms.indices[index])

        prebind = (
            atoms.positions[:, index],
            atoms.charges[index],
            atoms.indices[index],
            atoms.elements[index])
        
        return cls(n_atoms, prebind=prebind)

    def __len__(self):
        return self.n_atoms

    def __iter__(self):
        for index in range(len(self)):
            yield self.from_atoms(self, index)

    def __getitem__(self, index):
        return self.from_atoms(self, index)

    def __setitem__(self, index, value):
        atoms = self.from_atoms(self, index)
        atoms.positions = value.positions
        atoms.charges = value.charges
        atoms.indices = value.indices
        atoms.elements = value.elements

    @property
    def _n_real_atoms(self):
        return np.count_nonzero(self.indices != -1)

    @property
    def real(self):
        return self[:self._n_real_atoms]

    @property
    def virtual(self):
        return self[self._n_real_atoms:]