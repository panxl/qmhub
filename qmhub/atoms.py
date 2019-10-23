import numpy as np

from .utils.darray import DependArray


class Atoms(object):
    def __init__(self, positions=None, charges=None, elements=None):
        self.positions = positions
        self.charges = charges
        self.elements = elements

    @classmethod
    def new(cls, n_atoms):
        positions = DependArray(np.zeros((3, n_atoms)))
        charges = DependArray(np.zeros(n_atoms))
        elements = DependArray(np.zeros(n_atoms, dtype=str))

        kwargs = {
            'positions': positions,
            'charges': charges,
            'elements': elements,
        }

        return cls(**kwargs)

    @classmethod
    def from_atoms(cls, atoms, index=None):
        kwargs = {
            'positions': atoms.positions[:, index],
            'charges': atoms.charges[index],
            'elements': atoms.elements[index],
        }

        return cls(**kwargs)

    def __len__(self):
        return len(self.charges)

    def __iter__(self):
        for index in range(len(self)):
            yield self.from_atoms(self, index)

    def __getitem__(self, index):
        return self.from_atoms(self, index)

    def __setitem__(self, index, value):
        atoms = self.from_atoms(self, index)
        atoms.positions[:] = value.positions
        atoms.charges[:] = value.charges
        atoms.elements[:] = value.elements
