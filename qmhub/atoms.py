import numpy as np

from .utils import DependArray


class Atoms(object):
    def __init__(self, positions=None, charges=None, indices=None, elements=None, _real_mask=None):
        self.positions = positions
        self.charges = charges
        self.indices = indices
        self.elements = elements
        self._real_mask = _real_mask

    @classmethod
    def new(cls, n_atoms):
        positions = DependArray(np.zeros((3, n_atoms)))
        charges = DependArray(np.zeros(n_atoms))
        indices = DependArray(np.zeros(n_atoms, dtype=int))
        elements = DependArray(np.zeros(n_atoms, dtype=str))

        _real_mask = DependArray(
            np.zeros(n_atoms, dtype=bool),
            name="_real_mask",
            func=(lambda x: x != -1),
            dependencies=[indices],
        )

        kwargs = {
            'positions': positions,
            'charges': charges,
            'indices': indices,
            'elements': elements,
            '_real_mask': _real_mask,
        }

        return cls(**kwargs)

    @classmethod
    def from_atoms(cls, atoms, index=None):
        kwargs = {
            'positions': atoms.positions[:, index],
            'charges': atoms.charges[index],
            'indices': atoms.indices[index],
            'elements': atoms.elements[index],
            '_real_mask': atoms._real_mask[index],
        }

        return cls(**kwargs)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for index in range(len(self)):
            yield self.from_atoms(self, index)

    def __getitem__(self, index):
        return self.from_atoms(self, index)

    def __setitem__(self, index, value):
        atoms = self.from_atoms(self, index)
        atoms.positions[:] = value.positions
        atoms.charges[:] = value.charges
        atoms.indices[:] = value.indices
        atoms.elements[:] = value.elements

    @property
    def real(self):
        return self[self._real_mask.view(np.ndarray)]

    @property
    def virtual(self):
        return self[~self._real_mask.view(np.ndarray)]
