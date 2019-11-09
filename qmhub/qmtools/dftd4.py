from ctypes import c_int, c_double

import numpy as np

from dftd4.interface import DFTD4Library

from .qmbase import QMBase
from .templates.dftd4 import default_options
from ..units import CODATA08_BOHR_TO_A


class DFTD4(QMBase):

    OUTPUT = None
    default_options = default_options

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lib = DFTD4Library()

    def _get_qm_cache(self, *args, output=None):
        return self.gen_input().values()

    def gen_input(self):
        """Generate input file for QM software."""

        kwargs = {
            "natoms": len(self.qm_elements),
            "numbers":  np.asarray(self.qm_elements, dtype=c_int),
            "charge": self.charge,
            "positions": np.ascontiguousarray(self.qm_positions.T / CODATA08_BOHR_TO_A).view(c_double),
            "options": self.options,
            "output": "/dev/null",
        }

        return self._lib.D4Calculation(**kwargs)

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""
        return qm_cache[0]

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""
        return qm_cache[1].T

    def _get_mulliken_charges(self, qm_cache=None):
        """Get Mulliken charges from output of QM calculation."""
        return qm_cache[4]
