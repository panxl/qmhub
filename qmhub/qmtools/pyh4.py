from copy import copy

import numpy as np

from pyh4 import H4Library

from .qmbase import QMBase
from .templates.pyh4 import default_options
from ..units import CODATA08_BOHR_TO_A, CODATA08_HARTREE_TO_KCAL


class PyH4(QMBase):

    OUTPUT = None
    default_options = default_options

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lib = H4Library()

    def _get_qm_cache(self, *args, output=None):
        return self.gen_input()

    def gen_input(self):
        """Generate input file for QM software."""

        kwargs = {
            "natoms": len(self.qm_elements),
            "positions": self.qm_positions.T,
            "numbers": self.qm_elements,
            "parameters": self.options,
        }

        return self._lib.H4Calculation(**kwargs)

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""
        return qm_cache[0] / CODATA08_HARTREE_TO_KCAL

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""
        return qm_cache[1].T / (CODATA08_HARTREE_TO_KCAL / CODATA08_BOHR_TO_A)
