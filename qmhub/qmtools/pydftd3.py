import numpy as np

from pydftd3 import DFTD3Library

from .qmbase import QMBase
from .templates.pydftd3 import default_options
from ..units import CODATA08_BOHR_TO_A


class PyDFTD3(QMBase):

    OUTPUT = None
    default_options = default_options

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lib = DFTD3Library()

    def _get_qm_cache(self, *args, output=None):
        return self.gen_input()

    def gen_input(self):
        """Generate input file for QM software."""

        kwargs = {
            "natoms": len(self.qm_elements),
            "positions": self.qm_positions / CODATA08_BOHR_TO_A,
            "numbers": self.qm_elements,
            "parameters": self.options,
        }

        return self._lib.DFTD3Calculation(**kwargs)

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""
        return qm_cache[0]

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""
        return qm_cache[1]
