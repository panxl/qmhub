from ctypes import c_int, c_double
from copy import copy

import numpy as np

from dftd4.interface import DFTD4Library
from dftd4.interface import P_MBD_APPROX_ATM, P_REFQ_EEQ
from ctypes import c_int, c_double

from .qmbase import QMBase
from ..units import CODATA08_BOHR_TO_A


class DFTD4(QMBase):

    OUTPUT = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lib = DFTD4Library()

        self._default_options = {
            'lmbd': P_MBD_APPROX_ATM,
            'refq': P_REFQ_EEQ,
            'wf': 6.0,
            'g_a': 3.0,
            'g_c': 2.0,
            'properties': True,
            'energy': True,
            'forces': True,
            'hessian': False,
            'print_level': 1,
            's6': 1.0000,  # B3LYP-D4-ATM parameters
            's8': 1.93077774,
            's9': 1.0,
            's10': 0.0,
            'a1': 0.40520781,
            'a2': 4.46255249,
            'alp': 16,
        }

        self.options = copy(self._default_options)

    def _get_qm_cache(self, *args, output=None):
        return self.gen_input().values()

    def gen_input(self):
        """Generate input file for QM software."""

        self.options.update(self.options)

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
