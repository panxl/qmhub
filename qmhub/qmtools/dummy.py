import numpy as np

from .qmbase import QMBase


class Dummy(QMBase):

    OUTPUT = None
    default_options = None

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""
        return 0.

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""
        return np.zeros_like(self.qm_positions)
