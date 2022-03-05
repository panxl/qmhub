import numpy as np
import torch

from .qmbase import QMBase
from ..units import CODATA08_BOHR_TO_A, CODATA08_HARTREE_TO_EV


class Torch(QMBase):

    default_options = {'model': None}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = torch.jit.load(self.options['model']).double()

    def _get_qm_cache(self, *args, output=None):
        return self.gen_input()

    def gen_input(self):
        """Generate input file for QM software."""

        if self.mm_charges is not None:
            args = (
                torch.from_numpy(np.ascontiguousarray(self.qm_positions.T)[None]),
                torch.from_numpy(self.qm_element_ids[:]),
                torch.from_numpy(self.mm_positions.T[None]),
                torch.from_numpy(self.mm_charges[None]),
                )
        else:
            args = (
                torch.from_numpy(np.ascontiguousarray(self.qm_positions.T)[None]),
                torch.from_numpy(self.qm_element_ids[:]),
                )

        return self._model(*args)

    def _get_qm_energy(self, qm_cache):
        """Get QM energy from output of QM calculation."""
        return qm_cache[0][0].detach().numpy() / CODATA08_HARTREE_TO_EV

    def _get_qm_energy_gradient(self, qm_cache):
        """Get QM energy gradient from output of QM calculation."""
        return qm_cache[1][0].detach().numpy().T / (CODATA08_HARTREE_TO_EV / CODATA08_BOHR_TO_A)

    def _get_mm_esp(self, qm_cache):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        mm_esp = np.zeros((4, len(self.mm_charges)))
        mm_esp[0] = qm_cache[2][0].detach().numpy() / CODATA08_HARTREE_TO_EV
        mm_esp[1:] = qm_cache[3][0].detach().numpy().T / (CODATA08_HARTREE_TO_EV / CODATA08_BOHR_TO_A)
        return mm_esp
