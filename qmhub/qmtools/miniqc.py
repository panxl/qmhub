import os
from pathlib import Path

import numpy as np

from qchem.miniqc import run_job
from qchem.ctx import KeyType, QCStorage

from .templates.miniqc import get_qm_template, default_options
from .qmbase import QMBase


class MiniQC(QMBase):

    OUTPUT = None
    default_options = default_options

    def _get_qm_cache(self, *args, output=None):
        input_str = self.gen_input()
        store = QCStorage()
        run_job(input_str, str(self.cwd.resolve()), store, file=False)

        return [store]

    def gen_input(self):
        """Generate input file for QM software."""

        input_str = get_qm_template(self.options)

        input_str += "$molecule\n"
        input_str += f"{self.charge} {self.mult}\n"
        for e, x, y, z, in zip(
            self.qm_elements,
            self.qm_positions[0],
            self.qm_positions[1],
            self.qm_positions[2],
        ):
            input_str += f"{e:3} {x:21.14e} {y:21.14e} {z:21.14e}\n"
        input_str += "$end\n\n"

        input_str += "$external_charges\n"
        if self.mm_charges is not None:
            for x, y, z, c in zip(
                self.mm_positions[0],
                self.mm_positions[1],
                self.mm_positions[2],
                self.mm_charges,
            ):
                input_str += f"{x:21.14e} {y:21.14e} {z:21.14e} {c:21.14e}\n"
        input_str += "$end\n"

        return input_str


    def _get_qm_energy(self, qm_cache):
        """Get QM energy from output of QM calculation."""
        return qm_cache[0].get_data(KeyType.TOTAL_ENERGY)


    def _get_qm_energy_gradient(self, qm_cache):
        """Get QM energy gradient from output of QM calculation."""
        return qm_cache[0].get_data(KeyType.TOTAL_ENERGY_GRADIENT)

    def _get_mm_esp(self, qm_cache):
        """Get electrostatic potential  at MM atoms in the near field from QM density."""

        mm_esp = np.zeros((4, len(self.mm_charges)))

        mm_esp[0] = qm_cache[0].get_data(KeyType.ESP_ELECTRON_DENSITY) + qm_cache[0].get_data(KeyType.ESP_NUCLEI)
        mm_esp[1:] = -(qm_cache[0].get_data(KeyType.EFIELD_ELECTRON_DENSITY) + qm_cache[0].get_data(KeyType.EFIELD_NUCLEI))

        return mm_esp
