import os
from pathlib import Path

import numpy as np

from .templates.qchem import get_qm_template, default_options
from .qmbase import QMBase


class QChem(QMBase):

    OUTPUT = None
    default_options = default_options

    def gen_input(self):
        """Generate input file for QM software."""

        with open(Path(self.cwd).joinpath("qchem.inp"), "w") as f:
            f.write(get_qm_template(self.options))

            f.write("$molecule\n")
            f.write(f"{self.charge} {self.mult}\n")
            for e, x, y, z, in zip(
                self.qm_elements,
                self.qm_positions[0],
                self.qm_positions[1],
                self.qm_positions[2],
            ):
                f.write(f"{e:3} {x:21.14e} {y:21.14e} {z:21.14e}\n")
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            if self.mm_charges is not None:
                for x, y, z, c in zip(
                    self.mm_positions[0],
                    self.mm_positions[1],
                    self.mm_positions[2],
                    self.mm_charges,
                ):
                    f.write(f"{x:21.14e} {y:21.14e} {z:21.14e} {c:21.14e}\n")
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        os.environ["QCSCRATCH"] = str(self.cwd.resolve())

        cmdline = f"cd {self.cwd}; "
        cmdline += f"qchem -nt {self.nproc} qchem.inp qchem.out save > qchem_run.log"

        return cmdline

    def _get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or "save/99.0"

        try:
            energy = np.fromfile(Path(self.cwd).joinpath(output), dtype="f8", count=1, offset=8).item()
        except:
            raise
        else:
            os.remove(Path(self.cwd).joinpath(output))

        return energy

    def _get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or "save/131.0"

        try:
            gradient = np.fromfile(Path(self.cwd).joinpath(output), dtype="f8", count=(len(self.qm_elements)*3)).reshape(-1, 3).T
        except:
            raise
        else:
            os.remove(Path(self.cwd).joinpath(output))

        return gradient

    def _get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential  at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or ("save/5001.0", "save/5002.0")

        mm_esp = np.zeros((4, len(self.mm_charges)))

        try:
            mm_esp[0] = np.fromfile(Path(self.cwd).joinpath(output[0]), dtype="f8", count=len(self.mm_charges))
            mm_esp[1:] = -np.fromfile(Path(self.cwd).joinpath(output[1]), dtype="f8", count=(len(self.mm_charges)*3)).reshape(-1, 3).T
        except:
            raise
        else:
            os.remove(Path(self.cwd).joinpath(output[0]))
            os.remove(Path(self.cwd).joinpath(output[1]))

        return mm_esp

    def _get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or ("qchem.out")

        try:
            output = Path(self.cwd).joinpath(output).read_text().split("\n")
        except:
            output = Path(output).read_text().split("\n")

        for i in range(len(output)):
            if "Ground-State Mulliken Net Atomic Charges" in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + j + 4]
                    mulliken_charges[j] = float(line.split()[2])
                break

        return mulliken_charges
