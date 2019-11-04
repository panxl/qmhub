from pathlib import Path
import numpy as np

from ..utils.sys import get_nproc
from .templates.qchem import get_qm_template
from .qmbase import QMBase


class QChem(QMBase):

    QMTOOL = "Q-Chem"
    OUTPUT = "qchem.out"

    def gen_input(self):
        """Generate input file for QM software."""

        qm_elements = np.asarray(self.qm_elements, dtype=self.qm_elements.dtype)
        qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)
        mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)
        mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)

        with open(Path(self.cwd).joinpath("qchem.inp"), "w") as f:
            f.write(get_qm_template(self.keywords))
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            for i in range(len(qm_elements)):
                f.write(
                    "".join(
                        [
                            "%3d " % qm_elements[i],
                            "%22.14e" % qm_positions[0, i],
                            "%22.14e" % qm_positions[1, i],
                            "%22.14e" % qm_positions[2, i],
                            "\n",
                        ]
                    )
                )
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            if mm_charges is not None:
                for i in range(len(mm_charges)):
                    f.write(
                        "".join(
                            [
                                "%22.14e" % mm_positions[0, i],
                                "%22.14e" % mm_positions[1, i],
                                "%22.14e" % mm_positions[2, i],
                                " %22.14e" % mm_charges[i],
                                "\n",
                            ]
                        )
                    )
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = get_nproc()
        cmdline = "cd " + str(self.cwd) + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def _get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()
            output = qm_cache.array
        else:
            if output is None:
                output=self.OUTPUT
            output = Path(self.cwd).joinpath(output).read_text().split("\n")

        cc_energy = 0.0
        for line in output:
            if "Charge-charge energy" in line:
                cc_energy = line.split()[-2]

            if "Total energy" in line:
                scf_energy = line.split()[-1]
                break

        return float(scf_energy) - float(cc_energy)

    def _get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()

        if output is None:
            output = "efield.dat"

        return np.loadtxt(Path(self.cwd).joinpath("efield.dat"), skiprows=len(self.mm_charges), dtype=float).T

    def _get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential  at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            qm_cache.update_cache()

        if output is None:
            output = ("esp.dat", "efield.dat")

        mm_esp = np.zeros((4, len(self.mm_charges)))

        mm_esp[0] = np.loadtxt(Path(self.cwd).joinpath(output[0]), dtype=float)
        mm_esp[1:] = -np.loadtxt(Path(self.cwd).joinpath(output[1]), max_rows=len(self.mm_charges), dtype=float).T

        return mm_esp

    def _get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()
            output = qm_cache.array
        else:
            if output is None:
                output=self.OUTPUT
            output = Path(self.cwd).joinpath(output).read_text().split("\n")

        for i in range(len(output)):
            if "Ground-State Mulliken Net Atomic Charges" in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + j + 4]
                    mulliken_charges[j] = float(line.split()[2])
                break

        return mulliken_charges
