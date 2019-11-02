from pathlib import Path
import numpy as np

from ..units import AMBER_HARTREE_TO_KCAL, AMBER_BOHR_TO_A
from ..utils.sys import get_nproc

from .templates.sqm import get_qm_template, Elements
from .qmbase import QMBase


class SQM(QMBase):

    QMTOOL = 'SQM'

    def gen_input(self):
        """Generate input file for QM software."""

        qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)
        qm_elements = np.asarray(self.qm_elements, dtype=self.qm_elements.dtype)
        mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)
        mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)

        qm_element_num = []
        for element in qm_elements:
            qm_element_num.append(Elements.index(element))

        if not "qmcharge" in self.keywords:
            self.keywords["qmcharge"] = str(self.charge)
        
        if not "spin" in self.keywords:
            self.keywords["spin"] = str(self.mult)

        with open(Path(self.basedir).joinpath("sqm.inp"), 'w') as f:
            f.write(get_qm_template(self.keywords))

            for i in range(len(qm_elements)):
                f.write("".join(["%4d" % qm_element_num[i],
                                 "%4s " % qm_elements[i],
                                 "%22.14e" % qm_positions[0, i],
                                 "%22.14e" % qm_positions[1, i],
                                 "%22.14e" % qm_positions[2, i], "\n"]))

            if mm_charges is not None:
                f.write("#EXCHARGES\n")
                for i in range(len(mm_charges)):
                    f.write("".join(["   1   H ",
                                        "%22.14e" % mm_positions[0, i],
                                        "%22.14e" % mm_positions[1, i],
                                        "%22.14e" % mm_positions[2, i],
                                        " %22.14e" % mm_charges[i], "\n"]))
                f.write("#END" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + str(self.basedir) + "; "
        cmdline += "sqm -O -i sqm.inp -o sqm.out"

        return cmdline

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        output = Path(self.basedir).joinpath("sqm.out").read_text().split("\n")

        for line in output:
            line = line.strip().expandtabs()

            if "QMMM: SCF Energy" in line:
                return float(line.split()[4]) / AMBER_HARTREE_TO_KCAL

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        output = Path(self.basedir).joinpath("sqm.out").read_text().split("\n")

        for i in range(len(output)):
            if "Forces on QM atoms from SCF calculation" in output[i]:
                gradient = np.empty((3, len(self.qm_elements)), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + 1 + j]
                    gradient[0, j] = float(line[18:38])
                    gradient[1, j] = float(line[38:58])
                    gradient[2, j] = float(line[58:78])
                return gradient / (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)

    def _get_mm_esp(self, qm_cache=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        output = Path(self.basedir).joinpath("sqm.out").read_text().split("\n")

        mm_esp = np.zeros((4, len(self.mm_charges)))

        for i in range(len(output)):
            if "Electrostatic Potential on MM atoms from QM Atoms" in output[i]:
                for j in range(len(self.mm_charges)):
                    line = output[i + 1 + j]
                    mm_esp[0, j] = float(line.split()[-1])
                break

        for i in range(len(output)):
            if output[i].strip() == "QMMM: Forces on MM atoms from SCF calculation":
                for j in range(len(self.mm_charges)):
                    line = output[i + 1 + j]
                    mm_esp[1:, j] = [float(n) / self.mm_charges[j] for n in line.split()[-3:]]
                break

        mm_esp[0] /= AMBER_HARTREE_TO_KCAL
        mm_esp[1:] /= (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
        return mm_esp

    def _get_mulliken_charges(self, qm_cache=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        output = Path(self.basedir).joinpath("sqm.out").read_text().split("\n")

        for i in range(len(output)):
            if "Atomic Charges" in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + 2 + j]
                    mulliken_charges[j] = float(line.split()[-1])
                return mulliken_charges
