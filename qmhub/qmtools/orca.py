from pathlib import Path
import numpy as np

from ..units import ORCA_BOHR_TO_A
from ..utils.sys import get_nproc
from .templates.orca import get_qm_template
from .qmbase import QMBase


class ORCA(QMBase):

    OUTPUT = "orca.out"

    def gen_input(self):
        """Generate input file for QM software."""

        qm_element_symbols = np.asarray(self.qm_element_symbols, dtype=self.qm_element_symbols.dtype)
        qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)
        mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)
        mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)

        nproc = get_nproc()

        with open(Path(self.cwd).joinpath("orca.inp"), 'w') as f:
            f.write(get_qm_template(self.keywords, nproc=nproc, pointcharges="orca.pc"))

            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write("  Charge %d\n" % self.charge)
            f.write("  Mult %d\n" % self.mult)
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for i in range(len(qm_element_symbols)):
                f.write(" ".join(["%6s" % qm_element_symbols[i],
                                  "%22.14e" % qm_positions[0, i],
                                  "%22.14e" % qm_positions[1, i],
                                  "%22.14e" % qm_positions[2, i], "\n"]))
            f.write("  end\n")
            f.write("end\n")

        with open(Path(self.cwd).joinpath("orca.pc"), 'w') as f:
            f.write("%d\n" % len(mm_charges))
            for i in range(len(mm_charges)):
                f.write("".join(["%22.14e " % mm_charges[i],
                                 "%22.14e" % mm_positions[0, i],
                                 "%22.14e" % mm_positions[1, i],
                                 "%22.14e" % mm_positions[2, i], "\n"]))

        with open(Path(self.cwd).joinpath("orca.vpot.xyz"), 'w') as f:
            f.write("%d\n" % len(mm_charges))
            for i in range(len(mm_charges)):
                f.write("".join(["%22.14e" % (mm_positions[0, i] / ORCA_BOHR_TO_A),
                                 "%22.14e" % (mm_positions[1, i] / ORCA_BOHR_TO_A),
                                 "%22.14e" % (mm_positions[2, i] / ORCA_BOHR_TO_A), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + str(self.cwd) + "; "
        cmdline += "orca orca.inp > orca.out; "
        cmdline += "orca_vpot orca.gbw orca.scfp orca.vpot.xyz orca.vpot.out >> orca.out"

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

        for line in output:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])

    def _get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()
        
        if output is None:
            output = "orca.engrad"

        return np.loadtxt(Path(self.cwd).joinpath(output), skiprows=11, max_rows=len(self.qm_elements) * 3).reshape(len(self.qm_elements), 3).T

    def _get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            qm_cache.update_cache()

        if output is None:
            output = ("orca.vpot.out", "orca.pcgrad")

        mm_esp = np.zeros((4, len(self.mm_charges)))

        mm_esp[0] = np.loadtxt(Path(self.cwd).joinpath(output[0]), skiprows=1, max_rows=len(self.mm_charges), usecols=3)
        mm_esp[1:] = np.loadtxt(Path(self.cwd).joinpath(output[1]), skiprows=1, max_rows=len(self.mm_charges)).T / self.mm_charges

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
            if "MULLIKEN ATOMIC CHARGES" in output[i]:
                charges = []
                for line in output[(i + 2):(i + 2 + len(self.qm_elements))]:
                    charges.append(float(line.split()[3]))
                break

        return np.array(charges)
