import shutil
from pathlib import Path
import numpy as np

from ..units import BOHR_IN_ANGSTROM
from ..utils import get_nproc, run_cmdline
from .templates.orca import get_qm_template
from .qmbase import QMBase


class ORCA(QMBase):

    QMTOOL = 'ORCA'

    def gen_input(self):
        """Generate input file for QM software."""

        nproc = get_nproc()

        qm_elements = np.asarray(self.qm_elements, dtype=self.qm_elements.dtype)
        qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)
        with open(Path(self.basedir).joinpath("orca.inp"), 'w') as f:
            f.write(get_qm_template(self.keywords, nproc=nproc, pointcharges="orca.pc"))

            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write("  Charge %d\n" % self.charge)
            f.write("  Mult %d\n" % self.mult)
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for i in range(len(qm_elements)):
                f.write(" ".join(["%6s" % qm_elements[i],
                                  "%22.14e" % qm_positions[0, i],
                                  "%22.14e" % qm_positions[1, i],
                                  "%22.14e" % qm_positions[2, i], "\n"]))
            f.write("  end\n")
            f.write("end\n")

        mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)
        mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)
        with open(Path(self.basedir).joinpath("orca.pc"), 'w') as f:
            f.write("%d\n" % len(mm_charges))
            for i in range(len(mm_charges)):
                f.write("".join(["%22.14e " % mm_charges[i],
                                 "%22.14e" % mm_positions[0, i],
                                 "%22.14e" % mm_positions[1, i],
                                 "%22.14e" % mm_positions[2, i], "\n"]))

        with open(Path(self.basedir).joinpath("orca.vpot.xyz"), 'w') as f:
            f.write("%d\n" % len(mm_charges))
            for i in range(len(mm_charges)):
                f.write("".join(["%22.14e" % (mm_positions[0, i] / BOHR_IN_ANGSTROM),
                                 "%22.14e" % (mm_positions[1, i] / BOHR_IN_ANGSTROM),
                                 "%22.14e" % (mm_positions[2, i] / BOHR_IN_ANGSTROM), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.basedir + "; "
        cmdline += shutil.which("orca") + " orca.inp > orca.out; "
        cmdline += shutil.which("orca_vpot") + " orca.gbw orca.scfp orca.vpot.xyz orca.vpot.out >> orca.out"

        return cmdline

    def get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("orca.out")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        for line in output:
            line = line.strip().expandtabs()

            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])

    def get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("orca.engrad")
        else:
            output = Path(output)

        output = output.read_text().split("\n")
        start = 11
        stop = start + len(self.qm_elements) * 3
        return np.loadtxt(output[start:stop]).reshape((len(self.qm_elements), 3)).T

    def get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("orca.vpot.out")
        else:
            output = Path(output)
    
        output = output.read_text().split("\n")

        return np.loadtxt(output[1:(len(self.mm_charges) + 1)], usecols=3)

    def get_mm_esp_gradient(self, qm_cache=None, output=None):
        """Get electrostatic potential gradient at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("orca.pcgrad")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        return np.loadtxt(output[1:(len(self.mm_charges) + 1)]).T / self.mm_charges

    def get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("orca.out")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        for i in range(len(output)):
            if "MULLIKEN ATOMIC CHARGES" in output[i]:
                charges = []
                for line in output[(i + 2):(i + 2 + len(self.qm_elements))]:
                    charges.append(float(line.split()[3]))
                break

        return np.array(charges)
