from pathlib import Path
import numpy as np

from ..utils.sys import get_nproc
from .templates.qchem import get_qm_template
from .qmbase import QMBase


class QChem(QMBase):

    QMTOOL = "Q-Chem"

    def gen_input(self):
        """Generate input file for QM software."""

        qm_elements = np.asarray(self.qm_elements, dtype=self.qm_elements.dtype)
        qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)
        mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)
        mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)

        with open(Path(self.basedir).joinpath("qchem.inp"), "w") as f:
            f.write(get_qm_template(self.keywords))
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            for i in range(len(qm_elements)):
                f.write(
                    "".join(
                        [
                            "%3s" % qm_elements[i],
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
        cmdline = "cd " + str(self.basedir) + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        output = Path(self.basedir).joinpath("qchem.out").read_text().split("\n")

        for line in output:
            line = line.strip().expandtabs()

            cc_energy = 0.0
            if "Charge-charge energy" in line:
                cc_energy = line.split()[-2]

            if "Total energy" in line:
                scf_energy = line.split()[-1]
                break

        return float(scf_energy) - float(cc_energy)

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        return np.loadtxt(Path(self.basedir).joinpath("efield.dat"), skiprows=len(self.mm_charges), dtype=float).T

    def _get_mm_esp(self, qm_cache=None):
        """Get electrostatic potential  at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        mm_esp = np.zeros((4, len(self.mm_charges)))

        mm_esp[0] = np.loadtxt(Path(self.basedir).joinpath("esp.dat"), dtype=float)
        mm_esp[1:] = -np.loadtxt(Path(self.basedir).joinpath("efield.dat"), max_rows=len(self.mm_charges), dtype=float).T

        return mm_esp

    def _get_mulliken_charges(self, qm_cache=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        output = Path(self.basedir).joinpath("qchem.out").read_text().split("\n")

        charge_string = "Ground-State Mulliken Net Atomic Charges"

        for i in range(len(output)):
            if charge_string in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + 4 + j]
                    mulliken_charges[j] = float(line.split()[2])
                break

        return mulliken_charges
