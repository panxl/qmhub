import os
from pathlib import Path
import numpy as np

from ..utils import DependArray, get_nproc, run_cmdline
from .templates.qchem import get_qm_template


class QChem(object):

    QMTOOL = "Q-Chem"

    def __init__(
        self,
        qm_positions,
        qm_elements,
        mm_positions,
        mm_charges,
        charge=None,
        mult=None,
        basedir=None,
        keywords=None,
        ):
        """
        Creat a QM object.
        """

        self.qm_positions = qm_positions
        self.qm_elements = qm_elements
        self.mm_positions = mm_positions
        self.mm_charges = mm_charges

        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for QM calculation.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1

        if basedir is not None:
            self.basedir = basedir
        else:
            self.basedir = os.getcwd()

        if keywords is not None:
            self.keywords = keywords
        else:
            self.keywords = {}

        self._qm_cache = DependArray(
            name="qm_updated",
            func=self._get_qm_cache,
            dependencies=[
                self.qm_positions,
                self.qm_elements,
                self.mm_positions,
                self.mm_charges,
            ]
        )
        self.qm_energy = DependArray(
            name="qm_energy",
            func=self.get_qm_energy,
            dependencies=[self._qm_cache],
        )
        self.qm_forces = DependArray(
            name="qm_forces ",
            func=self.get_qm_forces,
            dependencies=[self._qm_cache],
        )
        self.mm_efield= DependArray(
            name="qm_efield",
            func=self.get_mm_efield,
            dependencies=[self._qm_cache],
        )
        self.mm_esp = DependArray(
            name="mm_esp",
            func=self.get_mm_esp,
            dependencies=[self._qm_cache],
        )
        self.mulliken_charges = DependArray(
            name="mulliken_charges",
            func=self.get_mulliken_charges,
            dependencies=[self._qm_cache],
        )

    def _get_qm_cache(self, *args):
        self.gen_input()
        run_cmdline(self.gen_cmdline())

        return True

    def gen_input(self):
        """Generate input file for QM software."""

        with open(Path(self.basedir).joinpath("qchem.inp"), "w") as f:
            f.write(get_qm_template(self.keywords))
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            qm_elements = np.asarray(self.qm_elements, dtype=self.qm_elements.dtype)
            qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)

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
            if self.mm_charges is not None:
                mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)
                mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)
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
        cmdline = "cd " + self.basedir + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("qchem.out")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        for line in output:
            line = line.strip().expandtabs()

            cc_energy = 0.0
            if "Charge-charge energy" in line:
                cc_energy = line.split()[-2]

            if "Total energy" in line:
                scf_energy = line.split()[-1]
                break

        return float(scf_energy) - float(cc_energy)

    def get_qm_forces(self, qm_cache=None, output=None):
        """Get QM forces from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("efield.dat")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        return np.loadtxt(output[len(self.mm_charges):], dtype=float)

    def get_mm_efield(self, qm_cache=None, output=None):
        """Get QM forces from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("efield.dat")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        return np.loadtxt(output[:len(self.mm_charges)], dtype=float)

    def get_mm_esp(self, qm_cache=None, output=None):
        """Get ESP at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("esp.dat")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        return np.loadtxt(output)

    def get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            assert np.asscalar(qm_cache) == True

        if output is None:
            output = Path(self.basedir).joinpath("qchem.out")
        else:
            output = Path(output)

        output = output.read_text().split("\n")

        charge_string = "Ground-State Mulliken Net Atomic Charges"

        for i in range(len(output)):
            if charge_string in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + 4 + j]
                    mulliken_charges[j] = float(line.split()[2])
                break

        return mulliken_charges
