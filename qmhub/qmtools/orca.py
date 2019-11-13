from pathlib import Path
import numpy as np

from ..units import ORCA_BOHR_TO_A
from .templates.orca import get_qm_template, default_options
from .qmbase import QMBase


class ORCA(QMBase):

    OUTPUT = "orca.out"
    default_options = default_options

    def gen_input(self):
        """Generate input file for QM software."""

        with open(Path(self.cwd).joinpath("orca.inp"), 'w') as f:
            f.write(get_qm_template(self.options, nproc=self.nproc, pointcharges="orca.pc"))

            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write(f"  Charge {self.charge}\n")
            f.write(f"  Mult {self.mult}\n")
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for s, x, y, z, in zip(
                self.qm_element_symbols,
                self.qm_positions[0],
                self.qm_positions[1],
                self.qm_positions[2],
            ):
                f.write(f"    {s:2} {x:21.14e} {y:21.14e} {z:21.14e}\n")

            f.write("  end\n")
            f.write("end\n")

        if self.mm_charges is not None:
            with open(Path(self.cwd).joinpath("orca.pc"), 'w') as f:
                f.write(f"{len(self.mm_charges)}\n")
                for c, x, y, z in zip(
                    self.mm_charges,
                    self.mm_positions[0],
                    self.mm_positions[1],
                    self.mm_positions[2],
                ):
                    f.write(f"{c:21.14e} {x:21.14e} {y:21.14e} {z:21.14e}\n")

            with open(Path(self.cwd).joinpath("orca.vpot.xyz"), 'w') as f:
                f.write(f"{len(self.mm_charges)}\n")
                for x, y, z in zip(
                    self.mm_positions[0] / ORCA_BOHR_TO_A,
                    self.mm_positions[1] / ORCA_BOHR_TO_A,
                    self.mm_positions[2] / ORCA_BOHR_TO_A,
                ):
                    f.write(f"{x:21.14e} {y:21.14e} {z:21.14e}\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = f"cd {self.cwd}; "
        cmdline += "orca orca.inp > orca.out; "
        cmdline += "orca_vpot orca.gbw orca.scfp orca.vpot.xyz orca.vpot.out >> orca.out"

        return cmdline

    def _get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        if qm_cache is not None:
            output = qm_cache
        else:
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = Path(self.cwd).joinpath(self.OUTPUT).read_text().split("\n")
            else:
                raise ValueError("Can not open output.")

        for line in output:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])

    def _get_qm_energy_gradient(self, qm_cache=None, output=None):
        """Get QM energy gradient from output of QM calculation."""

        if qm_cache is not None:
            qm_cache.update_cache()
        
        output = output or "orca.engrad"

        return np.loadtxt(Path(self.cwd).joinpath(output), skiprows=11, max_rows=len(self.qm_elements) * 3).reshape(len(self.qm_elements), 3).T

    def _get_mm_esp(self, qm_cache=None, output=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        if qm_cache is not None:
            qm_cache.update_cache()

        output = output or ("orca.vpot.out", "orca.pcgrad")

        mm_esp = np.zeros((4, len(self.mm_charges)))

        mm_esp[0] = np.loadtxt(Path(self.cwd).joinpath(output[0]), skiprows=1, max_rows=len(self.mm_charges), usecols=3)
        mm_esp[1:] = np.loadtxt(Path(self.cwd).joinpath(output[1]), skiprows=1, max_rows=len(self.mm_charges)).T / self.mm_charges

        return mm_esp

    def _get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if qm_cache is not None:
            output = qm_cache
        else:
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = Path(self.cwd).joinpath(self.OUTPUT).read_text().split("\n")
            else:
                raise ValueError("Can not open output.")

        for i in range(len(output)):
            if "MULLIKEN ATOMIC CHARGES" in output[i]:
                charges = []
                for line in output[(i + 2):(i + 2 + len(self.qm_elements))]:
                    charges.append(float(line.split()[3]))
                break

        return np.array(charges)
