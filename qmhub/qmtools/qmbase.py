import os
import numpy as np

from ..utils import DependArray, run_cmdline


class QMBase(object):

    QMTOOL = None

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

        raise NotImplementedError()

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        raise NotImplementedError()

    def get_qm_energy(self, qm_cache=None, output=None):
        """Get QM energy from output of QM calculation."""

        raise NotImplementedError()

    def get_qm_forces(self, qm_cache=None, output=None):
        """Get QM forces from output of QM calculation."""

        raise NotImplementedError()

    def get_mm_efield(self, qm_cache=None, output=None):
        """Get QM forces from output of QM calculation."""

        raise NotImplementedError()

    def get_mm_esp(self, qm_cache=None, output=None):
        """Get ESP at MM atoms in the near field from QM density."""

        raise NotImplementedError()

    def get_mulliken_charges(self, qm_cache=None, output=None):
        """Get Mulliken charges from output of QM calculation."""

        raise NotImplementedError()
