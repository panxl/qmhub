import os
import subprocess as sp
import numpy as np

from .qmtools import choose_qmtool


class Engine(object):

    def __init__(
        self,
        qm_positions,
        qm_elements,
        mm_positions=None,
        mm_charges=None,
        charge=None,
        mult=None,
    ):
        """
        Creat a Engine object.
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

        self._engines = {}

    def add_engine(self, engine, name=None, **kwargs):
        if name is None:
            name = engine

        engine_obj = choose_qmtool(engine)(**kwargs)
        setattr(self, name, engine_obj)
        self._engines[name] = engine_obj

    def gen_input(self, calc_forces=True, read_guess=False):
        for engine in self._engines.values():
            engine.gen_input(
                np.asarray(self.qm_positions),
                np.asarray(self.qm_elements, dtype=self.qm_elements.dtype),
                np.asarray(self.mm_positions),
                np.asarray(self.mm_charges),
                self.charge,
                self.mult,
                calc_forces=calc_forces,
                read_guess=read_guess,
            )

    def run(self, read_guess=False):
        for engine in self._engines.values():
            engine.run(read_guess=read_guess)
