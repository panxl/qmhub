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

    def add_engine(self, engine, name=None):
        if name is None:
            name = engine

        engine_obj = choose_qmtool(engine)(
            qm_positions = self.qm_positions,
            qm_elements = self.qm_elements,
            mm_positions = self.mm_positions,
            mm_charges = self.mm_charges,
            charge = self.charge,
            mult = self.mult,
        )

        setattr(self, name, engine_obj)
        self._engines[name] = engine_obj

    def gen_input(self, name=None, basedir=None, calc_forces=True, read_guess=False, **kwargs):
        if name is not None:
            engines = {name: self._engines[name]}
        else:
            engines = self._engines

        for engine in engines.values():
            engine.gen_input(
                basedir=basedir,
                calc_forces=calc_forces,
                read_guess=read_guess,
                **kwargs,
            )

    def run(self, name=None, basedir=None, read_guess=False):
        if name is not None:
            engines = {name: self._engines[name]}
        else:
            engines = self._engines

        for engine in engines.values():
            engine.run(basedir=basedir, read_guess=read_guess)

    def parse_output(self, name=None, basedir=None, calc_forces=True):
        if name is not None:
            engines = {name: self._engines[name]}
        else:
            engines = self._engines

        for engine in engines.values():
            engine.parse_output(basedir=basedir, calc_forces=calc_forces)
