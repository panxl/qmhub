"""
QMHub
A universal QM/MM interface.
"""

from .simulation import Simulation
from .model import Model
from .engine import Engine


class QMMM(object):
    def __init__(self, driver=None):
        self.driver = driver.lower()

    def setup_simulation(self, protocol="md", n_steps=None):
        self.simulation = Simulation(protocol, n_steps)

    def load_system(self, input):
        if self.driver == "sander":
            from .mmtools.sander import load_from_file

            self.system = load_from_file(input, simulation=self.simulation)

    def build_model(self, switching_type=None, cutoff=None, swdist=None, pbc=None):
        self.model = Model(
            self.system.qm.atoms.positions,
            self.system.atoms.positions,
            self.system.atoms.charges,
            self.system.cell_basis,
            switching_type=switching_type,
            cutoff=cutoff,
            swdist=swdist,
            pbc=pbc
        )

    def add_engine(self, engine, name=None, basedir=None, keywords=None):
        try:
            self.engine.add_engine(engine, name=name, basedir=basedir, keywords=keywords)
        except AttributeError:
            self.engine = Engine(
                self.system.qm.atoms.positions,
                self.system.qm.atoms.elements,
                self.model.elec.embedding_mm_positions,
                self.model.elec.embedding_mm_charges,
                charge=self.system.qm_charge,
                mult=self.system.qm_mult,
            )

            self.engine.add_engine(engine, name=name, basedir=basedir, keywords=keywords)
