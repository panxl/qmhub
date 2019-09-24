"""
QMHub
A universal QM/MM interface.
"""

from .simulation import Simulation
from .model import Model

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
        self.model = Model(self.system, switching_type, cutoff, swdist, pbc)
