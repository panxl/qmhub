import numpy as np


class Simulation(object):
    def __init__(self, protocol, n_steps):
        self.protocol = protocol
        self.n_steps = n_steps
        self.step = 0