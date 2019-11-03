import numpy as np

from .electools import Elec, Result
from .engine import Engine
from .utils.darray import DependArray


class Model(object):
    def __init__(
        self,
        qm_positions,
        positions,
        qm_charges,
        charges,
        cell_basis,
        qm_total_charge,
        switching_type=None,
        cutoff=None,
        swdist=None,
        pbc=None,
        ):
        """
        Creat a Model object.
        """

        self.qm_positions = qm_positions
        self.positions = positions
        self.qm_charges = qm_charges
        self.charges = charges
        self.cell_basis = cell_basis
        self.qm_total_charge = qm_total_charge

        self.switching_type = switching_type

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            raise ValueError("cutoff is not set")

        if swdist is None:
            self.swdist = cutoff * .75
        else:
            self.swdist = swdist

        if pbc is not None:
            self.pbc = pbc
        elif np.any(self.cell_basis != 0.0):
            self.pbc = True
        else:
            self.pbc = False

        self.elec = Elec(
            self.qm_positions,
            self.positions,
            self.qm_charges,
            self.charges,
            self.qm_total_charge,
            self.cell_basis,
            switching_type=self.switching_type,
            cutoff=self.cutoff,
            swdist=self.swdist,
            pbc=self.pbc,
        )

    def get_result(
        self,
        name,
        qm_energy,
        qm_energy_gradient,
        mm_esp,
        ):

        result_obj = Result(
            qm_energy=qm_energy,
            qm_energy_gradient=qm_energy_gradient,
            mm_esp=mm_esp,
            qm_charges=self.qm_charges,
            mm_charges=self.elec.near_field.charges,
            near_field_mask=self.elec.near_field.near_field_mask,
            scaling_factor=self.elec.near_field.scaling_factor,
            scaling_factor_gradient=self.elec.near_field.scaling_factor_gradient,
            qmmm_coulomb_tensor=self.elec.near_field.qmmm_coulomb_tensor,
            qmmm_coulomb_tensor_gradient=self.elec.near_field.qmmm_coulomb_tensor_gradient,
            weighted_qmmm_coulomb_tensor=self.elec.near_field.weighted_qmmm_coulomb_tensor,
            weighted_qmmm_coulomb_tensor_inv=self.elec.near_field.weighted_qmmm_coulomb_tensor_inv,
            elec=self.elec,
        )

        setattr(self, name, result_obj)
