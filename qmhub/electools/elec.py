import math
import numpy as np

from ..utils.darray import DependArray
from .distance import *
from .elec_near import ElecNear

try:
    from .pme import Ewald
except ImportError:
    from .ewald import Ewald

from .ewald import Ewald as EwaldQMQM


class Elec(object):

    def __init__(
        self,
        qm_positions,
        positions,
        qm_charges,
        charges,
        qm_total_charge,
        cell_basis,
        switching_type=None,
        cutoff=None,
        swdist=None,
        pbc=False,
        ):

        self.rij = DependArray(
            name="rij",
            func=get_rij,
            dependencies=[qm_positions, positions],
        )
        self.dij = DependArray(
            name="dij",
            func=get_dij,
            dependencies=[self.rij],
        )
        self.dij_gradient = DependArray(
            name="dij_gradient",
            func=get_dij_gradient,
            dependencies=[self.rij, self.dij],
        )
        self.dij_inverse = DependArray(
            name="dij_inverse",
            func=get_dij_inverse,
            dependencies=[self.dij],
        )
        self.dij_inverse_gradient = DependArray(
            name="dij_inverse_gradient",
            func=get_dij_inverse_gradient,
            dependencies=[self.dij_inverse, self.dij_gradient],
        )
        self.dij_min = DependArray(
            name="dij_min",
            func=get_dij_min,
            dependencies=[self.dij_inverse],
        )
        self.dij_min_gradient = DependArray(
            name="dij_min_gradient",
            func=get_dij_min_gradient,
            dependencies=[self.dij_min, self.dij_inverse, self.dij_inverse_gradient],
        )
        self.coulomb_exclusion = DependArray(
            name="coulomb_exclusion",
            func=(lambda x: np.nonzero(x.min(axis=0) < .8)[0]),
            dependencies=[self.dij],
        )
        self.mm1_index = DependArray(
            name="mm1_index",
            func=(lambda x: np.nonzero(np.logical_and(x > 0., x < .8))[0]),
            dependencies=[self.dij_min],
        )
        self.mm2_index = DependArray(
            name="mm2_index",
            func=get_mm2_index,
            dependencies=[
                positions,
                self.mm1_index,
                self.dij,
            ]
        )
    
        if pbc:
            self.full = Ewald(
                qm_positions=qm_positions,
                positions=positions,
                charges=charges,
                cell_basis=cell_basis,
                exclusion=self.coulomb_exclusion,
                cutoff=cutoff,
            )
            self.qmqm = EwaldQMQM(
                qm_positions=qm_positions,
                positions=qm_positions,
                charges=qm_charges,
                cell_basis=cell_basis,
                exclusion=np.arange(len(qm_charges)),
            )
        else:
            import importlib
            NonPBC = importlib.import_module(".nonpbc", package='qmhub.electools').__getattribute__('NonPBC')

            self.full = NonPBC(
                rij=self.rij,
                charges=charges,
                cell_basis=cell_basis,
                dij_inverse=self.dij_inverse,
                dij_inverse_gradient=self.dij_inverse_gradient,
                exclusion=self.coulomb_exclusion,
            )
            self.qmqm = None

        self.near_field = ElecNear(
            rij=self.rij,
            dij=self.dij,
            charges=charges,
            switching_type=switching_type,
            cutoff=cutoff,
            swdist=swdist,
        )

        self.qm_residual_esp = DependArray(
            name="qm_residual_esp",
            func=Elec._get_qm_residual_esp,
            dependencies=[
                self.full.qm_total_esp,
                self.near_field.qm_scaled_esp,
           ],
        )
        self.scaled_mm_charges = DependArray(
            name="scaled_mm_charges",
            func=Elec._get_scaled_mm_charges,
            dependencies=[
                self.near_field.charges,
                self.near_field.scaling_factor,
            ],
        )
        self.projected_mm_charges = DependArray(
            name="projected_mm_charges",
            func=Elec._get_projected_mm_charges,
            dependencies=[
                self.near_field.weighted_qmmm_coulomb_tensor_inv,
                self.qm_residual_esp,
                self.near_field.scaling_factor,
            ],
        )
        self.embedding_mm_charges = DependArray(
            name="embedding_mm_charges",
            func=Elec._get_embedding_mm_charges,
            dependencies=[
                self.near_field.scaling_factor,
                self.near_field.weighted_qmmm_coulomb_tensor_inv,
                self.qm_residual_esp,
                self.near_field.charges,
            ],
        )
        self.embedding_mm_positions = DependArray(
            name="embedding_mm_positions",
            func=Elec._get_embedding_mm_positions,
            dependencies=[
                positions,
                self.near_field.near_field_mask,
            ],
        )

    @staticmethod
    def _get_qm_residual_esp(qm_total_esp, qm_scaled_esp):
        return qm_total_esp - qm_scaled_esp

    @staticmethod
    def _get_scaled_mm_charges(charges, scaling_factor):
        return charges * scaling_factor

    @staticmethod
    def _get_projected_mm_charges(wt_inv, qm_esp, w):
        return (wt_inv @ qm_esp[0]) * w

    @staticmethod
    def _get_embedding_mm_charges(w, wt_inv, qm_esp, charges):
        return (wt_inv @ qm_esp[0] + charges) * w

    @staticmethod
    def _get_embedding_mm_positions(positions, near_field_mask):
        return positions[:, near_field_mask]
