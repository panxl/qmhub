import numpy as np


class MMBase(object):

    MMTOOL = None

    @property
    def qm_energy(self):
        return self.qm_atoms.qm_energy

    @property
    def qm_force(self):
        return self.qm_atoms.force

    @property
    def mm_force(self):
        return self.mm_atoms.force

    @staticmethod
    def parse_output(qm):
        """Parse the output of QM calculation."""

        if qm.calc_forces:
            qm.parse_output()

    def apply_corrections(self, embed):
        """Correct the results."""

        if self.qm_energy != 0.0:
            self.calc_me(embed)
            self.corr_scaling(embed)

    @staticmethod
    def calc_me(embed):
        """Calculate forces and energy for mechanical embedding."""

        if embed.mm_atoms_near.charge_me is not None:

            mm_esp_me = embed.get_mm_esp_me()
            mm_efield_me = embed.get_mm_efield_me()
            mm_charge = embed.mm_atoms_near.charge_me

            energy = mm_charge[:, np.newaxis] * mm_esp_me

            force = -1 * mm_charge[:, np.newaxis, np.newaxis] * mm_efield_me

            embed.mm_atoms_near.force += force.sum(axis=1)
            embed.qm_atoms.force -= force.sum(axis=0)

            embed.qm_atoms.qm_energy += energy.sum()

        # Cancel QM-MM and QM-QM image interactions in MM package
        if embed.mm_atoms_far.charge_eeq is not None:

            energy = (embed.qmmm_esp_far.sum(axis=0) + 0.5 * embed.qmqm_esp_far.sum(axis=0)) * embed.qm_atoms.charge
            force_qmmm = -1 * embed.qmmm_efield_far * embed.qm_atoms.charge[np.newaxis, :, np.newaxis]
            force_qmqm = -0.5 * embed.qmqm_efield_far * embed.qm_atoms.charge[np.newaxis, :, np.newaxis]

            embed.qm_atoms.qm_energy -= energy.sum()

            embed.mm_atoms_far.force -= force_qmmm.sum(axis=1)
            embed.qm_atoms.force += force_qmmm.sum(axis=0)

            embed.qm_atoms.force -= force_qmqm.sum(axis=1)
            embed.qm_atoms.force += force_qmqm.sum(axis=0)

    @staticmethod
    def corr_scaling(embed):
        """Correct forces due to charge scaling."""

        if embed.qmSwitchingType is not None:

            mm_charge = embed.mm_atoms_near.charge
            mm_esp = embed.get_mm_esp()
            scale_deriv = embed.scale_deriv
            dij_min_j = embed.mm_atoms_near.dij_min_j

            energy = mm_charge * mm_esp

            force_corr = -1 * energy[:, np.newaxis] * scale_deriv

            embed.mm_atoms_near.force -= force_corr
            np.add.at(embed.qm_atoms.force, dij_min_j, force_corr)
