"""Unit conversion and physical constants.
Internal units:
    Energy: kcal/mol
    Length: angstrom (A)
Atomic units:
    Energy: hartree (Eh)
    Length: bohr (a0)
"""

# 2018 CODATA
AVOGADRO_CONSTANT = 6.02214076e+23
HARTREE_IN_JOULE = 4.3597447222071e-18
EV_IN_JOULE = 1.602176634e-19
BOHR_IN_METER = 5.29177210903e-11

# Unit ratios
KCAL_IN_JOULE = 4.184e+03
ANGSTROM_IN_METER = 1e-10

# Atomic units in internal units
HARTREE_IN_KCAL_PER_MOLE = HARTREE_IN_JOULE / KCAL_IN_JOULE * AVOGADRO_CONSTANT
BOHR_IN_ANGSTROM = BOHR_IN_METER / ANGSTROM_IN_METER
FORCE_AU_IN_IU = HARTREE_IN_KCAL_PER_MOLE / BOHR_IN_ANGSTROM

# Unit conversion
HARTREE_IN_EV = HARTREE_IN_JOULE / EV_IN_JOULE

# Physical constants in internal units
COULOMB_CONSTANT = HARTREE_IN_KCAL_PER_MOLE * BOHR_IN_ANGSTROM
