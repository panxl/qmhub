"""Unit conversion and physical constants.
Internal units:
    Energy: kcal/mol
    Length: angstrom (A)
Atomic units:
    Energy: hartree (Eh)
    Length: bohr (a0)
"""

# Atomic units in internal units
E_AU = 6.275094737775374e+02
L_AU = 5.2917721067e-01
F_AU = E_AU / L_AU

# Physical constants in internal units
KE = E_AU * L_AU

# Unit conversion
EH_TO_EV = 2.721138602e+01
KCAL_TO_JOULE = 4.184