from dftd4.interface import P_MBD_APPROX_ATM, P_REFQ_EEQ


default_options = {
    'lmbd': P_MBD_APPROX_ATM,
    'refq': P_REFQ_EEQ,
    'wf': 6.0,
    'g_a': 3.0,
    'g_c': 2.0,
    'properties': True,
    'energy': True,
    'forces': True,
    'hessian': False,
    'print_level': 1,
    's6': 1.0000,  # B3LYP-D4-ATM parameters
    's8': 1.93077774,
    's9': 1.0,
    's10': 0.0,
    'a1': 0.40520781,
    'a2': 4.46255249,
    'alp': 16,
}
