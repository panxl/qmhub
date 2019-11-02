from string import Template


Elements = ["None", 'H', 'He',
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']


template = """\
&qmmm
${keywords}\
 /
"""


def get_qm_template(keywords_dict=None):

    keywords = {
        "qm_theory": "pm3",
        "qmcharge": "0",
        "spin": "1",
        "maxcyc": "0",
        "qmmm_int": "1",
        "verbosity": "6",
    }

    if keywords_dict is not None:
        keywords.update(keywords_dict)

    keywords = "".join([f" {key} = {value},\n" for key, value in keywords.items()])

    return Template(template).safe_substitute(keywords=keywords)
