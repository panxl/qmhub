from string import Template


template = """\
&qmmm
${options}\
 /
"""


def get_qm_template(options_dict=None):

    options = {
        "qm_theory": "pm3",
        "qmcharge": "0",
        "spin": "1",
        "maxcyc": "0",
        "qmmm_int": "1",
        "verbosity": "6",
    }

    if options_dict is not None:
        options.update(options_dict)

    options = "".join([f" {key} = {value},\n" for key, value in options.items()])

    return Template(template).safe_substitute(options=options)
