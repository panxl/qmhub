from string import Template


template = """\
&qmmm
${options}\
 maxcyc = 0,
 verbosity = 6,
 /
"""


default_options = {
    "qm_theory": "pm3",
    "qmmm_int": "1",
}


def get_qm_template(options):

    if options is None:
        options = default_options

    options = "".join([f" {key} = {value},\n" for key, value in options.items()])

    return Template(template).safe_substitute(options=options)
