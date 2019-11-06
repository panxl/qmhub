from string import Template


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
