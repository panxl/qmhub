from .mdi import load_from_mdi, write_to_mdi
from .bin import load_from_bin, write_to_bin
from .file import load_from_file, write_to_file


__all__ = ["load_system", "return_results"]


def load_system(mode=None):
    if mode is None:
        mode = "file"

    if mode.lower() == "mdi":
        return load_from_mdi
    elif mode.lower() == "bin":
        return load_from_bin
    elif mode.lower() == "file":
        return load_from_file
    else:
        raise ValueError("Only 'file' (default), 'bin', and 'mdi' modes are supported.")


def return_results(mode=None):
    if mode is None:
        mode = "file"

    if mode.lower() == "mdi":
        return write_to_mdi
    elif mode.lower() == "bin":
        return write_to_bin
    elif mode.lower() == "file":
        return write_to_file
    else:
        raise ValueError("Only 'file' (default), 'bin', and 'mdi' modes are supported.")
