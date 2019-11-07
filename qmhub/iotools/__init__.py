import importlib


IO_TYPE_TO_CLASS_MAP = {
    'fifo': "IOFifo",
    'bin': "IOBin",
    'text': "IOText",
}


class IO(object):
    @classmethod
    def create(cls, io_type, *args, **kwargs):
        if io_type not in IO_TYPE_TO_CLASS_MAP:
            raise ValueError("Only 'text', 'bin', and 'fifo' modes are supported.")

        io_module = importlib.import_module("qmhub.iotools." + io_type)
        io_cls = io_module.__getattribute__(IO_TYPE_TO_CLASS_MAP[io_type])

        return io_cls(*args, **kwargs)
