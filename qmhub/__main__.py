from pathlib import Path
import argparse
import configparser
from IPython import embed

from qmhub import QMMM


def main():
    parser = argparse.ArgumentParser(description='QMHub: A QM/MM interface.')
    parser.add_argument("config", help="QMHub config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-b", "--binfile", help="Path of the binary exchange file")
    group.add_argument("-f", "--file", help="Path of the text exchange file")

    parser.add_argument("-d", "--driver", help="Driver")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(args.config)

    qmmm = QMMM(args.driver)

    protocol=config.get('simulation', 'protocol', fallback='md')
    nrespa=config.getint('simulation', 'nrespa', fallback=None)
    qmmm.setup_simulation(protocol, nrespa=nrespa)

    if args.binfile is not None:
        fin = Path(args.binfile)
        qmmm.load_system(fin, mode="binfile")
    elif args.file is not None:
        fin = Path(args.file)
        qmmm.load_system(fin, mode="file")

    qmmm.build_model(
        switching_type=config.get('model', 'switching_function', fallback='lrec'),
        cutoff=config.getfloat('model', 'cutoff', fallback=10.),
        swdist=config.getfloat('model', 'swdist', fallback=None),
        pbc=config.getboolean('model', 'pbc', fallback=True),
    )

    for name, engine in config['engine'].items():
        if engine is None:
            engine = name

        if engine in config:
            keywords = config[engine]
        else:
            keywords = {}

        qmmm.add_engine(
            engine,
            name=name,
            group_name="engine",
            basedir=fin.parent,
            keywords=keywords,
        )

    if 'engine2' in config:
        for name, engine in config['engine2'].items():
            if engine is None:
                engine = name

            if engine in config:
                keywords = config[engine]
            else:
                keywords = {}

            qmmm.add_engine(
                engine,
                name=name,
                group_name="engine2",
                basedir=fin.parent,
                keywords=keywords,
            )

    if args.binfile is not None:
        qmmm.return_results(fin.with_suffix('.out'), mode="binfile")
    elif args.file is not None:
        qmmm.return_results(fin.with_suffix('.out'), mode="file")

    if args.interactive:
        embed()


if __name__ == "__main__":
    main()
