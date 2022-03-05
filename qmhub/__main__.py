from pathlib import Path
import argparse
import configparser
from IPython import embed

from qmhub import QMMM


def main():
    parser = argparse.ArgumentParser(description='QMHub: A QM/MM interface.')
    parser.add_argument("config", help="QMHub config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", help="Path to text exchange file")
    group.add_argument("-b", "--bin", help="Path to binary exchange file")
    group.add_argument("-f", "--fifo", help="Path to FIFO exchange file")

    parser.add_argument("-d", "--driver", help="Driver")
    parser.add_argument("-c", "--cwd", help="Working directory for engine calculations")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(args.config)

    if args.fifo is not None:
        mode = "fifo"
        input = Path(args.fifo)
    elif args.bin is not None:
        mode = "bin"
        input = Path(args.bin)
    elif args.text is not None:
        mode = "text"
        input = Path(args.text)

    qmmm = QMMM(mode, args.driver, args.cwd)

    protocol = config.get('simulation', 'protocol', fallback='md')
    nrespa = config.getint('simulation', 'nrespa', fallback=None)
    scaling_factor = config.getfloat('simulation', 'scaling_factor', fallback=None)
    save_input = config.getboolean('simulation', 'save_input', fallback=False)
    qmmm.setup_simulation(protocol, nrespa=nrespa, scaling_factor=scaling_factor)

    qmmm.load_system(input, save_input=save_input)

    qmmm.build_model(
        switching_type=config.get('model', 'switching_function', fallback='lrec'),
        cutoff=config.getfloat('model', 'cutoff', fallback=10.),
        swdist=config.getfloat('model', 'swdist', fallback=None),
        pbc=config.getboolean('model', 'pbc', fallback=True),
    )

    for name, engine in config['engine'].items():
        engine = engine or name

        if name in config:
            options = config[name]
        else:
            options = {}

        qmmm.add_engine(
            engine,
            name=name,
            group_name="engine",
            options=options,
        )

    if 'engine2' in config:
        for name, engine in config['engine2'].items():
            engine = engine or name

            if name in config:
                options = config[name]
            else:
                options = {}

            qmmm.add_engine(
                engine,
                name=name,
                group_name="engine2",
                options=options,
            )

    qmmm.return_results()

    if args.interactive:
        embed()


if __name__ == "__main__":
    main()
