from pathlib import Path
import argparse
import configparser


from qmhub import QMMM


def main():
    parser = argparse.ArgumentParser(description='QMHub: A QM/MM interface.')
    parser.add_argument("config", help="QMHub config file")
    parser.add_argument("driver", help="Driver")
    parser.add_argument("file", help="Path of the exchange file")
    args = parser.parse_args()

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(args.config)

    qmmm = QMMM(args.driver)
    qmmm.setup_simulation()

    fin = Path(args.file)
    qmmm.load_system(fin)

    qmmm.build_model(
        switching_type=config.get('model', 'switching_function', fallback='lrec'),
        cutoff=config.getfloat('model', 'cutoff', fallback=10.),
        swdist=config.getfloat('model', 'swdist', fallback=None),
        pbc=config.getboolean('model', 'pbc', fallback=True),
    )

    for engine in config['engine'].keys():
        if engine in config:
            qmmm.add_engine(engine, keywords=config[engine], basedir=fin.parent)
        else:
            qmmm.add_engine(engine, keywords={}, basedir=fin.parent)

    qmmm.return_results(fin.with_suffix('.out'))


if __name__ == "__main__":
    main()
