import os
import sys
from pathlib import Path

CURRENT_FILE = Path(os.path.abspath(__file__))
sys.path.append(str(CURRENT_FILE.parent.parent.parent))
from main.utils import *

from generators import Generator_XCAT, Generator_Invivo, Generator_Phantom


CLASS_MAP = {
    "xcat": Generator_XCAT,
    "invivo": Generator_Invivo,
    "phantom": Generator_Phantom
}


def main(params, verbose, **kwargs):

    if params.get("mode") is None:
        raise ValueError("Please provide a mode. Choose from 'xcat', 'invivo', 'phantom'.")
    if params["mode"] not in CLASS_MAP.keys():
        raise ValueError(f"Mode {params['mode']} not recognized. Choose from 'xcat', 'invivo', 'phantom'.")
    
    mode = params["mode"]
    generator = CLASS_MAP[mode](params, verbose=verbose)
    generator.generate_backgrounds()


if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Script to run a motion simulation.")
    parser.add_argument("-c", "--config", required=True, help="Parameters config file.")
    parser.add_argument("-v", "--verbose", default=1, type=int, help="Logging level (0: no logs, 1: info logs).")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    main(**config, **args.__dict__)
