import os
import sys
import numpy as np
from pathlib import Path

CURRENT_FILE = Path(os.path.abspath(__file__))
sys.path.append(str(CURRENT_FILE.parent.parent.parent))
from main.utils import *

from run_DENSE_sim import main_DENSE

def main(general_params, sim_params, verbose, **kwargs):
    
    t0 = time.time()
    if verbose:
        print("\nRunning DENSE simulation...\n")

    if general_params.get("background_input") == None:

        if general_params["subjects"]:
            subject_folders = [os.path.join(general_params["cardiac_motion_folder"], sub) for sub in general_params["subjects"]]
        else:
            subject_folders = sorted(glob.glob(os.path.join(general_params["cardiac_motion_folder"], "subject_*")))

        if verbose:
            print("Starting simulation(s) from {} motion case(s)..".format(len(subject_folders)))
        
        for i, subject_folder in enumerate(subject_folders):
            output_dim = 128
            if ("output_dim" in sim_params.keys()) and sim_params["output_dim"]:
                output_dim = int(np.random.choice(
                    sim_params["output_dim"]["sizes"], 
                    1, 
                    p=sim_params["output_dim"]["weights"]
                ))
            FOV = sim_params["output_dim"]["FOVs"][sim_params["output_dim"]["sizes"].index(output_dim)]
            Nt = 35
            if ("Nt" in sim_params.keys()) and sim_params["Nt"]:
                if type(sim_params["Nt"]) == int:
                    Nt = sim_params["Nt"]
                elif type(sim_params["Nt"]) == list and len(sim_params["Nt"]) == 2:
                    Nt = np.random.randint(sim_params["Nt"][0], sim_params["Nt"][1]+1)
            sim_params["Nt"] = Nt

            config_motion = os.path.join(subject_folder, "config.yaml")
            with open(config_motion, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)

            tt = config["tt"]
            general_params["subject_folder"] = os.path.basename(subject_folder)
            sim_params["N_im"] = output_dim
            sim_params["FOV"] = FOV
            sim_params["tt"] = tt

            main_DENSE(general_params=general_params, sim_params=sim_params, **kwargs)
            if verbose:
                print("Simulation {}/{} completed: from subject '{}'. Time elapsed: {}".format(
                    i+1, len(subject_folders), os.path.basename(subject_folder), nice_time(time.time()-t0)
                ))

    else:
        raise NotImplementedError("Background input not implemented yet.")

    if verbose:
        print("\nMotion simulation complete.")


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
