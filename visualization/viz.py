import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main(params, verbose, **kwargs):

    if params.get("mode") is None:
        raise ValueError("Please provide a mode.")
    if params.get("data") is None:
        raise ValueError("Please provide a data file.")

    if params["mode"] == "2D":
        data = np.load(params["data"])
        if params.get("vmin") is not None and params.get("vmax") is not None:
            plt.imshow(data, vmin=params["vmin"], vmax=params["vmax"])
        else:
            plt.imshow(data)
        plt.show()
    elif params["mode"] == "2D+time":
        data = np.load(params["data"])
        fig, ax = plt.subplots(1,1,squeeze=False)
        if params.get("vmin") is not None and params.get("vmax") is not None:
            I = ax[0,0].imshow(data[0], vmin=params["vmin"], vmax=params["vmax"])
        else:
            I = ax[0,0].imshow(data[0])
        def animate(i):
            I.set_array(data[i])
            return I,
        ani = animation.FuncAnimation(fig, animate, interval=50, frames=range(0,data.shape[0]), repeat=True)
        plt.show()
    else:
        raise ValueError("Mode {} not recognized. Choose from '2D', '2D+time'.".format(params["mode"]))


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