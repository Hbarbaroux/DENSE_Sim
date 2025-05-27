import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main(params, verbose, **kwargs):

    if params.get("mode") is None:
        raise ValueError("Please provide a mode.")
    if params.get("data") is None:
        raise ValueError("Please provide a data file.")
    
    if params.get("data_format") not in ["numpy", "nifti"]:
        raise ValueError("Data format {} not recognized. Choose from 'numpy', 'nifti'.".format(params["data_format"]))
    if params["data_format"] == "nifti":
        data = nib.load(params["data"]).get_fdata()
    else:
        data = np.load(params["data"])

    if params.get("cmap") is None: params["cmap"] = "gray"

    if params["mode"] == "2D":
        if len(data.shape) == 3:
            if params.get("frame") is None or params["frame"] >= data.shape[0] or params["frame"] < 0:
                raise ValueError("Please provide a valid frame.")
            data = data[params["frame"]]
        if params.get("vmin") is not None and params.get("vmax") is not None:
            plt.imshow(data, vmin=params["vmin"], vmax=params["vmax"], cmap=params["cmap"])
        else:
            plt.imshow(data, cmap=params["cmap"])
        plt.show()
    elif params["mode"] == "2D+time":
        fig, ax = plt.subplots(1,1,squeeze=False)
        plt.axis('off')
        if params.get("vmin") is not None and params.get("vmax") is not None:
            I = ax[0,0].imshow(data[0], vmin=params["vmin"], vmax=params["vmax"], cmap=params["cmap"])
        else:
            I = ax[0,0].imshow(data[0], cmap=params["cmap"])
        def animate(i):
            I.set_array(data[i])
            return I,
        ani = animation.FuncAnimation(fig, animate, interval=50, frames=range(0,data.shape[0]), repeat=True)
        if params.get("save"):
            writer = animation.PillowWriter(fps=15,
                                            bitrate=1800)
            ani.save(params["save_file"], writer=writer)
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