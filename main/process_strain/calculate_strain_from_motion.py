import os
import sys
import glob
import time
import numpy as np
from pathlib import Path
from scipy.signal import convolve2d
from scipy.ndimage import center_of_mass

CURRENT_FILE = Path(os.path.abspath(__file__))
sys.path.append(str(CURRENT_FILE.parent.parent.parent))

from main.utils import nice_time


FILTERS = {
    4: np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
}


def cart2pol(x, y):
    """
    Convert cartesian coordinates to polar coordinates.
    :param x: X coordinate.
    :param y: Y coordinate.
    :return: Polar coordinates (theta, rho).
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho


def calculate_strain(dispx, dispy, X, Y, centre, mask, filter=None):
    """
    Calculate strain from displacement fields. Similar to the method used in DENSEanalysis software (https://github.com/denseanalysis/denseanalysis).
    :param dispx: Displacement field in the x direction.
    :param dispy: Displacement field in the y direction.
    :param X: X coordinates of the meshgrid.
    :param Y: Y coordinates of the meshgrid.
    :param centre: Centre of the mask.
    :param mask: Mask of the region of interest.
    :param filter: Filter to use for the strain calculation.
    :return: Strain fields (Circumferential and Radial).
    """

    xtrj = dispx + X
    ytrj = dispy + Y

    strain = {"Ecc": np.zeros_like(xtrj) * np.nan,
              "Err": np.zeros_like(ytrj) * np.nan}

    theta, _ = np.array(cart2pol(centre[0]-X, centre[1]-Y))

    ct = np.cos(theta)
    st = np.sin(theta)

    h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) if filter is None else filter
    neighbors = convolve2d(mask, h, mode="same")

    filter_coords = list(zip(*np.array(np.where(h))-(len(h)//2)))
    max_neighbors = len(filter_coords)

    dx = np.zeros((2,max_neighbors))
    dX = np.zeros((2,max_neighbors))
    tf = np.zeros((max_neighbors,)).astype(bool)

    for frame in range(xtrj.shape[0]):

        for j in range(xtrj.shape[2]):
            for i in range(xtrj.shape[1]):
                
                if (mask[i,j]) and (neighbors[i,j] > 1):

                    dx[:,:] = np.nan
                    dX[:,:] = np.nan
                    tf[:] = False

                    for k, filter_coord in enumerate(filter_coords):
                        filter_i = i + filter_coord[0]
                        filter_j = j + filter_coord[1]

                        if (0 <= filter_i < xtrj.shape[1]) and (0 <= filter_j < xtrj.shape[2]) and mask[filter_i, filter_j]:
                            # Prevent numerical instabilities between mask and U/V matrices
                            if (dispx[frame,filter_i,filter_j] != 0) or \
                                (dispy[frame,filter_i,filter_j] != 0) or \
                                ((np.abs(xtrj[frame,filter_i,filter_j] - xtrj[frame,i,j]) < 0.1) and \
                                (np.abs(ytrj[frame,filter_i,filter_j] - ytrj[frame,i,j]) < 0.1)) \
                            :
                                tf[k] = True
                                dx[:, k] = [xtrj[frame,filter_i,filter_j] - xtrj[frame,i,j], 
                                            ytrj[frame,filter_i,filter_j] - ytrj[frame,i,j]]
                                dX[:, k] = [X[filter_i,filter_j] - X[i,j],
                                            Y[filter_i,filter_j] - Y[i,j]]

                    # Prevent numerical instabilities between mask and U/V matrices
                    if (dispx[frame,i,j] != 0) or (dispy[frame,i,j] != 0) or (np.nanmedian(np.abs(dx), axis=1) < 0.1).all():

                        Fave = np.linalg.lstsq(dX[:,tf].T, dx[:,tf].T, rcond=None)[0].T
                        E = 0.5*(np.dot(Fave.T,Fave) - np.eye(2))

                        rot = np.array([[ct[i,j], st[i,j]], [-st[i,j], ct[i,j]]])
                        Erot = np.dot(rot, np.dot(E, rot.T))

                        strain["Err"][frame,i,j] = Erot[0,0]
                        strain["Ecc"][frame,i,j] = Erot[1,1]

    return strain


def main(params, verbose, **kwargs):

    t0 = time.time()
    if verbose:
        print("\nRunning strain calculations...\n")

    if params["files"]:
        files = [os.path.join(params["base_folder"], file) for file in params["files"]]
    else:
        files = glob.glob(os.path.join(params["base_folder"], "*"))

    if verbose:
        print("Found {} cases to process.\n".format(len(files)))

    for i, file in enumerate(files):
        try:
            U_mx = np.load(os.path.join(file, params["path_dx"]))
            V_mx = np.load(os.path.join(file, params["path_dy"]))
            mask = (np.load(os.path.join(file, params["path_mask"]))[0] == 1)

        except:
            print("Case {}/{} from {} could not be processed: no data found.".format(
                i+1, len(files), os.path.basename(file)
            ))
            continue

        if params["transpose"]:
            U_mx = U_mx.transpose(0,2,1)
            V_mx = V_mx.transpose(0,2,1)
        if params["invert"]:
            U_mx, V_mx = V_mx, U_mx

        mask_center = center_of_mass(mask)
        mask_center = np.flip(mask_center)

        X, Y = np.meshgrid(np.arange(U_mx.shape[-1]), np.arange(U_mx.shape[-1]))
        if params["filter"]:
            filter = FILTERS[params["filter"]]
        else:
            filter = None
        strain = calculate_strain(U_mx, V_mx, X, Y, mask_center, mask, filter)
        
        os.makedirs(os.path.join(file, params["output_folder_subpath"]), exist_ok=True)
        for strain_cat, strain_values in strain.items():
            np.save(os.path.join(file, params["output_folder_subpath"], strain_cat + ".npy"), strain_values.astype(np.float32))

        if verbose:
            print("Case {}/{} from {} processed, time elapsed: {}".format(
                i+1, len(files), os.path.basename(file), nice_time(time.time()-t0)
            ))

    if verbose:
        print("\nStrain calculations done, total time elapsed: {}\n".format(nice_time(time.time()-t0)))


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