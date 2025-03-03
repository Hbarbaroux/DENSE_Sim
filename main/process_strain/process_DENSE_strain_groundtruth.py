import os
import sys
import glob
import time
import numpy as np
from pathlib import Path
from scipy import interpolate
from skimage.transform import resize
from scipy.ndimage import center_of_mass

CURRENT_FILE = Path(os.path.abspath(__file__))
sys.path.append(str(CURRENT_FILE.parent.parent.parent))

from main.utils import nice_time

from calculate_strain_from_motion import calculate_strain


def undersample_strain_from_strain(Ecc, Err, ED_DENSE_mask, motion_mask, DENSE_shape, DENSE_frames, motion_shape):
    """
    Undersample strain from full resolution to DENSE resolution.
    :param Ecc: Ground-truth circumferential strain.
    :param Err: Ground-truth radial strain.
    :param ED_DENSE_mask: DENSE myocardial mask for the end-diastolic frame.
    :param motion_mask: High-resolution myocardial mask for the end-diastolic frame.
    :param DENSE_shape: DENSE resolution.
    :param DENSE_frames: Number of frames in the DENSE sequence.
    :param motion_shape: High-resolution shape.
    :return: Undersampled strain fields (Circumferential and Radial).
    """
    Ecc_DENSE = np.zeros((DENSE_frames, DENSE_shape, DENSE_shape))
    Err_DENSE = np.zeros((DENSE_frames, DENSE_shape, DENSE_shape))

    if DENSE_frames < 40:
        Ecc = Ecc[::2]
        Err = Err[::2]

    for frame in range(DENSE_frames):

        # Interpolating ground-truth strain on full matrix grid
        x = np.arange(0, motion_shape)
        y = np.arange(0, motion_shape)
        xx, yy = np.meshgrid(x, y)

        xx_myo = xx[motion_mask]
        yy_myo = yy[motion_mask]
        Err_myo = Err[frame][motion_mask]
        Ecc_myo = Ecc[frame][motion_mask]
        GDErr = interpolate.griddata((xx_myo, yy_myo), Err_myo.ravel(), (xx, yy), method='nearest')
        GDEcc = interpolate.griddata((xx_myo, yy_myo), Ecc_myo.ravel(), (xx, yy), method='nearest')

        # Resizing to DENSE resolution
        resized_Err = resize(GDErr, (DENSE_shape,DENSE_shape), anti_aliasing=True, order=3)
        resized_Ecc = resize(GDEcc, (DENSE_shape,DENSE_shape), anti_aliasing=True, order=3)

        # Masking with DENSE ED LV mask
        resized_Err[~ED_DENSE_mask] = np.nan
        resized_Ecc[~ED_DENSE_mask] = np.nan

        Err_DENSE[frame] = resized_Err
        Ecc_DENSE[frame] = resized_Ecc

    return Err_DENSE, Ecc_DENSE


def undersample_strain_from_disp(U, V, ED_DENSE_mask, motion_mask, DENSE_shape, DENSE_frames, motion_shape):
    """
    Undersample strain from full resolution to DENSE resolution.
    :param U: Ground-truth displacement field in the x direction.
    :param V: Ground-truth displacement field in the y direction.
    :param ED_DENSE_mask: DENSE myocardial mask for the end-diastolic frame.
    :param motion_mask: High-resolution myocardial mask for the end-diastolic frame.
    :param DENSE_shape: DENSE resolution.
    :param DENSE_frames: Number of frames in the DENSE sequence.
    :param motion_shape: High-resolution shape.
    :return: Undersampled strain fields (Circumferential and Radial).
    """
    U_DENSE = np.zeros((DENSE_frames, DENSE_shape, DENSE_shape))
    V_DENSE = np.zeros((DENSE_frames, DENSE_shape, DENSE_shape))

    if DENSE_frames < 40:
        U = U[::2]
        V = V[::2]

    for frame in range(DENSE_frames):

        # Interpolating ground-truth strain on full matrix grid
        x = np.arange(0, motion_shape)
        y = np.arange(0, motion_shape)
        xx, yy = np.meshgrid(x, y)

        xx_myo = xx[motion_mask]
        yy_myo = yy[motion_mask]
        U_myo = U[frame][motion_mask]
        V_myo = V[frame][motion_mask]
        GDU = interpolate.griddata((xx_myo, yy_myo), U_myo.ravel(), (xx, yy), method='nearest')
        GDV = interpolate.griddata((xx_myo, yy_myo), V_myo.ravel(), (xx, yy), method='nearest')

        # Resizing to DENSE resolution
        resized_U = resize(GDU, (DENSE_shape,DENSE_shape), anti_aliasing=True)
        resized_V = resize(GDV, (DENSE_shape,DENSE_shape), anti_aliasing=True)

        # Masking with DENSE ED LV mask
        resized_U[~ED_DENSE_mask] = np.nan
        resized_V[~ED_DENSE_mask] = np.nan

        U_DENSE[frame] = resized_U * DENSE_shape / motion_shape
        V_DENSE[frame] = resized_V * DENSE_shape / motion_shape

    # Calculating strain from displacements
    centre = center_of_mass(~np.isnan(U_DENSE[0]))
    X, Y = np.meshgrid(np.arange(U_DENSE.shape[1]), np.arange(U_DENSE.shape[2]))
    strain = calculate_strain(U_DENSE, V_DENSE, X, Y, centre, ~np.isnan(U_DENSE[0]))

    return strain["Err"], strain["Ecc"]


def main(params, verbose, **kwargs):

    t0 = time.time()
    if verbose:
        print("\nRunning strain calculations...\n")

    if params["files"]:
        files = [os.path.join(params["motion_folder"], file) for file in params["files"]]
    else:
        files = glob.glob(os.path.join(params["motion_folder"], "*"))

    if verbose:
        print("Found {} cases to process.\n".format(len(files)))

    for i, file in enumerate(files):
        try:
            Ecc = np.load(os.path.join(file, params["path_Ecc"])).transpose(0,2,1)
            Err = np.load(os.path.join(file, params["path_Err"])).transpose(0,2,1)
            U = np.load(os.path.join(file, params["path_dx"])).transpose(0,2,1)
            V = np.load(os.path.join(file, params["path_dy"])).transpose(0,2,1)
            if ("path_motion_mask" in params.keys()) and (params["path_motion_mask"] is not None):
                motion_mask = (np.load(os.path.join(file, params["path_motion_mask"]))[0] == 1)
            else:
                motion_mask = ~np.isnan(Ecc[0])
            DENSE_mask = (np.load(os.path.join(params["DENSE_folder"], os.path.basename(file), params["path_DENSE_mask"])) == 1)
            ED_DENSE_mask = DENSE_mask[0]

        except:
            print("Case {}/{} from {} could not be processed: no data found.".format(
                i+1, len(files), os.path.basename(file)
            ))
            continue

        DENSE_shape = ED_DENSE_mask.shape[0]
        motion_shape = Ecc.shape[-1]
        DENSE_frames = DENSE_mask.shape[0] - 1 # Do not count ED frame

        if params["method"] == "from_disps":
            Err_DENSE, Ecc_DENSE = undersample_strain_from_disp(U, V, ED_DENSE_mask, motion_mask, DENSE_shape, DENSE_frames, motion_shape)
        elif params["method"] == "from_strain":
            Err_DENSE, Ecc_DENSE = undersample_strain_from_strain(Ecc, Err, ED_DENSE_mask, motion_mask, DENSE_shape, DENSE_frames, motion_shape)
        else:
            raise ValueError("Unknown method for strain calculation. Choose from ['from_disps', 'from_strain'].")

        # Saving the results
        np.save(os.path.join(params["DENSE_folder"], os.path.basename(file), "GT_Err"+params["output_suffix"]+".npy"), Err_DENSE.astype(np.float32))
        np.save(os.path.join(params["DENSE_folder"], os.path.basename(file), "GT_Ecc"+params["output_suffix"]+".npy"), Ecc_DENSE.astype(np.float32))

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