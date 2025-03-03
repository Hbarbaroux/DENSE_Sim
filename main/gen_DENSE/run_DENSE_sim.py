import os
import numpy as np
import sys
import matlab.engine as mtlb
from scipy.ndimage import binary_fill_holes, morphology
from skimage.morphology import disk
from skimage.transform import rescale
from scipy.ndimage import center_of_mass, binary_closing
from skimage.measure import profile_line
from scipy.interpolate import interp1d
import cv2 as cv
import matlab
import time
import yaml
import sigpy as sp
from sigpy.mri import dcf
from skimage.filters import window
import matplotlib.pyplot as plt
from scipy.signal.windows import kaiser, hamming
from skimage.transform import resize

from SimObject import SimObject
from SimPSD import SimInstant

from pathlib import Path

CURRENT_FILE = Path(os.path.abspath(__file__))

from pygrid_internal import pygrid as pygrid


def get_epi_endo_LA(mask, contour):
    """
    From a single contour array, extract epi and endo borders
        1. Compute the center of mass
        2. Pre-determine endo by running mask profile between contour points and
           the center of mass, and checking whether no myocardial points are present
           on that image profile
        3. Get the furthest epi points from the center of mass from each "leg" of the LA mask
        4. Refine epi/endo selection by seperating the contour between the two points

    :param mask: binary myocardial mask image
    :param contour: [N,2] contour points coordinates
    :returns: epi points coordinates, endo points coordinates ([N1,2] and [N2,2])
    """

    #########################
    # 1. Get center of mass #
    #########################

    barycentre = center_of_mass(mask)

    #######################
    # 2. First estimation #
    #######################

    # epi_endo is 0 for contour points determined as endo
    #             1 for contour points determined as epi
    epi_endo = []
    for point in contour:
        profile = profile_line(mask, (point[1], point[0]), barycentre)
        if not any(profile[1:] == 1): # endo
            epi_endo.append(0)
        else:                    # epi
            epi_endo.append(1)
    epi_endo = np.array(epi_endo, dtype=bool)

    ###########################
    # 3. Furthest endo points #
    ###########################

    distances = np.linalg.norm(contour[:, [1, 0]]-barycentre, axis=1)

    # Get the furthest one
    subset_idx = np.argmax(distances[epi_endo])
    furthest_idx_1 = np.arange(distances.shape[0])[epi_endo][subset_idx]
    
    # Remove the countour half around this point and check the new furthest one
    plus_minus_idx = len(contour)//4
    epi_endo_filtered = np.array(epi_endo)
    if furthest_idx_1-plus_minus_idx < 0:
        epi_endo_filtered[furthest_idx_1-plus_minus_idx:] = 0
        epi_endo_filtered[:furthest_idx_1+plus_minus_idx] = 0
    elif furthest_idx_1+plus_minus_idx > len(epi_endo_filtered):
        epi_endo_filtered[:(furthest_idx_1+plus_minus_idx)-len(epi_endo_filtered)] = 0
        epi_endo_filtered[furthest_idx_1-plus_minus_idx:] = 0
    else:
        epi_endo_filtered[furthest_idx_1-plus_minus_idx:furthest_idx_1+plus_minus_idx] = 0
    subset_idx = np.argmax(distances[epi_endo_filtered])
    furthest_idx_2 = np.arange(distances.shape[0])[epi_endo_filtered][subset_idx]

    # Check which one comes first in terms of index position in the contour array
    if furthest_idx_1 > furthest_idx_2:
        furthest_idx_1, furthest_idx_2 = furthest_idx_2, furthest_idx_1

    ############################
    # 4. Refine endo/epi split #
    ############################

    # Check which half contains more pre-determined epi points
    # and consider the entire half to be epi
    inner_sum = sum(epi_endo[furthest_idx_1:furthest_idx_2])
    outer_sum = sum(epi_endo[furthest_idx_2:]) + sum(epi_endo[:furthest_idx_1])

    if outer_sum > inner_sum:
        epi_contour = np.concatenate([contour[furthest_idx_2:], contour[:furthest_idx_1+1]])
        endo_contour = contour[furthest_idx_1+1:furthest_idx_2]
    else:
        endo_contour = np.concatenate([contour[furthest_idx_2+1:], contour[:furthest_idx_1]])
        epi_contour = contour[furthest_idx_1:furthest_idx_2+1]

    return epi_contour, endo_contour


def get_flip_angles(Nt, max_angle=20, cycle_duration=1000, T1=850, dt=30):
    """
    Generate variable flip angles for a DENSE acquisition.
    :param Nt: number of time points
    :param max_angle: maximum flip angle (last time point)
    :param cycle_duration: duration of a cardiac cycle (ms, only used if dt is None)
    :param T1: T1 relaxation time (ms)
    :param dt: time step (ms)
    :returns: list of flip angles
    """
    dt = dt if dt is not None else cycle_duration / Nt
    angles = []
    for i in range(Nt):
        if i == 0:
            angles.append(max_angle)
        else:
            alpha_2 = angles[-1] * np.pi / 180.0
            alpha_1 = np.arcsin(np.sin(alpha_2) * np.exp(-1.0 * dt / T1))
            angles.append(alpha_1 * 180 / np.pi)
    angles.reverse()
    return angles


def load_data(folder):
    """
    Load the data from a folder containing the cardiac motion data.
    :param folder: folder containing the simulation data
    :returns: tissue segmentation, x displacements, y displacements
    """
    tissues_seg_time = np.load(os.path.join(folder, "tissues_seg_time.npy"))
    U_disps = np.load(os.path.join(folder, "dx_pixel.npy"))
    V_disps = np.load(os.path.join(folder, "dy_pixel.npy"))

    return tissues_seg_time, U_disps, V_disps


def generate_sampling(
    pixel_fov, 
    sampling_type="spiral", 
    params={
        "interleaves": 4, 
        "spDWELL": 10e-6, 
        "ADCdwell": 2e-6,
        "spiral_form_file": "spiral.txt"
    }):
    """
    Generate the k-space sampling trajectory.
    :param pixel_fov: size of the field of view in pixels
    :param sampling_type: type of sampling (spiral or cartesian)
    :param params: dictionary of parameters for the sampling
    :returns: k-space trajectory coordinates (nbr_interleaves, nbr_samples, 2)
    """

    if sampling_type == "spiral":

        # Need to integrate the base spiral
        with open(params["spiral_form_file"], 'r') as f:
            spiral = f.readlines()
        spiral = np.array([i.strip().split("\t") for i in spiral]).astype(float)
        
        gxi = interp1d(
            np.arange(0,spiral.shape[0])*float(params["spDWELL"]), spiral[:,0]
        )(np.arange(0,(spiral.shape[0]-1)*float(params["spDWELL"]),float(params["ADCdwell"])))
        gyi = interp1d(
            np.arange(0,spiral.shape[0])*float(params["spDWELL"]), spiral[:,1]
        )(np.arange(0,(spiral.shape[0]-1)*float(params["spDWELL"]),float(params["ADCdwell"])))

        # Integrated base spiral, to form final trajectory
        # k = (n,2) are the coordinates of the sampled points,
        # n being the number of sampling points 
        k = np.zeros((len(gxi),2))
        for i in range(len(gxi)):
            k[i, 0] = np.sum(gxi[:i])
            k[i, 1] = np.sum(gyi[:i]) 

        # Creating interleaves from previous spiral, rotated at regular angles
        k_tot = np.zeros((params["interleaves"], k.shape[0], 2))
        for i in range(params["interleaves"]):
            rotangle=2*np.pi*(i-1)/params["interleaves"]
            k_tot[i,:,0] = k[:,0] * np.cos(rotangle) - k[:,1] * np.sin(rotangle)
            k_tot[i,:,1] = k[:,0] * np.sin(rotangle) + k[:,1] * np.cos(rotangle)

        # Resampling the spiral sampling over the desired diameter
        k_fov = (((k_tot - k_tot.min()) / (k_tot.max() - k_tot.min())) - 0.5 ) * pixel_fov

    else:
        x_sampling = np.linspace(-pixel_fov/2, pixel_fov/2, pixel_fov)
        X, Y = np.meshgrid(x_sampling, x_sampling)
        k_fov = np.stack([X.flatten(), Y.flatten()],axis=-1)[None,...]

    return k_fov


def run_DENSE_sim(
    imgs, # NtxNNxNN
    dx, # NtxNNxNN
    dy, # NtxNNxNN
    t1, # NNxNN
    t2, # NNxNN
    s,  # NNxNN
    N_im = 90,
    tt = None,
    Nt = None,
    FOV = [200,200,8],
    N_im_pre_spiral = 60,
    spiral_time = 5.5,
    t2_star = 40,
    SNR = (10, 30),
    use_gpu = True,
    ke_dir = [1.0, 0.0, 0.0],
    ke = 0.1,
    kd = 0.08,
    te = 1.08,
    max_flip = 15,
    fov_se_ratio = 0.6,
    phase_cycling_nbr = 2,
    fov_se_window_params = {
        "main": ('hamming'),
        "additional": {
            "ramp_ratio": 0.2
        }
    },
    k_sampling_type = "spiral", 
    k_sampling_params = {
        "interleaves": 4, 
        "interleaves_per_cycle": 2,
        "spDWELL": 10e-6, 
        "ADCdwell": 2e-6,
        "spiral_form_file": "spiral.txt"
    },
    do_density=True,
    outer_mask_file=None,
    **kwargs
):
    """
    Run a DENSE simulation.
    :param imgs: tissue segmentation images (Nt x NN x NN)
    :param dx: x displacements (Nt x NN x NN)
    :param dy: y displacements (Nt x NN x NN)
    :param t1: T1 relaxation time (NN x NN)
    :param t2: T2 relaxation time (NN x NN)
    :param s: proton density (NN x NN)
    :param N_im: size of the final image (N_imxN_im)
    :param tt: time points of the cardiac motion sequences
    :param Nt: number of time points for the DENSE simulation
    :param FOV: field of view (mm)
    :param N_im_pre_spiral: size of the image before the spiral sampling
    :param spiral_time: time for one spiral interleave (ms) -> used for T2* decay
    :param t2_star: T2* relaxation time
    :param SNR: relative SNR range/level (not actual SNR, needs to be mapped to noise level empirically)
    :param use_gpu: use GPU for the simulation (if available)
    :param ke_dir: encoding direction for the phase encoding
    :param ke: encoding strength (cycles/mm)
    :param kd: dephasing strength
    :param te: echo time
    :param max_flip: maximum flip angle
    :param fov_se_ratio: ratio of the field of view for the SE window
    :param phase_cycling_nbr: number of phase cycling acquisitions
    :param fov_se_window_params: parameters for the SE window
    :param k_sampling_type: type of k-space sampling
    :param k_sampling_params: parameters for the k-space sampling
    :param do_density: do density correction
    :param outer_mask_file: file containing the outer mask to apply at the end
    :returns: simulated DENSE acquisitions
    """

    if type(SNR) == list and len(SNR) == 2:
        SNR = np.random.uniform(SNR[0], SNR[1])
    elif type(SNR) == float or type(SNR) == int:
        SNR = SNR
    elif SNR is None:
        SNR = SNR
    else:
        raise ValueError("SNR should be a list of two elements or a float/int. SNR to None would lead to no noise added.")

    k_fov = generate_sampling(N_im_pre_spiral, k_sampling_type, k_sampling_params)
    pm_dcf = dcf.pipe_menon_dcf(k_fov, show_pbar=False)

    if Nt is None:
        Nt = dx.shape[0]
    Nt_gt = dx.shape[0]
    NN = imgs.shape[-1]

    if fov_se_window_params["main"] == ("hamming"):
        ramp_ratio = fov_se_window_params["additional"]["ramp_ratio"]
        fov_se = round(fov_se_ratio*N_im_pre_spiral)
        window_range = int(N_im_pre_spiral*ramp_ratio//2)*4

        w = np.zeros(N_im_pre_spiral)

        w[(N_im_pre_spiral - fov_se)//2 - window_range//4:(N_im_pre_spiral - fov_se)//2 + window_range//4] = window(fov_se_window_params["main"], window_range)[:window_range//2]
        w[(N_im_pre_spiral + fov_se)//2 - window_range//4:(N_im_pre_spiral + fov_se)//2 + window_range//4] = window(fov_se_window_params["main"], window_range)[window_range//2:]
        w[:(N_im_pre_spiral - fov_se)//2 - window_range//4] = window(fov_se_window_params["main"], window_range)[0]
        w[(N_im_pre_spiral + fov_se)//2 + window_range//4:] = window(fov_se_window_params["main"], window_range)[0]
        w[(N_im_pre_spiral - fov_se)//2 + window_range//4:(N_im_pre_spiral + fov_se)//2 - window_range//4] = 1

        window_smoothing = np.outer(w, w)
    else:
        window_smoothing = 1

    x = np.arange(0, imgs.shape[-1])
    y = np.arange(0, imgs.shape[-2])
    Y, X = np.meshgrid(x, y)

    xtrj = (dy + X)/len(X) - 0.5
    ytrj = (dx + Y)/len(Y) - 0.5

    Y = (Y / len(Y)) - 0.5
    X = (X / len(X)) - 0.5

    r_total = np.stack([xtrj.reshape(xtrj.shape[0], -1), ytrj.reshape(ytrj.shape[0], -1)], -1)
    r_total = np.concatenate([r_total, np.zeros_like(r_total[:,:,:1]) ], 2)
    r0 = np.stack([X.reshape(1, -1), Y.reshape(1, -1)], -1)
    r0 = np.concatenate([r0, np.zeros_like(r0[:,:,:1]) ], 2)
    r_total = np.concatenate([r0, r_total], 0)

    mask_myo0 = (imgs == 1)[0] | (imgs == 2)[0]
    mask_cavity0 = (imgs == 5)[0] | (imgs == 6)[0]
    mask_blood0 = mask_cavity0 | mask_myo0
    
    mask_blood0 = binary_fill_holes(mask_myo0)
    mask_blood0d = morphology.binary_dilation(mask_blood0, disk(5), iterations = 3)
    
    final_mask = ~(mask_blood0.astype(bool)) & (t1 != 0)

    mask_myo0_flatten = mask_myo0.flatten()
    r_heart = r_total[:, mask_myo0_flatten, :]
    s_heart = s.flatten()[mask_myo0_flatten]
    t1_heart = t1.flatten()[mask_myo0_flatten]
    t2_heart = t2.flatten()[mask_myo0_flatten]

    x_blood = X[mask_blood0d > 0.5]
    y_blood = Y[mask_blood0d > 0.5]
    z_blood = np.zeros_like(x_blood)

    r0_blood = np.stack([x_blood, y_blood, z_blood], 1)
    r_blood = np.tile(r0_blood, [Nt_gt+1,1,1])
    s_blood = np.ones(r_blood.shape[1]) * np.random.uniform(0.45, 0.55)
    t1_blood = np.ones(r_blood.shape[1]) * np.random.uniform(80, 100)
    t2_blood = np.ones(r_blood.shape[1]) * np.random.uniform(10, 20)

    final_mask_flatten = final_mask.flatten()
    r_remain = r_total[:, final_mask_flatten, :]
    s_remain = s.flatten()[final_mask_flatten]
    t1_remain = t1.flatten()[final_mask_flatten]
    t2_remain = t2.flatten()[final_mask_flatten]

    r_all = np.concatenate([r_blood, r_remain, r_heart], 1)
    s_all = np.concatenate([s_blood, s_remain, s_heart])
    t1_all = np.concatenate([t1_blood, t1_remain, t1_heart])
    t2_all = np.concatenate([t2_blood, t2_remain, t2_heart])

    if Nt >= 40:
        Nt_acqs = int(np.floor(Nt/2+1))
    else:
        Nt_acqs = Nt

    # For DENSE specifically run the phase cycling acquisition
    if phase_cycling_nbr:
        extra_theta = np.linspace(0, 2*np.pi, phase_cycling_nbr+1)[1:-1]
    else:
        extra_theta = []

    acq_locs = [np.arange(0, Nt) * 1000 / Nt + 500/Nt]
    nbr_cycles = 1
    if k_sampling_type == "spiral":
        nbr_cycles = k_sampling_params["interleaves"]//k_sampling_params["interleaves_per_cycle"]
        if k_sampling_params["interleaves"] > 1:
            acq_locs = []
            for _ in range(nbr_cycles):
                for i in range(k_sampling_params["interleaves_per_cycle"]):
                    acq_locs += [np.arange(0,Nt_acqs) * 30 + 15*(i+1)]
                    # acq_locs += [np.arange(0, Nt_acqs) * 15 + 22.5]

    acq_locs = np.array(acq_locs).astype(float)
    extra_acqs_locs = np.stack([acq_locs for i in range(len(extra_theta))]).astype(float)

    if k_sampling_type == "spiral":
        for i in range(nbr_cycles):
            random_offset = (np.random.rand()-0.5) * 10
            acq_locs[i*k_sampling_params["interleaves_per_cycle"]:(i+1)*k_sampling_params["interleaves_per_cycle"]] += random_offset
            for j in range(len(extra_theta)):
                random_offset = (np.random.rand()-0.5) * 10
                extra_acqs_locs[j][i*k_sampling_params["interleaves_per_cycle"]:(i+1)*k_sampling_params["interleaves_per_cycle"]] += random_offset

    angles = get_flip_angles(Nt_acqs, max_angle=max_flip, T1=850)

    sim_object = SimObject()
    sim_object.gen_from_generator(r_all, s_all, t1_all, t2_all, periodic=False, tt=np.array(tt), FOV=np.array(FOV)*1e-3)

    acqs0s = []
    for i in range(len(acq_locs)):
        simulator = SimInstant(sim_object, use_gpu=use_gpu)
        simulator.sample_DENSE_PSD(ke_dir=ke_dir, ke=ke, kd=kd, flip=angles, acq_loc=acq_locs[i], te=te)
        acqs0s.append(simulator.run())

    extra_acqs = []
    for theta_i, extra_acq_loc in zip(extra_theta, extra_acqs_locs):
        extra_acq = []
        for i in range(len(extra_acq_loc)):
            simulator = SimInstant(sim_object, use_gpu=use_gpu)
            simulator.sample_DENSE_PSD(ke_dir=ke_dir, rf_dir = [np.cos(theta_i), np.sin(theta_i), 0], ke=ke, kd = 0.0, acq_loc=extra_acq_loc[i], te=te, flip=angles)
            extra_acq.append(simulator.run())
        extra_acqs.append(extra_acq)

    t0 = time.time()
    for acqs0, acq_loc in zip(acqs0s, acq_locs):
        for ii in range(Nt_acqs):
            mask_myo = np.zeros((NN,NN))
            mask_static = np.ones((NN,NN))
            for i in range(len(tt)):
                if tt[i] > acq_loc[ii]:
                    lo = i - 1
                    hi = i

                    lo_mod = (tt[hi] - acq_loc[ii]) / (tt[hi] - tt[lo])
                    hi_mod = (acq_loc[ii] - tt[lo]) / (tt[hi] - tt[lo])
                    break
                elif i == (len(tt) - 1):
                    lo = i
                    hi = 0
                    lo_mod = (tt[hi] + 1000 - acq_loc[ii]) / (tt[hi] + 1000 - tt[lo])
                    hi_mod = (acq_loc[ii] - tt[lo]) / (tt[hi] + 1000 - tt[lo])

            r_frame = r_heart[lo] * lo_mod + r_heart[hi] * hi_mod
            rint = np.round((r_frame + 0.5) * NN).astype(int)
            mask_myo[rint[:,0], rint[:,1]] = 1.0
            mask_myo = binary_closing(mask_myo, iterations=2)
            mask_blood = binary_fill_holes(mask_myo)
            mask_cavity = mask_blood.astype(float) - mask_myo.astype(float)
            mask_cavity_dil = morphology.binary_dilation(mask_cavity)
            mask_static -= mask_cavity_dil

            m_temp = mask_static[mask_blood0d > 0.5]
            m_temp = ~(m_temp > 0.5)
            point_mask = np.ones(s_all.size)
            point_mask[:m_temp.size] = m_temp
            acqs0[ii][0] = acqs0[ii][0][point_mask > 0.5, :]
            acqs0[ii][1] = acqs0[ii][1][point_mask > 0.5, :]

    
    for extra_acq, extra_acqs_loc in zip(extra_acqs, extra_acqs_locs):
        for acq, acq_loc in zip(extra_acq, extra_acqs_loc):
            for ii in range(Nt_acqs):
                mask_myo = np.zeros((NN,NN))
                mask_static = np.ones((NN,NN))

                for i in range(len(tt)):
                    if tt[i] > acq_loc[ii]:
                        lo = i - 1
                        hi = i

                        lo_mod = (tt[hi] - acq_loc[ii]) / (tt[hi] - tt[lo])
                        hi_mod = (acq_loc[ii] - tt[lo]) / (tt[hi] - tt[lo])
                        break
                r_frame = r_heart[lo] * lo_mod + r_heart[hi] * hi_mod
                rint = np.round((r_frame + 0.5) * NN).astype(int)
                mask_myo[rint[:,0], rint[:,1]] = 1.0
                mask_myo = binary_closing(mask_myo, iterations=2)
                mask_blood = binary_fill_holes(mask_myo)
                mask_cavity = mask_blood.astype(float) - mask_myo.astype(float)
                mask_cavity_dil = morphology.binary_dilation(mask_cavity)
                mask_static -= mask_cavity_dil

                m_temp = mask_static[mask_blood0d > 0.5]
                m_temp = ~(m_temp > 0.5)
                point_mask = np.ones(s_all.size)
                point_mask[:m_temp.size] = m_temp
                acq[ii][0] = acq[ii][0][point_mask > 0.5, :]
                acq[ii][1] = acq[ii][1][point_mask > 0.5, :]

    ##### Now we generate the images
    all_im_pc = np.zeros((Nt, N_im, N_im), np.complex64)

    dens_mods = [1.0]*len(acqs0s)
    if do_density:
        for i, acqs0 in enumerate(acqs0s):
            dd = get_dens(acqs0[0][0], use_gpu = use_gpu)
            dens_mod = np.median(dd)
            dens_mods[i] = dens_mod

    if SNR:
        noise_scale = 0.3*256*256/N_im/SNR
    else:
        noise_scale = 0.0

    kaiser_range = [2,6]
    kaiser_beta = np.random.rand() * (kaiser_range[1] - kaiser_range[0]) + kaiser_range[0]

    k_nu_acqs = []
    k_nu_extra_acqs = []
    im0s = []
    extra_ims = [[] for _ in range(len(extra_acqs))]
    for ii in range(Nt_acqs):

        k_nus = []
        for i, acqs0 in enumerate(acqs0s):
            if do_density:
                dd = get_dens(acqs0[ii][0], use_gpu = use_gpu)
                dd = dens_mod / (dd + dens_mods[i] * .1)
            else:
                dd = np.ones(acqs0[0][0].shape[0], np.float32)

            im0, _ = sim_object.grid_im_from_M(acqs0[ii][0], acqs0[ii][1], N_im = N_im_pre_spiral, use_gpu = use_gpu, dens = dd)
            
            im0_reduced = im0 * window_smoothing

            k_nu = sp.nufft(im0_reduced, k_fov[i])
            k_nu = k_nu*np.exp(-np.linspace(0, spiral_time, k_nu.shape[-1])/t2_star)
            k_nu += noise_scale * (np.random.standard_normal(k_nu.shape) + 1j * np.random.standard_normal(k_nu.shape))
            k_nus.append(k_nu)
        im0s.append(im0)

        k_nu = np.stack(k_nus)
        k_nu_acqs.append(k_nu)

        k_nu_extra = []
        for l, extra_acq in enumerate(extra_acqs):
            k_nus = []
            for i, acq in enumerate(extra_acq):
                if do_density:
                    dd = get_dens(acq[ii][0], use_gpu = use_gpu)
                    dd = dens_mod / (dd + dens_mods[i] * .1)
                else:
                    dd = np.ones(acq[0][0].shape[0], np.float32)
                
                im_temp, _ = sim_object.grid_im_from_M(acq[ii][0], acq[ii][1], N_im = N_im_pre_spiral, use_gpu = use_gpu, dens = dd)
                im_temp_reduced = im_temp * window_smoothing
                k_nu = sp.nufft(im_temp_reduced, k_fov[i])
                k_nu = k_nu*np.exp(-np.linspace(0, spiral_time, k_nu.shape[-1])/t2_star)
                k_nu += noise_scale * (np.random.standard_normal(k_nu.shape) + 1j * np.random.standard_normal(k_nu.shape))
                k_nus.append(k_nu)
            extra_ims[l].append(im_temp)
            k_nu = np.stack(k_nus)
            k_nu_extra.append(k_nu)
        k_nu_extra_acqs.append(np.stack(k_nu_extra))        

    if Nt_acqs < Nt:
        k_nu_total = []
        k_nu_extra_total = []
        for i in range(Nt_acqs):
            k_nu_total.append(k_nu_acqs[i])
            k_nu_extra_total.append(k_nu_extra_acqs[i])

            if i < Nt_acqs - 1:
                k_nu_new = np.zeros_like(k_nu_acqs[i])
                k_nu_new[1::2,:] = k_nu_acqs[i][1::2,:]
                k_nu_new[::2,:] = k_nu_acqs[i+1][::2,:]
                k_nu_total.append(k_nu_new)

                k_nu_extra_new = np.zeros_like(k_nu_extra_acqs[i])
                k_nu_extra_new[:,1::2,:] = k_nu_extra_acqs[i][:,1::2,:]
                k_nu_extra_new[:,::2,:] = k_nu_extra_acqs[i+1][:,::2,:]
                k_nu_extra_total.append(k_nu_extra_new)
        k_nu_total = k_nu_total[:Nt]
        k_nu_extra_total = k_nu_extra_total[:Nt]
    else:
        k_nu_total = k_nu_acqs
        k_nu_extra_total = k_nu_extra_acqs

    for ii in range(Nt):
        k_nu = k_nu_total[ii]
        im0_dcf = sp.nufft_adjoint(k_nu * pm_dcf, k_fov, oshape=(N_im, N_im))
        im0_dcf = proc_im(im0_dcf, N_im, None, kaiser_beta)

        extra_im = []
        for i, extra_acq in enumerate(extra_acqs):
            k_nu = k_nu_extra_total[ii][i]
            im_temp_dcf = sp.nufft_adjoint(k_nu * pm_dcf, k_fov, oshape=(N_im, N_im))
            im_temp_dcf = proc_im(im_temp_dcf, N_im, None, kaiser_beta)
            extra_im.append(im_temp_dcf)

        # Generates a phase cycled image for DENSE
        im_pc = im0_dcf.copy()
        for i in range(len(extra_im)):
            im_pc += np.conj(np.exp(1j * extra_theta[i])) * extra_im[i]
        all_im_pc[ii] = im_pc

    resized_outer_mask = resize(np.load(outer_mask_file), (N_im, N_im), 0)
    all_im_pc[:, resized_outer_mask==0] = 0+0j

    return {
        "imgs": all_im_pc,
        "final_mask": final_mask,
        "mask_blood0": mask_blood0,
        "t1": t1,
        "r_all": r_all,
        "t1_all": t1_all,
        "SNR": SNR,
    }


def proc_im(im, N_im = 256, noise_scale = 50, kaiser_beta = 4, do_hamming = False):
    """
    Process the image by adding noise and random blurring.
    :param im: image to process
    :param N_im: size of the final image
    :param noise_scale: scale of the noise to add
    :param kaiser_beta: beta parameter for the Kaiser window
    :param do_hamming: apply a Hamming window
    :returns: processed image
    """
    
    k0 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im)))
    if noise_scale is not None and noise_scale > 0:
        k0 += noise_scale * (np.random.standard_normal(k0.shape) + 1j * np.random.standard_normal(k0.shape))

    if kaiser_beta > 0:
        window = kaiser(N_im, kaiser_beta, sym=False)
        k0 *= np.outer(window, window)

    if do_hamming:
        window = hamming(N_im, sym=False)
        k0 *= np.outer(window, window)

    im = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(k0)))
    
    return im


def get_dens(pos, N_im=256, oversamp=2.0, krad=1.5, nthreads = 0, use_gpu = False, scaler = 0.8):
    """
    Get the density correction for the DENSE simulation.
    """
    gridder = pygrid.Gridder(
            imsize=(N_im, N_im), grid_dims=2, over_samp=oversamp, krad=krad, use_gpu=use_gpu
            )

    kx_all = pos[:, 0].astype(np.float32)
    ky_all = pos[:, 1].astype(np.float32)
    dens = np.ones_like(kx_all)

    traj = np.stack((kx_all, ky_all, np.zeros_like(ky_all)), 1).astype(np.float32) * scaler

    MM = np.ones_like(kx_all).astype(np.complex64)

    out = None
    if use_gpu:
        out, kdata = gridder.cu_k2im(MM.astype(np.complex64), traj, dens, imspace=True)
    else:
        out, kdata = gridder.k2im(MM.astype(np.complex64), traj, dens, imspace=True)

    dd = None
    if use_gpu:
        dd = gridder.cu_im2k(out, traj, dens, imspace=True)
    else:
        dd = gridder.im2k(out, traj, dens, imspace=True)
    
    return np.abs(dd)


def get_matlab_contours(seg_mask, N_im, is_short_axis=True):
    """
    Get the endo and epi contours from the binary masks.
    :param seg_mask: binary masks (NN x NN x Nt)
    :param N_im: size of the final corresponding DENSE image
    :param is_short_axis: whether the axis is short or long
    :returns: endo and epi contours
    """

    # Getting epi and endo contours from binary masks, frame by frame
    endo_contours, epi_contours = [], []
    ratio = N_im / seg_mask.shape[0]
    for frame in range(seg_mask.shape[-1]):
        mask = np.roll(seg_mask[:,:,frame].T, int(1/ratio), [0,1])
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        if is_short_axis:
            endo_contours.append(matlab.double(((contours[-2][:,0,:])*ratio).tolist()))
            epi_contours.append(matlab.double(((contours[-1][:,0,:])*ratio).tolist()))

        else:
            epi_contour, endo_contour = get_epi_endo_LA(mask, contours[0][:,0,:])
            endo_contours.append(matlab.double(((endo_contour+1)*ratio).tolist()))
            epi_contours.append(matlab.double(((epi_contour+1)*ratio).tolist()))
    
    return endo_contours, epi_contours


def save_dns_data(mag, phase_x, phase_y, seg_mask, template_file, out_file, ke, pixel_spacing, slice_cat):
    """
    Save the DENSE data to a .dns workspace file (for DENSEanalysis).
    :param mag: magnitude images (Nt x NN x NN)
    :param phase_x: x phase images (Nt x NN x NN)
    :param phase_y: y phase images (Nt x NN x NN)
    :param seg_mask: segmentation mask (NN x NN x Nt)
    :param template_file: template file to use
    :param out_file: output file to save the data
    :param ke: encoding strength (cycles/mm)
    :param pixel_spacing: pixel spacing (mm)
    :param slice_cat: slice category (base, mid, apex, 2ch, 3ch, 4ch)
    :returns: None
    """
    eng = mtlb.start_matlab()
    eng.addpath(str(CURRENT_FILE.parent))

    magnitude_imgs = mag.transpose((1,2,0))
    phase_x_imgs = phase_x.transpose((1,2,0))
    phase_y_imgs = phase_y.transpose((1,2,0))

    magnitude_imgs = magnitude_imgs / magnitude_imgs.max()

    magnitude_imgs = matlab.uint16((magnitude_imgs * 4000).astype(np.uint16).tolist())
    phase_x_imgs = matlab.uint16((((phase_x_imgs + np.pi) / (2*np.pi)) * (2**12 - 1)).astype(np.uint16).tolist())
    phase_y_imgs = matlab.uint16((((phase_y_imgs + np.pi) / (2*np.pi)) * (2**12 - 1)).astype(np.uint16).tolist())

    seg_mask = seg_mask.transpose((2,1,0))

    pixel_spacing = matlab.double([pixel_spacing, pixel_spacing])
    endo_contours, epi_contours = get_matlab_contours(seg_mask, N_im=mag.shape[1])

    eng.convert_to_dns(
        magnitude_imgs, phase_x_imgs, phase_y_imgs, 
        pixel_spacing, ke, 
        epi_contours, endo_contours, 
        slice_cat, template_file, out_file, 
        nargout=0
    )

    eng.quit()


def get_anat_info(log_file):
    dict_act = {}

    with open(log_file, "r") as f:
        lines = f.readlines()

    i = 0
    while "Activity Ratios" not in lines[i]:
        i += 1
    i += 2

    while "-----" not in lines[i]:
        split_data = lines[i].strip().split(" = ")
        dict_act[split_data[0]] = float(split_data[1])
        i += 1

    return dict_act


def map_tissue_types(dict_act, tissue_cats):
    dict_act_global = {tissue_cat: [] for tissue_cat in tissue_cats}

    for tissue, act in dict_act.items():
        if tissue in ['muscle_activity', 'pericardium_activity']:
            dict_act_global["muscle"].append(act)
        if tissue in ['myoLV_act','myoRV_act','myoLA_act','myoRA_act']:
            dict_act_global["heart"].append(act)
        elif tissue in ['bldplLV_act','bldplRV_act','art_activity','vein_activity','bldplLA_act','bldplRA_act']:
            dict_act_global["blood"].append(act)
        elif tissue in ['body_activity', 'st_wall_activity', 'pancreas_activity', 'gall_bladder_activity', 'sm_intest_activity', 'desc_li_activity']:
            dict_act_global["fat"].append(act)
        elif tissue in ['liver_activity']:
            dict_act_global["liver"].append(act)
        elif tissue in ['bone_marrow_activity']:
            dict_act_global["marrow"].append(act)
        elif tissue in ['rib_activity','cortical_bone_activity','spine_activity', 'cartilage_activity']:
            dict_act_global["bone"].append(act)
        elif tissue in ['right lung_activity', 'left lung_activity', 'st_cnts_activity']:
            dict_act_global["air"].append(act)
    return dict_act_global


def get_tissue_maps(label_img, dict_act_global, tissue_params):
    t1 = np.zeros_like(label_img).astype(np.int16)
    t2 = np.zeros_like(label_img).astype(np.int16)
    s0 = np.zeros_like(label_img).astype(np.float64)

    for tissue_cat in dict_act_global.keys():
        for act_level in dict_act_global[tissue_cat]:
            t1[label_img == act_level] = tissue_params["T1"][tissue_cat]
            t2[label_img == act_level] = tissue_params["T2"][tissue_cat]
            s0[label_img == act_level] = tissue_params["S0"][tissue_cat]

    return t1, t2, s0


def load_spacing_info(log_file, orig_img_size, out_img_size):
    with open(log_file, "r") as f:
        lines = f.readlines()

    i = 0
    while "pixel width" not in lines[i]:
        i += 1
    
    orig_spacing = float(lines[i].split(" ")[-2])
    ratio = orig_img_size/out_img_size
    
    return ratio * orig_spacing * 10


def compute_lag_masks(ED_mask, U, V, new_size):
    """
    Compute displaced masks for the LV.
    :param ED_mask: end-diastolic mask (NN x NN)
    :param U: x displacements (Nt x NN x NN)
    :param V: y displacements (Nt x NN x NN)
    :param new_size: image size for the final masks
    :returns: displaced masks
    """
    r0 = np.array(np.where(ED_mask == 1)).T

    dx = U[:, r0[:,1], r0[:,0]]
    dy = V[:, r0[:,1], r0[:,0]]

    r_x = np.round((dx + r0[:,1][None,:]) * new_size / U.shape[-1]).astype(int)
    r_y = np.round((dy + r0[:,0][None,:]) * new_size / U.shape[-2]).astype(int)

    r_0_new = np.round(r0 * new_size / U.shape[-1]).astype(int)

    mask_final = np.zeros((ED_mask.shape[0], new_size, new_size))
    for it in range(len(U)+1):
        if it == 0:
            mask_final[it, r_0_new[:,1], r_0_new[:,0]] = 1
        else:
            mask_final[it, r_x[it-1], r_y[it-1]] = 1

    return mask_final


def main_DENSE(general_params, sim_params, tissue_params, verbose=1, **kwargs):
    """
    Main function to run a DENSE simulation.
    :param general_params: general parameters for the simulation
    :param sim_params: simulation parameters for the DENSE simulation
    :param tissue_params: tissue parameters for the simulation
    :param verbose: logging level
    :returns: None
    """

    data_folder = os.path.join(general_params["cardiac_motion_folder"], general_params["subject_folder"])
    if general_params["output_folder"]:
        DENSE_data_folder = os.path.join(general_params["output_folder"], general_params["subject_folder"])
    else:
        DENSE_data_folder = os.path.join(data_folder)
    os.makedirs(DENSE_data_folder, exist_ok=True)

    imgs, U_disps, V_disps = load_data(os.path.join(data_folder))
    
    if general_params["dict_act"]:
        dict_act = general_params["dict_act"]
    else:
        dict_act = get_anat_info(os.path.join(data_folder, "sim_log_0"))
    dict_act_global = map_tissue_types(dict_act, list(tissue_params["T1"].keys()))

    t1_map, t2_map, s0_map = get_tissue_maps(imgs[0], dict_act_global, tissue_params)
    x_disps = np.stack([U_disps[i].transpose() for i in range(len(U_disps))])
    y_disps = np.stack([V_disps[i].transpose() for i in range(len(V_disps))])

    SNR = np.random.uniform(sim_params["SNR_range"][0], sim_params["SNR_range"][1])
    sim_params["SNR"] = SNR

    DENSE_sim_res_X = run_DENSE_sim(imgs, x_disps, y_disps, t1_map, t2_map, s0_map, **sim_params)
    DENSE_sim_res_Y = run_DENSE_sim(imgs, x_disps, y_disps, t1_map, t2_map, s0_map, ke_dir=[0.0,1.0,0.0], **sim_params)
    DENSE_sim_res_0 = run_DENSE_sim(imgs, x_disps, y_disps, t1_map, t2_map, s0_map, ke_dir=None, **sim_params)
    magnitude_imgs = (np.abs(DENSE_sim_res_X["imgs"]) + np.abs(DENSE_sim_res_Y["imgs"])) / 2
    phase_x_imgs = np.angle(DENSE_sim_res_X["imgs"])
    phase_y_imgs = np.angle(DENSE_sim_res_Y["imgs"])
    phase_0_imgs = np.angle(DENSE_sim_res_0["imgs"])

    phase_x_imgs = np.angle(np.exp(1j*phase_x_imgs)*np.exp(-1j*phase_0_imgs))
    phase_y_imgs = np.angle(np.exp(1j*phase_y_imgs)*np.exp(-1j*phase_0_imgs))

    np.save(os.path.join(DENSE_data_folder, "magnitude.npy"), magnitude_imgs)
    np.save(os.path.join(DENSE_data_folder, "phase_x.npy"), phase_x_imgs)
    np.save(os.path.join(DENSE_data_folder, "phase_y.npy"), phase_y_imgs)

    LV_mask = (np.load(os.path.join(data_folder, "LV_mask_time.npy")) == general_params["LV_label"])
    LV_mask_DENSE = compute_lag_masks(LV_mask[0], U_disps, V_disps, magnitude_imgs.shape[-1])
    if sim_params["Nt"] < 40:
        LV_mask_DENSE = np.concatenate([LV_mask_DENSE[0:1], LV_mask_DENSE[1::2][:sim_params["Nt"]]], axis=0)
    else:
        LV_mask_DENSE = LV_mask_DENSE[:sim_params["Nt"]+1]

    np.save(os.path.join(DENSE_data_folder, "LV_mask_time_lowres.npy"), LV_mask_DENSE.astype(bool))

    config = {"sim_params": sim_params}
    with open(os.path.join(DENSE_data_folder, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    pixel_spacing = sim_params["FOV"][0] / sim_params["N_im"]
    template_dns_file = general_params["template_dns_file"]
    out_dns_file = os.path.join(DENSE_data_folder, "workspace.dns")
    save_dns_data(
        magnitude_imgs, phase_x_imgs, phase_y_imgs, LV_mask_DENSE[1:],
        template_dns_file, out_dns_file, 
        sim_params["ke"], pixel_spacing, general_params["slice_cat"]
    )