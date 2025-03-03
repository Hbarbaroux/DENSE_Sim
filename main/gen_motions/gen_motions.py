import os
import cv2
import sys
import random
import numpy as np
from pathlib import Path
from scipy.signal import windows
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.ndimage import (
    distance_transform_edt, center_of_mass, 
    binary_closing, binary_dilation
)

CURRENT_FILE = Path(os.path.abspath(__file__))
sys.path.append(str(CURRENT_FILE.parent.parent.parent))
from main.utils import *

from perlin import generate_perlin_noise_2d


# TODO: Add control over te perlin noise, random amplitude to it
def gen_2Dpoly(NN=256, shift=True, fit_order = 3):
    """
    Generate a 2D polynomial field with random coefficients and perlin noise.
    :param NN: int, size of the field
    :param shift: bool, whether to shift the field to be centered around 0
    :param fit_order: int, order of the polynomial
    :return: np.array, 2D polynomial field
    """
    x = np.linspace(-1, 1, NN) + np.random.uniform(-0.5, 0.5)
    y = np.linspace(-1, 1, NN) + np.random.uniform(-0.5, 0.5)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing="xy")

    [px, py] = np.meshgrid(range(fit_order + 1), range(fit_order + 1), indexing="ij")

    idx = (px + py) <= fit_order

    px = px[idx]
    py = py[idx]
    powers = np.vstack((px, py)).T

    N = powers.shape[0]

    A = np.zeros((xv.size, N))

    for i in range(N):
        temp = xv ** px[i] + yv ** py[i]
        A[:, i] = temp.ravel()

    coeff = np.random.standard_normal(N)

    b = A @ coeff
    b = np.reshape(b, (NN, NN))

    perlin_res = np.random.choice([2,4], 2)
    nn = generate_perlin_noise_2d((NN,NN), perlin_res)

    # print(b.min(), b.max(), nn.min(), nn.max())

    b += 4.0*nn

    poly_range = b.max() - b.min()

    b /= poly_range

    if shift:
        b += 1 - b.max()

    return b

def draw_inverse_exp(n=1):
    # Get random samples from inverse exponential distribution
    # clipped between 1 and 2
    sample = np.random.exponential(0.43,n)
    return 2-(sample-(sample.astype(int)))


def blur_outer(im, mask, blur=4.0):
    """
    Given an image and a mask, blur the image outside the mask
    (Gaussian dilation)
    :param im: image to blur
    :param mask: mask defining the region to blur
    :param blur: blur factor
    :returns: blurred image
    """
    edt, inds = distance_transform_edt(mask < 0.5, return_indices=True)

    edt_g = np.exp(-(edt/blur)**2.0)
    im2 = edt_g * im[inds[0], inds[1]]
    
    im2[mask>0.5] = im[mask>0.5]
    return im2


def get_temporal_waveform(Nt):
    """
    Generate a temporal waveform. The waveform describes the relative displacement
    of the myocardium at each time point in time.
    :param Nt: number of time points
    :returns: temporal waveform (1D numpy array)
    """
    Nt2 = Nt + np.random.randint(10)

    mod_lim = [0.4, 0.9]
    skip_chance = 0.8
    sec_height = [0.0, 0.3]
    mod3_height = [0.0, 0.1]

    mod_max = np.random.uniform(mod_lim[0], mod_lim[1])

    mod_max = int(mod_max * Nt)
    mod = windows.cosine(mod_max)
    if np.random.rand() < skip_chance:
        mod = mod[1:]

    N2 = Nt2-mod.size
    height2 = np.random.uniform(sec_height[0], sec_height[1])
    mod2 = np.ones(N2) * height2

    height3 = np.random.uniform(mod3_height[0], mod3_height[1])
    mod3 = height3 * windows.hamming(N2)
    mod2 += mod3

    mod_upper = mod.copy()
    mod_upper[mod < height2] = height2
    mod[mod_max//2:] = mod_upper[mod_max//2:]
    mod[mod.size-1] = (mod[mod.size-2] + height2) / 1.9
    mod = np.hstack((0.0, mod, mod2))

    x_stop = np.linspace(np.pi, -np.random.rand()*np.pi, np.random.randint(3,7))
    y_stop = (np.tanh(x_stop) + 1)/2
    y_stop = np.hstack([y_stop, np.zeros(np.random.randint(3))])
    y_stop = np.hstack((np.ones(mod.size - y_stop.size), y_stop))

    mod *= y_stop
    mod += np.random.uniform(0.0, 0.03) * np.random.standard_normal(mod.size)
    mod[0] = 0
    
    t2 = np.cumsum(mod)
    t2 *= 2*np.pi/t2[-1]
    t2 = t2[:Nt]
    return t2


def gen_motion_params(NN=256, rseed=None, extra_poly=0):
    """
    Generate the underlying motion fields for the simulation. The motion fields
    are generated as a combination of two polynomial fields, each modulated by
    a different filter, and a rotation field. The two polynomial fields are generated as random polynomials
    with a filter applied to them. The extra_poly parameter controls the number
    of additional polynomial fields that are generated.
    :param NN: size of the motion fields
    :param rseed: random seed
    :param extra_poly: number of extra polynomial fields to generate
    :returns: r_a, r_b, theta, extra_p
    """
    
    if rseed is not None:
        np.random.seed(rseed)
    
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, NN, False), np.linspace(-1, 1, NN, False), indexing="ij"
    )
    rr = np.sqrt(xx * xx + yy * yy)

    a_range = (2, 8)
    b_range = (1.1, 1.5)

    beta = np.random.rand() * (a_range[1] - a_range[0]) + a_range[0]
    cutoff = np.random.rand() * (b_range[1] - b_range[0]) + b_range[0]

    filt = 0.5 + 1.0 / np.pi * np.arctan(beta * (cutoff - rr.ravel()) / cutoff)
    filt = np.reshape(filt, rr.shape)

    p0 = gen_2Dpoly(NN=NN)
    r_a = p0 * filt

    # The extra scale factor here controls how round the path is, with smaller numbers meaning less round
    p1 = gen_2Dpoly(NN=NN)
    r_b = np.random.rand() * p1 * filt

    theta = (
        np.random.rand() * ((gen_2Dpoly(NN=NN) * 2 * np.pi) - np.pi)
        + np.random.rand() * 2 * np.pi
    )

    extra_p = []
    for i in range(extra_poly):
        extra_p.append(gen_2Dpoly(NN=NN, shift=False) * filt)

    return r_a, r_b, theta, extra_p


def get_RV_wall_center(mask):
    """
    Get the center of the RV wall. The RV wall center is calculated as the midpoint
    between the two insertion points of the RV wall.
    :param mask: mask of the myocardium
    :returns: RV wall center coordinates
    """
    lv_mask = (mask == 1)
    rv_mask = (mask == 2)
    lv_center = np.array(center_of_mass(lv_mask))
    rv_center = np.array(center_of_mass(rv_mask))

    # Outer tensor of LV and RV coordinates differences
    diffs = np.stack([
        np.argwhere(lv_mask)[:,0].reshape(-1, 1) - np.argwhere(rv_mask)[:,0],
        np.argwhere(lv_mask)[:,1].reshape(-1, 1) - np.argwhere(rv_mask)[:,1]
    ], axis=-1)

    # Distance between LV and RV points
    norms = np.linalg.norm(diffs, axis=-1)

    # Find the LV points that are close to RV points
    array_lv_points = np.argwhere(lv_mask)[np.where(norms <= 1)[0]]

    # For each point, calculate the sin of the angle between the LV-RV vector and the LV-point vector
    # (this to understand which points are on which side of the LV-RV vector)
    sin_lv_points = []
    for point in array_lv_points:
        vector_1 = (point - lv_center) / np.linalg.norm(point - lv_center)
        vector_2 = (rv_center - lv_center) / np.linalg.norm(rv_center - lv_center)
        sin_lv_points.append(vector_1[0] * vector_2[1] - vector_2[0] * vector_1[1])
    sin_lv_points = np.array(sin_lv_points)

    # Get the two insertion points
    c0 = array_lv_points[sin_lv_points < 0].mean(axis=0)
    c1 = array_lv_points[sin_lv_points > 0].mean(axis=0)

    # Get the mid point between the two insertion points
    mid = (c0 + c1) / 2
    
    return mid


def load_mask(file):
    """
    Load a mask from a file.
    :param file: path to the file
    :returns: mask
    """
    mask = np.load(file)
    return mask


def augment_mask(mask, size, dilation=2):
    """
    Augment the mask to the desired size and apply dilation if needed.
    :param mask: mask to augment
    :param size: desired size
    :param dilation: number of dilation iterations
    :returns: augmented mask
    """
    if size and size != mask.shape[0]:
        gauss_mask = gaussian(mask, 1.0, preserve_range=True)
        resize_mask = resize(gauss_mask, (size,size), preserve_range=True)
        mask = np.round(resize_mask).astype(int)

    if dilation is not None:
        lv_mask = (mask == 1)
        rv_mask = (mask == 2)
        if rv_mask.sum() > 0:
            lv_mask = binary_dilation(lv_mask, iterations=dilation)
            rv_mask = binary_dilation(rv_mask, iterations=dilation)
            mask = lv_mask + 2*rv_mask
            mask[mask == 3] = 1 # This is to make sure that the LV mask is not overwritten by the RV mask
        else:
            mask = binary_dilation(lv_mask, iterations=dilation)

    return mask


def trans_mask_center(mask):
    """
    Translate the mask to the center of the frame. The simulation assumes that 
    the mask is centered in the frame.
    :param mask: mask to translate
    :returns: translated mask, frame center coordinates, mask center coordinates
    """
    mask_values = np.unique(mask)
    mask_values = mask_values[mask_values != 0] # No need to move the background
                                                # (avoid having to deal with wrapping value at the edges of the image)
    frame_center = np.array([mask.shape[0]/2, mask.shape[0]/2])
    mask_center = np.array(center_of_mass(mask == 1)) # mask sure to center around LV

    trans_mask = np.zeros_like(mask)
    for value in mask_values:
        new_indices = np.round(np.array(np.where(mask == value)) - mask_center[..., np.newaxis] + frame_center[..., np.newaxis]).astype(int)
        trans_mask[new_indices[0], new_indices[1]] = value

    return trans_mask, frame_center, mask_center


def trans_back_array_center(input_array, frame_center, mask_center):
    """
    Translate a coordinates array assuming LV being centered in the frame back to original position. 
    The simulation assumes that the mask is centered in the frame.
    :param input_array: array to translate
    :param frame_center: frame center coordinates
    :param mask_center: mask center coordinates
    :returns: translated array
    """
    trans_back_array = np.round(input_array + mask_center[np.newaxis] - frame_center[np.newaxis]).astype(int)
    return trans_back_array


def gen_deformation(mask, Nt=30, motion_scaler=1, motion_blur=1.0, twist_angle=0, radial_exponent=2, LV_only=True):
    """
    Generate through-time deformation fields for the myocardium. Apply a general contraction motion
    to the myocardium, with a twist angle and radial exponent to control the shape of the motion. Apply motion fields from 
    gen_motion_params to the myocardial mask, modulated by the temporal waveform.
    :param mask: mask of the myocardium
    :param Nt: number of time points
    :param motion_scaler: additional general scaling factor for the motion fields
    :param motion_blur: motion blur factor
    :param twist_angle: twist angle
    :param radial_exponent: radial exponent -> smaller values make the motion more constant radially, larger values make it more linear
    :param LV_only: set to True to only generate motion for the LV
    :returns: r0_lv_NN -> initial points in image coordinates for LV
    :returns: r0_NN -> initial points in image coordinates for LV and RV
    :returns: r -> final points through time in image coordinates
    :returns: dx -> x displacement through time (assuming image coordinates are in [-0.5,0.5])
    :returns: dy -> y displacement through time (assuming image coordinates are in [-0.5,0.5])
    :returns: nbr_LV_pts -> number of LV points
    :returns: xnew -> time points in ms
    :returns: r0_NN_float -> initial points in image coordinates for LV and RV in float format
    """
    NN = mask.shape[0]
    lv_mask = (mask == 1)
    rv_mask = (mask == 2)
    mesh_range = np.arange(NN)/NN - 0.5
    xx, yy = np.meshgrid(mesh_range, mesh_range, indexing = 'ij')
    im_coords = np.array([xx.ravel(), yy.ravel()])

    # This gives the radius from 1 (endocardium) to 0 (epicardium) for LV
    r0_lv = im_coords[:, lv_mask.ravel()>0].T
    lv_rad = np.hypot(r0_lv[:,0], r0_lv[:,1])
    lv_rad -= lv_rad.min()
    lv_rad /= lv_rad.max()
    lv_rad = 1.0 - lv_rad

    if not LV_only:
        mid_wall_center = get_RV_wall_center(mask)
        # This gives the radius from 1 (endocardium) to 0 (epicardium) for RV
        r0_rv = im_coords[:, rv_mask.ravel()>0].T
        rv_rad = np.hypot(r0_rv[:,0]-mid_wall_center[0], r0_rv[:,1]-mid_wall_center[1])
        rv_rad -= rv_rad.min()
        rv_rad /= rv_rad.max()
        rv_rad = 1.0 - rv_rad


    # Get all initial points together
    r0 = r0_lv if LV_only else np.vstack([r0_lv, r0_rv])
    init_rad = np.hypot(r0[:,0], r0[:,1])
    init_theta = np.arctan2(r0[:,1], r0[:,0])
    r0_NN = np.round((r0+0.5) * NN).astype(int) # r0 in image coords
    r0_lv_NN = r0_NN[:r0_lv.shape[0]]

    # Generate the motion parameters that define general contraction motion
    r_a = np.random.uniform(0.0, 0.006) * np.ones_like(init_rad)
    r_a[:r0_lv.shape[0]] += np.random.uniform(0.003, 0.008)*(lv_rad**radial_exponent)
    if not LV_only:
        r_a[r0_lv.shape[0]:] += np.random.uniform(0.04, 0.14)*(init_rad[r0_lv.shape[0]:]**2.0)*rv_rad
        r_a[r0_lv.shape[0]:] += np.random.uniform(0.003, 0.008)*rv_rad

    r_b = 0.75 * r_a * np.random.rand()

    r_a2, r_b2, theta2, extra_p2 = gen_motion_params(NN=NN, extra_poly=4)
    r_a2 = (r_a2 - r_a2.mean()) * np.random.uniform(.010, .030)
    r_b2 = (r_b2 - r_b2.mean()) * np.random.uniform(.005, .015)
    theta2 = (theta2 - theta2.mean()) * np.random.uniform(0.10, 0.20)

    # Add some random twist by changing the direction away from center of LV
    # theta_mod = np.random.rand() - 0.5
    theta_mod = twist_angle
    theta_c = init_theta + theta_mod

    # Generate the time curve for the motion
    t2 = get_temporal_waveform(Nt)
    filt = np.hstack([0, 0, np.diff(t2)])
    filt2 = np.hstack([0, 0, np.diff(t2)])
    filt = np.concatenate([filt, filt2])
    xorig = np.arange(0, len(filt)) * 15
    xnew = np.arange(1, len(filt)) * 15 + 7.5
    filtnew = np.interp(xnew, xorig, filt)
    filt = filtnew[:80, None, None]
    filt += .01 * np.random.standard_normal(filt.shape)
    cos_t2 = np.cos(t2)
    cos_t2 = np.concatenate([[1], cos_t2, [(cos_t2[-1]+1)/2], cos_t2])
    cos_t2 = np.interp(xnew, xorig, cos_t2)[:80]
    sin_t2 = np.sin(t2)
    sin_t2 = np.concatenate([[0], sin_t2, [(sin_t2[-1])/2], sin_t2])
    sin_t2 = np.interp(xnew, xorig, sin_t2)[:80]
    Nt = 80

    # Apply motion through time
    xx0 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)
    xx1 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)
    xx2 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)
    xx3 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)

    p0, p1 = extra_p2[0][np.newaxis], extra_p2[1][np.newaxis]
    xmod = (p0 * xx0**1.0 + p1 * xx1**2.0) * filt * np.random.uniform(0.015, 0.025)

    p2, p3 = extra_p2[2][np.newaxis], extra_p2[3][np.newaxis]
    ymod = (p2 * xx2**1.0 + p3 * xx3**2.0) * filt * np.random.uniform(0.015, 0.025)

    # Put existing motion fields into image versions
    mask_NN = np.zeros((NN,NN))
    mask_NN[r0_NN[:,0], r0_NN[:,1]] = 1.0

    r_a_NN = np.zeros((NN,NN))
    r_a_NN[r0_NN[:,0], r0_NN[:,1]] = r_a.copy()

    r_b_NN = np.zeros((NN,NN))
    r_b_NN[r0_NN[:,0], r0_NN[:,1]] = r_b.copy()

    theta_c_NN = np.zeros((NN,NN), complex)
    theta_c_NN[r0_NN[:,0], r0_NN[:,1]] = np.exp(1j*theta_c).copy()

    r_a_NN += r_a2 * mask_NN
    r_b_NN += r_b2 * mask_NN
    theta_c_NN.real += theta2 * mask_NN
    theta_c_NN.imag += theta2 * mask_NN

    mask_NN_b = gaussian(mask_NN, motion_blur, preserve_range=True) + 1e-16
    r_a_NN = gaussian(r_a_NN, motion_blur, preserve_range=True) / mask_NN_b
    r_b_NN = gaussian(r_b_NN, motion_blur, preserve_range=True) / mask_NN_b
    theta_c_NN.real = gaussian(theta_c_NN.real, motion_blur, preserve_range=True) / mask_NN_b
    theta_c_NN.imag = gaussian(theta_c_NN.imag, motion_blur, preserve_range=True) / mask_NN_b

    xmod_out = np.zeros_like(xmod)
    ymod_out = np.zeros_like(ymod)
    for it in range(Nt):
        xmod[it] *= mask_NN
        xmod[it] = gaussian(xmod[it], motion_blur, preserve_range=True) / mask_NN_b
        xmod[it] *= mask_NN
        xmod_out[it] = blur_outer(xmod[it], mask_NN)

        ymod[it] *= mask_NN
        ymod[it] = gaussian(ymod[it], motion_blur, preserve_range=True) / mask_NN_b
        ymod[it] *= mask_NN
        ymod_out[it] = blur_outer(ymod[it], mask_NN)

    r_a_NN *= mask_NN
    r_b_NN *= mask_NN
    theta_c_NN.real *= mask_NN
    theta_c_NN.imag *= mask_NN

    r_a_out = blur_outer(r_a_NN, mask_NN)
    r_b_out = blur_outer(r_b_NN, mask_NN)
    theta_c_out = np.zeros_like(theta_c_NN)
    theta_c_out.real = blur_outer(theta_c_NN.real, mask_NN)
    theta_c_out.imag = blur_outer(theta_c_NN.imag, mask_NN)

    scaler = motion_scaler #NN/512
    r_a_out *= scaler
    r_b_out *= scaler
    theta_c_out *= scaler
    xmod_out *= scaler
    ymod_out *= scaler

    r_a_ff = r_a_out[r0_NN[:,0], r0_NN[:,1]]
    r_b_ff = r_b_out[r0_NN[:,0], r0_NN[:,1]]
    theta_c_ff = np.angle(theta_c_out[r0_NN[:,0], r0_NN[:,1]])
    xmod_ff = xmod_out[:, r0_NN[:,0], r0_NN[:,1]]
    ymod_ff = ymod_out[:, r0_NN[:,0], r0_NN[:,1]]

    # Compute actual pointwise motion
    ell_x = r_a_ff[None, :] * (cos_t2 - 1.0)[:,None] 
    ell_y = r_b_ff[None, :] * sin_t2[:,None] 

    dx = np.cos(theta_c_ff)[None,:] * ell_x  - np.sin(theta_c_ff)[None,:] * ell_y + xmod_ff
    dy = np.sin(theta_c_ff)[None,:] * ell_x  + np.cos(theta_c_ff)[None,:] * ell_y + ymod_ff

    dx = np.concatenate([np.zeros((1, dx.shape[1])), dx], 0)
    dy = np.concatenate([np.zeros((1, dy.shape[1])), dy], 0)

    # Final point cloud motion paths for RV and LV
    r = r0[None,...] + np.stack((dx, dy),2)
    r = np.concatenate([ r, np.zeros_like(r[:,:,:1]) ], 2)

    nbr_LV_pts = r0_lv.shape[0]

    r0_NN_float = ((r0+0.5) * NN)

    return r0_lv_NN, r0_NN, r, dx, dy, nbr_LV_pts, xnew, r0_NN_float


def gen_image_from_motion(dx, dy, r0, r0_lv, r, NN, nbr_LV_points):
    Nt = dx.shape[0]
    U = np.zeros((Nt-1, NN, NN))
    V = np.zeros((Nt-1, NN, NN))
    mask = np.zeros((Nt, NN, NN))
    mask_lv = np.zeros((Nt, NN, NN))

    for it in range(Nt):
        if it != 0:
            U[it-1, r0[:,0], r0[:,1]] = dx[it] * NN
            V[it-1, r0[:,0], r0[:,1]] = dy[it] * NN

        # Using r0 to make sure the displacements
        # are only applied to the original points
        if it == 0:
            mask[it, r0[:,0], r0[:,1]] = 1.0
            mask_lv[it, r0[:nbr_LV_points,0], r0[:nbr_LV_points,1]] = 1.0
        else:
            mask[it, r[it][:,0], r[it][:,1]] = 1.0
            mask_lv[it, r[it][:nbr_LV_points,0], r[it][:nbr_LV_points,1]] = 1.0

    # mask_lv = np.zeros((NN, NN))
    # mask_lv[r0_lv[:,0], r0_lv[:,1]] = 1.0
    # print(mask_lv.shape)

    return U, V, mask, mask_lv


def post_process_mask(mask, closing=2):

    if closing:
        # Fill in the gaps in the myocardium due to rounding errors in the motion fields
        mask = np.stack([binary_closing(mask_i, iterations=closing) for mask_i in mask])

    # Fill in the mask to get blood_pool and myocardium labels
    mask_ = np.stack([cv2.floodFill(mask_i.astype(np.uint8), None, (0,0), 2)[1] for mask_i in mask])

    # Relabel the ROIs for consistency with XCAT (blood_pool = 5, myocardium = 1)
    final_mask = np.zeros(mask_.shape)
    final_mask[mask_ == 0] = 5
    final_mask[mask_ == 1] = 1

    return final_mask.astype(int)


def save_data(U, V, mask_time, mask_0, orig_mask, output_folder, subject_name):
    """
    Save the motion fields and the mask to disk.
    :param U: x displacement field
    :param V: y displacement field
    :param mask_time: mask through time
    :param mask_0: mask at time 0
    :param orig_mask: original mask
    :param output_folder: output folder
    :param subject_name: subject name
    """

    os.makedirs(os.path.join(output_folder, subject_name), exist_ok=True)

    # Save the motion fields and the mask
    np.save(os.path.join(output_folder, subject_name, "tissues_seg_time.npy"), mask_time.astype(np.int8))
    np.save(os.path.join(output_folder, subject_name, "dx_pixel.npy"), V.transpose(0,2,1).astype(np.float32))
    np.save(os.path.join(output_folder, subject_name, "dy_pixel.npy"), U.transpose(0,2,1).astype(np.float32))
    np.save(os.path.join(output_folder, subject_name, "LV_mask_time.npy"), mask_0.astype(np.int8))
    np.save(os.path.join(output_folder, subject_name, "ED_mask.npy"), orig_mask.astype(np.int8))


def main(params, verbose, **kwargs):
    
    config = {}
    nbr_fails = 0 # The generation can fail if the mask is too small

    if verbose:
        print("\nRunning motion simulation...\n")

    ###################
    # Load parameters #
    ###################

    if params["subject_name_rule"] == None:
        subject_prefix = "subject_"
        existing_subjects = glob.glob(os.path.join(params["output_folder"], subject_prefix + "*"))
        if len(existing_subjects) == 0:
            subject_start_index = 1
            if verbose:
                print("No pre-existing subjects found, simulations will start a new dataset from scratch.\n")
        else:
            subject_indices = sorted([int(subject.split("_")[-1]) for subject in existing_subjects])
            subject_start_index = subject_indices[-1] + 1
            if verbose:
                print("Found {} existing subject(s), simulations will expand this dataset.\n".format(len(existing_subjects)))
    else:
        raise NotImplementedError("Custom subject name rules are not yet implemented.")
    
    bin_dilation = None
    if params["bin_dilation"]:
        if type(params["bin_dilation"]) == int:
            bin_dilation = params["bin_dilation"]
        elif type(params["bin_dilation"]) == list and len(params["bin_dilation"]) == 2:
            bin_dilation = np.random.randint(params["bin_dilation"][0], params["bin_dilation"][1]+1)

    motion_scaler = 1.2
    if params["motion_scaler"]:
        if type(params["motion_scaler"]) in [float, int]:
            motion_scaler = params["motion_scaler"]
        elif type(params["motion_scaler"]) == list and len(params["motion_scaler"]) == 2:
            motion_scaler = np.random.uniform(params["motion_scaler"][0], params["motion_scaler"][1])

    twist_angle = 0
    if params["twist_angle"]:
        if type(params["twist_angle"]) in [float, int]:
            twist_angle = params["twist_angle"]
        elif type(params["twist_angle"]) == list and len(params["twist_angle"]) == 2:
            twist_angle = np.random.uniform(params["twist_angle"][0], params["twist_angle"][1])

    Nt = 66
    if params["Nt"]:
        if type(params["Nt"]) == int:
            Nt = params["Nt"]
        elif type(params["Nt"]) == list and len(params["Nt"]) == 2:
            Nt = np.random.randint(params["Nt"][0], params["Nt"][1]+1)

    if not params["radial_exponent"]:
        params["radial_exponent"] = float(draw_inverse_exp())


    if params["subjects"]:
        subject_paths = [os.path.join(params["input_folder"], sub) for sub in params["subjects"]]
    else:
        subject_paths = sorted(glob.glob(os.path.join(params["input_folder"], "*")))
    n_orig = len(subject_paths)

    if params["number"]:
        subject_paths = random.choices(subject_paths, k=params["number"])

    input_subjects = [os.path.basename(sub) for sub in subject_paths]
    files = [os.path.join(sub, "mask.npy") for sub in subject_paths]

    if verbose:
        print("Starting {} motion simulation(s) from {} shape(s), mode {}...".format(len(files), n_orig, params["mode"]))

    for i, file in enumerate(files):
        
        mask = load_mask(file)
        mask = augment_mask(mask, params["size"], bin_dilation)
        NN = params["size"]
        
        trans_mask, frame_center, mask_center = trans_mask_center(mask)

        trans_r0_lv, trans_r0, trans_r, dx, dy, nbr_LV_pts, tt, trans_r0_NN_float = gen_deformation(
            trans_mask, 
            Nt=Nt, 
            motion_scaler=motion_scaler,
            twist_angle=twist_angle,
            radial_exponent=params["radial_exponent"],
            LV_only=True
        )
        r0 = trans_back_array_center(trans_r0, frame_center, mask_center)
        r0_lv = trans_back_array_center(trans_r0_lv, frame_center, mask_center)
        r0_NN_float = trans_r0_NN_float + mask_center[np.newaxis] - frame_center[np.newaxis]

        center_diff = np.zeros((trans_r.shape[0], 1, 3))
        center_diff[:,:,:2] = mask_center[np.newaxis] - frame_center[np.newaxis]
        r = np.round((trans_r+0.5) * NN + center_diff).astype(int)

        U, V, mask_time, mask_lv_0 = gen_image_from_motion(dx, dy, r0, r0_lv, r, NN, nbr_LV_pts)
        final_mask = post_process_mask(mask_time)
        mask_lv_0 = post_process_mask(mask_lv_0)

        if (final_mask == 5).sum() == 0:
            print("Simulation {}/{} failed: No blood pool found in mask, skipping simulation.".format(
                i+1, len(files)
            ))
            nbr_fails += 1
            continue

        if params["subject_name_rule"] is None:
            subject_name = subject_prefix + str(subject_start_index + i)
        else:
            raise NotImplementedError("Custom subject name rules are not yet implemented.")
        
        save_data(U, V, final_mask, mask_lv_0, mask, params["output_folder"], subject_name)
        if ("debug" in params.keys()) and (params["debug"] == True):
            np.save(os.path.join(params["output_folder"], subject_name, "r0_NN_float.npy"), r0_NN_float)

        config["from"] = input_subjects[i]
        config["bin_dilation"] = bin_dilation
        config["subject_name"] = subject_name
        config["twist_angle"] = twist_angle
        config["size"] = params["size"]
        config["motion_scaler"] = motion_scaler
        #config["phantom_params"] = phantom_params
        config["radial_exponent"] = params["radial_exponent"]
        config["Nt"] = Nt
        config["tt"] = [0] + tt.astype(float)[:80].tolist()
        # print(config["tt"])

        with open(os.path.join(params["output_folder"], subject_name, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        if verbose:
            print("Simulation {}/{} completed: '{}' from '{}' contour".format(
                i+1, params["number"], subject_name, input_subjects[i]
            ))

    number = params["number"] if params["number"] else len(files)
    if verbose:
        print("\nMotion simulation complete. {} out {} successfully generated.\n".format(number - nbr_fails, number))


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
