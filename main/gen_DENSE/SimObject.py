import hdf5storage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import h5py

import sys, os

from pygrid_internal import pygrid as pygrid
from pygrid_internal import c_grid as c_grid
from pygrid_internal import utils as pygrid_utils

class SimObject:
    def __init__(self):
        self.Nt = 24  # Number of timeframes in the dataset
        self.period = 1000  # periodic phantom interval in [ms]
        self.dt = self.period / self.Nt
        self.tt = np.arange(self.Nt) * self.dt
        self.fov = np.array([200, 200, 8.0]) * 1.0e-3

    def gen_from_generator(self, r, s, t1, t2, Nz=4, periodic=True, period=None, tt=None, FOV=None):
        zz = np.linspace(-0.5, 0.5, Nz)

        if periodic:
            self.period = period if period is not None else 1000 # periodic phantom interval in [ms]
        else:
            self.period = None

        if FOV is not None: self.fov = FOV

        self.r = r
        self.sig0 = s
        self.T1 = t1
        self.T2 = t2

        self.Nt = self.r.shape[0]  # Number of timeframes in the dataset
        self.dt = self.period / self.Nt if self.period is not None else 1000 / self.Nt
        self.tt = tt if tt is not None else np.arange(self.Nt) * self.dt


    def shift_positions(self, dt):
        r_new = np.zeros_like(self.r)
        for i in range(self.r.shape[0]):
            t0 = self.tt[i]
            t1 = t0 + dt
            r_new[i] = self.get_pos_time(t1)
        self.r = r_new

    def get_pos_time(self, p_time):
        p_time = p_time % self.period
        for i in range(self.tt.size):
            if self.tt[i] > p_time:
                lo = i - 1
                hi = i

                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt
                break
            elif i == (self.tt.size - 1):
                lo = i
                hi = 0
                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt

        pos = self.r[lo] * lo_mod + self.r[hi] * hi_mod
        return pos

    def get_pos_time_r(self, p_time, r_in):
        p_time = p_time % self.period
        for i in range(self.tt.size):
            if self.tt[i] > p_time:
                lo = i - 1
                hi = i

                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt
                break
            elif i == (self.tt.size - 1):
                lo = i
                hi = 0
                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt

        pos = r_in[lo] * lo_mod + r_in[hi] * hi_mod
        return pos


    def grid_im_from_M(self, pos, M, N_im=256, w=64, oversamp=4.0, krad=1.5, nthreads = 0, use_gpu = False, dens = None):        
        gridder = pygrid.Gridder(
            imsize=(N_im, N_im), grid_dims=2, over_samp=oversamp, krad=krad, use_gpu=use_gpu
        )
        # print("test", N_im)

        kx_all = pos[:, 0].astype(np.float32)
        ky_all = pos[:, 1].astype(np.float32)
        # print(ky_all.shape)
        if dens is None:
            dens = np.ones_like(kx_all)
        else:
            dens = dens.astype(np.float32)
        # print(dens.shape)

        traj = np.stack((kx_all, ky_all, np.zeros_like(ky_all)), 1).astype(np.float32)
        # print(traj.shape)
        
        MM = M[:, 0] + 1j * M[:, 1]
        # print(MM.shape)

        out = None
        if use_gpu:
            out, kdata = gridder.cu_k2im(MM.astype(np.complex64), traj, dens, imspace=True)
        else:
            out, kdata = gridder.k2im(MM.astype(np.complex64), traj, dens, imspace=True)
        # print(out.shape, kdata.shape)

        return out, kdata


if __name__ == "__main__":
    print('Nothing in __main__ right now')
