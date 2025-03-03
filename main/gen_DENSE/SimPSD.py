import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import pprint

from SimObject import SimObject

try:
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False

class PosTime:
    def __init__(self, sim_object, use_gpu = False):
        self.use_gpu = use_gpu
        self.tt = sim_object.tt.copy()
        self.period = sim_object.period
        self.dt = sim_object.dt

        if HAS_CUPY and self.use_gpu:
            self.r = cp.asarray(sim_object.r, cp.float32)
            self.pos = cp.zeros_like(self.r[0], cp.float32)
        else:
            self.r = sim_object.r.copy()
            self.pos = np.zeros_like(self.r[0])
    
    def calc_pos(self, p_time):
        if self.period is not None:
            p_time = p_time % self.period
        ## Only works for regular intervals
        # for i in range(self.tt.size):
        #     if self.tt[i] > p_time:
        #         lo = i - 1
        #         hi = i

        #         lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
        #         hi_mod = (p_time - self.tt[lo]) / self.dt
        #         break
        #     elif i == (self.tt.size - 1):
        #         lo = i
        #         hi = 0
        #         lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
        #         hi_mod = (p_time - self.tt[lo]) / self.dt

        ## Works for any kind of sampling
        for i in range(self.tt.size):
            if self.tt[i] >= p_time:
                lo = i - 1
                hi = i

                lo_mod = (self.tt[hi] - p_time) / (self.tt[hi] - self.tt[lo])
                hi_mod = (p_time - self.tt[lo]) / (self.tt[hi] - self.tt[lo])
                break
            elif i == (self.tt.size - 1):
                lo = i
                hi = 0
                lo_mod = (self.tt[hi] + self.period - p_time) / (self.tt[hi] + self.period - self.tt[lo])
                hi_mod = (p_time - self.tt[lo]) / (self.tt[hi] + self.period - self.tt[lo])

        self.pos[:] = self.r[lo] * lo_mod + self.r[hi] * hi_mod


class InstantRF:
    def __init__(self, dirvec=[1.0, 0.0, 0.0], flip=90, profile=None, use_gpu = False):
        self.type = "rf"
        self.use_gpu = use_gpu
        self.dirvec = np.array(dirvec).astype(np.float32)
        self.dirvec /= np.linalg.norm(self.dirvec)
        self.flip = flip
        self.profile = profile

        rads = flip * np.pi / 180.0
        rotvec = rads * self.dirvec

        self.rot = R.from_rotvec(rotvec).as_matrix()
        
        if HAS_CUPY and self.use_gpu:
            self.rot = cp.asarray(self.rot, dtype=cp.float32)

    def apply(self, M, sim_object, postime, t):
        if HAS_CUPY and self.use_gpu:
            xp = cp.get_array_module(M)
        else:
            xp = np
        
        postime.calc_pos(t)

        

        if not self.profile is None:
            posz = (postime.pos[:,2] + 0.5) * sim_object.fov[2]
            mask = xp.ones_like(M[:,0])
            mask = (posz>self.profile[0]) & (posz<self.profile[1])
            M[mask>0, :] = xp.matmul(self.rot, M[mask>0, :].T).T
        else:
            M[:] = xp.matmul(self.rot, M.T).T

        return None


class InstantGrad:
    def __init__(self, dirvec=[1, 0, 0], M0=11.74, use_gpu = False):
        self.type = "grad"
        self.use_gpu = use_gpu
        self.dirvec = np.array(dirvec).astype(np.float32)
        self.dirvec /= np.linalg.norm(self.dirvec)
        self.M0 = M0

        if HAS_CUPY and self.use_gpu:
            self.dirvec = cp.asarray(self.dirvec[None, :], dtype=cp.float32)
        

    def apply(self, M, sim_object, postime, t):
        if HAS_CUPY and self.use_gpu:
            xp = cp.get_array_module(M)
        else:
            xp = np
        
        postime.calc_pos(t)
        fov = xp.asarray(sim_object.fov[None, :], dtype=xp.float32)

        rr = (postime.pos * self.dirvec * fov).sum(1)
        theta = rr * self.M0 * 267.522

        # print([postime.pos.min(), postime.pos.max(), fov])
        # print([rr.min(), rr.max()])
        # print([theta.min(), theta.max()])

        M_new = xp.zeros_like(M)
        M_new[:, 0] = M[:, 0] * xp.cos(theta) - M[:, 1] * xp.sin(theta)
        M_new[:, 1] = M[:, 0] * xp.sin(theta) + M[:, 1] * xp.cos(theta)
        M_new[:, 2] = M[:, 2]

        M[:] = M_new

        return None


class InstantAcq:
    def __init__(self, use_gpu = False):
        self.type = "acq"
        self.use_gpu = use_gpu

    def apply(self, M, sim_object, postime, t):
        postime.calc_pos(t)
        
        if HAS_CUPY and self.use_gpu:
            return [cp.asnumpy(postime.pos), cp.asnumpy(M)]
        else:
            return [postime.pos.copy(), M.copy()]

class InstantSpoil:
    def __init__(self, use_gpu = False):
        self.type = "spoil"
        self.use_gpu = use_gpu

    def apply(self, M, sim_object, postime, t):
        M[:, :2] *= 0.0
        return None


class SimInstant:
    def __init__(self, sim_object, use_gpu = False, cu_device = 0, acq_rephase = True):
        self.psd = []
        self.sim_object = sim_object
        self.use_gpu = use_gpu
        self.acq_rephase = acq_rephase

        self.init_tag = False
        
        self.ss_params = {}

        self.Meq = np.zeros((sim_object.sig0.size, 3))
        self.Meq[:, 2] = sim_object.sig0.copy()

        self.M = self.Meq.copy()

        if HAS_CUPY and self.use_gpu:
            cp.cuda.Device(cu_device).use()  
            self.M = cp.asarray(self.M, dtype=cp.float32)
            self.Meq = cp.asarray(self.Meq, dtype=cp.float32)
            self.T1 = cp.asarray(self.sim_object.T1, dtype=cp.float32)
            self.T2 = cp.asarray(self.sim_object.T2, dtype=cp.float32)
        else:
            self.T1 = self.sim_object.T1.copy()
            self.T2 = self.sim_object.T2.copy()

        self.postime = PosTime(sim_object, use_gpu = self.use_gpu)

    def relaxation(self, dt):
        
        if HAS_CUPY and self.use_gpu:
            xp = cp.get_array_module(self.M)
        else:
            xp = np

        self.M[:,0] = self.M[:,0] * xp.exp(-dt / self.T2)
        self.M[:,1] = self.M[:,1] * xp.exp(-dt / self.T2)
        self.M[:,2] = self.Meq[:,2] - (self.Meq[:,2] - self.M[:,2]) * xp.exp(-dt / self.T1)

    def steady_state(self):
        
        if self.ss_params:

            flip = self.ss_params['flip'] 
            dt = self.ss_params['dt']
            Nss = self.ss_params['Nss']

            rf = InstantRF(flip=flip, use_gpu = self.use_gpu)
            spoil = InstantSpoil(use_gpu = self.use_gpu)
            
            for i in range(Nss):
                rf.apply(self.M, self.sim_object, self.postime, 0)
                self.relaxation(dt)
                spoil.apply(self.M, self.sim_object, self.postime, 0)
            # self.relaxation(dt)
        
    def run(self):
        acqs = []
        
        # if self.init_tag:
        #     rf = InstantRF(flip=90, use_gpu = self.use_gpu)
        #     spoil = InstantSpoil(use_gpu = self.use_gpu)
            
        #     rf.apply(self.M, self.sim_object, self.postime, 0)
        #     spoil.apply(self.M, self.sim_object, self.postime, 0)
        #     self.relaxation(1000.0)

        self.steady_state()

        if self.init_tag:
            self.relaxation(300.0)
        
        # pprint.pprint(self.psd)
        self.psd.sort(key=lambda x: x[1])  # Make sure PSD is sorted by timing
        # pprint.pprint(self.psd)

        current_time = 0
        for event in self.psd:
            dt = event[1] - current_time
            if dt > 0:
                self.relaxation(dt)
            current_time = event[1]

            out = event[0].apply(self.M, self.sim_object, self.postime, current_time)
            if out is not None:
                if self.acq_rephase:
                    out_new = np.zeros_like(out[1])
                    out_new[:,0] = out[1][:,1]
                    out_new[:,1] = -out[1][:,0]
                    out_new[:,2] = out[1][:,2] 
                    out[1][:] = out_new[:]
                acqs.append(out)

        return acqs

    def set_psd(self, psd, ss_params = None):
        self.psd = psd
        if ss_params is not None:
            self.ss_params = ss_params

    ########################
    #  PSDs
    ########################

    
    def sample_DENSE_PSD(self, rf_dir=(1.0, 0.0, 0.0), ke=0.1, ke_dir=(1.0, 0.0, 0.0), te=1.0, kd=0.08, flip=[15], acq_loc=[100], profile=None):
        M0_ke = 1e3 * 2 * ke * np.pi / 267.522  # [mT * ms / m]
        M0_kd = 1e3 * 2 * kd * np.pi / 267.522  # [mT * ms / m]

        self.psd.append((InstantRF(flip=90, use_gpu = self.use_gpu), 0))
        if ke_dir is not None:
            self.psd.append((InstantGrad(dirvec=ke_dir, M0=M0_ke, use_gpu = self.use_gpu), .01))
        self.psd.append((InstantGrad(dirvec=[0, 0, 1], M0=M0_kd, use_gpu = self.use_gpu), .02))
        self.psd.append((InstantRF(dirvec=rf_dir, flip=90, use_gpu = self.use_gpu), .03))
        self.psd.append((InstantSpoil(use_gpu = self.use_gpu), .04))

        if len(flip) == 1:
            flip = flip * len(acq_loc)

        for i, t_acq in enumerate(acq_loc):
            self.psd.append((InstantRF(flip=flip[i], use_gpu = self.use_gpu, profile=profile), t_acq))
            if ke_dir is not None:
                self.psd.append((InstantGrad(dirvec=ke_dir, M0=M0_ke, use_gpu = self.use_gpu), t_acq + .01))
            self.psd.append((InstantGrad(dirvec=[0, 0, 1], M0=M0_kd, use_gpu = self.use_gpu), t_acq + .02))
            self.psd.append((InstantAcq(use_gpu = self.use_gpu), t_acq + te))
            self.psd.append((InstantSpoil(use_gpu = self.use_gpu), t_acq + te + 0.01))


if __name__ == "__main__":
    print('Nothing in __main__ right now')
