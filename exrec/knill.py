import numpy as np
import qutip as qt
import itertools
from constants import *

from noises import KnillNoise

import time


class KnillEC:

    def __init__(self, data, anc, meas_data, meas_anc,
                 gamma=1e-3, gamma_phi=1e-3, loss_in=None, dephasing_in=None, recovery=MAXIMUM_LIKELIHOOD):

        self.data = data
        self.anc = anc
        self.meas_data = meas_data
        self.meas_anc = meas_anc

        self.decoding_scheme = recovery
        self.mod = 2 * self.data.N
        self.nmeas_data = len(self.meas_data.povm_elements)
        self.nmeas_anc = len(self.meas_anc.povm_elements)

        # print("==INITIALIZE KNILL NOISE...")
        # st = time.time()
        self.noise = KnillNoise(gamma, gamma_phi, loss_in, dephasing_in, self.data, self.mod)
        # print("==DONE INITIALIZE KNILL NOISE...", time.time() - st)

        # apply dephasing noise due to waiting
        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)

    def run(self, gmat=None):
        """
        :param gmat: input coefficient matrix, for F_k and E_k0 and correction of X
        gmat[k,k0;a,b;i,j,m,n]
        """
        self.noise.update_zero_noise(gmat)  # update noise before update gmat
        # if gmat is None:
        #     gmat = np.zeros((1, 1, 2, 2, 2, 2, 2, 2), dtype=complex)
        #     gmat[0, 0, 0, 0, 0, 0, 0, 0] = 1
        cmat = self.measure(gmat)  # c(k3,k4;x1,x2;i,j,m,n)
        return cmat

    def measure(self, gmat):
        # print("====Calculate amat...")
        # st = time.time()
        # amat[k,k0,k1;x1;i,j,m,n]
        amat = np.zeros((self.mod, self.noise.nkraus_zero, self.noise.nkraus_in, self.nmeas_data, 2, 2), dtype=complex)
        for ka in range(self.mod):
            for k0 in range(self.noise.nkraus_zero):
                for k1 in range(self.noise.nkraus_in):
                    ep = []
                    op = self.noise.phase[ka] * self.noise.loss_in.kraus[k1] * self.noise.loss_zero.kraus[k0]
                    for m in self.meas_data.povm_elements:
                        for keti in [self.data.plus, self.data.minus]:
                            for ketj in [self.data.plus, self.data.minus]:
                                ep.append(qt.expect(m, op * keti * ketj.dag() * op.dag()))
                    amat[ka, k0, k1, :, :, :] = np.reshape(np.array(ep), amat[ka, k0, k1, :, :, :].shape)
        # print("====Done calculating amat...", time.time() - st)

        # print("====Calculate amatx...")
        # st = time.time()
        amatx = np.zeros((self.noise.nphase_zero, self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus,
                          self.noise.nkraus, self.nmeas_data, 2, 2), dtype=complex)
        for k in range(self.noise.nphase_zero):
            for k2 in range(self.noise.nkraus):
                for k3 in range(self.noise.nkraus):
                    amatx[k, :, :, k2, k3, :, :, :] += amat[(k + k2 + k3) % self.mod, :, :, :, :, :]
        # print("====Done calculating amatx...", time.time() - st)

        # print("====Calculate bmat...")
        # st = time.time()
        bmat = np.zeros((self.mod, self.noise.nkraus, self.noise.nkraus, self.nmeas_anc, 2, 2), dtype=complex)
        for kb in range(self.mod):
            for k2 in range(self.noise.nkraus):
                for k3 in range(self.noise.nkraus):
                    ep = []
                    op = self.noise.phase[kb] * self.noise.loss.kraus[k2] * self.noise.loss.kraus[k3]
                    for m in self.meas_anc.povm_elements:
                        for ketm in [self.anc.plus, self.anc.minus]:
                            for ketn in [self.anc.plus, self.anc.minus]:
                                ep.append(qt.expect(m, op * ketm * ketn.dag() * op.dag()))
                    bmat[kb, k2, k3, :, :, :] = np.reshape(np.array(ep), bmat[kb, k2, k3, :, :, :].shape)
        # print("====Done calculating bmat...", time.time() - st)

        # print("====Calculate bmatx...")
        # st = time.time()
        bmatx = np.zeros((self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus, self.noise.nkraus,
                          self.noise.nkraus, self.nmeas_anc, 2, 2), dtype=complex)
        for k0 in range(self.noise.nkraus_zero):
            for k1 in range(self.noise.nkraus_in):
                for k4 in range(self.noise.nkraus):
                    bmatx[k0, k1, :, :, k4, :, :, :] += bmat[(k0 + k1 + k4) % self.mod, :, :, :, :, :]
        # print("====Done calculating bmatx...", time.time() - st)

        # print("====Calculate cmatr...")
        # st = time.time()
        cmatr = 0.25 * np.einsum('abcdezuv,bcdefwst->abefzwuvst', amatx, bmatx)
        if gmat is None:
            cmat = np.einsum('abefzwuvst->efzwuvst', cmatr)
        else:
            cmatr = np.einsum('abefzwuvst,abxyijmn->efxyzwijmnuvst', cmatr, gmat)
            # print("====Done calculating cmatr...", time.time() - st)

            # print("====Calculate cmat...")
            # st = time.time()
            # clear up Pauli indices
            cmat = np.zeros((self.noise.nkraus, self.noise.nkraus, self.nmeas_data, self.nmeas_anc, self.nmeas_data,
                             self.nmeas_anc, 2, 2, 2, 2), dtype=complex)
            idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
            for ijmn in idx_list:
                for uvst in idx_list:
                    idx = tuple(a ^ b for a, b in zip(ijmn, uvst))
                    sign = (ijmn[2] * uvst[0]) ^ (ijmn[3] * uvst[1])
                    cmat_idx = tuple([...] + list(idx))
                    cmatr_idx = tuple([...] + list(ijmn + uvst))
                    if sign == 0:
                        cmat[cmat_idx] += cmatr[cmatr_idx]
                    else:
                        cmat[cmat_idx] -= cmatr[cmatr_idx]
        # print("====Done calculating cmat...", time.time() - st)
        return cmat

    def recovery(self, cmat):
        if len(cmat.shape) == 8:
            return self.recovery_one_ec(cmat)
        elif len(cmat.shape) == 10:
            return self.recovery_two_ec(cmat)
        else:
            raise Exception("Invalid shape for cmat.")

    def recovery_one_ec(self, cmat):
        # print("====Calculate xymat...")
        # st = time.time()
        abmat = np.empty((self.nmeas_data, self.nmeas_anc), dtype=object)  # matrix stores decoding results, Z^a.X^b
        if self.decoding_scheme == DIRECT:
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    abmat[x1, x2] = (x1, x2)
        elif self.decoding_scheme == MAXIMUM_LIKELIHOOD:
            cmat_diag = np.zeros((self.noise.nkraus, self.noise.nkraus, self.nmeas_data, self.nmeas_anc, 2, 2),
                                 dtype=complex)
            for i in range(2):
                for j in range(2):
                    cmat_diag[..., i, j] += cmat[..., i, i, j, j]
            # need to check the argument here, because cmat_diag depends also on k3, k4.
            emat = np.einsum('abxyim,b->xyim', cmat_diag, self.noise.trace_loss)
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    idx = (x1, x2, Ellipsis)
                    (a, b) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                    abmat[x1, x2] = (a, b)
        # print("====Done calculate xymat...", time.time() - st)

        # print("====Calculate dmat...")
        # st = time.time()
        # clear up indices: dmat[k,k0;a,b;i,j,m,n]. Index a,b is used to keep track of X,Z recovery need to apply.
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k3 in range(self.noise.nkraus):
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for idx in idx_list:
                        a, b = abmat[x1, x2]
                        dmat_idx = tuple([k3 % self.mod, ..., a, b] + list(idx))
                        cmat_idx = tuple([k3, ..., x1, x2] + list(idx))
                        dmat[dmat_idx] += cmat[cmat_idx]
        # print("====Done calculate dmat...", time.time() - st)
        return dmat

    def recovery_two_ec(self, cmat):
        # print("====Calculate xymat...")
        # st = time.time()
        abmat = np.empty((self.nmeas_data, self.nmeas_anc, self.nmeas_data, self.nmeas_anc), dtype=object)
        if self.decoding_scheme == DIRECT:
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for x3 in range(self.nmeas_data):
                        for x4 in range(self.nmeas_anc):
                            abmat[x1, x2, x3, x4] = (x1 ^ x3, x2 ^ x4)
        elif self.decoding_scheme == MAXIMUM_LIKELIHOOD:
            cmat_diag = np.zeros(
                (self.noise.nkraus, self.noise.nkraus, self.nmeas_data, self.nmeas_anc, self.nmeas_data,
                 self.nmeas_anc, 2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    cmat_diag[..., i, j] += cmat[..., i, i, j, j]
            # need to check the argument here, because cmat_diag depends also on k3, k4.
            emat = np.einsum('abxyzwim,b->xyzwim', cmat_diag, self.noise.trace_loss)
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for x3 in range(self.nmeas_data):
                        for x4 in range(self.nmeas_anc):
                            idx = (x1, x2, x3, x4, Ellipsis)
                            (a, b) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                            abmat[x1, x2, x3, x4] = (a, b)
        # print("====Done calculate xymat...", time.time() - st)

        # print("====Calculate dmat...")
        # st = time.time()
        # clear up indices: dmat[k,k0;a,b;i,j,m,n]. Index a,b is used to keep track of X,Z recovery need to apply.
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k3 in range(self.noise.nkraus):
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for x3 in range(self.nmeas_data):
                        for x4 in range(self.nmeas_anc):
                            a, b = abmat[x1, x2, x3, x4]
                            dmat[k3 % self.mod, :, a, b, :, :, :, :] += cmat[k3, :, x1, x2, x3, x4, :, :, :, :]
        # print("====Done calculate dmat...", time.time() - st)
        return dmat

    def update_alpha(self, data=None, anc=None):
        if data is not None:
            self.data = data
            self.noise.update_alpha(data)
        if anc is not None:
            self.anc = anc

    def update_in_noise(self, loss_in, dephasing_in):
        self.noise.update_in_noise(loss_in, dephasing_in)
        self.meas_data.apply_dephasing(self.noise.dephasing_in)

    def update_measurement(self, meas_data, meas_anc):
        self.meas_data = meas_data
        self.meas_anc = meas_anc
        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)
