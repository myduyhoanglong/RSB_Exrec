import numpy as np
import qutip as qt
import itertools
from constants import *

from noises import HybridNoise


class HybridEC:
    def __init__(self, data=None, anc=None, meas_data=None, meas_anc=None,
                 gamma=1e-3, gamma_phi=1e-3, loss_in=None, dephasing_in=None, recovery=MAXIMUM_LIKELIHOOD):
        self.data = data
        self.anc = anc

        self.meas_data = meas_data
        self.meas_anc = meas_anc

        self.decoding_scheme = recovery
        self.mod = 2 * self.data.N
        self.mod_anc = self.data.N
        self.nmeas_data = len(self.meas_data.povm_elements)
        self.nmeas_anc = len(self.meas_anc.povm_elements)

        self.noise = HybridNoise(gamma, gamma_phi, loss_in, dephasing_in, self.data.N, self.mod, self.mod_anc)

        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)

    def run(self, gmat=None):
        """
        :param gmat: input coefficient matrix, for F_k and E_k0
        """
        self.noise.update_zero_noise(gmat)  # update noise before update gmat
        # if gmat is None:
        #     gmat = np.zeros((1, 1, 2, 2, 2, 2), dtype=complex)
        #     gmat[0, 0, 0, 0, 0, 0] = 1
        cmat = self.measure(gmat)  # c(k0,k1,k2,k3;x1,x2;i,j,m,n).
        return cmat

    def measure(self, gmat):
        amat = np.zeros((self.mod_anc, self.nmeas_anc), dtype=complex)
        for ka in range(self.mod_anc):
            state = self.anc.plus * self.anc.plus.dag()
            op = self.noise.phase_anc[ka]
            ep = [qt.expect(m, op * state * op.dag()) for m in self.meas_anc.povm_elements]
            amat[ka, :] = np.reshape(np.array(ep), amat[ka, :].shape)

        amatx = np.zeros((self.noise.nkraus_zero, self.noise.nkraus_in, self.nmeas_anc), dtype=complex)
        for k0 in range(self.noise.nkraus_zero):
            for k1 in range(self.noise.nkraus_in):
                amatx[k0, k1, :] += amat[(k0 + k1) % self.mod_anc, :]

        bmat = np.zeros(
            (self.mod, self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus, self.nmeas_data, 2, 2),
            dtype=complex)
        for kb in range(self.mod):
            for k0 in range(self.noise.nkraus_zero):
                for k1 in range(self.noise.nkraus_in):
                    for k2 in range(self.noise.nkraus):
                        ep = []
                        op = self.noise.phase[kb] * self.noise.loss.kraus[k2] * self.noise.loss_in.kraus[k1] * \
                             self.noise.loss_zero.kraus[k0]
                        for m in self.meas_data.povm_elements:
                            for keti in [self.data.plus, self.data.minus]:
                                for ketj in [self.data.plus, self.data.minus]:
                                    ep.append(qt.expect(m, op * keti * ketj.dag() * op.dag()))
                        bmat[kb, k0, k1, k2, :, :, :] = np.reshape(np.array(ep), bmat[kb, k0, k1, k2, :, :, :].shape)

        bmatx = np.zeros(
            (self.noise.nphase_zero, self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus, self.noise.nkraus,
             self.nmeas_data, 2, 2), dtype=complex)
        for k in range(self.noise.nphase_zero):
            for k3 in range(self.noise.nkraus):
                bmatx[k, :, :, :, k3, :, :, :] += bmat[(k + k3) % self.mod, :, :, :, :, :, :]
        # print(bmatx.shape, np.sum(bmatx))

        cmatr = 0.5 * np.einsum('bcz,abcdewmn->abcdezwmn', amatx, bmatx)
        if gmat is None:
            cmat = np.einsum('abcdezwmn->cdezwmn', cmatr)
            if self.noise.nkraus_in == 1:
                cmat = np.einsum('cdezwmn->dezwmn', cmat)
        else:
            cmat = np.einsum('abcdezwmn,abxyij->bcdexyzwijmn', cmatr, gmat)

        return cmat

    def recovery(self, cmat):
        if len(cmat.shape) == 6:
            return self.recovery_one_ec(cmat)
        if len(cmat.shape) == 7:
            return self.recovery_one_ec_with_noise_in(cmat)
        elif len(cmat.shape) == 12:
            return self.recovery_two_ec(cmat)
        else:
            raise Exception("Invalid shape for cmat.")

    def recovery_one_ec(self, cmat):
        lamat = np.empty((self.nmeas_anc, self.nmeas_data), dtype=object)
        if self.decoding_scheme == DIRECT:
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    lamat[x1, x2] = (x1, x2)
        elif self.decoding_scheme == MAXIMUM_LIKELIHOOD:
            cmat_diag = np.zeros((self.mod, self.noise.nkraus, self.nmeas_anc, self.nmeas_data, 2), dtype=complex)
            for k2 in range(self.noise.nkraus):
                for m in range(2):
                    cmat_diag[k2 % self.mod, :, :, :, m] += cmat[k2, :, :, :, m, m]

            emat = np.einsum('kbxym,b->xykm', cmat_diag, self.noise.trace_loss)
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    idx = (x1, x2, Ellipsis)
                    (l, a) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                    lamat[x1, x2] = (l, a)

        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k2 in range(self.noise.nkraus):
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    (l, a) = lamat[x1, x2]
                    k = (k2 - l) % self.mod
                    dmat[k, :, 0, a, 0, 0, :, :] += cmat[k2, :, x1, x2, :, :]

        return dmat

    def recovery_one_ec_with_noise_in(self, cmat):
        lamat = np.empty((self.nmeas_anc, self.nmeas_data), dtype=object)
        if self.decoding_scheme == DIRECT:
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    lamat[x1, x2] = (x1, x2)
        elif self.decoding_scheme == MAXIMUM_LIKELIHOOD:
            cmat_diag = np.zeros((self.mod, self.noise.nkraus, self.nmeas_anc, self.nmeas_data, 2), dtype=complex)
            for k1 in range(self.noise.nkraus_in):
                for k2 in range(self.noise.nkraus):
                    k = (k1 + k2) % self.mod
                    for m in range(2):
                        cmat_diag[k, :, :, :, m] += cmat[k1, k2, :, :, :, m, m]

            emat = np.einsum('kbxym,b->xykm', cmat_diag, self.noise.trace_loss)
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    idx = (x1, x2, Ellipsis)
                    (l, a) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                    lamat[x1, x2] = (l, a)

        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k1 in range(self.noise.nkraus_in):
            for k2 in range(self.noise.nkraus):
                for x1 in range(self.nmeas_anc):
                    for x2 in range(self.nmeas_data):
                        (l, a) = lamat[x1, x2]
                        k = (k1 + k2 - l) % self.mod
                        dmat[k, :, 0, a, 0, 0, :, :] += cmat[k1, k2, :, x1, x2, :, :]

        return dmat

    def recovery_two_ec(self, cmat):
        labmat = np.empty((self.nmeas_anc, self.nmeas_data, self.nmeas_anc, self.nmeas_data), dtype=object)
        if self.decoding_scheme == DIRECT:
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    for x3 in range(self.nmeas_anc):
                        for x4 in range(self.nmeas_data):
                            labmat[x1, x2, x3, x4] = ((x1 + x3) % self.mod, x2, x4)
        elif self.decoding_scheme == MAXIMUM_LIKELIHOOD:
            cmat_diag = np.zeros(
                (self.mod, self.noise.nkraus, self.nmeas_anc, self.nmeas_data, self.nmeas_anc, self.nmeas_data, 2, 2),
                dtype=complex)
            for k0 in range(self.noise.nkraus_zero):
                for k1 in range(self.noise.nkraus_in):
                    for k2 in range(self.noise.nkraus):
                        k = (k0 + k1 + k2) % self.mod
                        for i in range(2):
                            for m in range(2):
                                cmat_diag[k, :, :, :, :, :, i, m] += cmat[k0, k1, k2, :, :, :, :, :, i, i, m, m]

            emat = np.einsum('kbxyzwim,b->xyzwkim', cmat_diag, self.noise.trace_loss)  # check this
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    for x3 in range(self.nmeas_anc):
                        for x4 in range(self.nmeas_data):
                            idx = (x1, x2, x3, x4, Ellipsis)
                            (l, a, b) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                            labmat[x1, x2, x3, x4] = (l, a, b)

        cmat_simp = np.zeros(
            (self.mod, self.noise.nkraus, self.nmeas_anc, self.nmeas_data, self.nmeas_anc, self.nmeas_data, 2, 2, 2, 2),
            dtype=complex)
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k0 in range(self.noise.nkraus_zero):
            for k1 in range(self.noise.nkraus_in):
                for k2 in range(self.noise.nkraus):
                    k = (k0 + k1 + k2) % self.mod
                    cmat_simp[k, ...] += cmat[k0, k1, k2, ...]
        for kp in range(self.mod):
            for x1 in range(self.nmeas_anc):
                for x2 in range(self.nmeas_data):
                    for x3 in range(self.nmeas_anc):
                        for x4 in range(self.nmeas_data):
                            l, a, b = labmat[x1, x2, x3, x4]
                            k = (kp - l) % self.mod
                            dmat[k, :, a, b, :, :, :, :] += cmat_simp[kp, :, x1, x2, x3, x4, :, :, :, :]
        # for k0 in range(self.noise.nkraus_zero):
        #     for k1 in range(self.noise.nkraus_in):
        #         for k2 in range(self.noise.nkraus):
        #             for x1 in range(self.nmeas_anc):
        #                 for x2 in range(self.nmeas_data):
        #                     for x3 in range(self.nmeas_anc):
        #                         for x4 in range(self.nmeas_data):
        #                             l, a, b = labmat[x1, x2, x3, x4]
        #                             k = (k0 + k1 + k2 - l) % self.mod
        #                             dmat[k, :, a, b, :, :, :, :] += cmat[k0, k1, k2, :, x1, x2, x3, x4, :, :, :, :]
        # print(dmat.shape, np.sum(dmat))

        return dmat

    def update_alpha(self, data=None, anc=None):
        if data is not None:
            self.data = data
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
