import time

import numpy as np
import qutip as qt
# import matlab.engine
import itertools

import channels


class DecoderException(Exception):
    pass


class SDPDecoder:
    """Find the optimal recovery channel using semi-definite programming."""

    def __init__(self, code, loss, phase):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.loss = loss
        self.phase = phase

        self.engine = matlab.engine.start_matlab()

    def __del__(self):
        self.engine.quit()

    def decode(self, dmat):
        # print("Making noise...")
        # st = time.time()
        noise = self.make_noise(dmat)
        # print("Done making noise...", time.time() - st)

        # print("Find recovery...")
        # st = time.time()
        recovery = self.find_recovery(noise)
        # print("Done find recovery...", time.time() - st)

        output_channel_matrix = recovery.channel_matrix * noise.channel_matrix
        output_channel = channels.Channel(channel_matrix=output_channel_matrix)

        return output_channel

    def find_recovery(self, noise):
        physdim = self.encoder.dims[0][0]
        codedim = self.encoder.dims[1][0]

        # print("PREPARE DECODER...")
        # st = time.time()
        C = (1. / codedim ** 2) * noise.channel_matrix.dag()
        C = qt.super_to_choi(C)
        C = matlab.double(C.data.toarray().tolist(), is_complex=True)
        # print('DONE PREPARE DECODER...', time.time() - st)

        # print("OPTIMIZING...")
        # st = time.time()
        X = self.engine.cvxsdp(C, physdim, codedim)
        X = np.array(X)
        # print("DONE OPTIMIZTING...", time.time() - st)

        # print("CLEAN UP...")
        # st = time.time()
        dims = [[[physdim], [codedim]], [[physdim], [codedim]]]
        choi = qt.Qobj(X, dims=dims, superrep='choi')
        recovery = channels.Channel(choi=choi)

        return recovery

    def make_noise(self, dmat):
        """
        Construct noise map from coefficient matrix dmat(k,k0;a,b;i,j,m,n).
        Assume noise of the form d(k,k0;a;i,j,m,n)F_k E_{k0} X^mZ^i(.)Z^jX^n E_{k0}^{dagger} F_k^{dagger}.
        K=1: F_k=I. K>1: F_k = e^{phi*k*n}. K0=1, E_k0=I. K0>1: E_{a,k0} = X^a E_k0 X^a.
        """

        K = dmat.shape[0]
        K0 = dmat.shape[1]
        physdim = self.encoder.dims[0][0]
        if K0 == 1:
            loss = channels.IdentityChannel(physdim)
        else:
            loss = self.loss

        phase = self.phase

        # jmat = np.sum(dmat, axis=(2, 3, 4, 5, 6, 7))  # sum over all logical errors

        # bosonic_noise is the noise to optimize. full_noise also includes logical errors.
        full_noise = qt.Qobj()
        # bosonic_noise = qt.Qobj()
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k in range(K):
            for k0 in range(K0):
                pre__ = phase[k] * loss.kraus[k0] * self.encoder
                post__ = self.encoder.dag() * loss.kraus[k0].dag() * phase[k].dag()
                # bosonic_noise += jmat[k, k0] * qt.sprepost(phase[k] * loss.kraus[k0] * self.encoder,
                #                                            self.encoder.dag() * loss.kraus[k0].dag() *
                #                                            phase[k].dag())
                for a in range(2):
                    for b in range(2):
                        pre_ = pre__
                        post_ = post__
                        if b == 1:
                            pre_ = self.code.logical_X_allspace * pre__
                            post_ = post__ * self.code.logical_X_allspace.dag()
                        if a == 1:
                            pre_ = self.code.logical_Z_allspace * pre__
                            post_ = post__ * self.code.logical_Z_allspace.dag()
                        for (i, j, m, n) in idx_list:
                            pre = pre_
                            post = post_
                            # if m == 1:
                            #     pre_ = pre__ * self.code.logical_X_allspace
                            # if i == 1:
                            #     pre_ = pre__ * self.code.logical_Z_allspace
                            # if n == 1:
                            #     post_ = self.code.logical_X_allspace.dag() * post__
                            # if j == 1:
                            #     post_ = self.code.logical_Z_allspace.dag() * post__

                            if m == 1:
                                pre = pre_ * qt.sigmax()
                            if i == 1:
                                pre = pre_ * qt.sigmaz()
                            if n == 1:
                                post = qt.sigmax() * post_
                            if j == 1:
                                post = qt.sigmaz() * post_

                            # pre = pre * self.encoder
                            # post = self.encoder.dag() * post
                            full_noise += dmat[k, k0, a, b, i, j, m, n] * qt.sprepost(pre, post)
        full_noise = channels.Channel(channel_matrix=full_noise)
        # bosonic_noise = channels.Channel(channel_matrix=bosonic_noise)
        # bosonic_noise.tp_check(silent=True)
        full_noise.tp_check(silent=True)

        return full_noise

    def update_code(self, code):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)


class LogicalFastDecoder:
    def __init__(self, code, loss):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.loss = loss

        # print("making recovery...")
        # st = time.time()
        self.recovery = self.make_recovery()
        # print("done making recovery...", time.time() - st)

    def make_recovery(self):
        # print("making norm op...")
        # st = time.time()
        norm = self.norm_op()
        # print("done norm op...", time.time() - st)
        N = self.code.N
        P = self.code.projector

        krs = []
        for k in range(N):
            krs.append(self.encoder.dag() * P * self.loss.kraus[k].dag() * norm)
        rev = channels.Channel(kraus=krs)
        # rev.tp_check(silent=True)
        return rev

    def norm_op(self):
        N = self.code.N
        P = self.code.projector
        eP = qt.Qobj(np.zeros(P.shape), dims=P.dims)
        for k in range(N):
            eP += self.loss.kraus[k] * P * self.loss.kraus[k].dag()
        norm = self.inverse_sqr(eP, safe=True)
        return norm

    @staticmethod
    def inverse_sqr(rho, safe=False):
        # Apply f(rho) to diagonalizable matrix rho
        abstol = 1e-8
        vals, vecs = rho.eigenstates()
        out = qt.Qobj(np.zeros(rho.shape), dims=rho.dims)
        for i, lam in enumerate(vals):
            if lam > abstol or not safe:
                out += (1 / np.sqrt(lam)) * vecs[i] * vecs[i].dag()
        return out

    def decode(self, dmat):
        """dmat(k,k0;a,b;i,j,m,n)"""
        K = dmat.shape[0]
        K0 = dmat.shape[1]

        emat = np.zeros((K0, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k in range(K):
            if 0 <= k < self.code.N:
                emat[...] += dmat[k, ...]
            else:
                for a in range(2):
                    emat[:, a, :, :, :, :, :] += dmat[k, :, 1 - a, :, :, :, :, :]

        noise = self.make_noise(emat)

        output_matrix = self.recovery.channel_matrix * noise.channel_matrix
        output_map = channels.Channel(channel_matrix=output_matrix)

        return output_map

    def make_noise(self, emat):
        """emat(k0;a,b;i,j,m,n)"""
        K0 = emat.shape[0]
        noise = qt.Qobj()
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k0 in range(K0):
            pre__ = self.loss.kraus[k0] * self.encoder
            post__ = self.encoder.dag() * self.loss.kraus[k0].dag()
            for a in range(2):
                for b in range(2):
                    pre_ = pre__
                    post_ = post__
                    if b == 1:
                        pre_ = self.code.logical_X_allspace * pre__
                        post_ = post__ * self.code.logical_X_allspace.dag()
                    if a == 1:
                        pre_ = self.code.logical_Z_allspace * pre__
                        post_ = post__ * self.code.logical_Z_allspace.dag()
                    for (i, j, m, n) in idx_list:
                        pre = pre_
                        post = post_
                        if m == 1:
                            pre = pre_ * qt.sigmax()
                        if i == 1:
                            pre = pre_ * qt.sigmaz()
                        if n == 1:
                            post = qt.sigmax() * post_
                        if j == 1:
                            post = qt.sigmaz() * post_
                        noise += emat[k0, a, b, i, j, m, n] * qt.sprepost(pre, post)
        noise = channels.Channel(channel_matrix=noise)
        noise.tp_check(silent=True)
        return noise

    def update_code(self, code):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.recovery = self.make_recovery()


class FastDecoder:
    def __init__(self, code, loss):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.loss = loss

        # print("making recovery...")
        # st = time.time()
        self.recovery = self.make_recovery()
        # print("done making recovery...", time.time() - st)
        self.pauli_x = channels.Channel(kraus=[qt.sigmax()])
        self.pauli_z = channels.Channel(kraus=[qt.sigmaz()])

    def make_recovery(self):
        # print("making norm op...")
        # st = time.time()
        norm = self.norm_op()
        # print("done norm op...", time.time() - st)
        N = self.code.N
        P = self.code.projector

        krs = []
        for k in range(N):
            krs.append(self.encoder.dag() * P * self.loss.kraus[k].dag() * norm)
        rev = channels.Channel(kraus=krs)
        # rev.tp_check(silent=True)
        return rev

    def norm_op(self):
        N = self.code.N
        P = self.code.projector
        eP = qt.Qobj(np.zeros(P.shape), dims=P.dims)
        for k in range(N):
            eP += self.loss.kraus[k] * P * self.loss.kraus[k].dag()
        norm = self.inverse_sqr(eP, safe=True)
        return norm

    @staticmethod
    def inverse_sqr(rho, safe=False):
        # Apply f(rho) to diagonalizable matrix rho
        abstol = 1e-8
        vals, vecs = rho.eigenstates()
        out = qt.Qobj(np.zeros(rho.shape), dims=rho.dims)
        for i, lam in enumerate(vals):
            if lam > abstol or not safe:
                out += (1 / np.sqrt(lam)) * vecs[i] * vecs[i].dag()
        return out

    def decode(self, dmat):
        """dmat(k,k0;a,b;i,j,m,n)"""
        K = dmat.shape[0]
        K0 = dmat.shape[1]

        emat = np.zeros((K0, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k in range(K):
            if 0 <= k < self.code.N:
                emat[...] += dmat[k, ...]
            else:
                for a in range(2):
                    emat[:, a, :, :, :, :, :] += dmat[k, :, 1 - a, :, :, :, :, :]

        output_matrix = qt.Qobj()
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k0 in range(K0):
            pre___ = self.loss.kraus[k0] * self.encoder
            post___ = self.encoder.dag() * self.loss.kraus[k0].dag()
            for kr in self.recovery.kraus:
                pre__ = kr * pre___
                post__ = post___ * kr.dag()
                for a in range(2):
                    for b in range(2):
                        pre_ = pre__
                        post_ = post__
                        if b == 1:
                            pre_ = qt.sigmax() * pre__
                            post_ = post__ * qt.sigmax()
                        if a == 1:
                            pre_ = qt.sigmaz() * pre__
                            post_ = post__ * qt.sigmaz()
                        for (i, j, m, n) in idx_list:
                            pre = pre_
                            post = post_
                            if m == 1:
                                pre = pre_ * qt.sigmax()
                            if i == 1:
                                pre = pre_ * qt.sigmaz()
                            if n == 1:
                                post = qt.sigmax() * post_
                            if j == 1:
                                post = qt.sigmaz() * post_
                            output_matrix += emat[k0, a, b, i, j, m, n] * qt.sprepost(pre, post)

        # output_matrix = self.recovery.channel_matrix * noise.channel_matrix
        output_map = channels.Channel(channel_matrix=output_matrix)
        output_map.tp_check(silent=True)

        return output_map

    def make_noise(self, emat):
        """emat(k0;a,b;i,j,m,n)"""
        K0 = emat.shape[0]
        noise = qt.Qobj()
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k0 in range(K0):
            pre__ = self.loss.kraus[k0] * self.encoder
            post__ = self.encoder.dag() * self.loss.kraus[k0].dag()
            for a in range(2):
                for b in range(2):
                    pre_ = pre__
                    post_ = post__
                    if b == 1:
                        pre_ = pre__ * qt.sigmax()
                        post_ = qt.sigmax() * post__
                    if a == 1:
                        pre_ = pre__ * qt.sigmaz()
                        post_ = qt.sigmaz() * post__
                    for (i, j, m, n) in idx_list:
                        pre = pre_
                        post = post_
                        if m == 1:
                            pre = pre_ * qt.sigmax()
                        if i == 1:
                            pre = pre_ * qt.sigmaz()
                        if n == 1:
                            post = qt.sigmax() * post_
                        if j == 1:
                            post = qt.sigmaz() * post_
                        noise += emat[k0, a, b, i, j, m, n] * qt.sprepost(pre, post)

        noise = channels.Channel(channel_matrix=noise)
        noise.tp_check(silent=True)
        return noise

    def update_code(self, code):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.recovery = self.make_recovery()


class FastDecoderPhase:
    def __init__(self, code, loss, phase):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.loss = loss
        self.phase = phase

        print("making recovery...")
        st = time.time()
        self.recovery = self.make_recovery()
        print("done making recovery...", time.time() - st)

    def make_recovery(self):
        print("making norm op...")
        st = time.time()
        norm = self.norm_op()
        print("done norm op...", time.time() - st)
        N = self.code.N
        P = self.code.projector

        krs = []
        for k in range(N):
            for ka in range(N):
                krs.append(self.encoder.dag() * P * self.loss.kraus[ka].dag() * self.phase[k].dag() * norm)
        rev = channels.Channel(kraus=krs)
        # rev.tp_check(silent=True)
        return rev

    def norm_op(self):
        N = self.code.N
        P = self.code.projector
        eP = qt.Qobj(np.zeros(P.shape), dims=P.dims)
        for k in range(N):
            for ka in range(N):
                eP += self.phase[k] * self.loss.kraus[ka] * P * self.loss.kraus[ka].dag() * self.phase[k].dag()
        norm = self.inverse_sqr(eP, safe=True)
        return norm

    @staticmethod
    def inverse_sqr(rho, safe=False):
        # Apply f(rho) to diagonalizable matrix rho
        abstol = 1e-8
        vals, vecs = rho.eigenstates()
        out = qt.Qobj(np.zeros(rho.shape), dims=rho.dims)
        for i, lam in enumerate(vals):
            if lam > abstol or not safe:
                out += (1 / np.sqrt(lam)) * vecs[i] * vecs[i].dag()
        return out

    def decode(self, dmat):
        """dmat(k,k0;a,b;i,j,m,n)"""

        # emat = np.zeros((K0, 2, 2, 2, 2, 2, 2), dtype=complex)
        # for k in range(K):
        #     if 0 <= k < self.code.N:
        #         emat[...] += dmat[k, ...]
        #     else:
        #         for a in range(2):
        #             emat[:, a, :, :, :, :, :] += dmat[k, :, 1 - a, :, :, :, :, :]

        noise = self.make_noise(dmat)

        output_matrix = self.recovery.channel_matrix * noise.channel_matrix
        output_map = channels.Channel(channel_matrix=output_matrix)

        return output_map

    def make_noise(self, dmat):
        """emat(k0;a,b;i,j,m,n)"""
        K = dmat.shape[0]
        K0 = dmat.shape[1]
        noise = qt.Qobj()
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k in range(K):
            for k0 in range(K0):
                pre__ = self.phase[k] * self.loss.kraus[k0] * self.encoder
                post__ = self.encoder.dag() * self.loss.kraus[k0].dag() * self.phase[k].dag()
                for a in range(2):
                    for b in range(2):
                        pre_ = pre__
                        post_ = post__
                        if b == 1:
                            pre_ = self.code.logical_X_allspace * pre__
                            post_ = post__ * self.code.logical_X_allspace.dag()
                        if a == 1:
                            pre_ = self.code.logical_Z_allspace * pre__
                            post_ = post__ * self.code.logical_Z_allspace.dag()
                        for (i, j, m, n) in idx_list:
                            pre = pre_
                            post = post_
                            if m == 1:
                                pre = pre_ * qt.sigmax()
                            if i == 1:
                                pre = pre_ * qt.sigmaz()
                            if n == 1:
                                post = qt.sigmax() * post_
                            if j == 1:
                                post = qt.sigmaz() * post_
                            noise += dmat[k, k0, a, b, i, j, m, n] * qt.sprepost(pre, post)
        noise = channels.Channel(channel_matrix=noise)
        noise.tp_check(silent=True)
        return noise

    def update_code(self, code):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.recovery = self.make_recovery()
