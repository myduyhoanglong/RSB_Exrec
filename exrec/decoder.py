import numpy as np
import qutip as qt
import itertools
import channels


class DecoderException(Exception):
    pass


class SDPDecoder:
    """
    Find the optimal recovery channel using semi-definite programming. Noise channel has the form
    N(.) = sum_{k,k0,a,b,ijmn} d(k,k0;a,b;i,j,m,n) Z^aX^bF_k E_{k0} X^mZ^i(.)Z^jX^n E_{k0}^{dagger} F_k^{dagger}X^bZ^a
    F_k = e^{-i*(pi*k/(N*N))n}: propagated phase
    E_{k0}: loss kraus
    """

    def __init__(self, code, loss, phase):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.loss = loss
        self.phase = phase

        import matlab.engine
        self.engine = matlab.engine.start_matlab()

    def __del__(self):
        self.engine.quit()

    def decode(self, dmat):
        """Combines recovery and noise map to return the final channel."""
        noise = self.make_noise(dmat)
        recovery = self.find_recovery(noise)
        output_channel_matrix = recovery.channel_matrix * noise.channel_matrix
        output_channel = channels.Channel(channel_matrix=output_channel_matrix)
        return output_channel

    def find_recovery(self, noise):
        """
        Semi-definite program to find the optimal recovery.
        Args:
            noise: Channel
                Input noise channel.
        Returns:
            recovery: Channel
        """
        physdim = self.encoder.dims[0][0]
        codedim = self.encoder.dims[1][0]

        import matlab
        C = (1. / codedim ** 2) * noise.channel_matrix.dag()
        C = qt.super_to_choi(C)
        C = matlab.double(C.data.toarray().tolist(), is_complex=True)

        X = self.engine.cvxsdp(C, physdim, codedim)
        X = np.array(X)

        dims = [[[physdim], [codedim]], [[physdim], [codedim]]]
        choi = qt.Qobj(X, dims=dims, superrep='choi')
        recovery = channels.Channel(choi=choi)

        return recovery

    def make_noise(self, dmat):
        """
        Construct a noise map from coefficient matrix. Noise has the form
        N(.) = sum_{k,k0,a,b,ijmn} d(k,k0;a,b;i,j,m,n) Z^aX^bF_k E_{k0} X^mZ^i(.)Z^jX^n E_{k0}^{dagger} F_k^{dagger}X^bZ^a
        Args:
            dmat: ndarray
                Noise matrix dmat(k,k0;a,b;i,j,m,n)
        Returns:
            full_noise: Channel
        """
        K = dmat.shape[0]
        K0 = dmat.shape[1]
        physdim = self.encoder.dims[0][0]
        if K0 == 1:
            loss = channels.IdentityChannel(physdim)
        else:
            loss = self.loss
        phase = self.phase

        full_noise = qt.Qobj()
        idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
        for k in range(K):
            for k0 in range(K0):
                pre__ = phase[k] * loss.kraus[k0] * self.encoder
                post__ = self.encoder.dag() * loss.kraus[k0].dag() * phase[k].dag()
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

                            full_noise += dmat[k, k0, a, b, i, j, m, n] * qt.sprepost(pre, post)
        full_noise = channels.Channel(channel_matrix=full_noise)
        full_noise.tp_check(silent=True)

        return full_noise

    def update_code(self, code):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)


class TransposeChannelDecoder:
    """
    Recovery by transpose channel. The code space is span by logical cat codewords with degree of rotation N.
    Correctable errors: F_k, E_k with k=0,1,...,N-1
    """

    def __init__(self, code, loss):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.loss = loss

        self.recovery = self.make_recovery()
        self.pauli_x = channels.Channel(kraus=[qt.sigmax()])
        self.pauli_z = channels.Channel(kraus=[qt.sigmaz()])

    def make_recovery(self):
        N = self.code.N
        P = self.code.projector
        norm = self.norm_op()

        krs = []
        for k in range(N):
            krs.append(self.encoder.dag() * P * self.loss.kraus[k].dag() * norm)
        rev = channels.Channel(kraus=krs)

        return rev

    def norm_op(self):
        N = self.code.N
        P = self.code.projector
        eP = qt.Qobj(np.zeros(P.shape), dims=P.dims)
        for k in range(N):
            eP += self.loss.kraus[k] * P * self.loss.kraus[k].dag()
        norm = self.inverse_sqr(eP)
        return norm

    @staticmethod
    def inverse_sqr(X):
        """Computes X^{-1/2} for a positive matrix X."""
        abstol = 1e-8
        vals, vecs = X.eigenstates()
        out = qt.Qobj(np.zeros(X.shape), dims=X.dims)
        for i, lam in enumerate(vals):
            if lam > abstol:
                out += (1 / np.sqrt(lam)) * vecs[i] * vecs[i].dag()
        return out

    def decode(self, dmat):
        """
        Applies recovery to noise channel defined by matrix dmat. Logical recoveries from ECs are applied virtually,
        after decoding the information.
        Args:
            dmat: ndarray
                Noise matrix dmat(k,k0;a,b;i,j,m,n)
        Returns:
            output_map: Channel
        """
        K = dmat.shape[0]
        K0 = dmat.shape[1]

        # recovery for rotation errors
        emat = np.zeros((K0, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k in range(K):
            if 0 <= k < self.code.N:
                emat[...] += dmat[k, ...]
            else:
                for a in range(2):
                    emat[:, a, :, :, :, :, :] += dmat[k, :, 1 - a, :, :, :, :, :]

        # recovery for loss errors
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

        output_map = channels.Channel(channel_matrix=output_matrix)
        output_map.tp_check(silent=True)

        return output_map

    def update_code(self, code):
        self.code = code
        self.encoder = self.code.encoder(kraus=True)
        self.recovery = self.make_recovery()
