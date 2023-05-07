import numpy as np
import qutip as qt
import itertools
from constants import *

from noises import KnillNoise


class KnillEC:
    """
    Knill EC contains data mode, ancilla mode, and noise profile.
    Functions: Run the circuit without recovery, run the circuit with recovery, update code amplitude, update waiting
    noise (if any), update measurement offsets.
    """

    def __init__(self, data, anc, meas_data, meas_anc,
                 gamma=1e-3, gamma_phi=1e-3, loss_in=None, dephasing_in=None, recovery=MAXIMUM_LIKELIHOOD):
        """
        Initialize a KnillEC instance.
        Args:
            data: CatCode
                Data mode.
            anc: CatCode
                Ancilla mode.
            meas_data: LogicalMeasurement
                Data mode's measurement.
            meas_anc: LogicalMeasurement
                Ancilla mode's measurement.
            gamma: int
                Gate loss.
            gamma_phi: int
                Gate dephasing.
            loss_in: LossChannel
                Waiting loss.
            dephasing_in: DephasingChannel
                Waiting dephasing.
            recovery: int
                DIRECT or MAXIMUM_LIKELIHOOD.
        """

        self.data = data
        self.anc = anc
        self.meas_data = meas_data
        self.meas_anc = meas_anc

        self.recover_scheme = recovery
        self.mod = 2 * self.data.N
        self.nmeas_data = len(self.meas_data.povm_elements)
        self.nmeas_anc = len(self.meas_anc.povm_elements)

        self.noise = KnillNoise(gamma, gamma_phi, loss_in, dephasing_in, self.data.N, self.mod)

        # apply fixed noise to measurements
        self.update_measurement(meas_data, meas_anc)

    def run(self, gmat=None):
        """For Knill EC, running the EC only includes measure, recovery can be done at the end of extended gadget."""
        cmat = self.measure(gmat)
        return cmat

    def measure(self, gmat):
        """
        Returns a matrix describing the output channel, without recovery.
        Args:
            gmat: ndarray
                Input matrix describes noise from other EC, if not None.
                Indices: gmat(k,k0;x1,x2;i,j,m,n).
                k: index of propagated phase F_k = e^{-i*(pi*k/(N*N))n}.
                k0: index of loss E_k0.
                x1,x2: measurement outcomes.
                i,j,m,n: Logical operators. X^mZ^i(.)Z^jX^n.

        Returns:
            cmat: ndarray
                Output matrix describes the output channel.
                Indices: leading EC: cmat(k3,k4;x1,x2;i,j,m,n)
                         trailing EC: cmat(k3,k4;x1,x2,x3,x4;i,j,m,n)
                k3: index of propagated phase F_k3 = e^{-i*(pi*k3/(N*N))n}.
                k4: index of loss E_k4.
                x1,x2: measurement outcomes of leading EC.
                x3,x4: measurement outcomes of trailing EC.
                i,j,m,n: Logical operators. X^mZ^i(.)Z^jX^n.
        """

        # check if any noise from other EC
        self.noise.update_zero_noise(gmat)

        # Calculate compact measurement matrix for data mode: amat(ka,k0,k1;x1;i,j)
        # ka: generic propagated phase, ranging from 0 to N-1
        # k0: loss from other EC. k1: waiting loss. x1: measurement outcome. i,j: Logical Paulis Z^i(.)Z^j
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

        # Full measurement matrix for data mode, with phase indices unpacked: amatx(k,k0,k1,k2,k3;x1;i,j)
        # k,k2,k3: propagated phase from other EC, ancilla preparation, and ancilla CZ
        # k0: loss from other EC. k1: waiting loss. x1: measurement outcome. i,j: Logical Paulis Z^i(.)Z^j
        amatx = np.zeros((self.noise.nphase_zero, self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus,
                          self.noise.nkraus, self.nmeas_data, 2, 2), dtype=complex)
        for k in range(self.noise.nphase_zero):
            for k2 in range(self.noise.nkraus):
                for k3 in range(self.noise.nkraus):
                    amatx[k, :, :, k2, k3, :, :, :] += amat[(k + k2 + k3) % self.mod, :, :, :, :, :]

        # Calculate compact measurement matrix for ancilla mode: bmat(kb,k2,k3;x2;m,n)
        # kb: generic propagated phase, ranging from 0 to N-1
        # k2: loss in CZ. k3: loss in preparation. x2: measurement outcome. m,n: Logical Paulis X^m(.)X^n
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

        # Full measurement matrix for ancilla mode, with phase indices unpacked: bmatx(k0,k1,k2,k3,k4;x2;m,n)
        # k0,k1,k4: propagated phase from loss of other EC, waiting loss, and output mode preparation
        # k2: loss in CZ. k3: loss in preparation. x2: measurement outcome. m,n: Logical Paulis X^m(.)X^n
        bmatx = np.zeros((self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus, self.noise.nkraus,
                          self.noise.nkraus, self.nmeas_anc, 2, 2), dtype=complex)
        for k0 in range(self.noise.nkraus_zero):
            for k1 in range(self.noise.nkraus_in):
                for k4 in range(self.noise.nkraus):
                    bmatx[k0, k1, :, :, k4, :, :, :] += bmat[(k0 + k1 + k4) % self.mod, :, :, :, :, :]

        # sum over k1 (waiting loss) and k2 (ancilla CZ loss)
        # cmatr(k,k0,k3,k4;x1,x2;i,j,m,n) = (1/4)*sum_{k1,k2} amatx(k,k0,k1,k2,k3;x1;i,j)*bmatx(k0,k1,k2,k3,k4;x2;m,n)
        cmatr = 0.25 * np.einsum('abcdezuv,bcdefwst->abefzwuvst', amatx, bmatx)

        # sum over k,k0 (noise from other EC)
        if gmat is None:
            # no noise from other EC
            cmat = np.einsum('abefzwuvst->efzwuvst', cmatr)
        else:
            # noise from other EC gmat(k,k0;x1,x2;i,j,m,n)
            # cmats(k3,k4;x1,x2,x3,x4;ijmn,uvst) = sum_{k,k0}gmat(k,k0;x1,x2;ijmn)*cmatr(k,k0,k3,k4;x3,x4;uvst)
            cmats = np.einsum('abefzwuvst,abxyijmn->efxyzwijmnuvst', cmatr, gmat)

            # combine Pauli indices (ijmn) and (uvst) -> cmat(k3,k4;x1,x2,x3,x4;ijmn)
            cmat = np.zeros((self.noise.nkraus, self.noise.nkraus, self.nmeas_data, self.nmeas_anc, self.nmeas_data,
                             self.nmeas_anc, 2, 2, 2, 2), dtype=complex)
            idx_list = list(itertools.product(*list(itertools.repeat([0, 1], 4))))
            for ijmn in idx_list:
                for uvst in idx_list:
                    idx = tuple(a ^ b for a, b in zip(ijmn, uvst))

                    # sign = (-1)^(mu+nv)
                    sign = (ijmn[2] * uvst[0]) ^ (ijmn[3] * uvst[1])

                    cmat_idx = tuple([...] + list(idx))
                    cmats_idx = tuple([...] + list(ijmn + uvst))
                    if sign == 0:
                        cmat[cmat_idx] += cmats[cmats_idx]
                    else:
                        cmat[cmat_idx] -= cmats[cmats_idx]
        return cmat

    def recovery(self, cmat):
        if len(cmat.shape) == 8:
            return self.recovery_one_ec(cmat)
        elif len(cmat.shape) == 10:
            return self.recovery_two_ec(cmat)
        else:
            raise Exception("Invalid shape for cmat.")

    def recovery_one_ec(self, cmat):
        """
        Applies recovery after running one EC (either leading or trailing EC).
        Args:
            cmat: ndarray
                Measurement matrix cmat(k3,k4;x1,x2;i,j,m,n)

        Returns:
            dmat: nparray
                Matrix describes output map after recovery dmat(k,k0;a,b;ijmn)
                k: rotation phase, ranging from 0 to N-1
                k0: Kraus loss E_{k0}
                a,b: Z^aX^b recovery
        """

        # matrix stores decoding results, Z^a.X^b
        abmat = np.empty((self.nmeas_data, self.nmeas_anc), dtype=object)
        if self.recover_scheme == DIRECT:
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    abmat[x1, x2] = (x1, x2)
        elif self.recover_scheme == MAXIMUM_LIKELIHOOD:
            # cmat_diag(k3,k4;x1,x2;im) = cmat(k3,k4;x1,x2;iimm)
            cmat_diag = np.zeros((self.noise.nkraus, self.noise.nkraus, self.nmeas_data, self.nmeas_anc, 2, 2),
                                 dtype=complex)
            for i in range(2):
                for m in range(2):
                    cmat_diag[..., i, m] += cmat[..., i, i, m, m]

            # sum over k3,k4 to remove dependence on loss and phase. Weighted sum for kraus loss k4
            # need to check the argument here, because weights depend not only on noise but also state
            emat = np.einsum('abxyim,b->xyim', cmat_diag, self.noise.trace_loss)
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    idx = (x1, x2, Ellipsis)
                    (a, b) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                    abmat[x1, x2] = (a, b)

        # clear up indices -> dmat[k,k0;a,b;i,j,m,n]. Indices a,b keep track of X,Z recovery need to apply.
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k3 in range(self.noise.nkraus):
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    a, b = abmat[x1, x2]
                    dmat[k3 % self.mod, :, a, b, :, :, :, :] += cmat[k3, :, x1, x2, :, :, :, :]
        return dmat

    def recovery_two_ec(self, cmat):
        """
        Applies recovery after running both leading and trailing EC.
        Args:
            cmat: ndarray
                Measurement matrix cmat(k3,k4;x1,x2,x3,x4;i,j,m,n)

        Returns:
            dmat: nparray
                Matrix describes output map after recovery dmat(k,k0;a,b;ijmn)
                k: rotation phase, ranging from 0 to N-1
                k0: Kraus loss E_{k0}
                a,b: Z^aX^b recovery
        """

        # matrix stores decoding results, Z^a.X^b
        abmat = np.empty((self.nmeas_data, self.nmeas_anc, self.nmeas_data, self.nmeas_anc), dtype=object)
        if self.recover_scheme == DIRECT:
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for x3 in range(self.nmeas_data):
                        for x4 in range(self.nmeas_anc):
                            abmat[x1, x2, x3, x4] = (x1 ^ x3, x2 ^ x4)
        elif self.recover_scheme == MAXIMUM_LIKELIHOOD:
            cmat_diag = np.zeros(
                (self.noise.nkraus, self.noise.nkraus, self.nmeas_data, self.nmeas_anc, self.nmeas_data,
                 self.nmeas_anc, 2, 2), dtype=complex)
            for i in range(2):
                for m in range(2):
                    cmat_diag[..., i, m] += cmat[..., i, i, m, m]

            # sum over k3,k4 to remove dependence on loss and phase. Weighted sum for kraus loss k4
            # need to check the argument here, because weights depend not only on noise but also state
            emat = np.einsum('abxyzwim,b->xyzwim', cmat_diag, self.noise.trace_loss)
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for x3 in range(self.nmeas_data):
                        for x4 in range(self.nmeas_anc):
                            idx = (x1, x2, x3, x4, Ellipsis)
                            (a, b) = np.unravel_index(np.argmax(emat[idx]), emat[idx].shape)
                            abmat[x1, x2, x3, x4] = (a, b)

        # clear up indices -> dmat[k,k0;a,b;i,j,m,n]. Indices a,b keep track of X,Z recovery need to apply.
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for k3 in range(self.noise.nkraus):
            for x1 in range(self.nmeas_data):
                for x2 in range(self.nmeas_anc):
                    for x3 in range(self.nmeas_data):
                        for x4 in range(self.nmeas_anc):
                            a, b = abmat[x1, x2, x3, x4]
                            dmat[k3 % self.mod, :, a, b, :, :, :, :] += cmat[k3, :, x1, x2, x3, x4, :, :, :, :]
        return dmat

    def update_alpha(self, data=None, anc=None):
        if data is not None:
            self.data = data
        if anc is not None:
            self.anc = anc

    def update_alpha_proj_meas(self, data, anc, meas_data, meas_anc):
        if data is not None:
            self.data = data
        if anc is not None:
            self.anc = anc
        self.meas_data.noisy([self.noise.loss_meas, self.noise.dephasing_meas])
        self.meas_anc.noisy([self.noise.loss_meas, self.noise.dephasing_meas])
        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)

    def update_in_noise(self, loss_in, dephasing_in):
        """Updates waiting noise"""
        self.noise.update_in_noise(loss_in, dephasing_in)
        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)

    def update_measurement(self, meas_data, meas_anc):
        """Updates measurement offsets"""
        self.meas_data = meas_data
        self.meas_anc = meas_anc
        self.meas_data.noisy([self.noise.loss_meas, self.noise.dephasing_meas])
        self.meas_anc.noisy([self.noise.loss_meas, self.noise.dephasing_meas])
        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)
