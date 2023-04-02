import numpy as np
import qutip as qt
from constants import *

from noises import HybridNoise


class HybridEC:
    """
    Knill EC contains data mode, ancilla mode, and noise profile.
    Assumptions: ancilla mode has M=1 and alpha=ALPHA_MAX.
    Functions: Run the circuit without recovery, run the circuit with recovery, update code amplitude, update waiting
    noise (if any), update measurement offsets.
    """

    def __init__(self, data=None, anc=None, meas_data=None, meas_anc=None,
                 gamma=1e-3, gamma_phi=1e-3, loss_in=None, dephasing_in=None, recovery=MAXIMUM_LIKELIHOOD):
        """
        Initialize a HybridEC instance.
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

        self.recovery_scheme = recovery
        self.mod = 2 * self.data.N
        self.mod_anc = self.data.N
        self.nmeas_data = len(self.meas_data.povm_elements)
        self.nmeas_anc = len(self.meas_anc.povm_elements)

        self.noise = HybridNoise(gamma, gamma_phi, loss_in, dephasing_in, self.data.N, self.mod, self.mod_anc)

        # apply fixed noise to measurements
        self.update_measurement(meas_data, meas_anc)

    def run(self, gmat=None):
        """For Hybrid EC, running includes measure and partial recovery. Rotation of the output mode needs to be taken
        into account for measurement in the next EC."""
        cmatm = self.measure(gmat)
        cmat = self.partial_recovery(cmatm)
        return cmat

    def measure(self, gmat):
        """
        Returns a matrix describing the output channel, without recovery.
        Args:
            gmat: ndarray
                Input matrix describes noise from other EC, if not None.
                Indices: gmat(k,k0;x;i,j).
                k: index of propagated phase F_k = e^{-i*(pi*k/(N*N))n}.
                k0: index of loss E_k0.
                x: measurement outcome of logical X. Outcome of phase measurement should be already used for partial
                   recovery.
                i,j: Logical operators. X^iH(.)HX^j.

        Returns:
            cmat: ndarray
                Output matrix describes the output channel.
                Indices: leading EC: cmat(k2,k3;x1,x2;i,j)
                         trailing EC: cmat(k0,k1,k2,k3;x2,x3,x4;ijmn)
                k3: index of propagated phase F_k3 = e^{-i*(pi*k3/(N*N))n}.
                k4: index of loss E_k4.
                x2: measurement outcome of leading EC.
                x3,x4: measurement outcomes of trailing EC.
                i,j,m,n: Logical operators. X^mZ^i(.)Z^jX^n.
        """

        # check if any noise from other EC
        self.noise.update_zero_noise(gmat)

        # Calculate compact measurement matrix for ancilla mode: amat(ka;x1)
        # ka: generic propagated phase thorugh CROT, ranging from 0 to N-1
        # x1: measurement outcome
        amat = np.zeros((self.mod_anc, self.nmeas_anc), dtype=complex)
        for ka in range(self.mod_anc):
            state = self.anc.plus * self.anc.plus.dag()
            op = self.noise.phase_anc[ka]
            ep = [qt.expect(m, op * state * op.dag()) for m in self.meas_anc.povm_elements]
            amat[ka, :] = np.reshape(np.array(ep), amat[ka, :].shape)

        # Full measurement matrix for ancilla mode, with phase indices unpacked: amatx(k0,k1;x1)
        # k0,k1: propagated phase from loss of other EC and waiting loss
        amatx = np.zeros((self.noise.nkraus_zero, self.noise.nkraus_in, self.nmeas_anc), dtype=complex)
        for k0 in range(self.noise.nkraus_zero):
            for k1 in range(self.noise.nkraus_in):
                amatx[k0, k1, :] += amat[(k0 + k1) % self.mod_anc, :]

        # Calculate compact measurement matrix for data mode: bmat(kb,k0,k1,k2;x2;i,j)
        # kb: generic propagated phase, ranging from 0 to N-1
        # k0: loss from other EC. k1: waiting loss. k2: CZ loss
        # x1: measurement outcome. i,j: Logical Paulis X^iH(.)HX^j
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

        # Full measurement matrix for ancilla mode, with phase indices unpacked: bmatx(k,k0,k1,k2,k3;x2;i,j)
        # k,k3: propagated phase from other EC and output mode preparation
        # k0: loss from other EC. k1: waiting loss. k2: CZ loss
        # x1: measurement outcome. i,j: Logical Paulis X^iH(.)HX^j
        bmatx = np.zeros(
            (self.noise.nphase_zero, self.noise.nkraus_zero, self.noise.nkraus_in, self.noise.nkraus, self.noise.nkraus,
             self.nmeas_data, 2, 2), dtype=complex)
        for k in range(self.noise.nphase_zero):
            for k3 in range(self.noise.nkraus):
                bmatx[k, :, :, :, k3, :, :, :] += bmat[(k + k3) % self.mod, :, :, :, :, :, :]

        # combine two coefficients
        # cmatr(k,k0,k1,k2;x1,x2;ij) = 0.5*amatx(k0,k1;x1)*bmatx(k,k0,k1,k2,k3;x2;i,j)
        cmatr = 0.5 * np.einsum('bcz,abcdewmn->abcdezwmn', amatx, bmatx)

        # sum over k,k0 (noise from other EC)
        if gmat is None:
            cmat = np.einsum('abcdezwmn->cdezwmn', cmatr)
            # sum over k1 if no waiting noise
            if self.noise.nkraus_in == 1:
                cmat = np.einsum('cdezwmn->dezwmn', cmat)
        else:
            # cmat(k0,k1,k2,k3;x2,x3,x4;ijmn) = sum_{k} gmat(k,k0;x2;ij) * cmatr(k,k0,k1,k2;x3,x4;mn)
            cmat = np.einsum('abcdezwmn,abyij->bcdeyzwijmn', cmatr, gmat)

        return cmat

    def partial_recovery(self, cmatr):
        if len(cmatr.shape) == 6 or len(cmatr.shape) == 7:
            cmat = self.partial_recovery_one_ec(cmatr)
        elif len(cmatr.shape) == 11:
            cmat = self.partial_recovery_two_ec(cmatr)
        else:
            raise Exception("Partial recovery: Invalid shape for cmat.")
        return cmat

    def partial_recovery_one_ec(self, cmatr):
        """
        Recovery for phase rotation (that's why partial), leaving logical Pauli recovery till the end.
        Args:
            cmatr: ndarray
                Measurement matrix cmat(k2,k3;x1,x2;i,j) or cmat(k1,k2,k3;x1,x2;i,j)
        Returns:
            cmat: ndarray
                Partial recovery matrix cmat(k,k0;x;i,j)
                k: rotation phase, ranging from 0 to N-1
        """
        lmat = np.empty(self.nmeas_anc, dtype=object)
        if self.recovery_scheme == DIRECT:
            for x1 in range(self.nmeas_anc):
                lmat[x1] = x1
        else:
            raise Exception("Not implemented.")

        cmat = np.zeros((self.mod, self.noise.nkraus, self.nmeas_data, 2, 2), dtype=complex)
        if len(cmatr.shape) == 6:
            for x1 in range(self.nmeas_anc):
                for k2 in range(cmatr.shape[0]):
                    l = lmat[x1]
                    kp = (k2 - l) % self.mod
                    cmat[kp, :, :, :, :] += cmatr[k2, :, x1, :, :, :]
        elif len(cmatr.shape) == 7:
            for x1 in range(self.nmeas_anc):
                for k1 in range(cmatr.shape[0]):
                    for k2 in range(cmatr.shape[0]):
                        l = lmat[x1]
                        kp = (k1 + k2 - l) % self.mod
                        cmat[kp, :, :, :, :] += cmatr[k1, k2, :, x1, :, :, :]
        return cmat

    def partial_recovery_two_ec(self, cmatr):
        """
        Recovery for phase rotation (that's why partial), leaving logical Pauli recovery till the end.
        Args:
            cmatr: ndarray
                Measurement matrix cmat(k0,k1,k2,k3;x2,x3,x4;ijmn)
        Returns:
            cmat: ndarray
                Partial recovery matrix cmat(k,k0;x2,x4;i,j,m,n)
                k: rotation phase, ranging from 0 to N-1
                k0: loss kraus E_{k0}
        """
        lmat = np.empty(self.nmeas_anc, dtype=object)
        if self.recovery_scheme == DIRECT:
            for x1 in range(self.nmeas_anc):
                lmat[x1] = x1
        else:
            raise Exception("Not implemented.")

        cmat = np.zeros((self.mod, self.noise.nkraus, self.nmeas_data, self.nmeas_data, 2, 2, 2, 2), dtype=complex)
        for x3 in range(self.nmeas_anc):
            for k0 in range(cmatr.shape[0]):
                for k1 in range(cmatr.shape[1]):
                    for k2 in range(cmatr.shape[2]):
                        l = lmat[x3]
                        kp = (k0 + k1 + k2 - l) % self.mod
                        cmat[kp, :, :, :, :, :, :, :] += cmatr[k0, k1, k2, :, :, x3, :, :, :, :, :]
        return cmat

    def recovery(self, cmat):
        if len(cmat.shape) == 5:
            dmat = self.recovery_one_ec(cmat)
        elif len(cmat.shape) == 8:
            dmat = self.recovery_two_ec(cmat)
        else:
            raise Exception("Recovery: Invalid shape for cmat.")
        return dmat

    def recovery_one_ec(self, cmat):
        """
        Applies recovery for logical error after running one EC (either leading or trailing EC). Rotation error should
        be corrected by partial recovery.
        Args:
            cmat: ndarray
                Measurement matrix cmat(k,k0;x;i,j)

        Returns:
            dmat: nparray
                Matrix describes output map after recovery dmat(k,k0;a,b;ijmn)
                k: rotation phase, ranging from 0 to N-1
                k0: Kraus loss E_{k0}
                a,b: Z^aX^b recovery
        """
        bmat = np.empty(self.nmeas_data, dtype=object)
        if self.recovery_scheme == DIRECT:
            for x in range(self.nmeas_data):
                bmat[x] = x
        elif self.recovery_scheme == MAXIMUM_LIKELIHOOD:
            raise Exception("Not implemented.")

        # clear up indices -> dmat[k,k0;a,b;i,j,m,n]. Indices a,b keep track of X,Z recovery need to apply.
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for x in range(self.nmeas_data):
            b = bmat[x]
            dmat[:, :, 0, b, 0, 0, :, :] += cmat[:, :, x, :, :]

        return dmat

    def recovery_two_ec(self, cmat):
        """
        Applies recovery for logical error after running both ECs. Rotation error should be corrected
        by partial recovery.
        Args:
            cmat: ndarray
                Measurement matrix cmat(k,k0;x2,x4;i,j,m,n)

        Returns:
            dmat: nparray
                Matrix describes output map after recovery dmat(k,k0;a,b;ijmn)
                k: rotation phase, ranging from 0 to N-1
                k0: Kraus loss E_{k0}
                a,b: Z^aX^b recovery
        """
        abmat = np.empty((self.nmeas_data, self.nmeas_data), dtype=object)
        if self.recovery_scheme == DIRECT:
            for x2 in range(self.nmeas_data):
                for x4 in range(self.nmeas_data):
                    abmat[x2, x4] = (x2, x4)
        elif self.recovery_scheme == MAXIMUM_LIKELIHOOD:
            raise Exception("Not implemented.")

        # clear up indices -> dmat[k,k0;a,b;i,j,m,n]. Indices a,b keep track of X,Z recovery need to apply.
        dmat = np.zeros((self.mod, self.noise.nkraus, 2, 2, 2, 2, 2, 2), dtype=complex)
        for x2 in range(self.nmeas_data):
            for x4 in range(self.nmeas_data):
                a, b = abmat[x2, x4]
                dmat[:, :, a, b, :, :, :, :] += cmat[:, :, x2, x4, :, :, :, :]

        return dmat

    def update_alpha(self, data=None, anc=None):
        if data is not None:
            self.data = data
        if anc is not None:
            self.anc = anc

    def update_in_noise(self, loss_in, dephasing_in):
        """Updates waiting noise"""
        self.noise.update_in_noise(loss_in, dephasing_in)
        self.meas_data.apply_dephasing(self.noise.dephasing_in)

    def update_measurement(self, meas_data, meas_anc):
        """Updates measurement offsets"""
        self.meas_data = meas_data
        self.meas_anc = meas_anc
        self.meas_data.noisy([self.noise.loss_meas_data, self.noise.dephasing_meas_data])
        self.meas_anc.noisy([self.noise.loss_meas_anc, self.noise.dephasing_meas_anc])
        if self.noise.dephasing_in is not None:
            self.meas_data.apply_dephasing(self.noise.dephasing_in)
