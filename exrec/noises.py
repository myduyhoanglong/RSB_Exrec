import numpy as np
import qutip as qt

import channels
from codes import TrivialCode
from helpers import get_nkraus, average_infidelity
from constants import *


class BaseNoise:
    """
    A class constructs all noise maps.
    Base loss (or single gate loss): gamma
    Propagated phase through CZ: -2*pi*k/mod for k=0,1,...,mod-1.
    Waiting loss: gamma_wait
    Waiting dephasing: gamma_phi_wait
    """

    def __init__(self, scheme, gamma, gamma_wait, gamma_phi_wait, mod=1, rec=False):
        self.scheme = scheme
        self.gamma = gamma
        self.gamma_wait = gamma_wait
        self.gamma_phi_wait = gamma_phi_wait
        self.mod = mod
        self.rec = rec

        # single gate loss
        self.nkraus = get_nkraus(self.gamma)
        self.loss = channels.LossChannel(self.gamma, DIM, self.nkraus)

        # propagated phase
        phi = 2 * np.pi / self.mod
        self.phase = [(-1j * phi * k * qt.num(DIM)).expm() for k in range(self.mod)]

        # waiting noise
        self.nkraus_wait, self.loss_wait, self.dephasing_wait = self.make_wait_noise()

    def make_wait_noise(self):
        # Note that loss in one last location of leading EC is combined with waiting loss for extended gadget.
        # For rectangle, there is no leading EC.
        if self.rec:
            nkraus_wait = get_nkraus(self.gamma_wait)
            loss_wait = channels.LossChannel(self.gamma_wait, DIM, nkraus_wait)
        else:
            nkraus_wait = get_nkraus(self.gamma_wait + self.gamma)
            loss_wait = channels.LossChannel(self.gamma_wait + self.gamma, DIM, nkraus_wait)
        if self.gamma_phi_wait > 0:
            dephasing_wait = channels.DephasingChannel(self.gamma_phi_wait, DIM)
        else:
            dephasing_wait = None
        return nkraus_wait, loss_wait, dephasing_wait

    def update_wait_noise(self, gamma_wait, gamma_phi_wait):
        """Updates waiting loss and dephasing."""
        self.gamma_wait = gamma_wait
        self.gamma_phi_wait = gamma_phi_wait

        self.nkraus_wait, self.loss_wait, self.dephasing_wait = self.make_wait_noise()


class KnillNoise:
    """
    A class constructs noise specifically for Knill EC.
    Base loss (or single gate loss): gamma
    Propagated phase through CZ: -2*pi*k/mod for k=0,1,...,mod-1.
    Measurement loss: 2*gamma for both data and ancilla mode.
    Measurement dephasing: 4*gamma_phi for both data and ancilla mode.
    Waiting noise and leading EC noise (for trailing EC).
    """

    def __init__(self, gamma, gamma_phi, loss_in=None, dephasing_in=None, N=2, mod=1):
        self.gamma = gamma
        self.gamma_phi = gamma_phi
        self.N = N
        self.mod = mod

        # single gate loss
        self.nkraus = get_nkraus(gamma)
        self.loss = channels.LossChannel(gamma, DIM, self.nkraus)

        # propagated phase through CZ
        phi = 2 * np.pi / (mod * N)
        self.phase = [(-1j * phi * k * qt.num(DIM)).expm() for k in range(mod)]

        # array of tr(E_k^{dagger} E_k), used for MAXIMUM_LIKELIHOOD recovery
        self.trace_loss = np.array([np.trace(k.dag() * k) for k in self.loss.kraus])

        # measurement noise
        if self.gamma_phi > 0:
            self.dephasing_meas = channels.DephasingChannel(4 * self.gamma_phi, DIM)
        else:
            self.dephasing_meas = None
        self.loss_meas = channels.LossChannel(2 * self.gamma, DIM, get_nkraus(2 * self.gamma))

        # waiting noise
        if loss_in is None:
            self.nkraus_in = 1  # K1
            self.loss_in = channels.IdentityChannel(DIM)
        else:
            self.loss_in = loss_in
            self.nkraus_in = len(loss_in.kraus)
        self.dephasing_in = dephasing_in

        # noise from other EC, if any
        self.nphase_zero = 0
        self.nkraus_zero = 0
        self.loss_zero = None

    def update_zero_noise(self, gmat=None):
        """
        Updates noise from other EC, if any.
        Args:
            gmat: ndarray
                Input matrix describes noise from other EC, if not None. Indices: gmat(k,k0;x1,x2;i,j,m,n).
                k: index of propagated phase F_k = e^{-i*(pi*k/(N*N))n}.
                k0: index of loss E_k0.
                x1,x2: measurement outcomes.
                i,j,m,n: Logical operators. X^mZ^i(.)Z^jX^n.
        """
        if gmat is None:
            self.nphase_zero = 1  # K
            self.nkraus_zero = 1  # K0
            self.loss_zero = channels.IdentityChannel(DIM)
        else:
            self.nphase_zero = gmat.shape[0]
            self.nkraus_zero = gmat.shape[1]
            self.loss_zero = self.loss

    def update_in_noise(self, loss_in, dephasing_in):
        """
        Updates waiting noise.
        """
        self.nkraus_in = len(loss_in.kraus)
        self.loss_in = loss_in
        self.dephasing_in = dephasing_in


class HybridNoise:
    """
    A class constructs noise specifically for Hybrid EC.
    Base loss (or single gate loss): gamma
    Propagated phase through CZ: -2*pi*k/(N*N) for k=0,1,...,N-1.
    Propagated phase through CROT: -2*pi*k/N for k=0,1,...,N-1. Assume that M=1 for Hybrid EC.
    Measurement loss: 2*gamma for data mode, 3*gamma for ancilla mode.
    Measurement dephasing: 5*gamma_phi for data, 3*gamma_phi for ancilla.
    Waiting noise and leading EC noise (for trailing EC).
    """
    def __init__(self, gamma, gamma_phi, loss_in=None, dephasing_in=None, N=2, mod=1, mod_anc=1):
        self.gamma = gamma
        self.gamma_phi = gamma_phi
        self.N = N
        self.mod = mod
        self.mod_anc = mod_anc

        # single gate loss
        self.nkraus = get_nkraus(gamma)
        self.loss = channels.LossChannel(gamma, DIM, self.nkraus)

        # propagated phase through CZ
        phi = 2 * np.pi / (mod * N)
        self.phase = [(-1j * phi * k * qt.num(DIM)).expm() for k in range(mod)]  # base phase

        # propagated phase through CROT
        phi_anc = 2 * np.pi / mod_anc
        self.phase_anc = [(-1j * phi_anc * k * qt.num(DIM)).expm() for k in range(mod_anc)]

        # array of tr(E_k^{dagger} E_k), used for MAXIMUM_LIKELIHOOD recovery
        self.trace_loss = np.array([np.trace(k.dag() * k) for k in self.loss.kraus])

        # measurement noise
        if gamma_phi > 0:
            self.dephasing_meas_anc = channels.DephasingChannel(3 * self.gamma_phi, DIM)
            self.dephasing_meas_data = channels.DephasingChannel(5 * self.gamma_phi, DIM)
        else:
            self.dephasing_meas_anc = None
            self.dephasing_meas_data = None
        self.loss_meas_data = channels.LossChannel(2 * self.gamma, DIM, get_nkraus(2 * self.gamma))
        self.loss_meas_anc = channels.LossChannel(3 * self.gamma, DIM, get_nkraus(3 * self.gamma))

        # waiting noise
        if loss_in is None:
            self.nkraus_in = 1
            self.loss_in = channels.IdentityChannel(DIM)  # E1
        else:
            self.loss_in = loss_in
            self.nkraus_in = len(loss_in.kraus)
        self.dephasing_in = dephasing_in

        # noise from other EC, if any
        self.nphase_zero = 0
        self.nkraus_zero = 0
        self.loss_zero = None

    def update_zero_noise(self, gmat=None):
        """
        Updates noise from other EC, if any.
        Args:
            gmat: ndarray
                Input matrix describes noise from other EC, if not None. Indices: gmat(k,k0;x1,x2;i,j,m,n).
                k: index of propagated phase F_k = e^{-i*(pi*k/(N*N))n}.
                k0: index of loss E_k0.
                x1,x2: measurement outcomes.
                i,j,m,n: Logical operators. X^mZ^i(.)Z^jX^n.
        """
        if gmat is None:
            self.nphase_zero = 1  # K
            self.nkraus_zero = 1  # K0
            self.loss_zero = channels.IdentityChannel(DIM)
        else:
            self.nphase_zero = gmat.shape[0]
            self.nkraus_zero = gmat.shape[1]
            self.loss_zero = self.loss

    def update_in_noise(self, loss_in, dephasing_in):
        """
        Updates waiting noise.
        """
        self.nkraus_in = len(loss_in.kraus)
        self.loss_in = loss_in
        self.dephasing_in = dephasing_in


class BenchMark:
    """
    A class for unencoded scheme, here chosen to be (0,1)-Fock encoding.
    """
    def __init__(self, gamma_wait, gamma_phi_wait):
        self.gamma_wait = gamma_wait
        self.gamma_phi_wait = gamma_phi_wait
        self.dim = 2
        self.code = TrivialCode()
        self.loss, self.dephasing, self.noise = None, None, None
        self.update_noise(gamma_wait, gamma_phi_wait)
        self.infidelity = 1

    def get_infidelity(self):
        gate = self.code.decoder() * self.noise.channel_matrix * self.code.encoder()
        infidelity = average_infidelity(gate)
        self.infidelity = infidelity
        return infidelity

    def update_noise(self, gamma_wait, gamma_phi_wait):
        self.gamma_wait = gamma_wait
        self.gamma_phi_wait = gamma_phi_wait
        self.loss = channels.LossChannel(self.gamma_wait, self.dim, get_nkraus(self.gamma_wait))
        if self.gamma_phi_wait > 0:
            self.dephasing = channels.DephasingChannel(self.gamma_phi_wait, self.dim)
            self.noise = channels.Channel(channel_matrix=self.dephasing.channel_matrix * self.loss.channel_matrix)
        else:
            self.noise = channels.Channel(channel_matrix=self.loss.channel_matrix)
