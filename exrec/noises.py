import numpy as np
import qutip as qt

import channels
from codes import TrivialCode
from helpers import get_nkraus, average_infidelity
from constants import *

import time


class BaseNoise:
    def __init__(self, scheme, gamma, gamma_phi, gamma_wait, gamma_phi_wait, mod=1):
        self.nkraus = get_nkraus(gamma)
        self.loss = channels.LossChannel(gamma, DIM, self.nkraus)
        phi = 2 * np.pi / mod
        self.phase = [(-1j * phi * k * qt.num(DIM)).expm() for k in range(mod)]

        # measurement noise
        if scheme == KNILL:
            if gamma_phi > 0:
                self.dephasing_meas = channels.DephasingChannel(4 * gamma_phi, DIM)
            else:
                self.dephasing_meas = None
            self.loss_meas = channels.LossChannel(2 * gamma, DIM, get_nkraus(2 * gamma))
        elif scheme == HYBRID:
            if gamma_phi > 0:
                self.dephasing_meas_anc = channels.DephasingChannel(3 * gamma_phi, DIM)
                self.dephasing_meas_data = channels.DephasingChannel(5 * gamma_phi, DIM)
            else:
                self.dephasing_meas_anc = None
                self.dephasing_meas_data = None
            self.loss_meas_data = channels.LossChannel(2 * gamma, DIM, get_nkraus(2 * gamma))
            self.loss_meas_anc = channels.LossChannel(3 * gamma, DIM, get_nkraus(3 * gamma))

        # waiting noise
        # self.nkraus_wait = get_nkraus(gamma_wait)
        # self.loss_wait = channels.LossChannel(gamma_wait, DIM, self.nkraus_wait)
        self.nkraus_wait = get_nkraus(gamma_wait + gamma)
        self.loss_wait = channels.LossChannel(gamma_wait + gamma, DIM, self.nkraus_wait)
        if gamma_phi_wait > 0:
            self.dephasing_wait = channels.DephasingChannel(gamma_phi_wait, DIM)
        else:
            self.dephasing_wait = None

    def update_wait_noise(self, gamma_wait, gamma_phi_wait):
        self.nkraus_wait = get_nkraus(gamma_wait)
        self.loss_wait = channels.LossChannel(gamma_wait, DIM, self.nkraus_wait)
        if gamma_phi_wait > 0:
            self.dephasing_wait = channels.DephasingChannel(gamma_phi_wait, DIM)
        else:
            self.dephasing_wait = None


class KnillNoise:
    def __init__(self, gamma, gamma_phi, loss_in=None, dephasing_in=None, data=None, mod=1):
        self.code = data

        self.nkraus = get_nkraus(gamma)
        self.loss = channels.LossChannel(gamma, DIM, self.nkraus)  # base loss
        phi = 2 * np.pi / (mod * self.code.N)
        self.phase = [(-1j * phi * k * qt.num(DIM)).expm() for k in range(mod)]  # base phase
        self.trace_loss = np.array([np.trace(k.dag() * k) for k in self.loss.kraus])

        if loss_in is None:
            self.nkraus_in = 1
            self.loss_in = channels.IdentityChannel(DIM)  # E1
        else:
            self.loss_in = loss_in
            self.nkraus_in = len(loss_in.kraus)

        self.dephasing_in = dephasing_in

        self.nphase_zero = 0
        self.nkraus_zero = 0
        self.loss_zero = None
        self.phase_zero = None

    def update_zero_noise(self, gmat=None):
        if gmat is None:
            self.nphase_zero = 1  # K
            self.nkraus_zero = 1  # K0
            self.loss_zero = channels.IdentityChannel(DIM)
        else:
            self.nphase_zero = gmat.shape[0]
            self.nkraus_zero = gmat.shape[1]
            self.loss_zero = self.loss

    def update_in_noise(self, loss_in, dephasing_in):
        self.loss_in = loss_in
        self.nkraus_in = len(loss_in.kraus)
        self.dephasing_in = dephasing_in

    def update_alpha(self, data):
        self.code = data


class HybridNoise:
    def __init__(self, gamma, gamma_phi, loss_in=None, dephasing_in=None, N=2, mod=1, mod_anc=1):
        self.nkraus = get_nkraus(gamma)
        self.loss = channels.LossChannel(gamma, DIM, self.nkraus)  # base loss
        phi = 2 * np.pi / (mod * N)
        self.phase = [(-1j * phi * k * qt.num(DIM)).expm() for k in range(mod)]  # base phase
        phi_anc = 2 * np.pi / mod_anc
        self.phase_anc = [(-1j * phi_anc * k * qt.num(DIM)).expm() for k in range(mod_anc)]
        self.trace_loss = np.array([np.trace(k.dag() * k) for k in self.loss.kraus])

        if loss_in is None:
            self.nkraus_in = 1
            self.loss_in = channels.IdentityChannel(DIM)  # E1
        else:
            self.loss_in = loss_in
            self.nkraus_in = len(loss_in.kraus)

        self.dephasing_in = dephasing_in

        self.nphase_zero = 0
        self.nkraus_zero = 0
        self.loss_zero = None

    def update_zero_noise(self, gmat=None):
        if gmat is None:
            self.nphase_zero = 1  # K
            self.nkraus_zero = 1  # K0
            self.loss_zero = channels.IdentityChannel(DIM)
        else:
            self.nphase_zero = gmat.shape[0]
            self.nkraus_zero = gmat.shape[1]
            self.loss_zero = self.loss

    def update_in_noise(self, loss_in, dephasing_in):
        self.loss_in = loss_in
        self.nkraus_in = len(loss_in.kraus)
        self.dephasing_in = dephasing_in


class BenchMark:
    def __init__(self, noise_params):
        self.gamma, self.gamma_phi, self.eta = noise_params
        self.loss = channels.LossChannel(self.eta * self.gamma, 2, get_nkraus(self.gamma))
        self.dephasing = channels.DephasingChannel(self.eta * self.gamma_phi, 2)
        self.noise = channels.Channel(channel_matrix=self.dephasing.channel_matrix * self.loss.channel_matrix)
        self.code = TrivialCode()
        self.infidelity = 1

    def get_infidelity(self):
        gate = self.code.decoder() * self.noise.channel_matrix * self.code.encoder()
        infidelity = average_infidelity(gate)
        self.infidelity = infidelity
        return infidelity

    def update_noise(self, noise_params):
        self.gamma, self.gamma_phi, self.eta = noise_params
        self.loss = channels.LossChannel(self.eta * self.gamma, 2, get_nkraus(self.gamma))
        if self.gamma_phi > 0:
            self.dephasing = channels.DephasingChannel(self.eta * self.gamma_phi, 2)
            self.noise = channels.Channel(channel_matrix=self.dephasing.channel_matrix * self.loss.channel_matrix)
        else:
            self.noise = channels.Channel(channel_matrix=self.loss.channel_matrix)
