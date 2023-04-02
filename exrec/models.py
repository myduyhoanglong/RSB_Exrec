import math
import numpy as np
import qutip as qt

from codes import CatCode
from channels import LossChannel, DephasingChannel
from measurements import LogicalMeasurement, WedgeMeasurement
from noises import KnillNoise, HybridNoise
from constants import *
from helpers import get_nkraus


class Model:
    """A class that computes first order infidelity of an extended gadget by assigning each location a failure
        probability."""

    def __init__(self, scheme, code_params, meas_params, noise_params, recovery=DIRECT, decoder=TRANSPOSE):
        self.scheme = scheme
        self.code_params = code_params
        self.meas_params = meas_params
        self.noise_params = noise_params
        self.recovery = recovery
        self.decoder = decoder

        self.N, self.alpha_data, self.M, self.alpha_anc = code_params
        self.offset_data, self.offset_anc = meas_params
        self.gamma, self.gamma_phi, self.eta = noise_params

        self.gamma_wait = self.eta * self.gamma
        self.gamma_phi_wait = self.eta * self.gamma_phi

        self.data = CatCode(N=self.N, r=0, alpha=self.alpha_data, fockdim=DIM)
        self.anc = CatCode(N=self.M, r=0, alpha=self.alpha_anc, fockdim=DIM)
        self.meas_data, self.meas_anc = self.make_measurement()

        self.infidelity = 1

        if self.scheme == HYBRID and self.M > 1:
            raise Exception("Degree of rotation M should be 1 for HYBRID scheme.")

    def make_measurement(self):
        """
        Constructs measurements for data and ancilla mode.
        """
        if self.scheme == KNILL:
            meas_data = LogicalMeasurement(2 * self.N, DIM, -np.pi / (2 * self.N), self.offset_data)
            meas_anc = LogicalMeasurement(2 * self.M, DIM, -np.pi / (2 * self.M), self.offset_anc)
        elif self.scheme == HYBRID:
            meas_data = LogicalMeasurement(2 * self.N, DIM, -np.pi / (2 * self.N), self.offset_data)
            meas_anc = WedgeMeasurement(self.M * self.N, DIM, -np.pi / (self.M * self.N), self.offset_anc)
        else:
            raise Exception("Unknown scheme", self.scheme)

        return meas_data, meas_anc


class HybridModel(Model):
    """A class that computes first order infidelity of a Hybrid extended gadget by assigning each location a failure
        probability."""

    def __init__(self, code_params, meas_params, noise_params, recovery=DIRECT, decoder=TRANSPOSE):
        Model.__init__(self, HYBRID, code_params, meas_params, noise_params, recovery, decoder)
        self.mod = 2 * self.N
        self.mod_anc = self.N
        _, self.loss_wait, self.dephasing_wait = self.make_wait_noise()
        self.noise = HybridNoise(self.gamma, self.gamma_phi, loss_in=self.loss_wait, dephasing_in=self.dephasing_wait,
                                 N=self.N, mod=self.mod, mod_anc=self.mod_anc)

        self.meas_data.noisy([None, self.noise.dephasing_meas_data])
        self.meas_anc.noisy([None, self.noise.dephasing_meas_anc])

        self.gate_error = self.init_gate_error()
        self.wait_error = self.init_wait_error()
        self.meas_error_leading, self.meas_error_trailing = self.init_meas_error()

    def make_wait_noise(self):
        # Note that loss in one last location of leading EC is NOT combined with waiting loss in this model.
        nkraus_wait = get_nkraus(self.gamma_wait)
        loss_wait = LossChannel(self.gamma_wait, DIM, nkraus_wait)
        if self.gamma_phi_wait > 0:
            dephasing_wait = DephasingChannel(self.gamma_phi_wait, DIM)
        else:
            dephasing_wait = None
        return nkraus_wait, loss_wait, dephasing_wait

    def update_error(self):
        self.gate_error = self.init_gate_error()
        self.wait_error = self.init_wait_error()
        self.meas_error_leading, self.meas_error_trailing = self.init_meas_error()

    def init_gate_error(self):
        probs = []
        for k in range(self.N + 1):
            b = self.gamma * (self.alpha_data ** 2)
            p = np.exp(-b) * (b ** k) / math.factorial(k)
            probs.append(p)
        return probs

    def init_wait_error(self):
        probs = []
        for k in range(self.N + 1):
            b = self.gamma_wait * (self.alpha_data ** 2)
            p = np.exp(-b) * (b ** k) / math.factorial(k)
            probs.append(p)
        return probs

    def init_meas_error(self):
        probs_leading = []
        for k in range(self.N):
            s = self.noise.phase[k] * self.data.minus
            p = qt.expect(self.meas_data.povm_elements[0], s * s.dag())
            probs_leading.append(p)

        self.meas_data.apply_dephasing(self.dephasing_wait)
        probs_trailing = []
        for k in range(self.N):
            s = self.noise.phase[k] * self.data.minus
            p = qt.expect(self.meas_data.povm_elements[0], s * s.dag())
            probs_trailing.append(p)

        return probs_leading, probs_trailing

    def get_infidelity(self):
        if self.N == 2:
            p0 = self.meas_error_leading[0] + self.meas_error_trailing[0] + self.meas_error_leading[1] * \
                 self.gate_error[1] + 2 * self.meas_error_trailing[1] * self.gate_error[1]
            inf = p0 + (4 * (self.gate_error[1] ** 2) + 5 * self.gate_error[2]) + (
                    3 * self.gate_error[1] * self.wait_error[1] + self.wait_error[2])
        elif self.N == 3:
            p0 = self.meas_error_leading[0] + self.meas_error_trailing[0] + self.meas_error_leading[1] * \
                 self.gate_error[1] + 2 * self.meas_error_trailing[1] * self.gate_error[1] + self.meas_error_leading[
                     2] * self.gate_error[2] + self.meas_error_trailing[2] * (
                         2 * self.gate_error[2] + self.gate_error[1] ** 2)
            inf = p0 + (5 * self.gate_error[3] + 10 * self.gate_error[2] * self.gate_error[1] + self.gate_error[
                1] ** 3) + (3 * self.gate_error[2] * self.wait_error[1] + 3 * self.gate_error[1] * self.wait_error[2] +
                            self.wait_error[3])
        elif self.N == 4:
            p0 = self.meas_error_leading[0] + self.meas_error_trailing[0] + self.meas_error_leading[1] * \
                 self.gate_error[1] + 2 * self.meas_error_trailing[1] * self.gate_error[1] + self.meas_error_leading[
                     2] * self.gate_error[2] + self.meas_error_trailing[2] * (
                         2 * self.gate_error[2] + self.gate_error[1] ** 2) + 2 * self.gate_error[3] * \
                 self.meas_error_trailing[3]
            inf = p0 + (5 * self.gate_error[4] + 4 * (
                    2 * self.gate_error[3] * self.gate_error[1] + (self.gate_error[2] ** 2)) + 3 * (
                                self.gate_error[1] ** 2) * self.gate_error[2]) + (
                          self.gate_error[1] ** 3 * self.wait_error[1] + 3 * (self.gate_error[1] ** 2) *
                          self.gate_error[2] + 3 * self.gate_error[2] * self.wait_error[2] + 3 * self.gate_error[1] *
                          self.wait_error[3] + self.wait_error[4])
        else:
            raise Exception("Only support N <= 4")

        self.infidelity = inf
        return inf

    def update_alpha(self, alphas):
        self.alpha_data, self.alpha_anc = alphas
        self.code_params = [self.N, self.alpha_data, self.M, self.alpha_anc]
        self.data = CatCode(N=self.N, r=0, alpha=self.alpha_data, fockdim=DIM)
        self.anc = CatCode(N=self.M, r=0, alpha=self.alpha_anc, fockdim=DIM)
        self.update_error()

    def update_wait_noise(self, eta):
        self.gamma_wait = eta * self.gamma
        self.gamma_phi_wait = eta * self.gamma_phi
        self.eta = eta
        self.noise_params = [self.gamma, self.gamma_phi, self.eta]
        _, self.loss_wait, self.dephasing_wait = self.make_wait_noise()
        self.noise = HybridNoise(self.gamma, self.gamma_phi, loss_in=self.loss_wait, dephasing_in=self.dephasing_wait,
                                 N=self.N, mod=self.mod, mod_anc=self.mod_anc)
        self.update_error()

    def update_measurement(self, meas_params):
        self.offset_data, self.offset_anc = meas_params
        self.meas_params = [self.offset_data, self.offset_anc]
        self.meas_data, self.meas_anc = self.make_measurement()
        self.meas_data.noisy([None, self.noise.dephasing_meas_data])
        self.meas_anc.noisy([None, self.noise.dephasing_meas_anc])
