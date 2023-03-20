import numpy as np

from codes import CatCode
from measurements import LogicalMeasurement, WedgeMeasurement
from decoder import SDPDecoder, FastDecoder
from knill import KnillEC
from hybrid import HybridEC
from noises import BaseNoise

from constants import *
from helpers import average_infidelity

import time


class ExtendedGadgetException(Exception):
    pass


class ExtendedGadget:

    def __init__(self, scheme, code_params, meas_params, noise_params, recovery=MAXIMUM_LIKELIHOOD, decoder=SDP):
        """
        :param scheme: Knill, Knill without prep noise, Hybrid
        :param code_params: [N, alpha_data, M, alpha_anc]
        :param meas_params: [offset_data, offset_anc]
        :param noise_params: [gamma, gamma_phi, eta(waiting time)]
        """
        N, alpha_data, M, alpha_anc = code_params
        offset_data, offset_anc = meas_params
        gamma, gamma_phi, eta = noise_params
        self.scheme = scheme
        self.decoding_scheme = recovery
        self.ideal_decoder = decoder
        self.code_params = code_params
        self.meas_params = meas_params
        self.noise_params = noise_params
        if gamma_phi > 0 and gamma > 0:
            self.max_eta = min(0.1 / gamma, 1 / gamma_phi)
        elif gamma > 0:
            self.max_eta = 0.1 / gamma
        else:
            self.max_eta = 1000

        # print("INITIALIZE CODES...")
        # st = time.time()
        data = CatCode(N=N, r=0, alpha=alpha_data, fockdim=DIM)
        anc = CatCode(N=M, r=0, alpha=alpha_anc, fockdim=DIM)
        # print("DONE INITIALIZE CODES...", time.time() - st)

        # print("INITIALIZE BASE NOISE...")
        # st = time.time()
        gamma_wait = eta * gamma
        gamma_phi_wait = eta * gamma_phi
        self.base_noise = BaseNoise(scheme, gamma, gamma_phi, gamma_wait, gamma_phi_wait, mod=2 * N * M)
        # print("DONE INITIALIZE BASE NOISE...", time.time() - st)

        # print("INITIALIZE MEASUREMENT...")
        # st = time.time()
        meas_data, meas_anc = self.make_measurement(scheme, data.N, anc.N, offset_data, offset_anc)
        # print("DONE INITIALIZE MEASUREMENT...", time.time() - st)

        # print("INITIALIZE LEADING EC...")
        # st = time.time()
        self.leading_ec = self.make_ec(data=data, anc=anc, meas_data=meas_data, meas_anc=meas_anc, gamma=gamma,
                                       gamma_phi=gamma_phi, loss_in=None, dephasing_in=None, decoding=recovery)
        # print("DONE INITIALIZE LEADING EC...", time.time() - st)
        # print("INITIALIZE TRAILING EC...")
        # st = time.time()
        self.trailing_ec = self.make_ec(data=data, anc=anc, meas_data=meas_data, meas_anc=meas_anc, gamma=gamma,
                                        gamma_phi=gamma_phi, loss_in=self.base_noise.loss_wait,
                                        dephasing_in=self.base_noise.dephasing_wait, decoding=recovery)
        # print("DONE INITIALIZE TRAILING EC...", time.time() - st)

        if decoder == SDP:
            self.decoder = SDPDecoder(code=data, loss=self.base_noise.loss, phase=self.base_noise.phase)
        elif decoder == FAST:
            self.decoder = FastDecoder(code=data, loss=self.base_noise.loss)

        self.infidelity = 1
        self.infidelity_leading_ec = 1
        self.infidelity_trailing_ec = 1

    def make_ec(self, data=None, anc=None, meas_data=None, meas_anc=None, gamma=1e-3, gamma_phi=1e-3, loss_in=None,
                dephasing_in=None, decoding=MAXIMUM_LIKELIHOOD):
        if self.scheme == KNILL:
            return KnillEC(data, anc, meas_data, meas_anc, gamma, gamma_phi, loss_in, dephasing_in, decoding)
        elif self.scheme == KNILL_NO_PREP_NOISE:
            pass
        elif self.scheme == HYBRID:
            return HybridEC(data, anc, meas_data, meas_anc, gamma, gamma_phi, loss_in, dephasing_in, decoding)
        else:
            raise ExtendedGadgetException("Unknown scheme", self.scheme)

    def make_measurement(self, scheme, N, M, offset_data, offset_anc):
        if scheme == KNILL:
            meas_data = LogicalMeasurement(2 * N, DIM, -np.pi / (2 * N), offset_data)
            meas_anc = LogicalMeasurement(2 * M, DIM, -np.pi / (2 * M), offset_anc)
            meas_data.noisy([self.base_noise.loss_meas, self.base_noise.dephasing_meas])
            meas_anc.noisy([self.base_noise.loss_meas, self.base_noise.dephasing_meas])
        elif scheme == HYBRID:
            if M > 1:
                raise ExtendedGadgetException("Degree of rotation M should be 1 for HYBRID scheme.")
            meas_data = LogicalMeasurement(2 * N, DIM, -np.pi / (2 * N), offset_data)
            meas_anc = WedgeMeasurement(M * N, DIM, -np.pi / (M * N), offset_anc)
            meas_data.noisy([self.base_noise.loss_meas_data, self.base_noise.dephasing_meas_data])
            meas_anc.noisy([self.base_noise.loss_meas_anc, self.base_noise.dephasing_meas_anc])

        return meas_data, meas_anc

    def run(self):
        # print("RUNNING LEADING EC...")
        # st = time.time()
        cmat_leading_ec = self.leading_ec.run(gmat=None)
        # print("DONE RUNING LEADING EC...", time.time() - st)
        # print("RUNNING TRAILING EC...")
        # st = time.time()
        cmat_trailing_ec = self.trailing_ec.run(gmat=cmat_leading_ec)
        # print("DONE RUNNING TRAILING EC...", time.time() - st)
        # print("RUNNING DECODER...")
        # st = time.time()
        dmat = self.trailing_ec.recovery(cmat_trailing_ec)
        output = self.decoder.decode(dmat)
        # print("DONE RUNNING DECODER...", time.time() - st)

        return output

    def get_infidelity(self):
        # print(">>RUNNING...<<")
        # st = time.time()
        out_channel = self.run()
        infidelity = average_infidelity(out_channel.channel_matrix)
        self.infidelity = infidelity
        # print(">>DONE RUNNING...<<", time.time() - st)
        return infidelity

    def get_infidelity_leading_ec(self):
        cmat = self.leading_ec.run()
        dmat = self.leading_ec.recovery(cmat)
        output = self.decoder.decode(dmat)
        infidelity = average_infidelity(output.channel_matrix)
        self.infidelity_leading_ec = infidelity
        return infidelity

    def get_infidelity_trailing_ec(self):
        cmat = self.trailing_ec.run()
        dmat = self.trailing_ec.recovery(cmat)
        output = self.decoder.decode(dmat)
        infidelity = average_infidelity(output.channel_matrix)
        self.infidelity_trailing_ec = infidelity
        return infidelity

    def update_alpha(self, alphas):
        alpha_data, alpha_anc = alphas
        if alpha_data is not None:
            data = CatCode(N=self.code_params[0], r=0, alpha=alpha_data, fockdim=DIM)
            self.code_params[1] = alpha_data
        else:
            data = None
        if alpha_anc is not None:
            anc = CatCode(N=self.code_params[2], r=0, alpha=alpha_anc, fockdim=DIM)
            self.code_params[3] = alpha_anc
        else:
            anc = None

        self.leading_ec.update_alpha(data, anc)
        self.trailing_ec.update_alpha(data, anc)
        self.decoder.update_code(data)

    def update_wait_noise(self, eta=None):
        if eta is not None:
            gamma_wait = eta * self.noise_params[0]
            gamma_phi_wait = eta * self.noise_params[1]
            self.noise_params[2] = eta
            self.base_noise.update_wait_noise(gamma_wait, gamma_phi_wait)
            self.trailing_ec.update_in_noise(self.base_noise.loss_wait, self.base_noise.dephasing_wait)

    def update_measurement(self, meas_params):
        N = self.code_params[0]
        M = self.code_params[2]
        offset_data = meas_params[0]
        offset_anc = meas_params[1]
        meas_data, meas_anc = self.make_measurement(self.scheme, N, M, offset_data, offset_anc)
        self.leading_ec.update_measurement(meas_data, meas_anc)
        self.trailing_ec.update_measurement(meas_data, meas_anc)
        self.meas_params = meas_params
