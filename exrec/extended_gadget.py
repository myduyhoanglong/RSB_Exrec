import numpy as np

from codes import CatCode
from measurements import LogicalMeasurement, WedgeMeasurement
from decoder import SDPDecoder, TransposeChannelDecoder
from knill import KnillEC
from hybrid import HybridEC
from noises import BaseNoise
from helpers import average_infidelity
from constants import *


class ExtendedGadgetException(Exception):
    pass


class ExtendedGadget:
    """
    Extended gadget contains leading EC, trailing EC, and waiting step. For implementation, waiting step is absorbed
    into trailing EC. Loss in one last location of leading EC (CZ location) is combined with waiting loss.
    Assumptions: Cat code is used for encoding, canonical phase measurement is used for measurement.
    Functions: Get infidelity of extended gadget, of leading EC, of trailing EC. Update code's amplitude, measurement
    offsets, and waiting noise streng.
    """

    def __init__(self, scheme, code_params, meas_params, noise_params, recovery=DIRECT, decoder=TRANSPOSE):
        """
        Initialize an ExtendedGadget instance.
        Args:
            scheme: int
                KNILL or HYBRID
            code_params: list
                A list contains (data mode's rotation degree, data mode's amplitude, data mode's squeeze magnitude,
                ancilla mode's rotation degree, ancilla mode's amplitude, ancilla mode's squeeze magnitude),
                in the form [N, alpha_data, sq_data, M, alpha_anc, sq_anc].
            meas_params: list
                A list contains offsets of measurement of data mode and ancilla mode,
                in the form [offset_data, offset_anc].
            noise_params: list
                A list contains strengths of gate loss, gate dephasing, and multiplication factor of waiting noise
                compared to gate noise, in the form [gamma, gamma_phi, eta].
            recovery: int
                DIRECT or MAXIMUM_LIKELIHOOD.
                DIRECT: Recovery operators are deduced as in the ideal scenario.
                MAXIMUM_LIKELIHOOD: Recovery operators are deduced based on the noise as well.
            decoder: int
                SDP or TRANSPOSE. Ideal decoder attach to the end of the extended gadget
                SDP: semi-definite programming decoder.
                TRANSPOSE: transpose channel decoder.
        """

        self.scheme = scheme
        self.code_params = code_params
        self.meas_params = meas_params
        self.noise_params = noise_params
        self.recovery = recovery
        self.decoder = decoder

        self.N, self.alpha_data, self.sq_data, self.M, self.alpha_anc, self.sq_anc = code_params
        self.offset_data, self.offset_anc = meas_params
        self.gamma, self.gamma_phi, self.eta = noise_params

        self.gamma_wait = self.eta * self.gamma
        self.gamma_phi_wait = self.eta * self.gamma_phi

        self.base_noise = BaseNoise(scheme, self.gamma, self.gamma_wait, self.gamma_phi_wait, mod=2 * self.N * self.M)

        data = CatCode(N=self.N, r=self.sq_data, alpha=self.alpha_data, fockdim=DIM)
        anc = CatCode(N=self.M, r=self.sq_anc, alpha=self.alpha_anc, fockdim=DIM)

        meas_data, meas_anc = self.make_measurement()

        self.leading_ec = self.make_ec(data=data, anc=anc, meas_data=meas_data, meas_anc=meas_anc, gamma=self.gamma,
                                       gamma_phi=self.gamma_phi, loss_in=None, dephasing_in=None, recovery=recovery)

        self.trailing_ec = self.make_ec(data=data, anc=anc, meas_data=meas_data, meas_anc=meas_anc, gamma=self.gamma,
                                        gamma_phi=self.gamma_phi, loss_in=self.base_noise.loss_wait,
                                        dephasing_in=self.base_noise.dephasing_wait, recovery=recovery)

        if decoder == SDP:
            self.decoder = SDPDecoder(code=data, loss=self.base_noise.loss, phase=self.base_noise.phase)
        elif decoder == TRANSPOSE:
            self.decoder = TransposeChannelDecoder(code=data, loss=self.base_noise.loss)

        self.infidelity = 1
        self.infidelity_leading_ec = 1
        self.infidelity_trailing_ec = 1

        if self.scheme == HYBRID and self.M > 1:
            raise ExtendedGadgetException("Degree of rotation M should be 1 for HYBRID scheme.")

    def make_ec(self, data, anc, meas_data, meas_anc, gamma=1e-3, gamma_phi=1e-3, loss_in=None, dephasing_in=None,
                recovery=MAXIMUM_LIKELIHOOD):
        """
        Returns Knill EC or Hybrid EC. See comments in knill.py and hybrid.py .
        """
        if self.scheme == KNILL:
            ec = KnillEC(data, anc, meas_data, meas_anc, gamma, gamma_phi, loss_in, dephasing_in, recovery)
        elif self.scheme == HYBRID:
            ec = HybridEC(data, anc, meas_data, meas_anc, gamma, gamma_phi, loss_in, dephasing_in, recovery)
        else:
            raise ExtendedGadgetException("Unknown scheme", self.scheme)

        return ec

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
            raise ExtendedGadgetException("Unknown scheme", self.scheme)

        return meas_data, meas_anc

    def run(self):
        """
        Returns the final qubit-to-qubit channel of the extended gadget.
        Returns:
            output: channel
        """
        cmat_leading_ec = self.leading_ec.run(gmat=None)
        cmat_trailing_ec = self.trailing_ec.run(gmat=cmat_leading_ec)
        dmat = self.trailing_ec.recovery(cmat_trailing_ec)
        output = self.decoder.decode(dmat)

        return output

    def get_infidelity(self):
        out_channel = self.run()
        infidelity = average_infidelity(out_channel.channel_matrix)
        self.infidelity = infidelity
        return infidelity

    def get_infidelity_leading_ec(self):
        """Decodes only the leading EC."""
        cmat = self.leading_ec.run()
        dmat = self.leading_ec.recovery(cmat)
        output = self.decoder.decode(dmat)
        infidelity = average_infidelity(output.channel_matrix)
        self.infidelity_leading_ec = infidelity
        return infidelity

    def get_infidelity_trailing_ec(self):
        """Decodes only the trailing EC, ignoring the leading EC. Note that waiting noise is absorbed to trailing EC."""
        cmat = self.trailing_ec.run()
        dmat = self.trailing_ec.recovery(cmat)
        output = self.decoder.decode(dmat)
        infidelity = average_infidelity(output.channel_matrix)
        self.infidelity_trailing_ec = infidelity
        return infidelity

    def update_code(self, alphas, squeezes=(0, 0)):
        """
        Updates amplitudes of data and ancilla mode.
        Args:
            alphas: list
                A list contains amplitudes of data and ancilla mode, in the form [alpha_data, alpha_anc].
            squeezes: list
                A list contains squeezes of data and ancilla mode, in the form [sq_data, sq_anc].
        """
        self.alpha_data, self.alpha_anc = alphas
        self.sq_data, self.sq_anc = squeezes
        self.code_params = [self.N, self.alpha_data, self.sq_data, self.M, self.alpha_anc, self.sq_anc]
        data = CatCode(N=self.N, r=self.sq_data, alpha=self.alpha_data, fockdim=DIM)
        anc = CatCode(N=self.M, r=self.sq_anc, alpha=self.alpha_anc, fockdim=DIM)
        self.leading_ec.update_code(data, anc)
        self.trailing_ec.update_code(data, anc)
        self.decoder.update_code(data)

    def update_wait_noise(self, eta):
        """
        Updates waiting noise.
        Args:
            eta: int
                Multiplication factor, with gate loss and dephasing as base.
        """
        self.gamma_wait = eta * self.gamma
        self.gamma_phi_wait = eta * self.gamma_phi
        self.eta = eta
        self.noise_params = [self.gamma, self.gamma_phi, self.eta]
        self.base_noise.update_wait_noise(self.gamma_wait, self.gamma_phi_wait)
        self.trailing_ec.update_in_noise(self.base_noise.loss_wait, self.base_noise.dephasing_wait)

    def update_measurement(self, meas_params):
        """
        Updates measurement offsets.
        Args:
            meas_params: list
                A list contains offsets of measurement of data mode and ancilla mode,
                in the form [offset_data, offset_anc].
        """
        self.offset_data, self.offset_anc = meas_params
        self.meas_params = [self.offset_data, self.offset_anc]
        meas_data, meas_anc = self.make_measurement()
        self.leading_ec.update_measurement(meas_data, meas_anc)
        self.trailing_ec.update_measurement(meas_data, meas_anc)
