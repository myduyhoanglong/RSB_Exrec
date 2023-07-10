"""Functions for finding threshold, finding optimal configs for exgad, scanning landscape."""

import numpy as np
from optimizer import *
from constants import *
from extended_gadget import ExtendedGadget
from rectangle import Rectangle
from noises import BenchMark
from models import HybridModel, KnillModel


class Threshold:

    def __init__(self, scheme, N, gadget_type='exrec', group=False, opt_anc=False, fixed_wait=False, squeeze=False):
        """
        Args:
            gadget_type: 'exrec', 'rec', 'model'.
            group: KNILL only. If true, set data and ancilla mode to be the same.
            opt_anc: HYBRID only. If true, optimize ancilla mode. Else, set alpha_anc = ALPHA_MAX, offset_anc = 0.
            fixed_wait: If true, eta=1.
            squeeze: If true, use the squeezed cat code.
        """
        self.scheme = scheme
        self.N = N
        self.gadget_type = gadget_type
        self.group = group
        self.opt_anc = opt_anc
        self.fixed_wait = fixed_wait
        self.squeeze = squeeze

    def search_threshold(self, scan='gamma', fixed_param=1e-4, x_start=1e-4):
        """
        Args:
            scan: 'gamma' or 'gamma_phi'
                Which noise is scanned.
            fixed_param: float
                Value of the other noise.
            x_start: float
                Starting value for scanned noise.
        """
        cross = False
        cnt = 0
        maxcnt = 1
        cutoff = 1e-7
        ratio = None
        x = x_start
        x_low = x_start
        x_high = x_start
        while (not cross or cnt < maxcnt) and x > cutoff:
            if ratio is None:
                x = x_start
            elif not cross and ratio >= 1:
                x = x / 2
            elif not cross and ratio < 1:
                x = 2 * x
            elif cross:
                x = (x_low + x_high) / 2
            if cross:
                cnt += 1

            if scan == 'gamma':
                curr_params, curr_ratio = self.optimize_fixed_noise(gamma=x, gamma_phi=fixed_param)
            elif scan == 'gamma_phi':
                curr_params, curr_ratio = self.optimize_fixed_noise(gamma=fixed_param, gamma_phi=x)
            else:
                raise Exception("Unknown scan parameter.")
            print(x, curr_params, curr_ratio)

            if ratio is not None and not cross and (curr_ratio >= 1 > ratio):
                cross = True
                x_high = x
                x_low = x / 2
            elif ratio is not None and not cross and (curr_ratio < 1 <= ratio):
                cross = True
                x_low = x
                x_high = 2 * x
            if cross and curr_ratio >= 1:
                x_high = x
            elif cross and curr_ratio < 1:
                x_low = x
            ratio = curr_ratio
            params = curr_params

        return x, params

    def optimize_fixed_noise(self, gamma, gamma_phi):
        """Find the optimal ratio, fixing loss and dephasing strength. Use multiple initial parameters for optimizer."""
        init_pairs = self.get_init_pairs()
        best_ratio = None
        best_params = None
        for init_pair in init_pairs:
            curr_params, curr_ratio = self.optimize_fixed_noise_with_init_params(gamma, gamma_phi, init_pair)
            if best_ratio is None or curr_ratio < best_ratio:
                best_ratio = curr_ratio
                best_params = curr_params
        return best_params, best_ratio

    def optimize_fixed_noise_with_init_params(self, gamma, gamma_phi, init_params):
        gadget, benchmark = self.make_gadget(gamma, gamma_phi)
        op = Optimizer(gadget, benchmark)
        op = self.update_filename(op)
        op.update_function(scheme=self.scheme, group=self.group, opt_anc=self.opt_anc, fixed_wait=self.fixed_wait,
                           squeeze=self.squeeze)

        try:
            params, ratio = op.optimize(init_params)
            op.logger.write_data_log(op.gadget, op.benchmark, init_params, params)
        except:
            params, ratio = op.logger.write_last_log_line_to_data(op.gadget, init_params)
            op.logger.write_fail_log()
        return params, ratio

    def get_init_pairs(self):
        "must be fixed to include squeeze"
        if self.scheme == KNILL:
            N = self.N
            M = N
            default_offset = -(np.pi / (2 * N) - np.pi / (2 * N * M))
            if N == 2:
                offset = -0.3
            else:
                offset = default_offset
            if self.group:
                if self.fixed_wait:
                    if self.squeeze:
                        # (alpha, sqz, offset)
                        init_pairs = [[2, 0, default_offset], [2, 0, default_offset / 3], [4, 0, default_offset],
                                      [4, 0, default_offset / 3], [6, 0, default_offset], [6, 0, default_offset / 3]]
                    else:
                        # (alpha, offset)
                        init_pairs = [[2, default_offset], [2, default_offset / 3], [4, default_offset],
                                      [4, default_offset / 3], [6, default_offset], [6, default_offset / 3]]
                else:
                    if self.squeeze:
                        # (alpha, sqz, offset, eta)
                        init_pairs = [[2, 0, default_offset, 5], [2, 0, default_offset / 3, 5],
                                      [4, 0, default_offset, 10], [4, 0, default_offset / 3, 10],
                                      [6, 0, default_offset, 15], [6, 0, default_offset / 3, 15]]
                    else:
                        # (alpha, offset, eta)
                        init_pairs = [[2, default_offset, 5], [2, default_offset / 3, 5], [4, default_offset, 10],
                                      [4, default_offset / 3, 10], [6, default_offset, 15], [6, default_offset / 3, 15]]
            else:
                if self.fixed_wait:
                    if self.squeeze:
                        # (alpha_data, alpha_anc, sqz_data, sqz_anc, offset_data, offset_anc)
                        init_pairs = [[5, 5, 0, 0, default_offset, default_offset],
                                      [5, 5, 0, 0, default_offset / 3, default_offset / 3],
                                      [6, 6, 0, 0, default_offset, default_offset],
                                      [8, 8, 0, 0, default_offset / 3, default_offset / 3]]
                    else:
                        # (alpha_data, alpha_anc, offset_data, offset_anc)
                        init_pairs = [[5, 5, default_offset, default_offset],
                                      [5, 5, default_offset / 3, default_offset / 3],
                                      [6, 6, default_offset, default_offset],
                                      [8, 8, default_offset / 3, default_offset / 3]]
                else:
                    if self.squeeze:
                        # (alpha_data, alpha_anc, sqz_data, sqz_anc, offset_data, offset_anc, eta)
                        if N == 2:
                            init_pairs = [[4, 4, 0, 0, offset, offset, 10],
                                          [4, 4, 0.5, 0.5, offset, offset, 10],
                                          [4, 4, 1, 1, offset, offset, 10],
                                          [6, 6, 0, 0, offset, offset, 5],
                                          [6, 6, 0.5, 0.5, offset, offset, 5],
                                          [6, 6, 1, 1, offset, offset, 5]]
                        elif N == 3:
                            offset_data = -0.1
                            offset_anc = -0.3
                            init_pairs = [[5, 7, 0.5, 0.5, offset_data, offset_anc, 20],
                                          [5, 7, 1, 1, offset_data, offset_anc, 20],
                                          [6, 6, 0.5, 0.5, offset_data, offset_anc / 2, 10],
                                          [6, 6, 1, 1, offset_data, offset_anc / 2, 10]]
                        elif N == 4:
                            offset_data = -0.08
                            offset_anc = -0.1
                            init_pairs = [[8, 8, 0.5, 0.5, offset_data, offset_anc, 9],
                                          [8, 8, 1, 1, offset_data, offset_anc, 9]]
                    else:
                        # (alpha_data, alpha_anc, offset_data, offset_anc, eta)
                        init_pairs = [[2, 8, default_offset, default_offset, 5],
                                      [2, 8, default_offset / 3, default_offset / 3, 5],
                                      [4, 8, default_offset, default_offset, 10],
                                      [4, 8, default_offset / 3, default_offset / 3, 10],
                                      [6, 8, default_offset, default_offset, 15],
                                      [6, 8, default_offset / 3, default_offset / 3, 15]]
        elif self.scheme == HYBRID:
            N = self.N
            M = 1
            default_offset = -(np.pi / (2 * N) - np.pi / (2 * N * N))
            if self.opt_anc:
                if self.fixed_wait:
                    if self.squeeze:
                        # (alpha_data, alpha_anc, sqz_data, sqz_anc, offset_data, offset_anc)
                        init_pairs = [[5, 8, 0, 0, default_offset, 0], [5, 8, 0, 0, default_offset / 3, 0],
                                      [8, 8, 0, 0, default_offset, 0], [8, 8, 0, 0, default_offset / 3, 0]]
                    else:
                        # (alpha_data, alpha_anc, offset_data, offset_anc)
                        init_pairs = [[5, 8, default_offset, 0], [5, 8, default_offset / 3, 0],
                                      [8, 8, default_offset, 0], [8, 8, default_offset / 3, 0]]
                else:
                    if self.squeeze:
                        # (alpha_data, alpha_anc, sqz_data, sqz_anc, offset_data, offset_anc, eta)
                        if N == 2:
                            offset_data = -0.03
                            offset_anc = 0
                            init_pairs = [[3, 8, 0.5, 0, offset_data, offset_anc, 40],
                                          [3, 8, 1, 0.3, offset_data, offset_anc, 40],
                                          [4, 8, 0.5, 0, offset_data, offset_anc, 20],
                                          [4, 8, 1, 0.3, offset_data, offset_anc, 20]]
                        if N == 3:
                            offset_data = -0.1
                            offset_anc = 0
                            init_pairs = [[5, 8, 0.5, 0, offset_data, offset_anc, 10],
                                          [5, 8, 1, 0.3, offset_data, offset_anc, 10],
                                          [7, 8, 0.5, 0, offset_data, offset_anc, 5],
                                          [7, 8, 1, 0.3, offset_data, offset_anc, 5]]
                        if N == 4:
                            offset_data = -0.05
                            offset_anc = 0
                            init_pairs = [[6, 8, 0.5, 0, offset_data, offset_anc, 10],
                                          [6, 8, 1, 0.3, offset_data, offset_anc, 10],
                                          [8, 8, 0.5, 0, offset_data, offset_anc, 5],
                                          [8, 8, 1, 0.3, offset_data, offset_anc, 5]]
                    else:
                        # (alpha_data, offset_data, offset_anc, eta)
                        init_pairs = [[2, default_offset, 0, 5], [2, default_offset / 3, 0, 5],
                                      [4, default_offset, 0, 10], [4, default_offset / 3, 0, 10],
                                      [6, default_offset, 0, 15], [6, default_offset / 3, 0, 15]]
            else:
                if self.fixed_wait:
                    if self.squeeze:
                        # (alpha_data, sqz_data, offset_data)
                        init_pairs = [[2, 0, default_offset], [2, 0, default_offset / 3],
                                      [4, 0, default_offset], [4, 0, default_offset / 3],
                                      [6, 0, default_offset], [6, 0, default_offset / 3]]
                    else:
                        # (alpha_data, offset_data)
                        init_pairs = [[2, default_offset], [2, default_offset / 3],
                                      [4, default_offset], [4, default_offset / 3],
                                      [6, default_offset], [6, default_offset / 3]]
                else:
                    if self.squeeze:
                        # (alpha_data, sqz_data, offset_data, eta)
                        init_pairs = [[2, 0, default_offset, 5], [2, 0, default_offset / 3, 5],
                                      [4, 0, default_offset, 10], [4, 0, default_offset / 3, 10],
                                      [6, 0, default_offset, 15], [6, 0, default_offset / 3, 15]]
                    else:
                        # (alpha_data, offset_data, eta)
                        init_pairs = [[2, default_offset, 5], [2, default_offset / 3, 5],
                                      [4, default_offset, 10], [4, default_offset / 3, 10],
                                      [6, default_offset, 15], [6, default_offset / 3, 15]]
        else:
            raise Exception("Unknown scheme.")

        return init_pairs

    def make_gadget(self, gamma, gamma_phi):
        scheme = self.scheme
        N = self.N
        recovery = MAXIMUM_LIKELIHOOD
        decoder = TRANSPOSE

        # code params
        alpha_data = 2
        sq_data = 0
        sq_anc = 0
        if scheme == KNILL:
            M = N
            alpha_anc = alpha_data
        elif scheme == HYBRID:
            M = 1
            alpha_anc = ALPHA_MAX
        else:
            raise Exception("Unknown scheme")

        # measurement params
        offset_data = 0
        offset_anc = 0

        # noise params
        eta = 1
        gamma_wait = gamma * eta
        gamma_phi_wait = gamma_phi * eta

        code_params = [N, alpha_data, sq_data, M, alpha_anc, sq_anc]
        meas_params = [offset_data, offset_anc]
        noise_params = [gamma, gamma_phi, eta]

        benchmark = BenchMark(gamma_wait, gamma_phi_wait)

        if self.gadget_type == 'exrec':
            gadget = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                                    noise_params=noise_params, recovery=recovery, decoder=decoder)
        elif self.gadget_type == 'rec':
            gadget = Rectangle(scheme=scheme, code_params=code_params, meas_params=meas_params,
                               noise_params=noise_params, recovery=recovery, decoder=decoder)
        elif self.gadget_type == 'model':
            if scheme == KNILL:
                gadget = KnillModel(code_params=code_params, meas_params=meas_params, noise_params=noise_params,
                                    recovery=recovery, decoder=decoder)
            elif scheme == HYBRID:
                gadget = HybridModel(code_params=code_params, meas_params=meas_params, noise_params=noise_params,
                                     recovery=recovery, decoder=decoder)
            else:
                raise Exception("Unknown scheme.")
        else:
            raise Exception("Unknown gadget type.")

        return gadget, benchmark

    def update_filename(self, op):
        if self.scheme == KNILL:
            data_file = 'data_knill.txt'
            log_file = 'log_knill.txt'
        elif self.scheme == HYBRID:
            data_file = 'data_hybrid.txt'
            log_file = 'log_hybrid.txt'
        else:
            data_file = 'data.txt'
            log_file = 'log.txt'
        if self.gadget_type == 'rec':
            data_file = 'rec_' + data_file
            log_file = 'rec_' + log_file
        elif self.gadget_type == 'model':
            data_file = 'model_' + data_file
            log_file = 'model_' + log_file
        op.logger.update_path_data(data_file)
        op.logger.update_path_log(log_file)
        return op
