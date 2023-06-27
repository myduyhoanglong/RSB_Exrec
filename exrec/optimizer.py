import scipy.optimize as optimize
from logger import Logger
from constants import *
import time


class OptimizerException(Exception):
    pass


class Optimizer:
    """
    A class that finds optimal working point for extended gadget.
    Knill EC: optimize (alpha_data, alpha_anc, offset_data, offset_anc, eta)
    Hybrid EC: optimize (alpha_data, offset_data, offset_anc, eta)
    """

    def __init__(self, gadget, benchmark):
        self.gadget = gadget
        self.benchmark = benchmark
        self.logger = Logger(data_filename='data.txt', log_filename='log.txt')
        self.f = None
        self.maxiter = 100
        if self.gadget.gamma > 0 and self.gadget.gamma_phi > 0:
            self.max_eta = min(0.1 / self.gadget.gamma, 0.1 / self.gadget.gamma_phi)
        elif self.gadget.gamma > 0:
            self.max_eta = 0.1 / self.gadget.gamma
        else:
            self.max_eta = MAX_ETA

    def optimize(self, init_params):
        """Optimization."""
        self.logger.write_optimize_log_header(self.gadget, init_params)

        if self.f is None:
            raise OptimizerException("Function to optimize is not initialized.")
        result = optimize.minimize(self.f, init_params, method='Nelder-Mead',
                                   options={'maxiter': self.maxiter, 'disp': True})
        optimal_params = list(result.x)
        # update exrec with optimal params
        self.f(optimal_params)

        infid_exrec = self.gadget.infidelity
        infid_benchmark = self.benchmark.infidelity
        diff = infid_exrec - infid_benchmark
        ratio = infid_exrec / infid_benchmark
        print("Done:", optimal_params, infid_exrec, infid_benchmark, diff, ratio)
        return optimal_params, ratio

    def update_function(self, scheme, group=False, opt_anc=False, fixed_wait=False):
        """
        Update function f to optimize, based on which parameters to optimize.
        Args:
            scheme: KNILL or HYBRID
            group: boolean
                Affect only KNILL scheme, to set data and ancilla to be the same or different.
            opt_anc: boolean
                Affect only HYBRID scheme, to optimize measurement offset of ancilla or not.
            fixed_wait: boolean
                Keep waiting time fixed or vary.
        Returns:

        """
        if scheme == KNILL:
            if group:
                if fixed_wait:
                    self.f = self.function_knill_group_fixed_wait()
                else:
                    self.f = self.function_knill_group()
            else:
                if fixed_wait:
                    self.f = self.function_knill_separate_fixed_wait()
                else:
                    self.f = self.function_knill_separate()
        elif scheme == HYBRID:
            if opt_anc:
                if fixed_wait:
                    self.f = self.function_hybrid_anc_fixed_wait()
                else:
                    self.f = self.function_hybrid_anc()
            else:
                if fixed_wait:
                    self.f = self.function_hybrid_no_anc_fixed_wait()
                else:
                    self.f = self.function_hybrid_no_anc()
        else:
            raise OptimizerException("Unknown scheme.")

    def function_knill_group(self):
        """
        Assumptions: alpha_data = alpha_anc, offset_data = offset_anc.
        """

        def f(params):
            st = time.time()
            alpha, offset, eta = params

            if alpha < 0 or alpha > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.gadget.update_alpha([alpha, alpha])
            self.gadget.update_measurement([offset, offset])
            self.gadget.update_wait_noise(eta)
            self.benchmark.update_noise(self.gadget.gamma_wait, self.gadget.gamma_phi_wait)

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)
            return infid_exrec / infid_benchmark

        return f

    def function_knill_group_fixed_wait(self):
        """
        Assumptions: alpha_data = alpha_anc, offset_data = offset_anc, eta=1.
        """

        def f(params):
            st = time.time()
            alpha, offset = params

            if alpha < 0 or alpha > ALPHA_MAX:
                return 10000

            self.gadget.update_alpha([alpha, alpha])
            self.gadget.update_measurement([offset, offset])

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)
            return infid_exrec / infid_benchmark

        return f

    def function_knill_separate(self):

        def f(params):
            st = time.time()
            alpha_data, alpha_anc, offset_data, offset_anc, eta = params

            if min(alpha_data, alpha_anc) < 0 or max(alpha_data, alpha_anc) > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.gadget.update_alpha([alpha_data, alpha_anc])
            self.gadget.update_measurement([offset_data, offset_anc])
            self.gadget.update_wait_noise(eta)
            self.benchmark.update_noise(self.gadget.gamma_wait, self.gadget.gamma_phi_wait)

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)
            return infid_exrec / infid_benchmark

        return f

    def function_knill_separate_fixed_wait(self):
        """
        Assumption: eta=1.
        """

        def f(params):
            st = time.time()
            alpha_data, alpha_anc, offset_data, offset_anc = params

            if min(alpha_data, alpha_anc) < 0 or max(alpha_data, alpha_anc) > ALPHA_MAX:
                return 10000

            self.gadget.update_alpha([alpha_data, alpha_anc])
            self.gadget.update_measurement([offset_data, offset_anc])

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)
            return infid_exrec / infid_benchmark

        return f

    def function_hybrid_anc(self):

        def f(params):
            st = time.time()
            alpha_data, offset_data, offset_anc, eta = params

            if alpha_data < 0 or alpha_data > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.gadget.update_alpha([alpha_data, ALPHA_MAX])
            self.gadget.update_measurement([offset_data, offset_anc])
            self.gadget.update_wait_noise(eta)
            self.benchmark.update_noise(self.gadget.gamma_wait, self.gadget.gamma_phi_wait)

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)

            return infid_exrec / infid_benchmark

        return f

    def function_hybrid_anc_fixed_wait(self):
        """
        Assumption: eta=1.
        """

        def f(params):
            st = time.time()
            alpha_data, alpha_anc, offset_data, offset_anc = params

            if alpha_data < 0 or alpha_data > ALPHA_MAX or alpha_anc < 0 or alpha_anc > ALPHA_MAX:
                return 10000

            self.gadget.update_alpha([alpha_data, alpha_anc])
            self.gadget.update_measurement([offset_data, offset_anc])

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)

            return infid_exrec / infid_benchmark

        return f

    def function_hybrid_no_anc(self):

        def f(params):
            st = time.time()
            alpha_data, offset_data, eta = params

            if alpha_data < 0 or alpha_data > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.gadget.update_alpha([alpha_data, ALPHA_MAX])
            self.gadget.update_measurement([offset_data, 0])
            self.gadget.update_wait_noise(eta)
            self.benchmark.update_noise(self.gadget.gamma_wait, self.gadget.gamma_phi_wait)

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)

            return infid_exrec / infid_benchmark

        return f

    def function_hybrid_no_anc_fixed_wait(self):
        """
        Assumption: eta=1.
        """

        def f(params):
            st = time.time()
            alpha_data, offset_data = params

            if alpha_data < 0 or alpha_data > ALPHA_MAX:
                return 10000

            self.gadget.update_alpha([alpha_data, ALPHA_MAX])
            self.gadget.update_measurement([offset_data, 0])

            infid_exrec = self.gadget.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)

            return infid_exrec / infid_benchmark

        return f
