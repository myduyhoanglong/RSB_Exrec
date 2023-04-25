import scipy.optimize as optimize
from logger import Logger
from constants import *
import time


class Optimizer:
    """
    A class that finds optimal working point for extended gadget.
    Knill EC: optimize (alpha_data, alpha_anc, offset_data, offset_anc, eta)
    Hybrid EC: optimize (alpha_data, offset_data, offset_anc, eta)
    """

    def __init__(self, exrec, benchmark):
        self.exrec = exrec
        self.benchmark = benchmark
        self.logger = Logger(data_filename='data.txt', log_filename='log.txt')
        self.maxiter = 60
        if self.exrec.gamma > 0 and self.exrec.gamma_phi > 0:
            self.max_eta = min(0.1 / self.exrec.gamma, 0.1 / self.exrec.gamma_phi)
        elif self.exrec.gamma > 0:
            self.max_eta = 0.1 / self.exrec.gamma
        else:
            self.max_eta = MAX_ETA

    def optimize_exrec(self, scheme, init_params):
        if scheme == KNILL:
            result = self.optimize_knill(init_params)
        elif scheme == HYBRID:
            result = self.optimize_hybrid(init_params)
        else:
            raise Exception("Invalid scheme.")
        return result

    def optimize_knill(self, init_params):
        if len(init_params) == 3:
            return self.optimize_knill_group(init_params)
        elif len(init_params) == 5:
            return self.optimize_knill_separate(init_params)
        else:
            raise Exception("Undefined number of initial parameters for Knill optimization.")

    def optimize_knill_group(self, init_params):
        """
        Finds (alpha_data, offset_data, alpha_anc, offset_anc, eta) that minimizes ratio between encoded and
        benchmark infidelity.
        Assumptions: alpha_data = alpha_anc, offset_data = offset_anc.
        Args:
            init_params: list
                A list of initial parameters [alpha, offset, eta].
        """

        self.logger.write_optimize_log_header(self.exrec, init_params)

        def f(params):
            st = time.time()
            alpha, offset, eta = params

            if alpha < 0 or alpha > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.exrec.update_alpha([alpha, alpha])
            self.exrec.update_measurement([offset, offset])
            self.exrec.update_wait_noise(eta)
            self.benchmark.update_noise(self.exrec.gamma_wait, self.exrec.gamma_phi_wait)

            infid_exrec = self.exrec.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)
            return infid_exrec / infid_benchmark

        optimal_params, ratio = self.run_optimizer(init_params, f)
        return [optimal_params, ratio]

    def optimize_knill_separate(self, init_params):
        """
        Finds (alpha_data, offset_data, alpha_anc, offset_anc, eta) that minimizes ratio between encoded and
        benchmark infidelity.
        Args:
            init_params: list
                A list of initial parameters [alpha_data, alpha_anc, offset_data, offset_anc, eta].
        """

        self.logger.write_optimize_log_header(self.exrec, init_params)

        def f(params):
            st = time.time()
            alpha_data, alpha_anc, offset_data, offset_anc, eta = params

            if min(alpha_data, alpha_anc) < 0 or max(alpha_data, alpha_anc) > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.exrec.update_alpha([alpha_data, alpha_anc])
            self.exrec.update_measurement([offset_data, offset_anc])
            self.exrec.update_wait_noise(eta)
            self.benchmark.update_noise(self.exrec.gamma_wait, self.exrec.gamma_phi_wait)

            infid_exrec = self.exrec.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)
            return infid_exrec / infid_benchmark

        optimal_params, ratio = self.run_optimizer(init_params, f)
        return [optimal_params, ratio]

    def optimize_hybrid(self, init_params):
        """
        Finds (alpha_data, offset_data, offset_anc, eta) that minimizes ratio between encoded and benchmark infidelity.
        Args:
            init_params: list
                A list of initial parameters [alpha_data, offset_data, offset_anc, eta].
        """

        self.logger.write_optimize_log_header(self.exrec, init_params)

        def f(params):
            """Function returns ratio between encoded and benchmark infidelity."""
            st = time.time()
            alpha, offset_data, eta = params

            if alpha < 0 or alpha > ALPHA_MAX:
                return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            self.exrec.update_alpha([alpha, ALPHA_MAX])
            self.exrec.update_measurement([offset_data, 0])
            self.exrec.update_wait_noise(eta)
            self.benchmark.update_noise(self.exrec.gamma_wait, self.exrec.gamma_phi_wait)

            infid_exrec = self.exrec.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)

            return infid_exrec / infid_benchmark

        optimal_params, ratio = self.run_optimizer(init_params, f)
        return optimal_params, ratio

    def optimize_hybrid_fixed_meas(self, init_params):
        """
        Finds (alpha_data, eta) that minimizes ratio between encoded and benchmark infidelity.
        Args:
            init_params: list
                A list of initial parameters [alpha_data, eta].
        """
        self.logger.write_optimize_log_header(self.exrec, init_params)

        def f(params):
            """Function returns ratio between encoded and benchmark infidelity."""
            st = time.time()
            # alpha, eta = params
            eta = params[0]

            # if alpha < 0 or alpha > ALPHA_MAX:
            #     return 10000
            if eta < 0 or eta > self.max_eta:
                return 10000

            # self.exrec.update_alpha([alpha, ALPHA_MAX])
            self.exrec.update_wait_noise(eta)
            self.benchmark.update_noise(self.exrec.gamma_wait, self.exrec.gamma_phi_wait)

            infid_exrec = self.exrec.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            self.logger.write_optimize_log(list(params), infid_exrec, infid_benchmark, elapse)

            return infid_exrec / infid_benchmark

        optimal_params, ratio = self.run_optimizer(init_params, f)
        return optimal_params, ratio

    def run_optimizer(self, init_params, f):
        """Optimization."""
        result = optimize.minimize(f, init_params, method='Nelder-Mead',
                                   options={'maxiter': self.maxiter, 'disp': True})
        optimal_params = list(result.x)
        # update exrec with optimal params
        f(optimal_params)

        infid_exrec = self.exrec.infidelity
        infid_benchmark = self.benchmark.infidelity
        diff = infid_exrec - infid_benchmark
        ratio = infid_exrec / infid_benchmark
        print("Done:", optimal_params, infid_exrec, infid_benchmark, diff, ratio)
        return [optimal_params, ratio]
