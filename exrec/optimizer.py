import scipy.optimize as optimize
from constants import *
import time


class Optimizer:
    def __init__(self, exrec, benchmark):
        self.exrec = exrec
        self.benchmark = benchmark
        self.logger = Logger()

    def optimize_code_and_measurement(self, init_params, separate=False):
        """Find the optimal params for codes and measurements, fixing all other params.
        Assume that alpha_data=alpha_ancilla and offset_data=offset_ancilla."""

        benchmark_infidelity = self.benchmark.get_infidelity()

        if separate:
            def f(params):
                st = time.time()
                alpha_data, alpha_anc, offset_data, offset_anc = params
                if min(alpha_data, alpha_anc) < 0 or max(alpha_data, alpha_anc) > ALPHA_MAX:
                    return 1
                self.exrec.update_alpha(alpha_data=alpha_data, alpha_anc=alpha_anc)
                self.exrec.update_measurement([offset_data, offset_anc])
                infidelity = self.exrec.get_infidelity()
                print(params, infidelity, benchmark_infidelity, time.time() - st)
                return infidelity
        else:
            def f(params):
                st = time.time()
                alpha, offset = params
                if alpha < 0 or alpha > ALPHA_MAX:
                    return 1
                self.exrec.update_alpha([alpha, alpha])
                self.exrec.update_measurement([offset, offset])
                infidelity = self.exrec.get_infidelity()
                print(params, infidelity, benchmark_infidelity, time.time() - st)
                return infidelity

        result = optimize.minimize(f, init_params, method='Nelder-Mead', options={'maxiter': 50, 'disp': True})
        optimal_params = result.x
        optimal_infidelity = f(optimal_params)
        print("Done:", optimal_params, optimal_infidelity)
        return [optimal_params, optimal_infidelity, self.benchmark.infidelity]

    def optimize_threshold(self, init_params, scheme=KNILL):
        if scheme == KNILL:
            self.optimize_threshold_knill(init_params)
        elif scheme == HYBRID:
            self.optimize_threshold_hybrid(init_params)
        else:
            raise Exception("Invalid scheme.")

    def optimize_threshold_knill(self, init_params):
        """Find the optimal code, measurement, and waiting time that minimize the difference between logical and
        physical infidelity. Assume that alpha_data=alpha_ancilla and offset_data=offset_ancilla.
        Version 2: optimizing for ratio between encoded and benchmark infidelity may be more suitable."""

        self.log_optimize_header(init_params)

        def f(params):
            st = time.time()
            alpha, offset, eta = params

            if alpha < 0 or alpha > ALPHA_MAX:
                return 10000
            if eta > self.exrec.max_eta:
                return 10000

            self.exrec.update_alpha([alpha, alpha])
            self.exrec.update_measurement([offset, offset])
            self.exrec.update_wait_noise(eta)

            self.benchmark.update_noise(self.exrec.noise_params)

            infid = self.exrec.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            # print(params, infid, infid_benchmark, infid - infid_benchmark, elapse)
            self.log_optimize(list(params), infid, infid_benchmark, elapse)
            # return infid - infid_benchmark
            return infid / infid_benchmark

        result = optimize.minimize(f, init_params, method='Nelder-Mead', options={'maxiter': 50, 'disp': True})
        optimal_params = list(result.x)
        optimal_result = f(optimal_params)
        optimal_infidelity = self.exrec.infidelity
        benchmark_infidelity = self.benchmark.infidelity
        optimal_diff = optimal_infidelity - benchmark_infidelity
        optimal_ratio = optimal_infidelity / benchmark_infidelity
        print("Done:", optimal_params, optimal_infidelity, benchmark_infidelity, optimal_diff, optimal_ratio)
        self.log_exrec(init_params, optimal_params)
        return [optimal_params, optimal_diff, optimal_ratio]

    def optimize_threshold_hybrid(self, init_params):
        """Find the optimal code, measurement, and waiting time that minimize the difference between logical and
        physical infidelity. Assume that alpha_anc = ALPHA_MAX.
        Version 2: optimizing for ratio between encoded and benchmark infidelity may be more suitable."""

        self.log_optimize_header(init_params)

        def f(params):
            st = time.time()
            alpha, offset_data, offset_anc, eta = params

            if alpha < 0 or alpha > ALPHA_MAX:
                return 10000
            if eta > self.exrec.max_eta:
                return 10000

            self.exrec.update_alpha([alpha, ALPHA_MAX])
            self.exrec.update_measurement([offset_data, offset_anc])
            self.exrec.update_wait_noise(eta)

            self.benchmark.update_noise(self.exrec.noise_params)

            infid = self.exrec.get_infidelity()
            infid_benchmark = self.benchmark.get_infidelity()
            elapse = time.time() - st
            # print(params, infid, infid_benchmark, infid - infid_benchmark, elapse)
            self.log_optimize(list(params), infid, infid_benchmark, elapse)
            # return infid - infid_benchmark
            return infid / infid_benchmark

        result = optimize.minimize(f, init_params, method='Nelder-Mead', options={'maxiter': 50, 'disp': True})
        optimal_params = list(result.x)
        optimal_result = f(optimal_params)
        optimal_infidelity = self.exrec.infidelity
        benchmark_infidelity = self.benchmark.infidelity
        optimal_diff = optimal_infidelity - benchmark_infidelity
        optimal_ratio = optimal_infidelity / benchmark_infidelity
        print("Done:", optimal_params, optimal_infidelity, benchmark_infidelity, optimal_diff, optimal_ratio)
        self.log_exrec(init_params, optimal_params)
        return [optimal_params, optimal_diff, optimal_ratio]

    def log_exrec(self, init_params, optimal_params):
        content = ""
        if self.exrec.scheme == KNILL:
            content += "scheme:KNILL, "
        elif self.exrec.scheme == HYBRID:
            content += "scheme:HYBRID, "
        if self.exrec.decoding_scheme == MAXIMUM_LIKELIHOOD:
            content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        elif self.exrec.decoding_scheme == DIRECT:
            content += "decoding_scheme:DIRECT, "
        if self.exrec.ideal_decoder == SDP:
            content += "ideal_decoder:SDP, "
        elif self.exrec.ideal_decoder == FAST:
            content += "ideal_decoder:FAST, "
        content += "code_params:" + str(self.exrec.code_params) + ", "
        content += "meas_params:" + str(self.exrec.meas_params) + ", "
        content += "noise_params:" + str(self.exrec.noise_params) + ", "
        content += "init_params:" + str(init_params) + ", "
        content += "optimal_params:" + str(optimal_params) + ", "
        content += "encoded_infidelity:" + str(self.exrec.infidelity) + ", "
        content += "benchmark_infidelity:" + str(self.benchmark.infidelity) + ", "
        content += "diff:" + str(self.exrec.infidelity - self.benchmark.infidelity) + ", "
        content += "ratio:" + str(self.exrec.infidelity / self.benchmark.infidelity) + "\n"

        self.logger.log(content, log_type=EXREC_LOG)

        return content

    def log_optimize_header(self, init_params):
        content = "====================================\n"
        if self.exrec.scheme == KNILL:
            content += "scheme:KNILL, "
        elif self.exrec.scheme == HYBRID:
            content += "scheme:HYBRID, "
        if self.exrec.decoding_scheme == MAXIMUM_LIKELIHOOD:
            content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        elif self.exrec.decoding_scheme == DIRECT:
            content += "decoding_scheme:DIRECT, "
        if self.exrec.ideal_decoder == SDP:
            content += "ideal_decoder:SDP, "
        elif self.exrec.ideal_decoder == FAST:
            content += "ideal_decoder:FAST, "
        content += "code_params:" + str(self.exrec.code_params) + ", "
        content += "meas_params:" + str(self.exrec.meas_params) + ", "
        content += "noise_params:" + str(self.exrec.noise_params) + ", "
        content += "init_params:" + str(init_params) + "\n"

        self.logger.log(content, log_type=OPTIMIZE_LOG)

        return content

    def log_optimize(self, params, infid, infid_benchmark, elapse):
        content = ""
        content += "params:" + str(params) + ", "
        content += "encoded_infidelity:" + str(infid) + ", "
        content += "benchmark_infidelity:" + str(infid_benchmark) + ", "
        content += "diff:" + str(infid - infid_benchmark) + ", "
        content += "ratio:" + str(infid / infid_benchmark) + ", "
        content += "time:" + str(elapse) + "\n"

        self.logger.log(content, log_type=OPTIMIZE_LOG)

        return content

    def log_fail(self):
        content = "FAIL RUN\n"
        self.logger.log(content, log_type=OPTIMIZE_LOG)

    def write_last_log_line_to_data(self, init_params):
        content = ""
        if self.exrec.scheme == KNILL:
            content += "scheme:KNILL, "
        elif self.exrec.scheme == HYBRID:
            content += "scheme:HYBRID, "
        if self.exrec.decoding_scheme == MAXIMUM_LIKELIHOOD:
            content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        elif self.exrec.decoding_scheme == DIRECT:
            content += "decoding_scheme:DIRECT, "
        if self.exrec.ideal_decoder == SDP:
            content += "ideal_decoder:SDP, "
        elif self.exrec.ideal_decoder == FAST:
            content += "ideal_decoder:FAST, "
        content += "code_params:" + str(self.exrec.code_params) + ", "
        content += "meas_params:" + str(self.exrec.meas_params) + ", "
        content += "noise_params:" + str(self.exrec.noise_params) + ", "
        content += "init_params:" + str(init_params) + ", "

        with open(self.logger.log_path_optimize, 'r') as reader:
            line = reader.readlines()[-1]

        from helpers import reformat_line_in_log_file
        import json
        dict_str = reformat_line_in_log_file(line)
        data = json.loads(dict_str)

        content += "optimal_params:" + data['params'] + ", "
        content += "encoded_infidelity:" + data['encoded_infidelity'] + ", "
        content += "benchmark_infidelity:" + data['benchmark_infidelity'] + ", "
        content += "diff:" + data['diff'] + ", "
        content += "ratio:" + data['ratio'] + "\n"

        self.logger.log(content, log_type=EXREC_LOG)

        ratio = float(data['ratio'])
        return ratio


class Logger:
    def __init__(self):
        self.log_dir = './logs/'
        self.log_path_exrec = self.log_dir + 'data.txt'
        self.log_path_optimize = self.log_dir + 'log.txt'

    def update_path_exrec(self, path):
        self.log_path_exrec = self.log_dir + path

    def update_path_optimize(self, path):
        self.log_path_optimize = self.log_dir + path

    def log(self, log_content, log_type=EXREC_LOG):
        if log_type == EXREC_LOG:
            with open(self.log_path_exrec, 'a') as writer:
                writer.write(log_content)
        elif log_type == OPTIMIZE_LOG:
            with open(self.log_path_optimize, 'a') as writer:
                writer.write(log_content)
