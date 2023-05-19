from constants import *


class Logger:
    def __init__(self, data_filename='data.txt', log_filename='log.txt'):
        self.log_dir = './logs/'
        self.path_data = self.log_dir + data_filename
        self.path_log = self.log_dir + log_filename

    def update_path_data(self, filename):
        self.path_data = self.log_dir + filename

    def update_path_log(self, filename):
        self.path_log = self.log_dir + filename

    def write(self, log_content, log_type):
        if log_type == DATA_LOG:
            with open(self.path_data, 'a') as writer:
                writer.write(log_content)
        elif log_type == OPTIMIZE_LOG:
            with open(self.path_log, 'a') as writer:
                writer.write(log_content)

    def write_data_log(self, exrec, benchmark, init_params, optimal_params):
        content = ""
        if exrec.scheme == KNILL:
            content += "scheme:KNILL, "
        elif exrec.scheme == HYBRID:
            content += "scheme:HYBRID, "
        if exrec.recovery == MAXIMUM_LIKELIHOOD:
            content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        elif exrec.recovery == DIRECT:
            content += "decoding_scheme:DIRECT, "
        if exrec.decoder == SDP:
            content += "ideal_decoder:SDP, "
        elif exrec.decoder == TRANSPOSE:
            content += "ideal_decoder:TRANSPOSE, "
        content += "code_params:" + str(exrec.code_params) + ", "
        content += "meas_params:" + str(exrec.meas_params) + ", "
        content += "noise_params:" + str(exrec.noise_params) + ", "
        content += "init_params:" + str(init_params) + ", "
        content += "optimal_params:" + str(optimal_params) + ", "
        content += "encoded_infidelity:" + str(exrec.infidelity) + ", "
        content += "benchmark_infidelity:" + str(benchmark.infidelity) + ", "
        content += "diff:" + str(exrec.infidelity - benchmark.infidelity) + ", "
        content += "ratio:" + str(exrec.infidelity / benchmark.infidelity) + "\n"

        self.write(content, log_type=DATA_LOG)

        return content

    def write_optimize_log_header(self, exrec, init_params):
        content = "====================================\n"
        if exrec.scheme == KNILL:
            content += "scheme:KNILL, "
        elif exrec.scheme == HYBRID:
            content += "scheme:HYBRID, "
        if exrec.recovery == MAXIMUM_LIKELIHOOD:
            content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        elif exrec.recovery == DIRECT:
            content += "decoding_scheme:DIRECT, "
        if exrec.decoder == SDP:
            content += "ideal_decoder:SDP, "
        elif exrec.decoder == TRANSPOSE:
            content += "ideal_decoder:TRANSPOSE, "
        content += "code_params:" + str(exrec.code_params) + ", "
        content += "meas_params:" + str(exrec.meas_params) + ", "
        content += "noise_params:" + str(exrec.noise_params) + ", "
        if init_params is not None:
            content += "init_params:" + str(init_params) + "\n"
        else:
            content += "init_params:[]\n"

        self.write(content, log_type=OPTIMIZE_LOG)

        return content

    def write_optimize_log(self, params, infid, infid_benchmark, elapse):
        content = ""
        content += "params:" + str(params) + ", "
        content += "encoded_infidelity:" + str(infid) + ", "
        content += "benchmark_infidelity:" + str(infid_benchmark) + ", "
        content += "diff:" + str(infid - infid_benchmark) + ", "
        content += "ratio:" + str(infid / infid_benchmark) + ", "
        content += "time:" + str(elapse) + "\n"

        self.write(content, log_type=OPTIMIZE_LOG)

        return content

    def write_fail_log(self):
        content = "FAIL RUN\n"
        self.write(content, log_type=OPTIMIZE_LOG)

    def write_last_log_line_to_data(self, exrec, init_params):
        """Write the last line in optimize log file to data log file. Used when optimization is terminated by
        exceptions."""
        content = ""
        if exrec.scheme == KNILL:
            content += "scheme:KNILL, "
        elif exrec.scheme == HYBRID:
            content += "scheme:HYBRID, "
        if exrec.recovery == MAXIMUM_LIKELIHOOD:
            content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        elif exrec.recovery == DIRECT:
            content += "decoding_scheme:DIRECT, "
        if exrec.decoder == SDP:
            content += "ideal_decoder:SDP, "
        elif exrec.decoder == TRANSPOSE:
            content += "ideal_decoder:TRANSPOSE, "
        content += "code_params:" + str(exrec.code_params) + ", "
        content += "meas_params:" + str(exrec.meas_params) + ", "
        content += "noise_params:" + str(exrec.noise_params) + ", "
        content += "init_params:" + str(init_params) + ", "

        from data_processing import process_log
        with open(self.path_log, 'r') as reader:
            line = reader.readlines()[-1]

        data = process_log([line])[0]

        content += "optimal_params:" + str(data['params']) + ", "
        content += "encoded_infidelity:" + str(data['encoded_infidelity']) + ", "
        content += "benchmark_infidelity:" + str(data['benchmark_infidelity']) + ", "
        content += "diff:" + str(data['diff']) + ", "
        content += "ratio:" + str(data['ratio']) + "\n"

        self.write(content, log_type=DATA_LOG)

        ratio = float(data['ratio'])
        params = data['params']
        return params, ratio

    def write_knill_data_point(self, ptn):
        content = ""
        content += "scheme:KNILL, "
        content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
        content += "ideal_decoder:TRANSPOSE, "
        content += "code_params:" + str(ptn['code_params']) + ", "
        content += "meas_params:" + str(ptn['meas_params']) + ", "
        content += "noise_params:" + str(ptn['noise_params']) + ", "
        content += "init_params:" + str(ptn['init_params']) + ", "
        content += "optimal_params:" + str(ptn['optimal_params']) + ", "
        content += "encoded_infidelity:" + str(ptn['encoded_infidelity']) + ", "
        content += "benchmark_infidelity:" + str(ptn['benchmark_infidelity']) + ", "
        content += "diff:" + str(ptn['encoded_infidelity'] - ptn['benchmark_infidelity']) + ", "
        content += "ratio:" + str(ptn['encoded_infidelity'] / ptn['benchmark_infidelity']) + "\n"
        print(content)

        self.write(content, log_type=DATA_LOG)

        return content
