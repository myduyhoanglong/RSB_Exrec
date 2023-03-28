import time

import numpy as np

from optimizer import *
from constants import *
from extended_gadget import ExtendedGadget
from noises import BenchMark


def main():
    scan_hybrid()
    exit()
    scheme = HYBRID
    gamma_phi_list = [0]
    N_list = [2]
    thresholds = []
    for gamma_phi in gamma_phi_list:
        for N in N_list:
            thres = find_threshold(N, gamma_phi, gamma_start=1e-4, scheme=scheme)
            thresholds.append(thres)
    print(thresholds)


def find_threshold(N, gamma_phi, gamma_start=1e-4, scheme=KNILL):
    cross = False
    cnt = 0
    ratio = None
    gamma = gamma_start
    gamma_low = gamma_start
    gamma_high = gamma_start
    while not cross or cnt == 0:
        if ratio is None:
            gamma = gamma_start
        elif not cross and ratio > 1:
            gamma = gamma / 2
        elif not cross and ratio < 1:
            gamma = 2 * gamma
        elif cross:
            gamma = (gamma_low + gamma_high) / 2
        if cross:
            cnt += 1
        curr_ratio = find_optimal(N, gamma, gamma_phi, scheme)
        print(gamma, curr_ratio)
        if ratio is not None and not cross and (curr_ratio > 1 > ratio):
            cross = True
            gamma_high = gamma
            gamma_low = gamma / 2
        elif ratio is not None and not cross and (curr_ratio < 1 < ratio):
            cross = True
            gamma_low = gamma
            gamma_high = 2 * gamma
        if cross and curr_ratio > 1:
            gamma_high = gamma
        elif cross and curr_ratio < 1:
            gamma_low = gamma
        ratio = curr_ratio

    return gamma


def find_optimal(N, gamma, gamma_phi, scheme):
    init_pairs = [(2, 5), (4, 10), (6, 15), (8, 20)]
    best = None
    for init_pair in init_pairs:
        print(N, gamma, gamma_phi, init_pair)
        if scheme == KNILL:
            curr = collect_knill(N, gamma, gamma_phi, init_pair)
        else:
            curr = collect_hybrid(N, gamma, gamma_phi, init_pair)
        if best is None or curr < best:
            best = curr
    return best


def get_gamma(N, gamma_phi):
    if N == 1:
        gamma_list = [8e-5, 9e-5, 1e-4, 2e-4, 4e-4]
    if N == 2:
        gamma_list = [8e-5, 9e-5, 1e-4, 2e-4, 4e-4]
    if N == 3:
        gamma_list = [8e-5, 9e-5, 1e-4]
    if N == 4:
        # gamma_list = [8e-5, 9e-5, 1e-4, 2e-4, 4e-4]
        gamma_list = [5e-4, 6e-4, 7e-4]
    if N == 5:
        # gamma_list = [8e-5, 9e-5, 1e-4, 2e-4, 4e-4]
        gamma_list = [5e-4, 6e-4, 7e-4]
    return gamma_list


def collect_knill(N, gamma, gamma_phi, init_pair):
    scheme = KNILL
    decoding_scheme = MAXIMUM_LIKELIHOOD
    ideal_decoder = FAST

    # code params
    alpha_data = 4.85
    M = N
    alpha_anc = alpha_data

    # measurement params
    offset_data = 0.3469
    offset_anc = offset_data

    # noise params
    eta = 1

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(noise_params)

    print(">>INITIALIZE EXREC...<<")
    st = time.time()
    exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                           noise_params=noise_params, recovery=decoding_scheme, decoder=ideal_decoder)
    print(">>DONE INITIALIZE EXREC...<<", time.time() - st)

    offset = -(np.pi / (2 * N) - np.pi / (2 * N * M))
    (alpha, eta) = init_pair
    init_params = [alpha, offset, eta]
    op = Optimizer(exrec, benchmark)
    try:
        result = op.optimize_threshold(init_params=init_params)
        ratio = result[2]
    except:
        ratio = op.write_last_log_line_to_data(init_params)
        op.log_fail()
    return ratio


def collect_hybrid(N, gamma, gamma_phi, init_pair):
    scheme = HYBRID
    decoding_scheme = MAXIMUM_LIKELIHOOD
    ideal_decoder = FAST

    # code params
    alpha_data = 4.85
    M = 1
    alpha_anc = ALPHA_MAX

    # measurement params
    offset_data = 0.3469
    offset_anc = offset_data

    # noise params
    eta = 1

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(noise_params)

    print(">>INITIALIZE EXREC...<<")
    st = time.time()
    exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                           noise_params=noise_params, recovery=decoding_scheme, decoder=ideal_decoder)
    print(">>DONE INITIALIZE EXREC...<<", time.time() - st)

    offset_data = -(np.pi / (2 * N) - np.pi / (2 * N * N))
    offset_anc = 0
    (alpha, eta) = init_pair
    init_params = [alpha, offset_data, offset_anc, eta]
    op = Optimizer(exrec, benchmark)
    try:
        result = op.optimize_threshold(init_params=init_params, scheme=HYBRID)
        ratio = result[2]
    except:
        ratio = op.write_last_log_line_to_data(init_params)
        op.log_fail()
    return ratio


def content_log_optimize_header(exrec):
    content = "====================================\n"
    if exrec.scheme == KNILL:
        content += "scheme:KNILL, "
    elif exrec.scheme == HYBRID:
        content += "scheme:HYBRID, "
    if exrec.decoding_scheme == MAXIMUM_LIKELIHOOD:
        content += "decoding_scheme:MAXIMUM_LIKELIHOOD, "
    elif exrec.decoding_scheme == DIRECT:
        content += "decoding_scheme:DIRECT, "
    if exrec.ideal_decoder == SDP:
        content += "ideal_decoder:SDP, "
    elif exrec.ideal_decoder == FAST:
        content += "ideal_decoder:FAST, "
    content += "code_params:" + str(exrec.code_params) + ", "
    content += "meas_params:" + str(exrec.meas_params) + ", "
    content += "noise_params:" + str(exrec.noise_params) + "\n"

    return content


def content_log_optimize(params, infid, infid_benchmark, elapse):
    content = ""
    content += "params:" + str(params) + ", "
    content += "encoded_infidelity:" + str(infid) + ", "
    content += "benchmark_infidelity:" + str(infid_benchmark) + ", "
    content += "diff:" + str(infid - infid_benchmark) + ", "
    content += "ratio:" + str(infid / infid_benchmark) + ", "
    content += "time:" + str(elapse) + "\n"

    return content


def scan_hybrid():
    scheme = HYBRID
    decoding_scheme = MAXIMUM_LIKELIHOOD
    ideal_decoder = FAST

    # code params
    N = 3
    alpha_data = 4.85
    M = 1
    alpha_anc = ALPHA_MAX

    # measurement params
    offset_data = -0.1
    offset_anc = 0.001

    # noise params
    gamma = 0.0016
    gamma_phi = 0
    eta = 1

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(noise_params)

    print(">>INITIALIZE EXREC...<<")
    st = time.time()
    exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                           noise_params=noise_params, recovery=decoding_scheme, decoder=ideal_decoder)
    print(">>DONE INITIALIZE EXREC...<<", time.time() - st)

    logger = Logger()
    logger.update_path_exrec('scan.txt')
    logger.update_path_optimize('log_scan.txt')

    logger.log(content_log_optimize_header(exrec), log_type=OPTIMIZE_LOG)

    alphas = np.linspace(start=3, stop=6, num=30)
    etas = np.linspace(start=1, stop=26, num=50)
    for alpha in alphas:
        for eta in etas:
            st = time.time()
            init_params = [alpha, eta]

            exrec.update_alpha([alpha, ALPHA_MAX])
            exrec.update_wait_noise(eta)
            benchmark.update_noise(exrec.noise_params)

            infid = exrec.get_infidelity()
            infid_benchmark = benchmark.get_infidelity()
            elapse = time.time() - st
            print(init_params, infid, infid_benchmark, infid / infid_benchmark, elapse)

            logger.log(content_log_optimize(list(init_params), infid, infid_benchmark, elapse), log_type=OPTIMIZE_LOG)


if __name__ == "__main__":
    main()
