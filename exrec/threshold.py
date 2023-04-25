"""Functions for finding threshold, finding optimal configs for exgad, scanning landscape."""

import numpy as np
from optimizer import *
from constants import *
from extended_gadget import ExtendedGadget
from noises import BenchMark
from models import HybridModel, KnillModel


def find_boundary(scheme, N, model=False):
    gamma_phi_power = np.linspace(-4, -2, 11)
    bdr_set = []
    path = './logs/boundary.txt'
    for power in gamma_phi_power:
        gamma_phi = 10 ** power
        gamma, params = search_threshold_fixed_dephasing(scheme, N, gamma_phi, gamma_start=1e-4, model=model)
        bdr_set.append((gamma, gamma_phi, params))
        with open(path, 'a') as writer:
            writer.write(str((N, gamma, gamma_phi, params)) + '\n')
    return bdr_set


def search_threshold_fixed_dephasing(scheme, N, gamma_phi, gamma_start=1e-4, model=False):
    """Binary search for loss threshold (gamma), fixing dephasing strength. Search until cross
    encoded=benchmark line, then go back once."""
    cross = False
    cnt = 0
    maxcnt = 2
    cutoff = 1e-6
    ratio = None
    gamma = gamma_start
    gamma_low = gamma_start
    gamma_high = gamma_start
    while (not cross or cnt < maxcnt) and gamma > cutoff:
        if ratio is None:
            gamma = gamma_start
        elif not cross and ratio >= 1:
            gamma = gamma / 2
        elif not cross and ratio < 1:
            gamma = 2 * gamma
        elif cross:
            gamma = (gamma_low + gamma_high) / 2
        if cross:
            cnt += 1
        curr_params, curr_ratio = optimize_fixed_noise(scheme, N, gamma, gamma_phi, model)
        print(gamma, curr_params, curr_ratio)
        if ratio is not None and not cross and (curr_ratio >= 1 > ratio):
            cross = True
            gamma_high = gamma
            gamma_low = gamma / 2
        elif ratio is not None and not cross and (curr_ratio < 1 <= ratio):
            cross = True
            gamma_low = gamma
            gamma_high = 2 * gamma
        if cross and curr_ratio >= 1:
            gamma_high = gamma
        elif cross and curr_ratio < 1:
            gamma_low = gamma
        ratio = curr_ratio
        params = curr_params

    return gamma, params


def search_gamma_phi_threshold_fixed_loss(scheme, N, gamma, gamma_phi_start=1e-4, model=False):
    """Binary search for loss threshold (gamma), fixing dephasing strength. Search until cross
    encoded=benchmark line, then go back once."""
    cross = False
    cnt = 0
    maxcnt = 2
    cutoff = 1e-6
    ratio = None
    gamma_phi = gamma_phi_start
    gamma_phi_low = gamma_phi_start
    gamma_phi_high = gamma_phi_start
    while (not cross or cnt < maxcnt) and gamma_phi > cutoff:
        if ratio is None:
            gamma_phi = gamma_phi_start
        elif not cross and ratio > 1:
            gamma_phi = gamma_phi / 2
        elif not cross and ratio < 1:
            gamma_phi = 2 * gamma_phi
        elif cross:
            gamma_phi = (gamma_phi_low + gamma_phi_high) / 2
        if cross:
            cnt += 1
        curr_params, curr_ratio = optimize_fixed_noise(scheme, N, gamma, gamma_phi, model)
        print(gamma_phi, curr_ratio)
        if ratio is not None and not cross and (curr_ratio > 1 > ratio):
            cross = True
            gamma_phi_high = gamma_phi
            gamma_phi_low = gamma_phi / 2
        elif ratio is not None and not cross and (curr_ratio < 1 < ratio):
            cross = True
            gamma_phi_low = gamma_phi
            gamma_phi_high = 2 * gamma_phi
        if cross and curr_ratio > 1:
            gamma_phi_high = gamma_phi
        elif cross and curr_ratio < 1:
            gamma_phi_low = gamma_phi
        ratio = curr_ratio

    return gamma_phi


def optimize_fixed_noise(scheme, N, gamma, gamma_phi, model=False):
    """Find the optimal ratio, fixing loss and dephasing strength. Use multiple initial parameters for optimizer."""
    if scheme == KNILL:
        init_pairs = [(2, 8, 5), (4, 8, 10), (6, 8, 15), (8, 8, 20)]
    elif scheme == HYBRID:
        init_pairs = [(2, 5), (4, 10), (6, 15)]
    best_ratio = None
    best_params = None
    for init_pair in init_pairs:
        print(N, gamma, gamma_phi, init_pair)
        if model:
            curr_params, curr_ratio = optimize_model_fixed_noise_with_init_params(scheme, N, gamma, gamma_phi,
                                                                                  init_pair)
        else:
            curr_params, curr_ratio = optimize_fixed_noise_with_init_params(scheme, N, gamma, gamma_phi, init_pair)
        if best_ratio is None or curr_ratio < best_ratio:
            best_ratio = curr_ratio
            best_params = curr_params
    return best_params, best_ratio


def optimize_fixed_noise_with_init_params(scheme, N, gamma, gamma_phi, init_pair):
    """Find the optimal ratio, fixing loss and dephasing strength and initial parameters."""
    recovery = MAXIMUM_LIKELIHOOD
    decoder = TRANSPOSE

    # code params
    alpha_data = 4
    if scheme == KNILL:
        M = 1
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

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(gamma_wait, gamma_phi_wait)

    print(">>INITIALIZE EXREC...<<")
    st = time.time()
    exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                           noise_params=noise_params, recovery=recovery, decoder=decoder)
    print(">>DONE INITIALIZE EXREC...<<", time.time() - st)

    if scheme == KNILL:
        if len(init_pair) == 2:
            offset = -(np.pi / (2 * N) - np.pi / (2 * N * M))
            (alpha, eta) = init_pair
            init_params = [alpha, offset, eta]
        elif len(init_pair) == 3:
            offset_data = -(np.pi / (2 * N) - np.pi / (2 * N * M))
            offset_anc = -(np.pi / (2 * M) - np.pi / (2 * N * M))
            (alpha_data, alpha_anc, eta) = init_pair
            init_params = [alpha_data, alpha_anc, offset_data, offset_anc, eta]
        elif len(init_pair) == 5:
            (alpha_data, alpha_anc, offset_data, offset_anc, eta) = init_pair
            init_params = [alpha_data, alpha_anc, offset_data, offset_anc, eta]
    elif scheme == HYBRID:
        if len(init_pair) == 2:
            offset_data = -(np.pi / (2 * N) - np.pi / (2 * N * N))
            offset_anc = 0
            (alpha, eta) = init_pair
            init_params = [alpha, offset_data, eta]
        elif len(init_pair) == 3:
            offset_anc = 0
            (alpha, offset_data, eta) = init_pair
            init_params = [alpha, offset_data, eta]
    else:
        raise Exception("Unknown scheme")

    op = Optimizer(exrec, benchmark)
    if scheme == KNILL:
        data_file = 'data_knill.txt'
        log_file = 'log_knill.txt'
    elif scheme == HYBRID:
        data_file = 'data_hybrid.txt'
        log_file = 'log_hybrid.txt'
    else:
        data_file = 'data.txt'
        log_file = 'log.txt'
    op.logger.update_path_data(data_file)
    op.logger.update_path_log(log_file)

    try:
        params, ratio = op.optimize_exrec(scheme, init_params)
        op.logger.write_data_log(op.exrec, op.benchmark, init_params, params)
    except:
        params, ratio = op.logger.write_last_log_line_to_data(op.exrec, init_params)
        op.logger.write_fail_log()
    return params, ratio


def optimize_model_fixed_noise_with_init_params(scheme, N, gamma, gamma_phi, init_pair):
    """Find the optimal ratio, fixing loss and dephasing strength and initial parameters."""
    recovery = MAXIMUM_LIKELIHOOD
    decoder = TRANSPOSE

    # code params
    alpha_data = 4
    if scheme == KNILL:
        M = 1
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

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(gamma_wait, gamma_phi_wait)

    if scheme == KNILL:
        model = KnillModel(code_params=code_params, meas_params=meas_params, noise_params=noise_params,
                           recovery=recovery, decoder=decoder)
    elif scheme == HYBRID:
        model = HybridModel(code_params=code_params, meas_params=meas_params, noise_params=noise_params,
                            recovery=recovery, decoder=decoder)

    if scheme == KNILL:
        if len(init_pair) == 2:
            offset = -(np.pi / (2 * N) - np.pi / (2 * N * M))
            (alpha, eta) = init_pair
            init_params = [alpha, offset, eta]
        elif len(init_pair) == 3:
            offset_data = -(np.pi / (2 * N) - np.pi / (2 * N * M))
            offset_anc = -(np.pi / (2 * M) - np.pi / (2 * N * M))
            print(offset_data, offset_anc)
            (alpha_data, alpha_anc, eta) = init_pair
            init_params = [alpha_data, alpha_anc, offset_data, offset_anc, eta]
        elif len(init_pair) == 5:
            (alpha_data, alpha_anc, offset_data, offset_anc, eta) = init_pair
            init_params = [alpha_data, alpha_anc, offset_data, offset_anc, eta]
    elif scheme == HYBRID:
        if len(init_pair) == 2:
            offset_data = -(np.pi / (2 * N) - np.pi / (2 * N * N))
            offset_anc = 0
            (alpha, eta) = init_pair
        elif len(init_pair) == 3:
            offset_anc = 0
            (alpha, offset_data, eta) = init_pair
        init_params = [alpha, offset_data, eta]
    else:
        raise Exception("Unknown scheme")

    op = Optimizer(model, benchmark)
    op.logger.update_path_data('model_data.txt')
    op.logger.update_path_log('model_log.txt')
    try:
        print(init_params)
        params, ratio = op.optimize_exrec(scheme, init_params)
        op.logger.write_data_log(op.exrec, op.benchmark, init_params, params)
    except:
        ratio = None
        params = None
    return params, ratio


def scan_ec_fixed_meas(scheme, model=False):
    """Scan (alpha, eta) landscape, fixing measurement offsets."""
    alpha_low = 3
    alpha_high = 6
    alpha_num = 40
    eta_low = 1
    eta_high = 25
    eta_num = 50

    if model:
        log_filename = 'model_log_scan_2.txt'
    else:
        log_filename = 'log_scan.txt'

    recovery = MAXIMUM_LIKELIHOOD
    decoder = TRANSPOSE

    # code params
    N = 3
    alpha_data = 4.85
    if scheme == KNILL:
        M = N
        alpha_anc = alpha_data
    elif scheme == HYBRID:
        M = 1
        alpha_anc = ALPHA_MAX
    else:
        raise Exception("Unknown scheme")

    # measurement params
    offset_data = -0.1
    offset_anc = 0.001

    # noise params
    gamma = 1.6e-3
    gamma_phi = 0
    eta = 1

    gamma_wait = gamma * eta
    gamma_phi_wait = gamma_phi * eta

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(gamma_wait, gamma_phi_wait)

    if model:
        if scheme == HYBRID:
            exrec = HybridModel(code_params=code_params, meas_params=meas_params, noise_params=noise_params,
                                recovery=recovery, decoder=decoder)
        else:
            raise Exception("Unknown scheme")
    else:
        exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                               noise_params=noise_params, recovery=recovery, decoder=decoder)

    logger = Logger(log_filename=log_filename)
    logger.write_optimize_log_header(exrec, init_params=None)

    alphas = np.linspace(start=alpha_low, stop=alpha_high, num=alpha_num)
    etas = np.linspace(start=eta_low, stop=eta_high, num=eta_num)
    for alpha in alphas:
        for eta in etas:
            st = time.time()
            init_params = [alpha, eta]

            exrec.update_alpha([alpha, ALPHA_MAX])
            exrec.update_wait_noise(eta)
            benchmark.update_noise(exrec.gamma_wait, exrec.gamma_phi_wait)

            infid_exrec = exrec.get_infidelity()
            infid_benchmark = benchmark.get_infidelity()
            elapse = time.time() - st
            print(init_params, infid_exrec, infid_benchmark, infid_exrec / infid_benchmark, elapse)

            logger.write_optimize_log(list(init_params), infid_exrec, infid_benchmark, elapse)


def scan_ec_varied_meas(scheme, model=False):
    """Scan (offset_data) landscape, with optimal (alpha_data, eta) for each value of offset_data."""
    offset_low = -0.3
    offset_high = 0
    offset_num = 29

    if model:
        data_filename = 'model_meas_data.txt'
        log_filename = 'model_meas_log.txt'
    else:
        data_filename = 'meas_data.txt'
        log_filename = 'meas_log.txt'

    recovery = MAXIMUM_LIKELIHOOD
    decoder = TRANSPOSE

    # code params
    N = 2
    alpha_data = 4.85
    if scheme == KNILL:
        M = N
        alpha_anc = alpha_data
    elif scheme == HYBRID:
        M = 1
        alpha_anc = ALPHA_MAX
    else:
        raise Exception("Unknown scheme")

    # measurement params
    offset_data = -0.25
    offset_anc = 0.00

    # noise params
    gamma = 1.6e-3
    gamma_phi = 0
    eta = 1

    gamma_wait = gamma * eta
    gamma_phi_wait = gamma_phi * eta

    code_params = [N, alpha_data, M, alpha_anc]
    meas_params = [offset_data, offset_anc]
    noise_params = [gamma, gamma_phi, eta]

    benchmark = BenchMark(gamma_wait, gamma_phi_wait)

    if model:
        if scheme == HYBRID:
            exrec = HybridModel(code_params=code_params, meas_params=meas_params, noise_params=noise_params,
                                recovery=recovery, decoder=decoder)
        else:
            raise Exception("Unknown scheme")
    else:
        exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                               noise_params=noise_params, recovery=recovery, decoder=decoder)

    op = Optimizer(exrec, benchmark)
    op.logger.update_path_data(data_filename)
    op.logger.update_path_log(log_filename)

    # offsets = np.linspace(start=offset_low, stop=offset_high, num=offset_num)
    offsets = [-0.289, -0.049, -0.039, -0.032, -0.016, -0.017]
    alphas = [4.51, 2.61, 2.43, 2.23, 1.91]
    gammas = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
    # for offset in offsets:
    #     st = time.time()
    #     alpha = 3
    #     eta = 5
    #     init_params = [alpha, eta]
    #     exrec.update_measurement([offset, offset_anc])
    #     params, ratio = op.optimize_hybrid_fixed_meas(init_params)
    #     init_params = [alpha, offset, offset_anc, eta]
    #     params = [params[0], offset, offset_anc, params[1]]
    #     op.logger.write_data_log(op.exrec, op.benchmark, init_params, params)
    #     elapse = time.time() - st
    #     print(offset, params, ratio, elapse)

    for i, alpha in enumerate(alphas):
        code_params = [N, alpha, M, ALPHA_MAX]
        meas_params = [offsets[i], 0.0]
        noise_params = [gammas[i], 0, 1]
        exrec = ExtendedGadget(scheme=scheme, code_params=code_params, meas_params=meas_params,
                               noise_params=noise_params,
                               recovery=recovery, decoder=decoder)
        op = Optimizer(exrec, benchmark)
        init_params = [5]
        params, ratio = op.optimize_hybrid_fixed_meas(init_params)
        print(alpha, params, ratio)
