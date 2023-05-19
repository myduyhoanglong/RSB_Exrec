from threshold import *


def main():
    for N in [3]:
        threshold = Threshold(scheme=KNILL, N=N, gadget_type='model', group=False, opt_anc=False, fixed_wait=True)
        threshold.search_threshold(scan='gamma', fixed_param=0, x_start=1e-4)
        gamma_phi_power = np.linspace(-5, -2.6, 13)
        for power in gamma_phi_power:
            gamma_phi = 10 ** power
            threshold.search_threshold(scan='gamma', fixed_param=gamma_phi, x_start=1e-4)


if __name__ == "__main__":
    main()