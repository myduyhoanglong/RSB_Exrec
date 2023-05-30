from threshold import *


def main():
    for N in [4]:
        threshold = Threshold(scheme=KNILL, N=N, gadget_type='exrec', group=False, opt_anc=False, fixed_wait=True)
        gamma_phi_power = np.linspace(-5, -3, 11)
        for power in gamma_phi_power:
            gamma_phi = 10 ** power
            threshold.search_threshold(scan='gamma', fixed_param=gamma_phi, x_start=1e-4)


if __name__ == "__main__":
    main()