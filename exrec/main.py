from threshold import *


def main():
    threshold = Threshold(scheme=HYBRID, N=2, gadget_type='exrec', group=False, opt_anc=True, fixed_wait=False,
                          squeeze=True)

    threshold.search_threshold(scan='gamma', fixed_param=0, x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-4), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-3.8), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-3.6), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-3.4), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-3.2), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-3), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-2.8), x_start=1e-4)
    threshold.search_threshold(scan='gamma', fixed_param=10 ** (-2.6), x_start=1e-4)


if __name__ == "__main__":
    main()
