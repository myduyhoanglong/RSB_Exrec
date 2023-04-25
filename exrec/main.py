from threshold import *


def main():
    search_threshold_fixed_dephasing(scheme=KNILL, N=2, gamma_phi=0, gamma_start=1e-4, model=True)
    find_boundary(scheme=KNILL, N=2, model=True)


if __name__ == "__main__":
    main()
