from threshold import *


def main():
    search_threshold_fixed_dephasing(scheme=KNILL, N=4, gamma_phi=0, gamma_start=1e-4, model=False)
    search_gamma_phi_threshold_fixed_loss(scheme=KNILL, N=4, gamma=0, gamma_phi_start=1e-4, model=False)
    find_boundary(scheme=KNILL, N=4, model=True)


if __name__ == "__main__":
    main()
