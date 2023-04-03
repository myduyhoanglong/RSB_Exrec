from threshold import *


def main():
    optimize_fixed_noise_with_init_params(scheme=HYBRID, N=3, gamma=1e-4, gamma_phi=0, init_pair=[5, 10])
    optimize_fixed_noise_with_init_params(scheme=HYBRID, N=3, gamma=2e-4, gamma_phi=0, init_pair=[5, 11])
    optimize_fixed_noise_with_init_params(scheme=HYBRID, N=3, gamma=4e-4, gamma_phi=0, init_pair=[5, 10])
    optimize_fixed_noise_with_init_params(scheme=HYBRID, N=3, gamma=8e-4, gamma_phi=0, init_pair=[4, 12])
    optimize_fixed_noise_with_init_params(scheme=HYBRID, N=3, gamma=1.6e-3, gamma_phi=0, init_pair=[4, 12])
    optimize_fixed_noise_with_init_params(scheme=HYBRID, N=3, gamma=3.2e-3, gamma_phi=0, init_pair=[3, 14])


if __name__ == "__main__":
    main()
