from threshold import *


def main():
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=1e-4, gamma_phi=0, init_pair=[5, 5, 10])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=2e-4, gamma_phi=0, init_pair=[5, 5, 10])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=4e-4, gamma_phi=0, init_pair=[4, 4, 20])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=8e-4, gamma_phi=0, init_pair=[3, 3, 70])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=1.6e-3, gamma_phi=0, init_pair=[3, 3, 40])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=1.8e-3, gamma_phi=0, init_pair=[3, 3, 40])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=2e-3, gamma_phi=0, init_pair=[3, 3, 40])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=2.2e-3, gamma_phi=0, init_pair=[3, 3, 40])
    optimize_fixed_noise_with_init_params(scheme=KNILL, N=2, gamma=2.4e-3, gamma_phi=0, init_pair=[3, 3, 40])


if __name__ == "__main__":
    main()
