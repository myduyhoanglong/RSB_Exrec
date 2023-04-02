from threshold import *


def main():
    N_list = [2, 3, 4]
    gamma_list = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3]
    for N in N_list:
        for gamma in gamma_list:
            optimize_fixed_noise(scheme=HYBRID, N=N, gamma=gamma, gamma_phi=0, model=True)


if __name__ == "__main__":
    main()
