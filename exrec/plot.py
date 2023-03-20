import matplotlib.pyplot as plt
import numpy as np
from constants import *
import json
import re


def main():
    plot = Plotter()
    plot.plot_threshold()


class Plotter:
    def __init__(self):
        self.data_path = './logs/data.txt'
        with open(self.data_path, 'r') as reader:
            self.lines = reader.read().splitlines()
        self.num_data = len(self.lines)
        self.data = self.process_data(self.lines)

    def process_data(self, lines):
        data = []
        for k, line in enumerate(lines):
            line = self.reformat_line(line)
            x = self.reformat_dict(line)
            data.append(x)
        return data

    def plot_threshold(self):
        gamma_phi = 0
        N_list = [1, 2, 3, 4, 5]
        gamma = []
        diff = [{}, {}, {}, {}, {}]
        ratio = [{}, {}, {}, {}, {}]
        for ptn in self.data:
            kt, kt_phi, _ = ptn['noise_params']
            N = ptn['code_params'][0]
            encoded_infid = ptn['encoded_infidelity']
            benchmark = ptn['benchmark_infidelity']
            r = encoded_infid / benchmark
            if kt_phi == gamma_phi:
                if kt not in gamma:
                    gamma.append(kt)
                if kt not in diff[N - 1].keys():
                    diff[N - 1][kt] = ptn['diff']
                    ratio[N - 1][kt] = r
                else:
                    if ptn['diff'] < diff[N - 1][kt]:
                        diff[N - 1][kt] = ptn['diff']
                    if r < ratio[N - 1][kt]:
                        ratio[N - 1][kt] = r
        x = sorted(gamma)
        markers = ['.', 'o', '+', '*', 'x']
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for N in N_list[1:]:
            xN = diff[N - 1].keys()
            xN = sorted(xN)
            # yN = [diff[N - 1][i] for i in xN]
            yN = [ratio[N - 1][i] for i in xN]
            # plt.scatter(xN, yN, label='N=%d' % N)
            plt.plot(xN, yN, label='N=%d' % N, linestyle='--', marker=markers[N - 1], color=colors[N - 1])
        # trivial = np.zeros_like(x)
        trivial = np.ones_like(x)
        plt.plot(x, trivial)
        # plt.yscale('symlog')
        plt.xscale('log')
        plt.xticks(ticks=[9e-5, 1e-4, 5e-4], labels=[r'$9 \times 10^{-5}$', r'$10^{-4}$', r'$5 \times 10^{-4}$'])
        # plt.yticks(ticks=[0.5, 1, 2, 3, 4, 5], labels=['0.5', '1', '2', '3', '4', '5'])

        plt.xlabel('$\gamma$')
        plt.ylabel('$IF_{encoded}$/$IF_{benchmark}$')
        # plt.ylabel('$IF_{encoded}$ - $IF_{benchmark}$')
        plt.title('$\gamma_{\phi}=0$')
        plt.legend(loc='lower right')

        plt.show()

    @staticmethod
    def reformat_line(line):
        line = re.sub(r':', '\":\"', line)
        subs = re.split(r', ', line)
        new_line = '\"' + subs[0] + '\"'
        for sub in subs[1:]:
            if (sub[0].isnumeric() or sub[0] == '-') and sub[-1].isnumeric():
                new_line += ',' + sub
            elif sub[-1].isnumeric() and '[' in sub:
                new_line += ',\"' + sub
            elif (sub[0].isnumeric() or sub[0] == '-') and ']' in sub:
                new_line += ',' + sub + '\"'
            else:
                new_line += ',\"' + sub + '\"'
        new_line = '{' + new_line + '}'
        return new_line

    @staticmethod
    def reformat_dict(dict_str):
        data = json.loads(dict_str)
        data['code_params'] = json.loads(data['code_params'])
        data['meas_params'] = json.loads(data['meas_params'])
        data['noise_params'] = json.loads(data['noise_params'])
        data['init_params'] = json.loads(data['init_params'])
        data['optimal_params'] = json.loads(data['optimal_params'])
        data['encoded_infidelity'] = float(data['encoded_infidelity'])
        data['benchmark_infidelity'] = float(data['benchmark_infidelity'])
        data['diff'] = float(data['diff'])
        return data


if __name__ == "__main__":
    main()
