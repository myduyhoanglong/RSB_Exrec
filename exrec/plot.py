import matplotlib.pyplot as plt
import numpy as np

from constants import *
from data_processing import *


def main():
    # path = './logs/model_data_knill_scan_boundary_N_3.txt'
    path = './logs/data_knill_gm_phi_zero.txt'
    plot = Plotter(path)
    plot.plot_boundary()
    # plot.plot_threshold_multiple_gm_phi()
    # plot.plot_threshold_with_full_data(gamma_phi=10**(-4))
    plot.plot_threshold_with_optimal_data(gamma_phi=0)
    exit()


def plot_boundary_group(scheme):
    if scheme == HYBRID:
        # N=2
        gm_2 = [0.0004, 0.00045081262897053026, 0.0004767095559115071, 0.0005001517672278046, 0.000534328179026622,
                0.0005591037825396579, 0.000559510459497694, 0.0005345680485596268, 0.0004428037217785514, 0.00029, 0]
        gm_phi_2 = [0, 0.0001, 0.00015848931924611142, 0.00025118864315095795, 0.00039810717055349735,
                    0.000630957344480193, 0.001, 0.001584893192461114, 0.0025118864315095794, 0.003981071705534973,
                    0.009]

        # N=3
        gm_3 = [0.0028, 0.002498704122999332, 0.002389420009396805, 0.0021330347567881684, 0.001758152021098984,
                0.001226095838681096, 0.0006988775963423187, 0.0002974428600639679, 0]
        gm_phi_3 = [0, 0.0001, 0.00015848931924611142, 0.00025118864315095795, 0.00039810717055349735,
                    0.000630957344480193, 0.001, 0.001584893192461114, 0.0025]

        # N=4
        gm_4 = [0.003750300002209621, 0.0033535328600611852, 0.0027543420337948436, 0.0017708713109561393,
                0.0008602854023008191, 0.0003536400165309926, 0.00010963454312795038]
        gm_phi_4 = [0.0001, 0.00015848931924611142, 0.00025118864315095795, 0.00039810717055349735,
                    0.000630957344480193, 0.0007943282347242813, 0.001]
    elif scheme == KNILL:
        # N=2
        gm_2 = [0.0002254997710763093, 0.00026574033498226517, 0.0002775692553976777, 0.0002888039291570443,
                0.00029231197088438657, 0.0002870320952882233, 0.0002613661062033355, 0.00020063684045711674]
        gm_phi_2 = [0, 0.0001, 0.00015848931924611142, 0.00025118864315095795, 0.00039810717055349735,
                    0.000630957344480193, 0.001, 0.001584893192461114]

        # N=3
        gm_3 = [0.0005079548427202572, 0.0004141630661508438, 0.0003561687748990968, 0.00030372567596689446,
                0.000252532303827931, 0.00021079078706956702, 0.00016903418319059332, 8.009750346915894e-05]
        gm_phi_3 = [0, 0.0001, 0.00015848931924611142, 0.00025118864315095795, 0.00039810717055349735,
                    0.000630957344480193, 0.001, 0.001584893192461114]

        # N=4
        gm_4 = []
        gm_phi_4 = []

    plt.plot(gm_2, gm_phi_2, linestyle='--', marker='.', label='N=2')
    plt.plot(gm_3, gm_phi_3, linestyle='--', marker='.', label='N=3')
    plt.plot(gm_4, gm_phi_4, linestyle='--', marker='.', label='N=4')
    plt.xlabel('$\gamma$')
    plt.ylabel('$\gamma_{\phi}$')
    plt.legend(loc='upper right')

    # plt.xscale('log')
    # plt.yscale('log')

    plt.show()


class Plotter:
    def __init__(self, path):
        self.data_path = path
        with open(self.data_path, 'r') as reader:
            self.lines = reader.read().splitlines()
        self.num_data = len(self.lines)
        self.data = None
        # self.data = process_data(self.lines)
        # self.data = process_log(self.lines[1:])

    def get_optimal_data(self, gamma=None, gamma_phi=None):
        """Given (scheme, N, gamma_phi, gamma), find the optimal point over many starting points. Returns the list of
        optimal points over the whole data file."""
        self.data = process_data(self.lines)
        ratio = [{}, {}, {}, {}, {}]
        opt_ptns = [{}, {}, {}, {}, {}]
        opt_data = []

        for ptn in self.data:
            kt, kt_phi, _ = ptn['noise_params']
            N = ptn['code_params'][0]
            if 'ratio' in ptn.keys():
                r = ptn['ratio']
            else:
                encoded_infid = ptn['encoded_infidelity']
                benchmark = ptn['benchmark_infidelity']
                r = encoded_infid / benchmark
            val = None
            if gamma_phi is not None and kt_phi == gamma_phi:
                val = kt
            elif gamma is not None and kt == gamma:
                val = kt_phi
            if val is None:
                continue
            elif val not in ratio[N - 1].keys():
                ratio[N - 1][val] = r
                opt_ptns[N - 1][val] = ptn
            else:
                if r < ratio[N - 1][val]:
                    ratio[N - 1][val] = r
                    opt_ptns[N - 1][val] = ptn

        for opt_N in opt_ptns:
            for ptn in opt_N.values():
                opt_data.append(ptn)
        return opt_data

    def get_full_data(self, gamma_phi):
        """Given (scheme, N, gamma_phi, gamma), find the optimal point over many starting points. Returns the list of
        all optimized points over the whole data file."""
        self.data = process_data(self.lines)
        ratio = [{}, {}, {}, {}, {}]
        opt_ptns = [{}, {}, {}, {}, {}]
        opt_data = []

        for ptn in self.data:
            kt, kt_phi, _ = ptn['noise_params']
            N = ptn['code_params'][0]
            if 'ratio' in ptn.keys():
                r = ptn['ratio']
            else:
                encoded_infid = ptn['encoded_infidelity']
                benchmark = ptn['benchmark_infidelity']
                r = encoded_infid / benchmark
            if kt_phi == gamma_phi:
                if kt not in ratio[N - 1].keys():
                    ratio[N - 1][kt] = [r]
                    opt_ptns[N - 1][kt] = [ptn]
                else:
                    if r not in ratio[N - 1][kt]:
                        ratio[N - 1][kt].append(r)
                        opt_ptns[N - 1][kt].append(ptn)

        for opt_N in opt_ptns:
            for ptns in opt_N.values():
                for ptn in ptns:
                    opt_data.append(ptn)
        return opt_data

    def plot_boundary(self):
        self.data = process_data(self.lines)
        gm_phi_list = []
        sep_data = {}
        for ptn in self.data:
            scheme = ptn['scheme']
            N = ptn['code_params'][0]

            gm, gm_phi, _ = ptn['noise_params']
            if gm_phi > 4e-3:
                continue
            if gm_phi not in gm_phi_list:
                gm_phi_list.append(gm_phi)
                sep_data[gm_phi] = []
            new_list = list(sep_data[gm_phi])
            new_list.append(ptn)
            sep_data[gm_phi] = new_list

        gm_phi_list = sorted(gm_phi_list)
        gm_list = []
        for gm_phi in gm_phi_list:
            ptns = sep_data[gm_phi]
            gms = []
            rs = []
            for ptn in ptns:
                gm, _, _ = ptn['noise_params']
                r = ptn['ratio']
                gms.append(gm)
                rs.append(r)
            fit_curve = np.poly1d(np.polyfit(np.array(rs), np.array(gms), deg=2))
            gm_list.append(fit_curve(1))

        print(gm_phi_list)
        print(gm_list)

        plt.scatter(gm_list, gm_phi_list, marker='.')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()

    def plot_threshold_multiple_gm_phi(self):
        self.data = process_data(self.lines)
        gm_phi_list = []
        sep_data = {}
        for ptn in self.data:
            scheme = ptn['scheme']
            N = ptn['code_params'][0]

            gm, gm_phi, _ = ptn['noise_params']
            if gm_phi > 4e-3:
                continue
            if gm_phi not in gm_phi_list:
                gm_phi_list.append(gm_phi)
                sep_data[gm_phi] = []
            new_list = list(sep_data[gm_phi])
            new_list.append(ptn)
            sep_data[gm_phi] = new_list

        gm_phi_list = sorted(gm_phi_list)
        gm_max = 0
        for gm_phi in gm_phi_list:
            ptns = sep_data[gm_phi]
            gms = []
            rs_dict = {}
            for ptn in ptns:
                gm, _, _ = ptn['noise_params']
                r = ptn['ratio']
                gms.append(gm)
                rs_dict[gm] = r
                if gm > gm_max:
                    gm_max = gm
            gms = sorted(gms)
            rs = []
            for gm in gms:
                rs.append(rs_dict[gm])
            plt.plot(gms, rs, label='$\gamma_{\phi}=%.2e$' % gm_phi, linestyle='--', marker='.')

        plt.plot([0, gm_max], [1, 1])
        plt.xscale('log')
        plt.xlabel('$\gamma$')
        plt.ylabel('$IF_{encoded}$/$IF_{benchmark}$')
        plt.legend(loc='lower right')

        plt.show()

    def plot_threshold_with_optimal_data(self, gamma=None, gamma_phi=None):
        data = self.get_optimal_data(gamma, gamma_phi)
        noises = []
        opt_ptns = [{}, {}, {}, {}, {}]
        ratio = [{}, {}, {}, {}, {}]
        opt_params = [{}, {}, {}, {}, {}]
        for ptn in data:
            kt, kt_phi, _ = ptn['noise_params']
            N = ptn['code_params'][0]
            if gamma_phi is not None and kt_phi == gamma_phi:
                val = kt
            elif gamma is not None and kt == gamma:
                val = kt_phi
            if val not in noises:
                noises.append(val)
            opt_ptns[N - 1][val] = ptn
            ratio[N - 1][val] = ptn['ratio']
            opt_params[N - 1][val] = ptn['optimal_params']

        N_list = [2, 3, 4, 5]
        markers = ['.', 'o', '+', '*', 'x']
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        noises = sorted(noises)

        for N in N_list:
            xN = sorted(ratio[N - 1].keys())
            yN = [ratio[N - 1][i] for i in xN]
            print(xN)
            print(yN)
            plt.plot(xN, yN, label='N=%d' % N, linestyle='--', marker=markers[N - 1], color=colors[N - 1])
            for i, x in enumerate(xN):
                mul = 0.0
                ptn = opt_ptns[N - 1][x]
                scheme = ptn['scheme']
                r = ptn['ratio']
                if scheme == 'HYBRID':
                    alpha = ptn['optimal_params'][0]
                    offset_data = ptn['optimal_params'][1]
                    if len(ptn['optimal_params']) == 3:
                        eta = ptn['optimal_params'][2]
                    else:
                        eta = ptn['optimal_params'][3]
                    txt = "(%.2f,%.3f,%.2f,%.3f)" % (alpha, offset_data, eta, r)
                elif scheme == 'KNILL':
                    if len(ptn['optimal_params']) == 5:
                        alpha_data = ptn['optimal_params'][0]
                        alpha_anc = ptn['optimal_params'][1]
                        offset_data = ptn['optimal_params'][2]
                        offset_anc = ptn['optimal_params'][3]
                        eta = ptn['optimal_params'][4]
                        txt = "(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)" % (
                            alpha_data, alpha_anc, offset_data, offset_anc, eta, r)
                    else:
                        alpha = ptn['optimal_params'][0]
                        offset = ptn['optimal_params'][1]
                        eta = ptn['optimal_params'][2]
                        txt = "(%.2f,%.2f,%.2f,%.2f)" % (alpha, offset, eta, r)
                print(N, x, yN[i], txt)
                if N <= 4:
                    plt.text(x=xN[i] * (1 + mul), y=yN[i] * (1 + mul), s=txt, fontdict=dict(color='black', size=8))

        trivial = np.ones_like(noises)
        plt.plot(noises, trivial)
        plt.xscale('log')
        plt.xticks(ticks=[9e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                   labels=[r'$9 \times 10^{-5}$', r'$10^{-4}$', r'$5 \times 10^{-4}$', r'$10^{-3}$',
                           r'$5 \times 10^{-3}$', r'$10^{-2}$'])
        plt.xlabel('$\gamma$')
        plt.ylabel('$IF_{encoded}$/$IF_{benchmark}$')
        plt.title('$\gamma_{\phi}=0$')
        plt.legend(loc='upper right')

        plt.show()

    def plot_threshold_with_full_data(self, gamma_phi):
        data = self.get_full_data(gamma_phi)
        print(len(data))
        gamma = []
        opt_ptns = [{}, {}, {}, {}, {}]
        for ptn in data:
            kt, kt_phi, _ = ptn['noise_params']
            N = ptn['code_params'][0]
            if kt_phi == gamma_phi:
                if kt not in gamma:
                    gamma.append(kt)
                if kt not in opt_ptns[N - 1].keys():
                    opt_ptns[N - 1][kt] = [ptn]
                else:
                    opt_ptns[N - 1][kt].append(ptn)
                # opt_ptns[N - 1][kt] = ptn
                # ratio[N - 1][kt] = ptn['ratio']
                # opt_params[N - 1][kt] = (ptn['optimal_params'][0], ptn['optimal_params'][-1])

        N_list = [2]
        markers = ['.', 'o', '+', '*', 'x']
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        gamma = sorted(gamma)

        for N in N_list:
            xN = sorted(opt_ptns[N - 1].keys())
            for i, x in enumerate(xN):
                ptns = opt_ptns[N - 1][x]
                for ptn in ptns:
                    r = ptn['ratio']
                    plt.scatter(x, r, label='N=%d' % N, marker=markers[N - 1], color=colors[N - 1])
                    scheme = ptn['scheme']
                    if scheme == 'HYBRID':
                        alpha = ptn['optimal_params'][0]
                        offset_data = ptn['optimal_params'][1]
                        if len(ptn['optimal_params']) == 3:
                            eta = ptn['optimal_params'][2]
                        else:
                            eta = ptn['optimal_params'][3]
                        txt = "(%.2f,%.3f,%.2f,%.3f)" % (alpha, offset_data, eta, r)
                    elif scheme == 'KNILL':
                        alpha_data = ptn['optimal_params'][0]
                        alpha_anc = ptn['optimal_params'][1]
                        offset_data = ptn['optimal_params'][2]
                        offset_anc = ptn['optimal_params'][3]
                        eta = ptn['optimal_params'][4]
                        txt = "(%.2f,%.2f,%.2f,%.2f,%.2f)" % (alpha_data, alpha_anc, offset_data, offset_anc, eta)
                    mul = 0
                    plt.text(x=xN[i] * (1 + mul), y=r * (1 + mul), s=txt, fontdict=dict(color='black', size=8))

        trivial = np.ones_like(gamma)
        plt.plot(gamma, trivial)
        plt.xscale('log')
        plt.xticks(ticks=[9e-5, 1e-4, 5e-4, 1e-3],
                   labels=[r'$9 \times 10^{-5}$', r'$10^{-4}$', r'$5 \times 10^{-4}$', r'$10^{-3}$'])
        plt.xlabel('$\gamma$')
        plt.ylabel('$IF_{encoded}$/$IF_{benchmark}$')
        plt.title('$\gamma_{\phi}=0$')
        # plt.legend(loc='lower right')

        plt.show()

    def plot_threshold(self, gamma_phi):
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
