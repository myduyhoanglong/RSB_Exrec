import numpy as np
import qutip as qt
import re


def n_pow_op(base, powfac, dim):
    """Creates an operator base**(n**powfac) where n is the number operator"""
    op = np.zeros((dim, dim), dtype=np.complex_)
    for n in range(dim):
        op[n][n] = base ** (n ** powfac)
    return qt.Qobj(op)


def average_infidelity(gate):
    return 1 - qt.average_gate_fidelity(gate)


def get_nkraus(gamma):
    if gamma <= 1e-3:
        return 12
    elif gamma < 1e-2:
        return 21
    else:
        return 30


def matrixf(rho, f, safe=False):
    # Apply f(rho) to diagonalizable matrix rho
    abstol = 1e-8
    vals, vecs = rho.eigenstates()
    out = qt.Qobj(np.zeros(rho.shape), dims=rho.dims)
    for i, lam in enumerate(vals):
        if lam > abstol or not safe:
            out += f(lam) * vecs[i] * vecs[i].dag()
    return out


def phasestate(s, r, phi0=0., fockdim=None):
    if fockdim is None:
        fockdim = s
    phim = phi0 + (2.0 * np.pi * r) / s
    n = np.arange(s)
    data = 1.0 / np.sqrt(s) * np.exp(1.0j * n * phim)
    data = np.hstack((data, np.zeros(fockdim - s)))
    return qt.Qobj(data)


def povm_mat(theta1, theta2, dim):
    def matrixelement(n, m, theta1, theta2):
        return np.where(n == m,
                        theta2 - theta1,
                        1j / (n - m) * (np.exp(1j * theta1 * (n - m))
                                        - np.exp(1j * theta2 * (n - m))))

    povm = 1 / (2 * np.pi) * np.fromfunction(lambda n, m: matrixelement(n, m, theta1, theta2), (dim, dim))
    return qt.Qobj(povm)


def reformat_line_in_log_file(line):
    line = line.strip()
    line = re.sub(r':', '\":\"', line)
    subs = re.split(r', ', line)
    new_line = '\"' + subs[0]
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
