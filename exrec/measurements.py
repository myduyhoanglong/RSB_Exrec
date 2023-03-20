import numpy as np
import qutip as qt
from exrec.helpers import povm_mat


# Ported from git@github.com:arnelg/arXiv-1901.08071.git, with modifications


class POVMException(Exception):
    pass


class POVM():

    def __init__(self, povm_elements, kraus=None, outcomes=None):
        self._outcomes = outcomes
        self._povm_elements = povm_elements
        self._dims = povm_elements[0].dims
        self._kraus = kraus

    @property
    def dim(self):
        return self._dims[0]

    @property
    def sysdim(self):
        return self.dim[0]

    @property
    def povm_elements(self):
        return self._povm_elements

    @property
    def outcomes(self):
        return self._outcomes

    def complete_check(self, silent=False, atol=1e-9):
        if not qt.isequal(sum(self._povm_elements), qt.identity(self.dim), tol=atol):
            dist = qt.tracedist(sum(self._povm_elements), qt.identity(self.dim))
            raise POVMException("POVM not complete", dist)
        if not silent:
            print("POVM complete check done!")


class WedgeMeasurement(POVM):
    def __init__(self, slice, fockdim, base_offset, offset=0.):
        self.fockdim = fockdim
        self.base_offset = base_offset
        povm_list = []
        for m in range(slice):
            povm = povm_mat(m * 2 * np.pi / slice + self.base_offset + offset,
                            (m + 1) * 2 * np.pi / slice + self.base_offset + offset, self.fockdim)
            povm_list.append(povm)
        POVM.__init__(self, povm_list)
        self.noiseless_povm_elements = self.povm_elements
        self.noisy_povm_elements = None
        self.complete_check(silent=True)

    def noisy(self, noises):
        loss, dephasing = noises
        new_povm = self.povm_elements
        if loss is not None:
            new_povm = [sum([k.dag() * m * k for k in loss.kraus]) for m in self.noiseless_povm_elements]  # dual loss
        if dephasing is not None:
            new_povm = [dephasing(m) for m in new_povm]
        POVM.__init__(self, new_povm)
        self.noisy_povm_elements = self.povm_elements
        self.complete_check(silent=True)

    def apply_dephasing(self, dephasing):
        # Apply dephasing on top of base noise.
        if dephasing is not None:
            new_povm = [dephasing(m) for m in self.noisy_povm_elements]
            POVM.__init__(self, new_povm)


class LogicalMeasurement(POVM):
    def __init__(self, slice, fockdim, base_offset, offset=0.):
        self.fockdim = fockdim
        self.base_offset = base_offset
        povm_plus = 0
        povm_minus = 0
        for m in range(slice):
            povm = povm_mat(m * 2 * np.pi / slice + self.base_offset + offset,
                            (m + 1) * 2 * np.pi / slice + self.base_offset + offset, self.fockdim)
            if m % 2:
                povm_minus += povm
            else:
                povm_plus += povm
        povm_list = [povm_plus, povm_minus]
        POVM.__init__(self, povm_list)
        self.noiseless_povm_elements = self.povm_elements
        self.noisy_povm_elements = None
        self.complete_check(silent=True)

    def noisy(self, noises):
        loss, dephasing = noises
        new_povm = self.povm_elements
        if loss is not None:
            new_povm = [sum([k.dag() * m * k for k in loss.kraus]) for m in self.noiseless_povm_elements]  # dual loss
        if dephasing is not None:
            new_povm = [dephasing(m) for m in new_povm]
        POVM.__init__(self, new_povm)
        self.noisy_povm_elements = self.povm_elements
        self.complete_check(silent=True)

    def apply_dephasing(self, dephasing):
        # Apply dephasing on top of base noise.
        if dephasing is not None:
            new_povm = [dephasing(m) for m in self.noisy_povm_elements]
            POVM.__init__(self, new_povm)


class ADHWedgeMeasurement(POVM):
    def __init__(self, N, dim, adh_mat, offset=0.):
        self.s = dim
        self.adh = adh_mat
        povm_list = []
        for m in range(N):
            povm = self._povm(m * 2 * np.pi / N + offset, (m + 1) * 2 * np.pi / N + offset)
            povm_list.append(povm)
        POVM.__init__(self, povm_list)

    def _povm(self, theta1, theta2):
        def matrixelement(n, m, theta1, theta2):
            diag = theta2 - theta1
            off_diag = 1j / (n - m) * (np.exp(1j * theta1 * (n - m)) - np.exp(1j * theta2 * (n - m))) \
                       * self.adh[:self.s, :self.s]
            return np.where(n == m, diag, off_diag)

        povm = 1 / (2 * np.pi) * np.fromfunction(lambda n, m: matrixelement(n, m, theta1, theta2), (self.s, self.s))
        return qt.Qobj(povm)


class PhaseMeasurement(POVM):
    def __init__(self, dim, offset=0.):
        outcomes = []
        povm_list = []
        for r in range(dim):
            outcomes.append((2 * np.pi * r / dim + offset) % (2 * np.pi))
            phi = qt.phase_basis(dim, r, phi0=offset)
            povm_list.append(phi * phi.dag())
        POVM.__init__(self, povm_list, kraus=povm_list, outcomes=outcomes)


class ADHMatrix():
    def __init__(self, dim):
        self.dim = dim
        self.h = self._h_matrix(dim)
        np.save('adh_%d' % dim, self.h)

    def _h_matrix(self, dim):
        print("Gamma...")
        log_gm = self.log_gamma(dim)
        print("Cmat...")
        cmat = self.C(dim)
        print("Hmat...")
        hmat = np.zeros((dim, dim))
        for m in range(dim):
            for n in range(m, dim):
                h = 0
                for p in range(int(m / 2) + 1):
                    for q in range(int(n / 2) + 1):
                        h += np.exp(log_gm[m, p] + log_gm[n, q]) * cmat[m, n, p, q]
                hmat[m, n] = h
                hmat[n, m] = h
        return hmat

    def log_gamma(self, dim):
        gm = np.zeros((dim, int((dim + 1) / 2)))
        for m in range(dim):
            for p in range(int(m / 2) + 1):
                if p == 0:
                    gm[m, p] = -0.5 * self.log_fact(m)
                else:
                    gm[m, p] = gm[m, p - 1] + np.log(m - 2 * p + 1) + np.log(m - 2 * p + 2) - np.log(2 * p)
        return gm

    def C(self, dim):
        cmat = np.zeros((dim, dim, int((dim + 1) / 2), int((dim + 1) / 2)))
        print("Moment...")
        log_moment = self.log_moment(dim)
        print("binmat...")
        binmat = np.zeros((dim, dim, dim))
        binarr_plus = np.zeros((dim, dim))
        binarr_minus = np.zeros((dim, dim))
        for diff in range(dim):
            for i in range(dim):
                binarr_plus[diff, i] = self.bin(0.5 * diff, i)
                binarr_minus[diff, i] = self.bin(-0.5 * diff, i)
        for diff in range(dim):
            binmat[diff, :, :] = np.outer(binarr_plus[diff, :], binarr_minus[diff, :])
        # for diff in range(dim):
        #     for i in range(dim):
        #         for j in range(dim):
        #             binmat[diff, i, j] = self.bin(0.5 * diff, i) * self.bin(-0.5 * diff, j)
        print("cmat...")
        for m in range(dim):
            print(m)
            for n in range(m, dim):
                for p in range(int(m / 2) + 1):
                    for q in range(int(n / 2) + 1):
                        d = max(max(m, n) + 1, 5)
                        c = np.sum(np.multiply(binmat[n - m, :d, :d], np.exp(log_moment[p:p + d, q:q + d])))
                        cmat[m, n, p, q] = c
                        cmat[n, m, q, p] = c
        return cmat

    def log_moment(self, dim):
        d = dim + int((dim + 1) / 2) + 1
        moment = np.zeros((d, d))
        for m in range(d):
            cm = -self.log_fact(2 * m + 1, double=True)
            moment[m, 0] = cm
            moment[0, m] = cm
        for m in range(1, d):
            for n in range(1, d):
                moment[m, n] = np.log(m * np.exp(moment[m - 1, n]) + n * np.exp(moment[m, n - 1])) - np.log(
                    2 * ((m - n) ** 2) + m + n)
        return moment

    def bin(self, n, l):
        lb = 1
        if l != 0:
            for k in range(1, l + 1):
                lb *= (n - k + 1) / k
        return lb

    def log_fact(self, m, double=False):
        l = 0
        if m != 0:
            if not double:
                for k in range(1, m + 1):
                    l += np.log(k)
            else:
                if m % 2:
                    for k in range(1, m + 1, 2):
                        l += np.log(k)
        return l
