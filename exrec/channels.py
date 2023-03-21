import numpy as np
import qutip as qt
from helpers import n_pow_op


# Ported from git@github.com:arnelg/arXiv-1901.08071.git, with modifications

class ChannelException(Exception):
    pass


class Channel():

    def __init__(self, kraus=None, channel_matrix=None, choi=None):
        self._kraus = kraus
        self._channel_matrix = channel_matrix
        self._choi = choi
        self._dim = None
        self._kraus_dual = None
        self._dual = None

    def __call__(self, other):
        return self.channel_matrix(other)

    @property
    def sysdim(self):
        if self._dim is None:
            self._dim = self.kraus[0].dims
        return self._dim

    @property
    def kraus(self):
        if self._kraus is not None:
            return self._kraus
        elif self._choi is not None:
            self._kraus = qt.choi_to_kraus(self._choi)
        elif self._channel_matrix is not None:
            self._choi = qt.super_to_choi(self._channel_matrix)
            self._kraus = qt.choi_to_kraus(self._choi)
        return self._kraus

    @property
    def channel_matrix(self):
        if self._channel_matrix is not None:
            return self._channel_matrix
        elif self._choi is not None:
            self._channel_matrix = qt.choi_to_super(self._choi)
        elif self._kraus is not None:
            self._channel_matrix = qt.kraus_to_super(self._kraus)
        return self._channel_matrix

    @property
    def choi(self):
        if self._choi is not None:
            return self._choi
        elif self._kraus is not None:
            self._choi = qt.kraus_to_choi(self._kraus)
        elif self._channel_matrix is not None:
            self._choi = qt.super_to_choi(self._channel_matrix)
        return self._choi

    @property
    def eigenvalues(self):
        return self._choi.eigenenergies()

    @property
    def dual(self):
        if self._dual is None:
            if self._kraus_dual is None:
                self._kraus_dual = [k.dag() for k in self._kraus]
            self._dual = Channel(kraus=self._kraus_dual)
        return self._dual

    def tp_check(self, silent=False, atol=1e-7):
        """ QuTiP istp check is too strcit. """
        if self._kraus is not None:
            tmp = 0
            for kr in self._kraus:
                tmp += kr.dag() * kr
        elif self._choi is not None:
            tmp = self._choi.ptrace([0])
        else:
            tmp = qt.super_to_choi(self._channel_matrix).ptrace([0])
        ide = qt.identity(tmp.shape[0])
        try:
            elems = (tmp-ide).data
            dist = np.max(np.abs(elems))
        except:
            print(tmp)
            print(self.kraus)
            exit()
        if not qt.isequal(tmp, ide, tol=atol):
            print(tmp)
            raise ChannelException("channel not trace preserving", dist)
        if not silent:
            print("trace preserving check done!")

    def cp_check(self, silent=False, atol=1e-9):
        """ QuTiP iscp check is too strict. """
        if not self.channel_matrix.iscp:
            lam = self.choi.eigenenergies()
            iscp = (np.all(np.real(lam) > -atol)
                    and np.all(np.abs(np.imag(lam)) < atol))
        else:
            iscp = True
        if not iscp:
            raise ChannelException("channel not completely positive")
        if not silent:
            print("completely positive check done!")

    def cptp_check(self):
        self.cp_check()
        self.tp_check()


class IdentityChannel(Channel):

    def __init__(self, dim):
        k_list = [qt.identity(dim)]
        Channel.__init__(self, k_list)
        self.tp_check(silent=True)


class LossChannel(Channel):

    def __init__(self, gamma, dim, lmax):
        kt = gamma
        a = qt.destroy(dim)
        x = np.exp(-kt)
        n_fac = n_pow_op(np.sqrt(x), 1, dim)
        k_list = [n_fac]
        for l in range(1, lmax):
            k_list.append(np.sqrt((1 - x) / l) * k_list[l - 1] * a)
        Channel.__init__(self, kraus=k_list)
        self._kt = kt
        self._dim = dim
        self._num_kraus = lmax

        self.tp_check(silent=True)

    def propagate(self, dim, phi):
        """
        Get rotation errors propagated to the second mode, due to losses in the first mode, through a CROT gate.
        dim: second mode's dimension.
        phi: rotation angle of CROT gate.
        """
        n = qt.num(dim)
        if self._kt == 0:
            phase = [0 * n for k in range(len(self.kraus))]
            phase[0] = phase[0].expm()
        else:
            phase = [(1j * phi * k * n).expm() for k in range(len(self.kraus))]
        return phase


class DephasingChannel(Channel):
    """Exact dephasing channel."""

    def __init__(self, gamma, dim):
        def block(k, l):
            b = np.zeros((dim, dim))
            b[k, l] = np.exp(-0.5 * kt * ((k - l) ** 2))
            return b

        kt = gamma
        choi_blocks = np.array([[block(r_ix, c_ix) for c_ix in range(dim)] for r_ix in range(dim)])
        choi_mat = qt.Qobj(inpt=np.hstack(np.hstack(choi_blocks)),
                           dims=[[[dim], [dim]], [[dim], [dim]]],
                           type='super', superrep='choi')
        Channel.__init__(self, choi=choi_mat)
        self._channel_matrix = qt.choi_to_super(self._choi)
        self._kt = kt

        self.tp_check(silent=True)
