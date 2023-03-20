import numpy as np
import qutip as qt


# Ported from git@github.com:arnelg/arXiv-1901.08071.git, with modifications

class CodeException(Exception):
    pass


class RotationalCode(object):

    def __init__(self, zero=None, one=None, plus=None, minus=None, N=None,
                 encoder=None, purity_threshold=1e-10):
        if encoder is not None:
            zero = encoder * qt.basis(2, 0)
            one = encoder * qt.basis(2, 1)
        self._encoder = encoder
        if plus is None and zero is not None and one is not None:
            self._plus = (zero + one) / np.sqrt(2)
        elif plus is not None:
            self._plus = plus
        if minus is None and zero is not None and one is not None:
            self._minus = (zero - one) / np.sqrt(2)
        elif minus is not None:
            self._minus = minus
        if zero is None and plus is not None and minus is not None:
            self._zero = (plus + minus) / np.sqrt(2)
        elif zero is not None:
            self._zero = zero
        if one is None and plus is not None and minus is not None:
            self._one = (plus - minus) / np.sqrt(2)
        elif one is not None:
            self._one = one
        self._N = N
        self._name = 'rotcode'

    def encoder(self, kraus=False):
        if self._encoder is None:
            self._encoder = (self.zero * qt.basis(2, 0).dag()
                             + self.one * qt.basis(2, 1).dag())
        if kraus:
            return self._encoder
        else:
            return qt.sprepost(self._encoder, self._encoder.dag())

    def decoder(self, kraus=False):
        S = self.encoder(kraus=True)
        if kraus:
            return S.dag()
        else:
            return qt.sprepost(S.dag(), S)

    @property
    def name(self):
        return self._name

    @property
    def zero(self):
        return self._zero

    @property
    def one(self):
        return self._one

    @property
    def plus(self):
        return self._plus

    @property
    def minus(self):
        return self._minus

    @property
    def codewords(self):
        return self.zero, self.one

    @property
    def projector(self):
        # Projector onto code space P_code
        return self.zero * self.zero.dag() + self.one * self.one.dag()

    @property
    def logical_Z(self):
        S = self.encoder(kraus=True)
        return S * qt.sigmaz() * S.dag()

    @property
    def logical_X(self):
        S = self.encoder(kraus=True)
        return S * qt.sigmax() * S.dag()

    @property
    def logical_H(self):
        S = self.encoder(kraus=True)
        return S * qt.hadamard_transform() * S.dag()

    @property
    def logical_Z_allspace(self):
        Q = self.identity - self.projector
        return self.logical_Z + Q

    @property
    def logical_X_allspace(self):
        Q = self.identity - self.projector
        return self.logical_X + Q

    @property
    def logical_H_allspace(self):
        Q = self.identity - self.projector
        return self.logical_H + Q

    @property
    def dim(self):
        # Hilbert space dimension
        return self.zero.dims[0][0]

    @property
    def identity(self):
        # Identity operator on full Hilbert space
        return qt.identity(self.dim)

    @property
    def annihilation_operator(self):
        return qt.destroy(self.dim)

    @property
    def number_operator(self):
        return qt.num(self.dim)

    @property
    def N(self):
        return self._N

    def codeaverage(self, op):
        # Return average tr(P_code/2 op)
        return qt.expect(op, 0.5 * self.projector)

    def codecheck(self, silent=False, atol=1e-6):
        # Check if code words are normalized
        x = np.abs(self.zero.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x - 1)
        x = np.abs(self.one.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x - 1)
        x = np.abs(self.plus.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x - 1)
        x = np.abs(self.minus.norm())
        if not np.isclose(x, 1.0, atol=atol):
            raise CodeException("code word not normalized", x - 1)
        # Check if code words are orthogonal
        x = np.abs((self.zero.dag() * self.one).tr())
        if not np.isclose(x, 0, atol=atol):
            raise CodeException("code word not orthogonal", x)
        x = np.abs((self.plus.dag() * self.minus).tr())
        if not np.isclose(x, 0, atol=atol):
            raise CodeException("code word not orthogonal", x)
        if not silent:
            print("Code check done!")

    def commutator_check(self, silent=False, atol=1e-5):
        # Check if commutator is one
        a = self.annihilation_operator
        c = self.codeaverage(a * a.dag() - a.dag() * a)
        if not np.isclose(c, 1., atol=atol):
            raise CodeException("commutator not one", c)
        if not silent:
            print("Commutator check done!")

    def check_truncation(self, n):
        P = qt.Qobj(np.diag(np.hstack((np.zeros(n), np.ones(self.dim - n)))))
        return self.codeaverage(P)

    def check_rotationsymmetry(self, N=None, tol=1e-8):
        if N is None:
            N = self.N
        ck = self.zero.data.toarray()
        if not np.isclose(np.sum(np.abs(self.zero.data.toarray()[::2 * N]) ** 2),
                          1.0, rtol=tol, atol=tol):
            raise CodeException("zero not rotation symmetric")
        if not np.isclose(np.sum(np.abs(self.one.data.toarray()[N::2 * N]) ** 2),
                          1.0, rtol=tol, atol=tol):
            raise CodeException("one not rotation symmetric")

    def deleter(self, kraus=False):
        nothing = qt.basis(1, 0)
        k_list = [nothing * qt.basis(self.dim, i).dag() for i in range(self.dim)]
        if kraus:
            return k_list
        else:
            return qt.kraus_to_super(k_list)


class TrivialCode(RotationalCode):
    def __init__(self):
        zero = qt.basis(2, 0)
        one = qt.basis(2, 1)
        RotationalCode.__init__(self, zero=zero, one=one, N=1)
        self._name = 'trivialcode'


class CatCode(RotationalCode):

    def __init__(self, N, r, alpha, fockdim):
        zero = qt.Qobj()
        one = qt.Qobj()
        for m in range(2 * N):
            phi = m * np.pi / N
            D = qt.displace(fockdim, alpha * np.exp(1j * phi))
            S = qt.squeeze(fockdim, r * np.exp(2j * (phi - np.pi / 2)))
            blade = D * S * qt.basis(fockdim, 0)
            zero += blade
            one += (-1) ** m * blade
        zero = zero / zero.norm()
        one = one / one.norm()
        self._alpha = alpha
        self._r = r
        RotationalCode.__init__(self, zero=zero, one=one, N=N)
        self._name = 'cat'

        self.codecheck(silent=True)
        self.commutator_check(silent=True)
