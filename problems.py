"""
problems.py - Test problem definitions for IOE 511/MATH 562 project.
Provides all 12 required problems (P1-P12) via get_problem(name).

P1-P4 quadratic matrices Q are loaded from course-provided .mat files.
q vectors are generated with np.random.seed(0), matching Project_Problems.py.
"""

import numpy as np
import scipy.io
import os

_MAT_DIR = os.path.dirname(os.path.abspath(__file__))


class Problem:
    """Wraps an optimization problem and tracks evaluation counts."""

    def __init__(self, name, x0, f, g, H=None):
        self.name = name
        self.x0 = np.array(x0, dtype=float)
        self._f = f
        self._g = g
        self._H = H
        self.f_evals = 0
        self.g_evals = 0
        self.H_evals = 0

    def compute_f(self, x):
        self.f_evals += 1
        return float(self._f(x))

    def compute_g(self, x):
        self.g_evals += 1
        return np.array(self._g(x), dtype=float)

    def compute_H(self, x):
        if self._H is None:
            raise NotImplementedError(f"Hessian not provided for {self.name}")
        self.H_evals += 1
        return np.array(self._H(x), dtype=float)

    def reset_counters(self):
        self.f_evals = 0
        self.g_evals = 0
        self.H_evals = 0


# ===== P1-P4: Convex Quadratics =====
# f(x) = 0.5 x^T Q x + q^T x
# Q loaded from course-provided .mat files; q ~ N(0,1) with seed 0.

def _load_quad_mat(mat_name, n):
    """Load Q from a .mat file and generate q with the course-standard seed."""
    import scipy.sparse
    mat = scipy.io.loadmat(os.path.join(_MAT_DIR, mat_name))
    Q = mat['Q']
    if scipy.sparse.issparse(Q):
        Q = Q.toarray()   # .mat files store Q as sparse; convert to dense ndarray
    Q = np.array(Q, dtype=float)
    np.random.seed(0)
    q = np.random.normal(size=n)
    x0 = np.zeros(n)
    f = lambda x: float(0.5 * x @ Q @ x + q @ x)
    g = lambda x: Q @ x + q
    H = lambda x: Q.copy()
    return f, g, H, x0


def p1_quad_10_10():
    """Quadratic, n=10, kappa=10."""
    f, g, H, x0 = _load_quad_mat('quad_10_10_Q.mat', 10)
    return Problem('P1_quad_10_10', x0, f, g, H)


def p2_quad_10_1000():
    """Quadratic, n=10, kappa=1000."""
    f, g, H, x0 = _load_quad_mat('quad_10_1000_Q.mat', 10)
    return Problem('P2_quad_10_1000', x0, f, g, H)


def p3_quad_1000_10():
    """Quadratic, n=1000, kappa=10."""
    f, g, H, x0 = _load_quad_mat('quad_1000_10_Q.mat', 1000)
    return Problem('P3_quad_1000_10', x0, f, g, H)


def p4_quad_1000_1000():
    """Quadratic, n=1000, kappa=1000."""
    f, g, H, x0 = _load_quad_mat('quad_1000_1000_Q.mat', 1000)
    return Problem('P4_quad_1000_1000', x0, f, g, H)


# ===== P5-P6: Quartic =====
# f(x) = 0.5 x^T x + (sigma/4)(x^T Q x)^2
_Q_QUARTIC = np.array([[5, 1, 0, 0.5],
                        [1, 4, 0.5, 0],
                        [0, 0.5, 3, 0],
                        [0.5, 0, 0, 2]], dtype=float)
_X0_QUARTIC = np.array([np.cos(70), np.sin(70), np.cos(70), np.sin(70)])


def _make_quartic(sigma):
    Q = _Q_QUARTIC

    def f(x):
        return 0.5 * x @ x + (sigma / 4) * (x @ Q @ x) ** 2

    def g(x):
        # grad 0.5||x||^2 = x; grad (sigma/4)(x^T Q x)^2 = sigma*(x^T Q x)*(Q x)
        return x + sigma * (x @ Q @ x) * (Q @ x)

    def H(x):
        # Hess = I + sigma * [2*(Qx)(Qx)^T + (x^T Q x)*Q]
        Qx = Q @ x
        return np.eye(4) + sigma * (2 * np.outer(Qx, Qx) + (x @ Q @ x) * Q)

    return f, g, H


def p5_quartic_1():
    """Quartic with sigma=1e-4."""
    f, g, H = _make_quartic(1e-4)
    return Problem('P5_quartic_1', _X0_QUARTIC.copy(), f, g, H)


def p6_quartic_2():
    """Quartic with sigma=1e4."""
    f, g, H = _make_quartic(1e4)
    return Problem('P6_quartic_2', _X0_QUARTIC.copy(), f, g, H)


# ===== P7: Rosenbrock 2D =====

def p7_rosenbrock_2():
    """Rosenbrock, n=2."""
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def g(x):
        return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])

    def H(x):
        h11 = 2 - 400*(x[1] - x[0]**2) + 800*x[0]**2
        h12 = -400*x[0]
        return np.array([[h11, h12], [h12, 200.0]])

    return Problem('P7_rosenbrock_2', np.array([-1.2, 1.0]), f, g, H)


# ===== P8: Rosenbrock 100D =====

def p8_rosenbrock_100():
    """Rosenbrock, n=100."""
    n = 100
    x0 = np.ones(n); x0[0] = -1.2

    def f(x):
        return np.sum((1 - x[:-1])**2 + 100*(x[1:] - x[:-1]**2)**2)

    def g(x):
        grad = np.zeros(n)
        diff = x[1:] - x[:-1]**2          # shape (n-1,)
        grad[:-1] += -2*(1 - x[:-1]) - 400*x[:-1]*diff
        grad[1:]  += 200*diff
        return grad

    def H(x):
        hess = np.zeros((n, n))
        for i in range(n - 1):
            d = x[i+1] - x[i]**2
            hess[i, i]   += 2 - 400*d + 800*x[i]**2
            hess[i, i+1] += -400*x[i]
            hess[i+1, i] += -400*x[i]
            hess[i+1, i+1] += 200.0
        return hess

    return Problem('P8_rosenbrock_100', x0, f, g, H)


# ===== P9: DataFit 2D =====
# f(x) = sum_{i=1}^3 (y_i - x[0]*(1 - x[1]^i))^2

def p9_datafit_2():
    """DataFit, n=2."""
    y = np.array([1.5, 2.25, 2.625])

    def f(x):
        return sum((y[i] - x[0]*(1 - x[1]**(i+1)))**2 for i in range(3))

    def g(x):
        g0, g1 = 0.0, 0.0
        for i in range(3):
            ri = y[i] - x[0]*(1 - x[1]**(i+1))
            g0 += -2*ri*(1 - x[1]**(i+1))
            # dr/dx1 uses derivative of -x[0]*(-x1^i*(i+1)) = x[0]*(i+1)*x1^i
            g1 += 2*ri*x[0]*(i+1)*x[1]**i
        return np.array([g0, g1])

    def H(x):
        hess = np.zeros((2, 2))
        for i in range(3):
            ri = y[i] - x[0]*(1 - x[1]**(i+1))
            dr0 = -(1 - x[1]**(i+1))              # dr/dx0
            dr1 = x[0]*(i+1)*x[1]**i              # dr/dx1
            d2r_01 = (i+1)*x[1]**i                # d2r/dx0dx1
            d2r_11 = x[0]*i*(i+1)*x[1]**(i-1) if i > 0 else 0.0  # d2r/dx1^2
            hess[0, 0] += 2*dr0**2
            hess[0, 1] += 2*dr0*dr1 + 2*ri*d2r_01
            hess[1, 0] += 2*dr0*dr1 + 2*ri*d2r_01
            hess[1, 1] += 2*dr1**2 + 2*ri*d2r_11
        return hess

    return Problem('P9_datafit_2', np.array([1.0, 1.0]), f, g, H)


# ===== P10-P11: Exponential =====
# f(x) = (e^x1-1)/(e^x1+1) + 0.1*e^{-x1} + sum_{i=2}^n (x_i-1)^4

def _make_exponential(n):
    x0 = np.zeros(n); x0[0] = 1.0

    def f(x):
        e = np.exp(x[0])
        return (e - 1)/(e + 1) + 0.1*np.exp(-x[0]) + np.sum((x[1:] - 1)**4)

    def g(x):
        e = np.exp(x[0])
        grad = np.zeros(n)
        grad[0] = 2*e/(e + 1)**2 - 0.1*np.exp(-x[0])
        grad[1:] = 4*(x[1:] - 1)**3
        return grad

    def H(x):
        e = np.exp(x[0])
        hess = np.zeros((n, n))
        hess[0, 0] = 2*e*(1 - e)/(e + 1)**3 + 0.1*np.exp(-x[0])
        diag_rest = 12*(x[1:] - 1)**2
        hess[1:, 1:] = np.diag(diag_rest)
        return hess

    return f, g, H, x0


def p10_exponential_10():
    """Exponential, n=10."""
    f, g, H, x0 = _make_exponential(10)
    return Problem('P10_exponential_10', x0, f, g, H)


def p11_exponential_1000():
    """Exponential, n=100 (named '1000' in course materials)."""
    f, g, H, x0 = _make_exponential(100)
    return Problem('P11_exponential_1000', x0, f, g, H)


# ===== P12: Genhumps 5 =====
# f(x) = sum_{i=0}^3 [sin(2x_i)^2 * sin(2x_{i+1})^2 + 0.05*(x_i^2 + x_{i+1}^2)]

def p12_genhumps_5():
    """Genhumps, n=5."""
    x0 = np.array([-506.2, 506.2, 506.2, 506.2, 506.2])

    def f(x):
        s = np.sin(2*x)
        total = 0.0
        for i in range(4):
            total += s[i]**2 * s[i+1]**2 + 0.05*(x[i]**2 + x[i+1]**2)
        return total

    def g(x):
        s = np.sin(2*x)
        c4 = np.sin(4*x)   # sin(4x) = 2*sin(2x)*cos(2x), used for derivative of sin(2x)^2
        grad = np.zeros(5)
        for i in range(4):
            # d/dx[i] of term i: 2*sin(4x[i])*sin(2x[i+1])^2 + 0.1*x[i]
            grad[i]   += 2*c4[i]*s[i+1]**2 + 0.1*x[i]
            # d/dx[i+1] of term i: sin(2x[i])^2*2*sin(4x[i+1]) + 0.1*x[i+1]
            grad[i+1] += s[i]**2*2*c4[i+1] + 0.1*x[i+1]
        return grad

    def H(x):
        s = np.sin(2*x)
        c4 = np.sin(4*x)
        cos4 = np.cos(4*x)
        hess = np.zeros((5, 5))
        for i in range(4):
            # d2/dx[i]^2: 8*cos(4x[i])*sin(2x[i+1])^2 + 0.1
            hess[i, i]     += 8*cos4[i]*s[i+1]**2 + 0.1
            # d2/dx[i+1]^2: sin(2x[i])^2*8*cos(4x[i+1]) + 0.1
            hess[i+1, i+1] += s[i]**2*8*cos4[i+1] + 0.1
            # d2/dx[i]dx[i+1]: 4*sin(4x[i])*sin(4x[i+1])
            hess[i, i+1]   += 4*c4[i]*c4[i+1]
            hess[i+1, i]   += 4*c4[i]*c4[i+1]
        return hess

    return Problem('P12_genhumps_5', x0, f, g, H)


# ===== Dispatcher =====

_PROBLEM_MAP = {
    'P1_quad_10_10':    p1_quad_10_10,
    'P2_quad_10_1000':  p2_quad_10_1000,
    'P3_quad_1000_10':  p3_quad_1000_10,
    'P4_quad_1000_1000':p4_quad_1000_1000,
    'P5_quartic_1':     p5_quartic_1,
    'P6_quartic_2':     p6_quartic_2,
    'P7_rosenbrock_2':  p7_rosenbrock_2,
    'P8_rosenbrock_100':p8_rosenbrock_100,
    'P9_datafit_2':     p9_datafit_2,
    'P10_exponential_10':   p10_exponential_10,
    'P11_exponential_1000': p11_exponential_1000,
    'P12_genhumps_5':   p12_genhumps_5,
}


def get_problem(name):
    """Return a Problem instance by name. Raises KeyError for unknown names."""
    if name not in _PROBLEM_MAP:
        raise KeyError(f"Unknown problem '{name}'. Available: {list(_PROBLEM_MAP)}")
    return _PROBLEM_MAP[name]()


def all_problems():
    """Return a list of all 12 Problem instances."""
    return [fn() for fn in _PROBLEM_MAP.values()]
