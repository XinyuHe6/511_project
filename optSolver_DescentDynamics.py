"""
optSolver_DescentDynamics.py - Optimization solver for IOE 511/MATH 562 (Team DescentDynamics)
Usage: x, f, info = optSolver_DescentDynamics(problem, method, options)

Supported methods (method.name or just the string):
  GradientDescent, GradientDescentW,
  Newton, NewtonW,
  TRNewtonCG, TRSR1CG,
  BFGS, BFGSW, DFP, DFPW
"""

import numpy as np
import time
from types import SimpleNamespace


# ===== DEFAULT OPTIONS =====

_DEFAULTS = {
    'term_tol': 1e-6,        # gradient norm stopping tolerance
    'max_iterations': 1000,  # max outer iterations
    # Backtracking (Armijo) parameters
    'alpha_bar': 1.0,        # initial step size
    'alpha_max': 100.0,      # maximum step size considered by line search
    'max_ls_iterations': 100,  # maximum line-search iterations
    'c1_ls': 0.01,           # sufficient decrease constant
    'tau': 0.7,              # step reduction factor
    # Wolfe curvature constant
    'c2_ls': 0.44,
    # Trust region parameters
    'delta0': 1.0,           # initial TR radius
    'delta_max': 500.0,      # maximum TR radius
    'c1_tr': 0.25,            # accept step if rho >= c1_tr
    'c2_tr': 0.9,           # expand TR if rho >= c2_tr
    # CG subproblem (Steihaug)
    'term_tol_CG': 0.0001,
    'max_iterations_CG': 100,
}


def _parse_opts(options):
    """Merge user options with defaults; return a SimpleNamespace."""
    if options is None:
        return SimpleNamespace(**_DEFAULTS)
    d = dict(_DEFAULTS)
    if isinstance(options, dict):
        d.update(options)
    else:  # SimpleNamespace or similar
        d.update({k: getattr(options, k) for k in vars(options)})
    return SimpleNamespace(**d)


def _parse_method(method):
    """Accept method as a string, dict, or SimpleNamespace; return SimpleNamespace."""
    if isinstance(method, str):
        return SimpleNamespace(name=method)
    if isinstance(method, dict):
        return SimpleNamespace(**method)
    return method


# ===== MAIN ENTRY POINT =====

def optSolver_DescentDynamics(problem, method, options=None):
    """
    Solve an unconstrained optimization problem.

    Parameters
    ----------
    problem : Problem (from problems.py)
        Must have .name, .x0, .compute_f(), .compute_g(), .compute_H()
    method  : str | dict | SimpleNamespace
        Must specify .name; one of the 10 supported algorithms.
    options : dict | SimpleNamespace | None
        Algorithm parameters; missing keys are filled from defaults.

    Returns
    -------
    x    : ndarray  - final iterate
    f    : float    - function value at x
    info : dict     - iterations, f_evals, g_evals, H_evals, cpu_time,
                      term_flag (0=converged, 1=max_iter), f_hist, g_norm_hist
    """
    if not hasattr(problem, 'x0') or not hasattr(problem, 'name'):
        raise ValueError("problem must have .x0 and .name attributes")

    method = _parse_method(method)
    if not hasattr(method, 'name'):
        raise ValueError("method must have a .name attribute")

    opts = _parse_opts(options)
    # Reset counters so every solve reports clean per-run diagnostics.
    problem.reset_counters()

    dispatch = {
        'GradientDescent':  _grad_descent,
        'GradientDescentW': _grad_descentW,
        'Newton':           _newton,
        'NewtonW':          _newtonW,
        'TRNewtonCG':       _tr_newton_cg,
        'TRSR1CG':          _tr_sr1_cg,
        'BFGS':             _bfgs,
        'BFGSW':            _bfgsW,
        'DFP':              _dfp,
        'DFPW':             _dfpW,
    }
    if method.name not in dispatch:
        raise ValueError(f"Unknown method '{method.name}'. Supported: {list(dispatch)}")

    t0 = time.time()
    x, f, info = dispatch[method.name](problem, opts)
    # Gather bookkeeping here so the method-specific routines can stay focused on the algorithm.
    info['cpu_time'] = time.time() - t0
    info['f_evals'] = problem.f_evals
    info['g_evals'] = problem.g_evals
    info['H_evals'] = problem.H_evals
    return x, f, info


# ===== LINE SEARCHES =====

def _backtracking(problem, x, f, g, d, opts):
    """Armijo (sufficient decrease) backtracking line search."""
    alpha = min(opts.alpha_bar, opts.alpha_max)
    gtd = g @ d   # directional derivative (must be < 0 for a descent direction)
    for _ in range(opts.max_ls_iterations):
        if problem.compute_f(x + alpha * d) <= f + opts.c1_ls * alpha * gtd:
            break
        alpha *= opts.tau
        if alpha < 1e-16:
            break
    return alpha


def _wolfe_ls(problem, x, f, g, d, opts):
    """
    Strong Wolfe line search (Nocedal & Wright, Algorithm 3.5/3.6).
    Guarantees both sufficient decrease and the curvature condition.
    """
    c1, c2 = opts.c1_ls, opts.c2_ls
    phi0 = f
    dphi0 = g @ d   # directional derivative at alpha=0; must be negative

    def phi(a):  return problem.compute_f(x + a * d)
    def dphi(a): return problem.compute_g(x + a * d) @ d

    alpha, alpha_prev = min(opts.alpha_bar, opts.alpha_max), 0.0
    phi_prev = phi0

    for i in range(opts.max_ls_iterations):
        phi_a = phi(alpha)
        # Armijo violated or function increased: bracket found, zoom in
        if phi_a > phi0 + c1*alpha*dphi0 or (i > 0 and phi_a >= phi_prev):
            return _zoom(phi, dphi, alpha_prev, alpha, phi0, dphi0, c1, c2, opts.max_ls_iterations)
        dphi_a = dphi(alpha)
        if abs(dphi_a) <= -c2*dphi0:
            return alpha        # strong Wolfe satisfied
        if dphi_a >= 0:
            return _zoom(phi, dphi, alpha, alpha_prev, phi0, dphi0, c1, c2, opts.max_ls_iterations)
        alpha_prev = alpha
        phi_prev = phi_a
        alpha = min(2.0 * alpha, opts.alpha_max)

    return alpha


def _zoom(phi, dphi, a_lo, a_hi, phi0, dphi0, c1, c2, max_ls_iterations):
    """Zoom phase: narrow [a_lo, a_hi] to a point satisfying strong Wolfe conditions."""
    phi_lo = phi(a_lo)
    for _ in range(max_ls_iterations):
        alpha = 0.5*(a_lo + a_hi)   # bisection; cubic interpolation also works
        phi_a = phi(alpha)
        if phi_a > phi0 + c1*alpha*dphi0 or phi_a >= phi_lo:
            a_hi = alpha            # upper bound tightened
        else:
            dphi_a = dphi(alpha)
            if abs(dphi_a) <= -c2*dphi0:
                return alpha        # found a strong Wolfe point
            if dphi_a*(a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = alpha
            phi_lo = phi_a
    return alpha


# ===== NEWTON DIRECTION (MODIFIED CHOLESKY) =====

def _modified_newton_dir(H, g, beta0=1e-8):
    """
    Compute d = -H_mod^{-1} g where H_mod = H + tau*I is positive definite.
    tau is increased until Cholesky factorization succeeds (Nocedal & Wright, Alg. 3.3).
    """
    n = len(g)
    tau = 0.0
    min_diag = np.min(np.diag(H))
    if min_diag < 0:
        tau = -min_diag + beta0   # initial regularization if H already not PD

    for _ in range(60):
        try:
            np.linalg.cholesky(H + tau*np.eye(n))   # test positive definiteness
            return np.linalg.solve(H + tau*np.eye(n), -g)
        except np.linalg.LinAlgError:
            tau = max(2.0*tau, beta0)
            beta0 *= 10
    return np.linalg.solve(H + tau*np.eye(n), -g)   # fallback after max iterations


# ===== TRUST REGION CG (STEIHAUG) =====

def _tr_boundary_tau(p, d, delta):
    """Find tau >= 0 such that ||p + tau*d||^2 = delta^2 (positive root of quadratic)."""
    dd, pd, pp = d@d, p@d, p@p
    disc = max(pd**2 - dd*(pp - delta**2), 0.0)
    return (-pd + np.sqrt(disc)) / dd


def _tr_cg_steihaug(B, g, delta, opts):
    """
    Steihaug CG method for the trust-region subproblem:
        min  g^T p + 0.5 p^T B p   subject to  ||p|| <= delta
    Stops at TR boundary (negative curvature or boundary hit) or CG convergence.
    Returns step p.
    """
    n = len(g)
    p = np.zeros(n)
    r = g.copy()        # residual: r = B*p + g = g initially
    d = -r.copy()       # CG search direction
    r_norm0 = np.linalg.norm(r)
    tol = opts.term_tol_CG * r_norm0

    if r_norm0 < 1e-14:
        return p   # already at optimum inside TR

    for _ in range(opts.max_iterations_CG):
        Bd = B @ d
        dBd = d @ Bd

        if dBd <= 0:
            # Negative curvature: follow d to the TR boundary
            return p + _tr_boundary_tau(p, d, delta) * d

        alpha = (r @ r) / dBd
        p_new = p + alpha * d

        if np.linalg.norm(p_new) >= delta:
            # Step would exit TR: intersect with boundary instead
            return p + _tr_boundary_tau(p, d, delta) * d

        p = p_new
        r_new = r + alpha * Bd

        if np.linalg.norm(r_new) <= tol:
            return p   # CG converged

        beta = (r_new @ r_new) / (r @ r)
        d = -r_new + beta * d
        r = r_new

    return p


# ===== TRUST REGION STEP ACCEPTANCE AND RADIUS UPDATE =====

def _tr_update(problem, x, f, g, p, B, delta, opts):
    """
    Compute rho = (actual reduction) / (predicted reduction),
    update TR radius, and accept or reject the step.
    Returns: x_new, f_new, g_new, delta_new, accepted
    """
    f_trial = problem.compute_f(x + p)
    actual    = f - f_trial
    predicted = -(g @ p + 0.5 * p @ B @ p)   # predicted decrease from quadratic model

    rho = actual / predicted if abs(predicted) > 1e-16 else 0.0

    # Radius update based on quality of the quadratic model
    if rho < opts.c1_tr:
        delta_new = 0.25 * np.linalg.norm(p)            # poor agreement: shrink
    elif rho > opts.c2_tr and np.linalg.norm(p) >= 0.9*delta:
        delta_new = min(2*delta, opts.delta_max)         # great step near boundary: expand
    else:
        delta_new = delta                                # acceptable: keep radius

    if rho > opts.c1_tr:
        x_new = x + p
        f_new = f_trial
        g_new = problem.compute_g(x_new)
        return x_new, f_new, g_new, delta_new, True
    else:
        return x.copy(), f, g.copy(), delta_new, False   # reject step


# ===== ALGORITHM IMPLEMENTATIONS =====

def _init_stats():
    """Create the common diagnostics dictionary shared by all solver variants."""
    return {'iterations': 0, 'term_flag': 1, 'f_hist': [], 'g_norm_hist': []}


def _grad_descent(problem, opts):
    """Gradient descent with Armijo backtracking line search."""
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        d = -g                                  # steepest descent direction
        alpha = _backtracking(problem, x, f, g, d, opts)
        x = x + alpha * d
        f = problem.compute_f(x)
        g = problem.compute_g(x)

    info['iterations'] = len(info['f_hist'])
    return x, f, info


def _grad_descentW(problem, opts):
    """Gradient descent with Wolfe line search."""
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        d = -g
        alpha = _wolfe_ls(problem, x, f, g, d, opts)
        x = x + alpha * d
        f = problem.compute_f(x)
        g = problem.compute_g(x)

    info['iterations'] = len(info['f_hist'])
    return x, f, info


def _newton(problem, opts):
    """Modified Newton with Armijo backtracking line search."""
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        H = problem.compute_H(x)
        d = _modified_newton_dir(H, g)          # regularized Newton step
        alpha = _backtracking(problem, x, f, g, d, opts)
        x = x + alpha * d
        f = problem.compute_f(x)
        g = problem.compute_g(x)

    info['iterations'] = len(info['f_hist'])
    return x, f, info


def _newtonW(problem, opts):
    """Modified Newton with Wolfe line search."""
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        H = problem.compute_H(x)
        d = _modified_newton_dir(H, g)
        alpha = _wolfe_ls(problem, x, f, g, d, opts)
        x = x + alpha * d
        f = problem.compute_f(x)
        g = problem.compute_g(x)

    info['iterations'] = len(info['f_hist'])
    return x, f, info


def _tr_newton_cg(problem, opts):
    """Trust-region Newton with Steihaug CG subproblem solver."""
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    delta = opts.delta0
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        H = problem.compute_H(x)
        p = _tr_cg_steihaug(H, g, delta, opts)  # solve TR subproblem with exact Hessian
        x, f, g, delta, _ = _tr_update(problem, x, f, g, p, H, delta, opts)

    info['iterations'] = len(info['f_hist'])
    return x, f, info


def _tr_sr1_cg(problem, opts):
    """Trust-region SR1 quasi-Newton with Steihaug CG subproblem solver."""
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    n = len(x)
    B = np.eye(n)       # Hessian approximation (starts as identity)
    delta = opts.delta0
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        p = _tr_cg_steihaug(B, g, delta, opts)
        x_new, f_new, g_new, delta, accepted = _tr_update(problem, x, f, g, p, B, delta, opts)

        if accepted:
            s = x_new - x
            y = g_new - g
            u = y - B @ s          # SR1 correction vector
            denom = u @ s
            # Skip update if denominator is too small (numerical safeguard)
            if abs(denom) >= 1e-8 * np.linalg.norm(u) * np.linalg.norm(s):
                B = B + np.outer(u, u) / denom
            x, f, g = x_new, f_new, g_new

    info['iterations'] = len(info['f_hist'])
    return x, f, info


# ===== QUASI-NEWTON INVERSE HESSIAN UPDATES =====

def _bfgs_update(H_inv, s, y):
    """
    BFGS inverse Hessian update (two-sided rank-2 update):
    H_{k+1} = (I - rho*s*y^T) H_k (I - rho*y*s^T) + rho*s*s^T
    where rho = 1/(y^T s). Skip if y^T s is too small (preserves PD property).
    """
    ys = y @ s
    if ys <= 1e-10 * np.linalg.norm(y) * np.linalg.norm(s):
        return H_inv
    rho = 1.0 / ys
    n = len(s)
    A = np.eye(n) - rho * np.outer(s, y)
    return A @ H_inv @ A.T + rho * np.outer(s, s)


def _dfp_update(H_inv, s, y):
    """
    DFP inverse Hessian update (rank-2 update, older than BFGS):
    H_{k+1} = H_k - (H_k y y^T H_k)/(y^T H_k y) + (s s^T)/(y^T s)
    Skip if y^T s is too small.
    """
    ys = y @ s
    if ys <= 1e-10 * np.linalg.norm(y) * np.linalg.norm(s):
        return H_inv
    Hy = H_inv @ y
    return H_inv - np.outer(Hy, Hy) / (y @ Hy) + np.outer(s, s) / ys


def _quasi_newton_loop(problem, opts, update_fn, ls_fn):
    """
    Generic quasi-Newton loop shared by BFGS and DFP (any line search).
    Maintains an inverse Hessian approximation H and uses d = -H*g as search direction.
    """
    x = problem.x0.copy()
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    n = len(x)
    H = np.eye(n)       # inverse Hessian approximation (starts as identity)
    info = _init_stats()

    for _ in range(opts.max_iterations):
        info['f_hist'].append(f)
        info['g_norm_hist'].append(np.linalg.norm(g))
        if np.linalg.norm(g) <= opts.term_tol:
            info['term_flag'] = 0; break

        d = -H @ g                              # quasi-Newton search direction
        alpha = ls_fn(problem, x, f, g, d, opts)
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

        s = x_new - x
        y = g_new - g
        H = update_fn(H, s, y)                 # update inverse Hessian approximation

        x, f, g = x_new, f_new, g_new

    info['iterations'] = len(info['f_hist'])
    return x, f, info


# Each of the four quasi-Newton variants is a one-liner using the shared loop
def _bfgs(problem,  opts): return _quasi_newton_loop(problem, opts, _bfgs_update, _backtracking)
def _bfgsW(problem, opts): return _quasi_newton_loop(problem, opts, _bfgs_update, _wolfe_ls)
def _dfp(problem,   opts): return _quasi_newton_loop(problem, opts, _dfp_update,  _backtracking)
def _dfpW(problem,  opts): return _quasi_newton_loop(problem, opts, _dfp_update,  _wolfe_ls)
