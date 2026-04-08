"""
run_experiments.py - Run all experiments for "Table: Summary of Results"
Runs all 10 algorithms on all 12 test problems and prints/saves results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import threading
warnings.filterwarnings('ignore')

# Maximum wall-clock seconds per (problem, method) run before giving up
RUN_TIMEOUT = 60  # seconds


def _run_with_timeout(fn, timeout):
    """Run fn() in a thread; return its result or raise TimeoutError after timeout seconds."""
    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"exceeded {timeout}s")
    if exc[0] is not None:
        raise exc[0]
    return result[0]

from problems import all_problems
from optSolver_DescentDynamics import optSolver_DescentDynamics

# ===== CONFIGURATION =====

METHODS = [
    'GradientDescent', 'GradientDescentW',
    'Newton', 'NewtonW',
    'TRNewtonCG', 'TRSR1CG',
    'BFGS', 'BFGSW',
    'DFP', 'DFPW',
]

# Problems that require a Hessian (Newton/TR methods need compute_H)
# All problems have Hessians implemented, but n=1000 ones may be slow for Newton
SKIP_HESSIAN_METHODS = {'Newton', 'NewtonW', 'TRNewtonCG'}
LARGE_PROBLEMS = {'P3_quad_1000_10', 'P4_quad_1000_1000', 'P8_rosenbrock_100', 'P11_exponential_1000'}

OPTIONS = {
    'term_tol': 1e-6,
    'max_iterations': 1000,
    'alpha_bar': 1.0,
    'c1_ls': 1e-4,
    'tau': 0.5,
    'c2_ls': 0.9,
    'delta0': 1.0,
    'delta_max': 100.0,
    'c1_tr': 0.1,
    'c2_tr': 0.75,
    'term_tol_CG': 1e-6,
    'max_iterations_CG': 200,
}


# ===== RUN EXPERIMENTS =====

def run_all(verbose=True):
    """Run all 12 x 10 experiments; return a dict of results."""
    problems = all_problems()
    results = {}  # key: (problem.name, method) -> dict

    for prob in problems:
        for method in METHODS:
            # Skip Hessian-based methods for large n to avoid excessive runtime
            if prob.name in LARGE_PROBLEMS and method in SKIP_HESSIAN_METHODS:
                results[(prob.name, method)] = {'skipped': True, 'reason': 'n too large for H'}
                if verbose:
                    print(f"  SKIP  {prob.name:25s}  {method:20s}  (large n, Hessian too expensive)")
                continue

            try:
                x, f, info = _run_with_timeout(
                    lambda m=method: optSolver_DescentDynamics(prob, m, OPTIONS),
                    RUN_TIMEOUT
                )
                converged = info['term_flag'] == 0
                results[(prob.name, method)] = {
                    'skipped': False,
                    'converged': converged,
                    'f': f,
                    'iterations': info['iterations'],
                    'f_evals': info['f_evals'],
                    'g_evals': info['g_evals'],
                    'H_evals': info['H_evals'],
                    'cpu_time': info['cpu_time'],
                    'f_hist': info['f_hist'],
                    'g_norm_hist': info['g_norm_hist'],
                }
                if verbose:
                    status = 'CONV' if converged else 'FAIL'
                    print(f"  {status}  {prob.name:25s}  {method:20s}  "
                          f"f={f:.3e}  iters={info['iterations']:4d}  "
                          f"fevals={info['f_evals']:5d}  t={info['cpu_time']:.3f}s")
            except Exception as e:
                results[(prob.name, method)] = {'skipped': True, 'reason': str(e)}
                if verbose:
                    print(f"  ERR   {prob.name:25s}  {method:20s}  {e}")

    return results, problems


# ===== BUILD SUMMARY TABLE =====

def build_table(results, problems, metric='iterations'):
    """
    Build a DataFrame (rows=problems, cols=methods) for a given metric.
    metric: 'iterations' | 'f_evals' | 'g_evals' | 'cpu_time'
    """
    rows = []
    for prob in problems:
        row = {'Problem': prob.name}
        for method in METHODS:
            r = results.get((prob.name, method), {})
            if r.get('skipped'):
                row[method] = 'SKIP'
            elif not r.get('converged'):
                row[method] = 'FAIL'
            else:
                val = r.get(metric, '—')
                row[method] = f'{val:.3f}' if isinstance(val, float) else str(val)
        rows.append(row)
    return pd.DataFrame(rows).set_index('Problem')


def print_summary_table(results, problems):
    """Print the four summary tables to stdout."""
    for metric, label in [('iterations', 'Iterations'),
                           ('f_evals',   'Function Evaluations'),
                           ('g_evals',   'Gradient Evaluations'),
                           ('cpu_time',  'CPU Time (s)')]:
        df = build_table(results, problems, metric)
        print(f"\n{'='*30} {label} {'='*30}")
        print(df.to_string())


# ===== CONVERGENCE PLOTS (f vs. iteration) =====

def plot_convergence(results, problems, save_dir=None):
    """Plot f(x_k) vs iteration for each problem (all methods on one axes)."""
    for prob in problems:
        fig, ax = plt.subplots(figsize=(8, 5))
        any_plotted = False
        for method in METHODS:
            r = results.get((prob.name, method), {})
            if r.get('skipped') or not r.get('f_hist'):
                continue
            ax.semilogy(r['f_hist'], label=method, alpha=0.8)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel('Iteration')
        ax.set_ylabel('f(x_k)')
        ax.set_title(f'Convergence — {prob.name}')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f'conv_{prob.name}.png'), dpi=120)
        plt.show()


# ===== PERFORMANCE PROFILE =====

def performance_profile(results, problems, metric='f_evals', tau_max=100, save_path=None):
    """
    Plot a performance profile (Dolan & Moré, 2002) for the given metric.
    Only considers runs that converged.
    """
    # Collect raw metric values; FAIL/SKIP -> inf
    data = {}  # method -> list of values (one per problem)
    for method in METHODS:
        vals = []
        for prob in problems:
            r = results.get((prob.name, method), {})
            if not r.get('skipped') and r.get('converged'):
                vals.append(r.get(metric, np.inf))
            else:
                vals.append(np.inf)
        data[method] = np.array(vals, dtype=float)

    # Best performance on each problem
    stacked = np.column_stack(list(data.values()))  # shape (n_probs, n_methods)
    best = np.min(stacked, axis=1, keepdims=True)   # best across methods per problem

    # Performance ratios
    ratios = stacked / best   # ratio >= 1 (smaller is better)

    taus = np.linspace(1, tau_max, 500)
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, method in enumerate(METHODS):
        rho = np.array([(ratios[:, j] <= t).mean() for t in taus])
        ax.plot(taus, rho, label=method)

    ax.set_xlabel(f'Performance ratio τ  (metric: {metric})')
    ax.set_ylabel('Fraction of problems solved  ρ(τ)')
    ax.set_title('Performance Profile')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.show()


# ===== SAVE TO CSV =====

def save_csv(results, problems, out_dir='results'):
    """Save one CSV per metric to out_dir/."""
    os.makedirs(out_dir, exist_ok=True)
    for metric in ['iterations', 'f_evals', 'g_evals', 'cpu_time']:
        df = build_table(results, problems, metric)
        path = os.path.join(out_dir, f'table_{metric}.csv')
        df.to_csv(path)
        print(f'Saved: {path}')


# ===== MAIN =====

if __name__ == '__main__':
    print('Running all experiments...\n')
    results, problems = run_all(verbose=True)

    print_summary_table(results, problems)
    save_csv(results, problems, out_dir='results')

    # Convergence plots saved to results/plots/
    plot_convergence(results, problems, save_dir='results/plots')

    # Performance profiles
    performance_profile(results, problems, metric='f_evals',
                        save_path='results/perf_profile_fevals.png')
    performance_profile(results, problems, metric='iterations',
                        save_path='results/perf_profile_iters.png')
