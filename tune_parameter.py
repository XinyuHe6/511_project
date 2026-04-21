"""
tune_parameter.py - Lightweight random search over solver hyperparameters.

The active routine below samples parameter configurations, evaluates them on the
full benchmark set, and ranks them by success rate and average iterations.
"""

# import numpy as np
# import pandas as pd
# import random
# import warnings
# from problems import _PROBLEM_MAP
# from optSolver_DescentDynamics import optSolver_DescentDynamics
# from types import SimpleNamespace

# # Ignore warnings during tuning to keep the terminal clean
# warnings.filterwarnings('ignore')

# def tune_everything():
#     # --- Define the Search Space ---
#     param_space = {
#         'c1_ls': [1e-4, 1e-3],              # Armijo constant
#         'tau': [0.5, 0.7, 0.85],            # Backtracking factor (0.85 is more precise)
#         'c2_ls': [0.1, 0.44, 0.9],          # Wolfe curvature (0.1 for BFGS, 0.9 for Newton)
#         'delta0': [0.1, 1.0, 2.0],          # Initial Trust Region radius
#         'delta_max': [50.0, 100.0, 500.0],  # Max TR radius
#         'c1_tr': [0.1, 0.2, 0.25],          # TR acceptance threshold
#         'c2_tr': [0.75, 0.8, 0.9],          # TR expansion threshold
#         'alpha_bar': [1.0, 0.5]             # Initial step size (0.5 is safer)
#     }

#     # Test on a mix of Small/Hard/High-Dim problems
#     test_subset = ['P6_quartic_2', 'P7_rosenbrock_2', 'P12_genhumps_5']
#     methods_to_test = ['BFGSW', 'TRNewtonCG']
    
#     num_trials = 15 
#     results = []

#     print(f"Starting Global Tuning ({num_trials} trials)...")
#     print(f"{'Trial':>5} | {'Success':>7} | {'Avg Iters':>10} | {'Parameters'}")
#     print("-" * 90)

#     for i in range(num_trials):
#         # 1. Randomly pick a configuration
#         cfg = {k: random.choice(v) for k, v in param_space.items()}
        
#         total_iters = 0
#         successes = 0
        
#         # 2. Run across methods and problems
#         for method in methods_to_test:
#             for p_name in test_subset:
#                 opts = SimpleNamespace(
#                     term_tol=1e-6,
#                     max_iterations=1000,
#                     term_tol_CG=1e-6,
#                     max_iterations_CG=200,
#                     **cfg
#                 )
                
#                 try:
#                     prob = _PROBLEM_MAP[p_name]()
#                     _, _, info = optSolver_DescentDynamics(prob, method, opts)
                    
#                     if info['term_flag'] == 0:
#                         successes += 1
#                         total_iters += info['iterations']
#                 except Exception:
#                     # Catching overflows/NaNs here
#                     continue

#         # 3. Score this configuration
#         score = successes / (len(methods_to_test) * len(test_subset))
#         avg_it = total_iters / successes if successes > 0 else 9999
        
#         results.append({'score': score, 'avg_iters': avg_it, 'params': cfg})
#         print(f"{i+1:>5} | {score:>7.0%} | {avg_it:>10.1f} | {cfg}")

#     # --- Find the Winner ---
#     df = pd.DataFrame(results)
#     # Sort by success rate (high to low) then iterations (low to high)
#     best_row = df.sort_values(['score', 'avg_iters'], ascending=[False, True]).iloc[0]
    
#     print("\n" + "="*50)
#     print("WINNING CONFIGURATION FOUND")
#     print("="*50)
#     for k, v in best_row['params'].items():
#         print(f"'{k}': {v},")

# if __name__ == "__main__":
#     tune_everything()




import numpy as np
import pandas as pd
import random
import warnings
import time
from problems import _PROBLEM_MAP
from optSolver_DescentDynamics import optSolver_DescentDynamics
from types import SimpleNamespace

# Ignore numerical overflow warnings to keep output clean
warnings.filterwarnings('ignore')

def super_tune():
    """Sample solver settings and report the best-performing configurations."""
    # --- 1. Define the full parameter search space ---
    param_space = {
        'term_tol': [1e-6],               # The outer stopping tolerance is usually fixed.
        'max_iterations': [1000],         # The outer iteration cap is usually fixed.
        'alpha_bar': [1.0, 0.5],          # Smaller initial steps can reduce overflow risk.
        'c1_ls': [1e-4, 1e-3, 1e-2],      # Armijo sufficient-decrease constant.
        'tau': [0.5, 0.7, 0.8, 0.9],      # Backtracking shrink factor.
        'c2_ls': [0.1, 0.44, 0.9],        # Wolfe curvature constant.
        'delta0': [0.1, 1.0, 5.0],        # Initial trust-region radius.
        'delta_max': [100.0, 500.0],      # Maximum trust-region radius.
        'c1_tr': [0.1, 0.2, 0.25],        # Trust-region acceptance threshold.
        'c2_tr': [0.75, 0.8, 0.9],        # Trust-region expansion threshold.
        'term_tol_CG': [1e-4, 1e-6, 1e-8],# Inner CG tolerance.
        'max_iterations_CG': [100, 200, 500] # Inner CG iteration cap.
    }

    # --- 2. Prepare the test environment ---
    all_problem_names = list(_PROBLEM_MAP.keys())
    all_methods = [
        'GradientDescent', 'GradientDescentW',
        'Newton', 'NewtonW',
        'TRNewtonCG', 'TRSR1CG',
        'BFGS', 'BFGSW',
        'DFP', 'DFPW'
    ]
    
    num_random_samples = 10  # A small sample count is a reasonable first pass.
    final_results = []

    print(f"Starting Super Tuning: {num_random_samples} random configurations")
    print(f"Each sample tests {len(all_methods)} methods x {len(all_problem_names)} problems.")
    print("-" * 100)

    for i in range(num_random_samples):
        # Randomly sample a set of parameters
        cfg = {k: random.choice(v) for k, v in param_space.items()}
        start_time = time.time()
        
        solve_count = 0
        total_iters = 0
        
        # Evaluate the sampled configuration on every method/problem pair.
        for method in all_methods:
            for p_name in all_problem_names:
                opts = SimpleNamespace(**cfg)
                
                try:
                    prob = _PROBLEM_MAP[p_name]()
                    # Successful solves contribute to the final ranking statistics.
                    _, _, info = optSolver_DescentDynamics(prob, method, opts)
                    
                    if info['term_flag'] == 0:
                        solve_count += 1
                        total_iters += info['iterations']
                except:
                    # Catch Overflow, NaN, and other numerical errors
                    continue
        
        elapsed = time.time() - start_time
        success_rate = solve_count / (len(all_methods) * len(all_problem_names))
        avg_it = total_iters / solve_count if solve_count > 0 else 9999
        
        result_entry = {
            'success_rate': success_rate,
            'avg_iters': avg_it,
            'time': elapsed,
            'params': cfg
        }
        final_results.append(result_entry)
        
        print(f"Sample {i+1:02d} | Success: {success_rate:>5.1%} | AvgIt: {avg_it:>6.1f} | Time: {elapsed:>5.1f}s")

    # --- 3. Result analysis and reporting ---
    df = pd.DataFrame(final_results)
    # Sort: highest success rate first, then lowest average iterations
    best_configs = df.sort_values(by=['success_rate', 'avg_iters'], ascending=[False, True])
    
    print("\n" + "="*60)
    print("BEST 3 CONFIGURATIONS FOR ALL PROBLEMS & METHODS")
    print("="*60)
    
    for idx in range(min(3, len(best_configs))):
        best = best_configs.iloc[idx]
        print(f"\nRANK {idx+1} (Success Rate: {best['success_rate']:.1%})")
        print("Copy this to your OPTIONS in run_experiments.py:")
        print("OPTIONS = {")
        for k, v in best['params'].items():
            print(f"    '{k}': {v},")
        print("}")

if __name__ == "__main__":
    super_tune()
