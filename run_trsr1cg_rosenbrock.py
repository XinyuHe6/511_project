"""
Run the team's trust-region SR1-CG method on the Rosenbrock test problems.

This script is intentionally narrow in scope for the course deliverable:
it runs TRSR1CG on Problem 7 and Problem 8 using the solver's built-in
default parameters and prints the requested diagnostics.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optSolver_DescentDynamics import optSolver_DescentDynamics
from problems import get_problem


METHOD = "TRSR1CG"
PROBLEM_NAMES = [
    "P7_rosenbrock_2",
    "P8_rosenbrock_100",
]


def _run_problem(problem_name):
    """Solve one Rosenbrock instance and return the requested metrics."""
    problem = get_problem(problem_name)
    _, f_final, info = optSolver_DescentDynamics(problem, METHOD, options=None)
    return {
        "problem": problem.name,
        "iterations": info["iterations"],
        "cpu_time": info["cpu_time"],
        "f_evals": info["f_evals"],
        "g_evals": info["g_evals"],
        "final_f": f_final,
        "converged": info["term_flag"] == 0,
    }


def _print_results(rows):
    """Print a compact summary table for the two Rosenbrock problems."""
    headers = [
        ("Problem", 20),
        ("Method", 10),
        ("Iterations", 12),
        ("CPU Time (s)", 14),
        ("f Evaluations", 15),
        ("g Evaluations", 15),
        ("Final f", 14),
        ("Status", 10),
    ]

    def format_cell(text, width):
        return f"{text:<{width}}"

    header_line = " ".join(format_cell(label, width) for label, width in headers)
    divider = "-" * len(header_line)
    print(header_line)
    print(divider)

    for row in rows:
        status = "CONVERGED" if row["converged"] else "MAX_ITER"
        values = [
            row["problem"],
            METHOD,
            str(row["iterations"]),
            f"{row['cpu_time']:.6f}",
            str(row["f_evals"]),
            str(row["g_evals"]),
            f"{row['final_f']:.6e}",
            status,
        ]
        print(" ".join(format_cell(value, width) for value, (_, width) in zip(values, headers)))


def main():
    rows = [_run_problem(problem_name) for problem_name in PROBLEM_NAMES]
    print("TRSR1CG on Rosenbrock problems with solver default parameters")
    print()
    _print_results(rows)


if __name__ == "__main__":
    main()
