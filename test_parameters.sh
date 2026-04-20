#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${1:-$PROJECT_ROOT/parameter_sweeps_$TIMESTAMP}"

# LS sweep plus second-stage TR-CG sweep requested by the user.
C1_LS_SET=(1e-5 3e-5 1e-4 3e-4 1e-3 1e-2)
C2_LS_SET=(0.1 0.2 0.4 0.6 0.8 0.9)
C1_TR_FIXED="0.03"
C2_TR_FIXED="0.7"
TERM_TOL_CG_SET=(0.1 0.05 0.01 0.005 0.001 0.0001)
MAX_ITERATIONS_CG_SET=(50 100 200 400 800 1600)

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/test_parameters_mpl_cache}"
mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"

python - "$PROJECT_ROOT" "$OUTPUT_ROOT" \
    "$(IFS=,; echo "${C1_LS_SET[*]}")" \
    "$(IFS=,; echo "${C2_LS_SET[*]}")" \
    "$C1_TR_FIXED" \
    "$C2_TR_FIXED" \
    "$(IFS=,; echo "${TERM_TOL_CG_SET[*]}")" \
    "$(IFS=,; echo "${MAX_ITERATIONS_CG_SET[*]}")" <<'PY'
import itertools
import json
import math
import os
import sys

import pandas as pd

project_root = sys.argv[1]
output_root = sys.argv[2]
c1_ls_values = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
c2_ls_values = [item.strip() for item in sys.argv[4].split(",") if item.strip()]
c1_tr_raw = sys.argv[5]
c2_tr_raw = sys.argv[6]
term_tol_cg_values = [item.strip() for item in sys.argv[7].split(",") if item.strip()]
max_iterations_cg_values = [item.strip() for item in sys.argv[8].split(",") if item.strip()]

sys.path.insert(0, project_root)

import run_experiments as re

LS_METHODS = [
    "GradientDescent",
    "GradientDescentW",
    "Newton",
    "NewtonW",
    "BFGS",
    "BFGSW",
    "DFP",
    "DFPW",
]

TR_METHODS = [
    "TRNewtonCG",
    "TRSR1CG",
]

BASE_OPTIONS = dict(re.OPTIONS)


def mean_or_nan(total, count):
    return total / count if count else math.nan


def save_metric_tables(results, problems, out_dir):
    for metric in ["iterations", "f_evals", "g_evals", "H_evals", "cpu_time"]:
        table = re.build_table(results, problems, metric)
        table.to_csv(os.path.join(out_dir, f"table_{metric}.csv"))


def summarize_results(results, problems, methods):
    total_pairs = len(problems) * len(methods)
    attempted_runs = 0
    successful_runs = 0
    total_iterations = 0.0
    total_evaluations = 0.0
    total_f_evals = 0.0
    total_g_evals = 0.0
    total_h_evals = 0.0
    per_method_rows = []

    for method in methods:
        method_attempted = 0
        method_successful = 0
        method_iterations = 0.0
        method_total_evals = 0.0
        method_f_evals = 0.0
        method_g_evals = 0.0
        method_h_evals = 0.0

        for problem in problems:
            result = results.get((problem.name, method), {})
            if result.get("skipped"):
                continue

            attempted_runs += 1
            method_attempted += 1

            if not result.get("converged"):
                continue

            successful_runs += 1
            method_successful += 1

            f_evals = float(result.get("f_evals", 0))
            g_evals = float(result.get("g_evals", 0))
            h_evals = float(result.get("H_evals", 0))
            eval_count = f_evals + g_evals + h_evals
            iterations = float(result.get("iterations", 0))

            total_iterations += iterations
            total_evaluations += eval_count
            total_f_evals += f_evals
            total_g_evals += g_evals
            total_h_evals += h_evals

            method_iterations += iterations
            method_total_evals += eval_count
            method_f_evals += f_evals
            method_g_evals += g_evals
            method_h_evals += h_evals

        per_method_rows.append(
            {
                "method": method,
                "total_pairs": len(problems),
                "attempted_runs": method_attempted,
                "skipped_runs": len(problems) - method_attempted,
                "successful_runs": method_successful,
                "success_rate": mean_or_nan(method_successful, method_attempted),
                "avg_iterations": mean_or_nan(method_iterations, method_successful),
                "avg_total_evaluations": mean_or_nan(method_total_evals, method_successful),
                "avg_f_evals": mean_or_nan(method_f_evals, method_successful),
                "avg_g_evals": mean_or_nan(method_g_evals, method_successful),
                "avg_H_evals": mean_or_nan(method_h_evals, method_successful),
            }
        )

    group_summary = {
        "total_pairs": total_pairs,
        "attempted_runs": attempted_runs,
        "skipped_runs": total_pairs - attempted_runs,
        "successful_runs": successful_runs,
        "success_rate": mean_or_nan(successful_runs, attempted_runs),
        "avg_iterations": mean_or_nan(total_iterations, successful_runs),
        "avg_total_evaluations": mean_or_nan(total_evaluations, successful_runs),
        "avg_f_evals": mean_or_nan(total_f_evals, successful_runs),
        "avg_g_evals": mean_or_nan(total_g_evals, successful_runs),
        "avg_H_evals": mean_or_nan(total_h_evals, successful_runs),
    }
    return group_summary, pd.DataFrame(per_method_rows)


def run_sweep(label, methods, overrides, folder_name, summary_fields):
    run_dir = os.path.join(output_root, label, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    effective_options = dict(BASE_OPTIONS)
    effective_options.update(overrides)

    re.METHODS = list(methods)
    re.OPTIONS.clear()
    re.OPTIONS.update(effective_options)

    results, problems = re.run_all(verbose=False)
    save_metric_tables(results, problems, run_dir)

    with open(os.path.join(run_dir, "options.json"), "w", encoding="utf-8") as handle:
        json.dump(effective_options, handle, indent=2)

    group_summary, method_summary = summarize_results(results, problems, methods)
    pd.DataFrame([group_summary]).to_csv(os.path.join(run_dir, "group_summary.csv"), index=False)
    method_summary.to_csv(os.path.join(run_dir, "method_summary.csv"), index=False)

    summary_row = dict(summary_fields)
    summary_row["result_dir"] = run_dir
    summary_row.update(group_summary)
    return summary_row


ls_rows = []
ls_total = len(c1_ls_values) * len(c2_ls_values)
for idx, (c1_ls_raw, c2_ls_raw) in enumerate(itertools.product(c1_ls_values, c2_ls_values), start=1):
    print(f"[LS {idx:02d}/{ls_total}] c1_ls={c1_ls_raw}, c2_ls={c2_ls_raw}", flush=True)
    ls_rows.append(
        run_sweep(
            label="ls",
            methods=LS_METHODS,
            overrides={
                "c1_ls": float(c1_ls_raw),
                "c2_ls": float(c2_ls_raw),
            },
            folder_name=f"results_c1ls_{c1_ls_raw}_c2ls_{c2_ls_raw}",
            summary_fields={
                "c1_ls": c1_ls_raw,
                "c2_ls": c2_ls_raw,
            },
        )
    )

tr_rows = []
tr_total = len(term_tol_cg_values) * len(max_iterations_cg_values)
for idx, (term_tol_cg_raw, max_iterations_cg_raw) in enumerate(
    itertools.product(term_tol_cg_values, max_iterations_cg_values),
    start=1,
):
    print(
        "[TR {idx:02d}/{total}] c1_tr={c1_tr}, c2_tr={c2_tr}, term_tol_CG={tol}, max_iterations_CG={cgmax}".format(
            idx=idx,
            total=tr_total,
            c1_tr=c1_tr_raw,
            c2_tr=c2_tr_raw,
            tol=term_tol_cg_raw,
            cgmax=max_iterations_cg_raw,
        ),
        flush=True,
    )
    tr_rows.append(
        run_sweep(
            label="tr",
            methods=TR_METHODS,
            overrides={
                "c1_tr": float(c1_tr_raw),
                "c2_tr": float(c2_tr_raw),
                "term_tol_CG": float(term_tol_cg_raw),
                "max_iterations_CG": int(max_iterations_cg_raw),
            },
            folder_name=(
                "results_c1tr_{c1_tr}_c2tr_{c2_tr}_termtolcg_{tol}_maxitercg_{cgmax}".format(
                    c1_tr=c1_tr_raw,
                    c2_tr=c2_tr_raw,
                    tol=term_tol_cg_raw,
                    cgmax=max_iterations_cg_raw,
                )
            ),
            summary_fields={
                "c1_tr": c1_tr_raw,
                "c2_tr": c2_tr_raw,
                "term_tol_CG": term_tol_cg_raw,
                "max_iterations_CG": max_iterations_cg_raw,
            },
        )
    )

ls_summary = pd.DataFrame(ls_rows).sort_values(
    by=["success_rate", "avg_iterations", "avg_total_evaluations"],
    ascending=[False, True, True],
)
tr_summary = pd.DataFrame(tr_rows).sort_values(
    by=["success_rate", "avg_iterations", "avg_total_evaluations"],
    ascending=[False, True, True],
)

ls_summary.to_csv(os.path.join(output_root, "ls_summary.csv"), index=False)
tr_summary.to_csv(os.path.join(output_root, "tr_summary.csv"), index=False)

notes_path = os.path.join(output_root, "README.txt")
with open(notes_path, "w", encoding="utf-8") as handle:
    handle.write(
        "Summary conventions:\n"
        "- success_rate = successful_runs / attempted_runs\n"
        "- attempted_runs excludes entries marked SKIP by run_experiments.py\n"
        "- averages are computed over successful_runs only\n"
        "- avg_total_evaluations = avg(f_evals + g_evals + H_evals)\n"
    )

print(f"All sweeps finished. Results saved under: {output_root}", flush=True)
PY

echo "Done. Output root: $OUTPUT_ROOT"
