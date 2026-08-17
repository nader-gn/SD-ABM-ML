#!/usr/bin/env python3
"""Repository-level reproduction driver for the Tehran SD–ABM–ML study.

The driver never mutates ``src/``, ``reference/``, or the saved ``paper_outputs/`` snapshot. It creates a fresh
``reproduced/`` workspace, reruns the model and supplementary analyses, exports
reported artifacts, checks reported numerical claims, and
finally compares the rerun with immutable reference outputs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
REPRO = REPO / "reproduced"
WS = REPRO / "workspace"
CORE = WS / "computational_bundle"
CAL = WS / "calibration_recovery"
LOGS = REPRO / "logs"
SCENARIOS = [f"SC{i}" for i in range(12)]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MPLBACKEND": "Agg",
}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(THREAD_ENV)
    return env


def log_run(name: str, cmd: list[str], cwd: Path, timeout: int = 1800) -> float:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"START {now()}\nCWD {cwd}\nCMD {' '.join(map(str, cmd))}\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True,
                              env=runtime_env(), timeout=timeout)
        handle.write(f"\nEND {now()} returncode={proc.returncode} elapsed_s={time.time()-start:.3f}\n")
    if proc.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Step {name} failed (see {log_path})\n{tail}")
    return time.time() - start


def materialize(force: bool = True) -> None:
    if force and REPRO.exists():
        shutil.rmtree(REPRO)
    CORE.mkdir(parents=True, exist_ok=True)
    CAL.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for directory in ["config", "scenarios", "scripts", "tables"]:
        shutil.copytree(REPO / "src" / "core" / directory, CORE / directory, dirs_exist_ok=True)
    shutil.copytree(REPO / "src" / "supplementary_analyses", CORE / "supplementary_analyses", dirs_exist_ok=True)
    shutil.copytree(REPO / "src" / "calibration_recovery", CAL, dirs_exist_ok=True)
    for directory in ["outputs", "figure_inputs", "figure_inputs_supplementary", "figures", "figures_supplementary", "verification"]:
        (CORE / directory).mkdir(exist_ok=True)
    (REPRO / "verification").mkdir(exist_ok=True)


def run_scenario(scenario: str):
    cmd = [sys.executable, str(CORE / "scripts" / "run_all_scenarios.py"), "--root", str(CORE), "--scenario", scenario]
    log_path = LOGS / f"core_{scenario}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=CORE, stdout=handle, stderr=subprocess.STDOUT, text=True,
                              env=runtime_env(), timeout=900)
    return scenario, proc.returncode, time.time() - start, log_path


def reproduce_core(workers: int):
    timings = []
    timings.append(("core_lint", log_run("core_lint", [sys.executable, str(CORE / "scripts" / "lint_config.py"), "--root", str(CORE)], CORE)))
    print("Running SC0-SC11 sequentially in isolated processes...", flush=True)
    failures = []
    for scenario in SCENARIOS:
        scenario, rc, elapsed, log_path = run_scenario(scenario)
        print(f"  {scenario}: rc={rc}, {elapsed:.1f}s", flush=True)
        timings.append((f"core_{scenario}", elapsed))
        if rc:
            failures.append((scenario, log_path))
    if failures:
        raise RuntimeError(f"Scenario failures: {failures}")

    scripts = [
        "extract_manuscript_results.py",
        "generate_decision_architecture.py",
        "generate_auc_input.py",
        "generate_historical_modal_validation_input.py",
        "generate_truth_validation.py",
        "run_figures.py",
        "audit_logic.py",
        "run_model_consistency_checks.py",
        "export_execution_audits.py",
    ]
    for script in scripts:
        name = "core_" + script[:-3]
        elapsed = log_run(name, [sys.executable, str(CORE / "scripts" / script), "--root", str(CORE)], CORE, timeout=1200)
        timings.append((name, elapsed))
        print(f"  {name}: {elapsed:.1f}s", flush=True)
    return timings


def reproduce_supplementary(workers: int, force: bool):
    ext = CORE / "supplementary_analyses"
    timings = []
    scripts = [
        ("prepare_ml_cache.py", []),
        ("run_walk_forward_validation.py", []),
        ("derive_ml_residual_scales.py", []),
        ("run_uncertainty_all_scenarios.py", ["--workers", str(workers), "--force"] if force else ["--workers", str(workers)]),
        ("run_uncertainty_blocks_all_scenarios.py", ["--workers", str(workers), "--force"] if force else ["--workers", str(workers)]),
        ("run_abm_policy_ablation.py", []),
        ("run_sc5_coefficient_sensitivity.py", []),
        ("run_sc5_diesel_accounting_sensitivity.py", []),
        ("run_sc6_emission_factor_sensitivity.py", []),
        ("run_sc6_pm25_decomposition.py", []),
        ("run_hard_budget_stress.py", []),
        ("run_exogenous_stress_from_definitions.py", ["--workers", str(workers), "--force"] if force else ["--workers", str(workers)]),
        ("summarize_exogenous_stress_clean.py", []),
        ("verify_central_outputs.py", []),
    ]
    for script, args in scripts:
        name = "supplementary_" + script[:-3]
        elapsed = log_run(name, [sys.executable, str(ext / script), *args], CORE, timeout=1800)
        timings.append((name, elapsed))
        print(f"  {name}: {elapsed:.1f}s", flush=True)
    return timings


def reproduce_calibration():
    timings = []
    elapsed = log_run("parameterization_evidence", [sys.executable, str(CAL / "reproduce_parameterization_evidence.py")], CAL, timeout=600)
    print(f"  parameterization_evidence: {elapsed:.1f}s", flush=True)
    timings.append(("parameterization_evidence", elapsed))
    elapsed = log_run("calibration_recovery", [sys.executable, str(CAL / "reproduce_full_calibration_evidence.py")], CAL, timeout=2400)
    print(f"  calibration_recovery: {elapsed:.1f}s", flush=True)
    timings.append(("calibration_recovery", elapsed))
    elapsed = log_run("calibration_fit_summary", [sys.executable, str(CAL / "summarize_historical_calibration_fit.py")], CAL, timeout=120)
    print(f"  calibration_fit_summary: {elapsed:.1f}s", flush=True)
    timings.append(("calibration_fit_summary", elapsed))
    elapsed = log_run("ml_propagation_validation", [sys.executable, str(CAL / "summarize_ml_propagation_validation.py")], CAL, timeout=1200)
    print(f"  ml_propagation_validation: {elapsed:.1f}s", flush=True)
    timings.append(("ml_propagation_validation", elapsed))
    return timings


def reproduce_supplementary_figures():
    timings = []
    for script in ["generate_figure_S8.py"]:
        name = "supplementary_" + script[:-3]
        elapsed = log_run(name, [sys.executable, str(CORE / "supplementary_analyses" / script)], CORE, timeout=600)
        timings.append((name, elapsed))
        print(f"  {name}: {elapsed:.1f}s", flush=True)
    return timings



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run the complete paper reproduction workflow")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--supplementary", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4)//2)))
    parser.add_argument("--no-force", action="store_true", help="Reuse existing supplementary task files when present")
    parser.add_argument("--keep", action="store_true", help="Keep an existing reproduced/ workspace")
    args = parser.parse_args()
    if not any([args.all, args.core, args.supplementary, args.calibration]):
        args.all = True
    if not args.keep or not CORE.exists():
        materialize(force=not args.keep)

    timings = []
    start = time.time()
    try:
        if args.all or args.core:
            timings += reproduce_core(args.workers)
        if args.all or args.supplementary:
            timings += reproduce_supplementary(args.workers, force=not args.no_force)
        if args.all or args.calibration:
            timings += reproduce_calibration()
        if args.all or args.supplementary:
            timings += reproduce_supplementary_figures()

        if args.all:
            timings.append(("paper_artifact_export", log_run("paper_artifact_export", [sys.executable, str(REPO / "tools" / "export_paper_artifacts.py"), "--repo", str(REPO)], REPO, timeout=300)))
            timings.append(("paper_result_checks", log_run("paper_result_checks", [sys.executable, str(REPO / "tools" / "check_paper_results.py"), "--repo", str(REPO)], REPO, timeout=300)))

        pd.DataFrame([{"step": name, "elapsed_s": seconds} for name, seconds in timings]).to_csv(REPRO / "verification" / "step_timings.csv", index=False)

        if args.all:
            log_run("reference_verification", [sys.executable, str(REPO / "tools" / "verify_against_reference.py"), "--repo", str(REPO)], REPO, timeout=1800)
            print(f"\nSUCCESS. Total elapsed: {(time.time()-start)/60:.1f} min")
            print(REPRO / "verification" / "REPRODUCTION_REPORT.md")
            print(REPRO / "verification" / "PAPER_RESULTS_REPORT.md")
            print(REPRO / "paper_outputs")
        else:
            print(f"\nPARTIAL RUN COMPLETE. Total elapsed: {(time.time()-start)/60:.1f} min")
            print("Run --all for the paper-level numerical and reference-output verification suite.")
    except Exception:
        print("\nREPRODUCTION FAILED. Inspect reproduced/logs/.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
