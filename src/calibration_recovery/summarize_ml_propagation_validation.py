"""Validate observed hybrid outputs downstream of the fitted ML modal-prior agents.

This diagnostic is deliberately broader than direct ML-target validation. It first
parses the executable dependency graph to identify outputs that are descendants of
one or more of the 12 fitted ML modal-prior agents and that also have an observed
historical counterpart in ``DATA_clean.csv``. Final modal shares remain the primary
ML/ABM validation targets (Table 5). This script adds a second validation layer for
observed *downstream consequences* propagated through the closed-loop SD-ABM-ML
system.

The 2022-2023 test is conditional: modal-share targets are not injected, whereas
non-target exogenous and mapped historical inputs use the information available for
those terminal years. Results therefore assess conditional endogenous reconstruction,
not an unconditional forecast and not causal attribution to ML alone.
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "rerun_output"
BUNDLE = HERE.parent / "computational_bundle"
if not BUNDLE.exists():
    BUNDLE = HERE.parent / "core"

HIST_YEARS = list(range(2013, 2022))
HOLDOUT_YEARS = [2022, 2023]

MODAL = {
    "Tehran": [
        "modal_share_car", "modal_share_taxi", "modal_share_bus",
        "modal_share_metro", "modal_share_motorcycle", "modal_share_other",
    ],
    "R12": [
        "modal_share_car_r12", "modal_share_tax_r12", "modal_share_bus_r12",
        "modal_share_met_r12", "modal_share_mot_r12", "modal_share_oth_r12",
    ],
}

DOWNSTREAM_GROUPS = {
    "mobility_activity": ["trips_bus_total", "trips_metro", "trips_per_year"],
    "energy_air_quality": [
        "fuel_CNG_kg_year", "fuel_gasoline_litre_year",
        "concentration_PM25_ugm3", "concentration_NO2_ugm3",
    ],
    "service_time": [
        "spd_mot", "spd_bus", "spd_car", "spd_tax",
        "travel_time_bus_hours", "travel_time_car_hours",
        "travel_time_motorcycle_hours", "travel_time_taxi_hours",
    ],
    "system_feedback": ["population_city", "GDP_per_capita_IRR_year"],
}
GROUP_LABELS = {
    "mobility_activity": "Mobility and activity",
    "energy_air_quality": "Energy and air quality",
    "service_time": "Service speed and travel time",
    "system_feedback": "System feedback",
}

# Two observed descendants are intentionally not treated as independent propagation
# validation targets because their historical/terminal construction contains a direct
# same-concept empirical anchor.
CONDITIONAL_ANCHORS = {
    "modal_share_bik": "derived from final 'other' share and the mapped historical bike split",
    "pop_r12": "uses observed R12 population through 2023 before switching to the projection ratio",
}


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("ml_prop_system_core", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def executable_dependencies(agents: dict[str, dict]) -> dict[str, set[str]]:
    known = set(agents)
    token_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    deps: dict[str, set[str]] = {}
    for name, cfg in agents.items():
        parents = set(map(str, cfg.get("dependencies") or []))
        parents.update(tok for tok in token_re.findall(str(cfg.get("expression") or "")) if tok in known and tok != name)
        if cfg.get("type") == "stock":
            parents.update(map(str, cfg.get("inflows") or []))
            parents.update(map(str, cfg.get("outflows") or []))
        deps[name] = {p for p in parents if p in known}
    return deps


def shortest_distance_from_ml(agents: dict[str, dict], deps: dict[str, set[str]]) -> dict[str, int]:
    children: dict[str, set[str]] = defaultdict(set)
    for child, parents in deps.items():
        for parent in parents:
            children[parent].add(child)
    ml = [name for name, cfg in agents.items() if cfg.get("type") == "ml"]
    distance = {name: 0 for name in ml}
    q = deque(ml)
    while q:
        node = q.popleft()
        for child in children.get(node, ()):
            candidate = distance[node] + 1
            if child not in distance or candidate < distance[child]:
                distance[child] = candidate
                q.append(child)
    return distance


def build_holdout_config(cfg: dict, raw_df: pd.DataFrame) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["time"]["projection_start_year"] = 2022
    cfg["time"]["end_year"] = 2024
    cfg["ml_training_end_year"] = 2021
    cfg["history_truth_mode"] = "features_only"
    cfg.setdefault("training", {})["inject_ml_truth_in_history"] = False
    cfg["training"]["prefer_truth_for_endogenous_deps"] = True
    exf = copy.deepcopy(cfg.get("exogenous_forecast", {}) or {})
    agents = cfg.get("agents", {}) or {}
    rows = {int(y): rec for y, rec in raw_df.set_index("YEAR_GRG").to_dict(orient="index").items() if pd.notna(y)}

    def put_series(key: str, value: float, year: int) -> None:
        exf.setdefault(key, {})
        exf[key][int(year)] = float(value)

    existing_keys = set(exf.keys())
    for year in HOLDOUT_YEARS:
        row = rows.get(year, {})
        for key in list(existing_keys):
            value = None
            if key in row and pd.notna(row.get(key)):
                value = row[key]
            elif f"{key}_truth" in row and pd.notna(row.get(f"{key}_truth")):
                value = row[f"{key}_truth"]
            elif key in agents:
                col = agents[key].get("column")
                if col in row and pd.notna(row.get(col)):
                    value = row[col]
                elif f"{key}_truth" in row and pd.notna(row.get(f"{key}_truth")):
                    value = row[f"{key}_truth"]
            if value is not None:
                put_series(str(key), float(value), year)
        for name, acfg in agents.items():
            if acfg.get("type") != "input":
                continue
            col = acfg.get("column") or name
            if col in row and pd.notna(row.get(col)):
                put_series(name, float(row[col]), year)
    cfg["exogenous_forecast"] = exf
    return cfg


def run_holdout(cfg: dict, raw: pd.DataFrame) -> pd.DataFrame:
    core = load_core(BUNDLE / "config" / "system_core.py")
    hold_cfg = build_holdout_config(cfg, raw)
    run_dir = OUT / "_ml_propagation_holdout_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(hold_cfg, tmp, sort_keys=False, allow_unicode=True)
        cfg_path = Path(tmp.name)
    try:
        core.run_simulation_from_config(
            str(cfg_path),
            data_file=str(BUNDLE / "config" / "DATA_clean.csv"),
            output_dir=str(run_dir),
        )
    finally:
        cfg_path.unlink(missing_ok=True)
    return pd.read_csv(run_dir / "simulation_data.csv")


def normalized_metrics(sim: pd.DataFrame, raw: pd.DataFrame, agent: str, years: list[int]) -> dict:
    truth_col = f"{agent}_truth"
    m = sim[["YEAR_GRG", agent]].merge(raw[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner")
    m = m[m["YEAR_GRG"].isin(years)].copy()
    y = pd.to_numeric(m[truth_col], errors="coerce").to_numpy(float)
    p = pd.to_numeric(m[agent], errors="coerce").to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        raise ValueError(f"No valid observations for {agent} in {years}")
    err = p - y
    denom = max(float(np.mean(np.abs(y))), 1e-12)
    nz = np.abs(y) > 1e-12
    return {
        "n": int(len(y)),
        "observed_mean": float(np.mean(y)),
        "simulated_mean": float(np.mean(p)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "nmae_percent": float(100.0 * np.mean(np.abs(err)) / denom),
        "nrmse_percent": float(100.0 * np.sqrt(np.mean(err**2)) / denom),
        "mape_percent": float(100.0 * np.mean(np.abs(err[nz] / y[nz]))) if nz.any() else np.nan,
        "relative_bias_percent": float(100.0 * np.mean(err) / denom),
    }


def pooled_modal(sim: pd.DataFrame, raw: pd.DataFrame, years: list[int], geo: str) -> dict:
    ys, ps = [], []
    for agent in MODAL[geo]:
        truth_col = f"{agent}_truth"
        m = sim[["YEAR_GRG", agent]].merge(raw[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner")
        m = m[m["YEAR_GRG"].isin(years)].dropna()
        ys.extend(pd.to_numeric(m[truth_col], errors="coerce").tolist())
        ps.extend(pd.to_numeric(m[agent], errors="coerce").tolist())
    y = np.asarray(ys, float); p = np.asarray(ps, float)
    mask = np.isfinite(y) & np.isfinite(p); y, p = y[mask], p[mask]
    err = p - y
    sst = float(np.square(y - y.mean()).sum())
    return {
        "n": int(len(y)),
        "R2": float(1.0 - np.square(err).sum() / sst) if sst > 0 else np.nan,
        "MAE_pp": float(np.mean(np.abs(err)) * 100.0),
        "RMSE_pp": float(np.sqrt(np.mean(err**2)) * 100.0),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((BUNDLE / "config" / "BASE_CONFIG.yaml").read_text(encoding="utf-8"))
    raw = pd.read_csv(BUNDLE / "config" / "DATA_clean.csv")
    raw["YEAR_GRG"] = pd.to_numeric(raw["YEAR_GRG"], errors="coerce")
    # Match the paper's modal-share convention: the Tehran "other" class includes
    # the separately reported bicycle share for validation/closure. This mirrors the
    # core data-preparation rule used by generate_truth_validation.py.
    raw_modal = raw.copy()
    if "modal_share_other_truth" in raw_modal.columns and "modal_share_bik_truth" in raw_modal.columns:
        raw_modal["modal_share_other_truth"] = (
            pd.to_numeric(raw_modal["modal_share_other_truth"], errors="coerce").fillna(0.0)
            + pd.to_numeric(raw_modal["modal_share_bik_truth"], errors="coerce").fillna(0.0)
        )
    hist_path = OUT / "reference_timeseries.csv"
    if not hist_path.exists():
        raise FileNotFoundError("reference_timeseries.csv is required; run reproduce_full_calibration_evidence.py first")
    hist = pd.read_csv(hist_path)
    hold = run_holdout(cfg, raw)

    agents = cfg["agents"]
    deps = executable_dependencies(agents)
    distance = shortest_distance_from_ml(agents, deps)
    ml_agents = {name for name, acfg in agents.items() if acfg.get("type") == "ml"}
    observed_descendants = sorted(
        name for name in agents
        if name in distance and name not in ml_agents and f"{name}_truth" in raw.columns and name in hist.columns
    )

    modal_agents = set(sum(MODAL.values(), []))
    downstream_agents = set(sum(DOWNSTREAM_GROUPS.values(), []))
    conditional_agents = set(CONDITIONAL_ANCHORS)
    expected = modal_agents | downstream_agents | conditional_agents
    missing = expected - set(observed_descendants)
    unexpected = set(observed_descendants) - expected
    if missing or unexpected:
        raise RuntimeError(f"Observed ML-descendant classification mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    metric_rows = []
    group_lookup = {agent: group for group, names in DOWNSTREAM_GROUPS.items() for agent in names}
    for agent in sorted(downstream_agents, key=lambda x: (group_lookup[x], x)):
        h = normalized_metrics(hist, raw, agent, HIST_YEARS)
        o = normalized_metrics(hold, raw, agent, HOLDOUT_YEARS)
        acfg = agents[agent]
        metric_rows.append({
            "group": group_lookup[agent],
            "group_label": GROUP_LABELS[group_lookup[agent]],
            "agent": agent,
            "unit": str(acfg.get("units", "")),
            "shortest_graph_distance_from_ml_prior": int(distance[agent]),
            "historical_n": h["n"],
            "historical_nrmse_percent": h["nrmse_percent"],
            "historical_nmae_percent": h["nmae_percent"],
            "historical_relative_bias_percent": h["relative_bias_percent"],
            "holdout_n": o["n"],
            "holdout_nrmse_percent": o["nrmse_percent"],
            "holdout_nmae_percent": o["nmae_percent"],
            "holdout_relative_bias_percent": o["relative_bias_percent"],
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "ml_propagation_validation_metrics.csv", index=False)

    group_rows = []
    for group, g in metrics.groupby("group", sort=False):
        group_rows.append({
            "group": group,
            "group_label": GROUP_LABELS[group],
            "output_series": int(len(g)),
            "historical_median_nrmse_percent": float(g["historical_nrmse_percent"].median()),
            "historical_max_nrmse_percent": float(g["historical_nrmse_percent"].max()),
            "holdout_median_nrmse_percent": float(g["holdout_nrmse_percent"].median()),
            "holdout_max_nrmse_percent": float(g["holdout_nrmse_percent"].max()),
            "holdout_series_nrmse_le_10_percent": int((g["holdout_nrmse_percent"] <= 10).sum()),
            "holdout_series_nrmse_le_15_percent": int((g["holdout_nrmse_percent"] <= 15).sum()),
            "holdout_series_nrmse_le_20_percent": int((g["holdout_nrmse_percent"] <= 20).sum()),
        })
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUT / "ml_propagation_validation_group_summary.csv", index=False)

    modal_rows = []
    for split, sim, years in [
        ("historical_calibration_2013_2021", hist, HIST_YEARS),
        ("terminal_holdout_2022_2023", hold, HOLDOUT_YEARS),
    ]:
        for geo in MODAL:
            row = pooled_modal(sim, raw_modal, years, geo)
            modal_rows.append({"split": split, "geography": geo, **row})
    modal_df = pd.DataFrame(modal_rows)
    modal_df.to_csv(OUT / "ml_modal_layer_validation_pooled.csv", index=False)

    coverage = pd.DataFrame([
        {"scope": "observed graph descendants of ML priors", "count": len(observed_descendants), "treatment": "complete graph/truth audit"},
        {"scope": "final modal-share outputs", "count": len(modal_agents), "treatment": "primary modal validation (Table 5; pooled summary repeated here for linkage)"},
        {"scope": "additional endogenous downstream consequences", "count": len(downstream_agents), "treatment": "historical + conditional terminal validation in Table S11"},
        {"scope": "same-concept conditional anchors", "count": len(conditional_agents), "treatment": "identified but excluded from independent propagation accuracy claims"},
    ])
    coverage.to_csv(OUT / "ml_propagation_validation_scope.csv", index=False)

    anchor_rows = [{"agent": k, "reason": v} for k, v in CONDITIONAL_ANCHORS.items()]
    pd.DataFrame(anchor_rows).to_csv(OUT / "ml_propagation_conditional_anchors.csv", index=False)

    summary = {
        "observed_ml_descendants": len(observed_descendants),
        "modal_outputs": len(modal_agents),
        "additional_downstream_outputs": len(downstream_agents),
        "conditional_anchors_not_counted_as_independent_validation": len(conditional_agents),
        "historical_window": [2013, 2021],
        "terminal_holdout_window": HOLDOUT_YEARS,
        "downstream_historical_median_nrmse_percent": float(metrics.historical_nrmse_percent.median()),
        "downstream_holdout_median_nrmse_percent": float(metrics.holdout_nrmse_percent.median()),
        "downstream_holdout_nrmse_le_10_percent": int((metrics.holdout_nrmse_percent <= 10).sum()),
        "downstream_holdout_nrmse_le_15_percent": int((metrics.holdout_nrmse_percent <= 15).sum()),
        "downstream_holdout_nrmse_le_20_percent": int((metrics.holdout_nrmse_percent <= 20).sum()),
        "largest_downstream_holdout_nrmse_percent": float(metrics.holdout_nrmse_percent.max()),
        "largest_downstream_holdout_nrmse_agent": str(metrics.loc[metrics.holdout_nrmse_percent.idxmax(), "agent"]),
        "interpretation": (
            "Validation of observed closed-loop consequences that are graph descendants of fitted ML modal priors. "
            "These outcomes are hybrid outputs and also depend on ABM, SD, exogenous, anchored, and calibrated quantities; "
            "the diagnostic evaluates propagated reconstruction, not causal attribution to ML alone. The 2022-2023 test is conditional on known non-target terminal-year inputs."
        ),
    }
    (OUT / "ml_propagation_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
