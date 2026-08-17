"""Recompute reported historical and holdout validation metrics."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True

MODAL_MAP = {
    "Tehran": {
        "car": "modal_share_car",
        "taxi": "modal_share_taxi",
        "bus": "modal_share_bus",
        "metro": "modal_share_metro",
        "motorcycle": "modal_share_motorcycle",
        "other": "modal_share_other",
    },
    "Region12": {
        "car": "modal_share_car_r12",
        "taxi": "modal_share_tax_r12",
        "bus": "modal_share_bus_r12",
        "metro": "modal_share_met_r12",
        "motorcycle": "modal_share_mot_r12",
        "other": "modal_share_oth_r12",
    },
}

SPLITS = {
    "all_hindcast_2012_2023": list(range(2012, 2024)),
    "holdout_train_2012_2021": list(range(2012, 2022)),
    "holdout_test_2022_2023": [2022, 2023],
}


def load_core(core_path: Path):
    spec = importlib.util.spec_from_file_location("system_core", str(core_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["system_core"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def metric_frame(sim_df: pd.DataFrame, truth_df: pd.DataFrame, mapping: Dict[str, str], years: list[int]) -> pd.DataFrame:
    sim = sim_df.copy()
    truth = truth_df.copy()
    sim["YEAR_GRG"] = pd.to_numeric(sim["YEAR_GRG"], errors="coerce")
    truth["YEAR_GRG"] = pd.to_numeric(truth["YEAR_GRG"], errors="coerce")
    sim = sim[sim["YEAR_GRG"].isin(years)]
    truth = truth[truth["YEAR_GRG"].isin(years)]
    rows = []
    for agent, truth_col in mapping.items():
        if agent not in sim.columns or truth_col not in truth.columns:
            continue
        joined = sim[["YEAR_GRG", agent]].merge(truth[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner")
        joined[agent] = pd.to_numeric(joined[agent], errors="coerce")
        joined[truth_col] = pd.to_numeric(joined[truth_col], errors="coerce")
        joined = joined.dropna()
        if len(joined) < 2:
            continue
        pred = joined[agent].to_numpy(float)
        obs = joined[truth_col].to_numpy(float)
        err = pred - obs
        sst = np.square(obs - obs.mean()).sum()
        rows.append({
            "agent": agent,
            "truth_col": truth_col,
            "n": int(len(joined)),
            "years": f"{int(joined.YEAR_GRG.min())}-{int(joined.YEAR_GRG.max())}",
            "RMSE": float(np.sqrt(np.mean(np.square(err)))),
            "MAE": float(np.mean(np.abs(err))),
            "R2": float(1.0 - np.square(err).sum() / sst) if sst > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["R2", "MAE"], na_position="last").reset_index(drop=True)


def build_holdout_config(cfg: Dict[str, Any], raw_df: pd.DataFrame) -> Dict[str, Any]:
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
    for year in (2022, 2023):
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


def modal_metrics(sim_df: pd.DataFrame, truth_df: pd.DataFrame, split_name: str, years: list[int]):
    sim = sim_df.copy()
    truth = truth_df.copy()
    sim["YEAR_GRG"] = pd.to_numeric(sim["YEAR_GRG"], errors="coerce")
    truth["YEAR_GRG"] = pd.to_numeric(truth["YEAR_GRG"], errors="coerce")
    sim = sim[sim["YEAR_GRG"].isin(years)]
    truth = truth[truth["YEAR_GRG"].isin(years)]
    pooled_rows, per_mode_rows = [], []
    for geo, mode_map in MODAL_MAP.items():
        pooled_values = []
        for mode, agent in mode_map.items():
            truth_col = f"{agent}_truth"
            if agent not in sim.columns or truth_col not in truth.columns:
                continue
            joined = sim[["YEAR_GRG", agent]].merge(truth[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner")
            joined[agent] = pd.to_numeric(joined[agent], errors="coerce")
            joined[truth_col] = pd.to_numeric(joined[truth_col], errors="coerce")
            joined = joined.dropna()
            if len(joined) < 2:
                continue
            pred = joined[agent].to_numpy(float)
            obs = joined[truth_col].to_numpy(float)
            err_pp = (pred - obs) * 100.0
            sst = np.square(obs - obs.mean()).sum()
            per_mode_rows.append({
                "convention": "paper_validation",
                "split": split_name,
                "geo": geo,
                "mode": mode,
                "n": int(len(joined)),
                "R2": float(1.0 - np.square(pred - obs).sum() / sst) if sst > 0 else np.nan,
                "MAE_pp": float(np.mean(np.abs(err_pp))),
                "RMSE_pp": float(np.sqrt(np.mean(np.square(err_pp)))),
            })
            pooled_values.extend(zip(obs, pred))
        if pooled_values:
            obs = np.array([v[0] for v in pooled_values], float)
            pred = np.array([v[1] for v in pooled_values], float)
            err_pp = (pred - obs) * 100.0
            sst = np.square(obs - obs.mean()).sum()
            pooled_rows.append({
                "convention": "paper_validation",
                "split": split_name,
                "geo": geo,
                "n": int(len(pooled_values)),
                "R2": float(1.0 - np.square(pred - obs).sum() / sst) if sst > 0 else np.nan,
                "MAE_pp": float(np.mean(np.abs(err_pp))),
                "RMSE_pp": float(np.sqrt(np.mean(np.square(err_pp)))),
            })
    return pd.DataFrame(pooled_rows), pd.DataFrame(per_mode_rows)


def main(root: Path) -> None:
    root = root.resolve()
    ver = root / "verification"
    ver.mkdir(exist_ok=True)
    raw_df = pd.read_csv(root / "config" / "DATA_clean.csv")
    cfg = yaml.safe_load((root / "config" / "BASE_CONFIG.yaml").read_text(encoding="utf-8"))
    sim_sc0 = pd.read_csv(root / "outputs" / "simulation_data_SC0.csv")

    truth_mapping = {c: f"{c}_truth" for c in sim_sc0.columns if f"{c}_truth" in raw_df.columns}
    allhist = metric_frame(sim_sc0, raw_df, truth_mapping, list(range(2012, 2024)))
    allhist.to_csv(ver / "agent_truth_validation_allhistory_2012_2023.csv", index=False)
    ml_agents = [name for name, spec in cfg.get("agents", {}).items() if spec.get("type") == "ml"]
    allhist[allhist.agent.isin(ml_agents)].to_csv(ver / "ml_agent_truth_validation_allhistory_2012_2023.csv", index=False)

    hold_cfg = build_holdout_config(cfg, raw_df)
    core = load_core(root / "config" / "system_core.py")
    run_dir = ver / "_holdout_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(hold_cfg, tmp, sort_keys=False, allow_unicode=True)
        cfg_path = Path(tmp.name)
    try:
        core.run_simulation_from_config(str(cfg_path), data_file=str(root / "config" / "DATA_clean.csv"), output_dir=str(run_dir))
    finally:
        cfg_path.unlink(missing_ok=True)
    hold_sim = pd.read_csv(run_dir / "simulation_data.csv")
    holdout = metric_frame(hold_sim, raw_df, truth_mapping, [2022, 2023])
    holdout.to_csv(ver / "agent_truth_validation_holdout_2022_2023.csv", index=False)
    holdout[holdout.agent.isin(ml_agents)].to_csv(ver / "ml_agent_truth_validation_holdout_2022_2023.csv", index=False)

    truth_cols = [f"{agent}_truth" for modes in MODAL_MAP.values() for agent in modes.values()]
    figure_aligned_truth = sim_sc0[["YEAR_GRG"] + truth_cols].copy()
    pooled, per_mode = [], []
    for split_name, years in SPLITS.items():
        source_sim = sim_sc0 if split_name == "all_hindcast_2012_2023" else hold_sim
        p, m = modal_metrics(source_sim, figure_aligned_truth, split_name, years)
        pooled.append(p)
        per_mode.append(m)
    pooled = pd.concat(pooled, ignore_index=True)
    per_mode = pd.concat(per_mode, ignore_index=True)
    pooled.to_csv(ver / "Table_05_validation_pooled.csv", index=False)
    per_mode.to_csv(ver / "Table_05_validation_per_mode.csv", index=False)

    holdout_pooled = pooled[pooled.split.eq("holdout_test_2022_2023")].copy()
    holdout_per_mode = per_mode[per_mode.split.eq("holdout_test_2022_2023")].copy()
    holdout_pooled[["geo", "R2", "MAE_pp", "RMSE_pp"]].rename(columns={"R2": "SC3"}).to_csv(ver / "real_holdout_pooled_metrics_2022_2023.csv", index=False)
    holdout_per_mode[["geo", "mode", "R2", "MAE_pp", "RMSE_pp"]].rename(columns={"R2": "SC3"}).to_csv(ver / "real_holdout_per_mode_metrics_2022_2023.csv", index=False)

    summary = {
        "modal_validation_convention": "figure-aligned observed modal shares",
        "strict_holdout_years": [2022, 2023],
        "pooled_metrics": pooled.to_dict(orient="records"),
    }
    (ver / "truth_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE truth validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    main(Path(parser.parse_args().root))
