from __future__ import annotations
import copy
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'config'))
import system_core


def reconstruct_training_df(config, df: pd.DataFrame) -> pd.DataFrame:
    year_col = str(config.history_align_col or 'YEAR_GRG')
    if year_col not in df.columns and 'YEAR_GRG' in df.columns:
        year_col = 'YEAR_GRG'
    years = pd.to_numeric(df[year_col], errors='coerce').dropna().astype(int)
    if years.empty:
        return df.copy()
    y_min, y_max = int(years.min()), int(years.max())
    cfg_h = copy.deepcopy(config)
    cfg_h.start_year = int(min(cfg_h.start_year, y_min))
    cfg_h.end_year = int(y_max + 1)
    cfg_h.hindcast_clamp_ml_to_observed = True
    cfg_h.hindcast_clamp_years = list(range(y_min, y_max + 1))
    cfg_h.inject_ml_truth_in_history = True
    cfg_h.prefer_truth_for_endogenous_deps = True
    runner_h = system_core.HybridSimulationRunner(cfg_h)
    res_h = runner_h.run_simulation(df, skip_offline_train=True)
    ts = res_h.get('timeseries', None)
    ts_df = pd.DataFrame(ts) if isinstance(ts, list) else (ts.copy() if isinstance(ts, pd.DataFrame) else pd.DataFrame())
    ts_df = ts_df.loc[:, ~ts_df.columns.duplicated()].copy()
    if ts_df.empty or year_col not in ts_df.columns:
        return df.copy()
    out = df.copy()
    ts_df[year_col] = pd.to_numeric(ts_df[year_col], errors='coerce').astype('Int64')
    out[year_col] = pd.to_numeric(out[year_col], errors='coerce').astype('Int64')
    agent_cols = [a.name for a in config.agents.values() if a.name in ts_df.columns and not str(a.name).endswith('_truth') and a.name != year_col]
    merged = out[[year_col]].merge(ts_df[[year_col] + agent_cols], on=year_col, how='left')
    if agent_cols:
        out.loc[:, agent_cols] = merged[agent_cols].to_numpy()
    return out


def main() -> None:
    cfg_path = ROOT / 'config' / 'BASE_CONFIG.yaml'
    data_path = ROOT / 'config' / 'DATA_clean.csv'
    config = system_core.load_config_from_yaml(cfg_path)
    config.data_file = str(data_path.resolve())
    df = pd.read_csv(data_path)
    train_df = reconstruct_training_df(config, df)

    runner = system_core.HybridSimulationRunner(config)
    train_raw = runner.data_manager.load_data(train_df)
    runner.data_manager.build_lagged_features(train_raw)
    cache = {'models': {}, 'training_rows': int(len(train_df))}
    for name, agent in runner.agents.items():
        mlb = next((b for b in agent.behaviors if isinstance(b, system_core.MLBehavior)), None)
        if mlb is None:
            continue
        X, y, feat_cols = runner.data_manager.prepare_training_data(agent.config, feature_selector=None)
        mlb.fit_from_arrays(X, y, feat_cols)
        cache['models'][name] = {
            'model': mlb.model,
            'n_features_fit': mlb._n_features_fit,
            'selected_features': list(mlb.selected_features or feat_cols),
            'active_dep_specs': list(mlb._active_dep_specs_for_online or []),
            'target_transform': mlb._target_transform,
            'n_samples': int(len(y)),
        }
    out = ROOT / 'supplementary_analyses' / 'ml_model_cache.joblib'
    joblib.dump(cache, out, compress=3)
    train_df.to_csv(ROOT / 'supplementary_analyses' / 'reconstructed_training_data_for_ml.csv', index=False)
    print(f'Wrote {out} with {len(cache["models"])} fitted ML models')


if __name__ == '__main__':
    main()
