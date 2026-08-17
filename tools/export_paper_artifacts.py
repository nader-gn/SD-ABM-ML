#!/usr/bin/env python3
"""Collect reported SVG figures and machine-readable source data."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import pandas as pd

MAIN_FIGURES = [f"Figure {i}.svg" for i in range(4, 13)]
SUPPLEMENTARY_FIGURES = ["Figure S8.svg"]


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main(repo: Path) -> None:
    repo = repo.resolve()
    core = repo / "reproduced" / "workspace" / "computational_bundle"
    out = repo / "reproduced" / "paper_outputs"
    if out.exists():
        shutil.rmtree(out)
    fig_out = out / "figures"
    data_out = out / "figure_data"
    fig_out.mkdir(parents=True)
    data_out.mkdir(parents=True)

    rows = []
    for name in MAIN_FIGURES:
        copy_required(core / "figures" / name, fig_out / name)
        rows.append({"paper_item": name[:-4], "format": "SVG", "artifact": (fig_out / name).relative_to(out).as_posix()})
    for name in SUPPLEMENTARY_FIGURES:
        copy_required(core / "figures_supplementary" / name, fig_out / name)
        rows.append({"paper_item": name[:-4], "format": "SVG", "artifact": (fig_out / name).relative_to(out).as_posix()})

    data_map = {
        "Figure_04_modal_share_series.csv": core / "figure_inputs" / "Figure_04_modal_share_series.csv",
        "Figure_04_validation_pooled.csv": core / "verification" / "Table_05_validation_pooled.csv",
        "Figure_04_validation_per_mode.csv": core / "verification" / "Table_05_validation_per_mode.csv",
        "Figure_05_06_timeseries.csv": core / "outputs" / "kpi_timeseries_selected_long_2024_2030.csv",
        "Figure_07_main_scenario_deltas.csv": core / "figure_data" / "Figure_07_main_scenario_deltas.csv",
        "Figure_07_delta_modes.csv": core / "verification" / "Figure_07_delta_modes.csv",
        "Figure_08_secondary_scenario_deltas.csv": core / "figure_data" / "Figure_08_secondary_scenario_deltas.csv",
        "Figure_08_delta_modes.csv": core / "verification" / "Figure_08_delta_modes.csv",
        "Figure_09_auc_input.csv": core / "figure_inputs" / "Figure_09_auc_input.csv",
        "Figure_10_outcome_lens_scores.csv": core / "outputs" / "Figure_10_outcome_lens_scores.csv",
        "Figure_11_equal_weight_scores.csv": core / "outputs" / "Figure_11_equal_weight_scores.csv",
        "Figure_11_rank_acceptability.csv": core / "outputs" / "Figure_11_rank_acceptability.csv",
        "Figure_11_priority_sweep_winners.csv": core / "outputs" / "Figure_11_priority_sweep_winners.csv",
        "Figure_12_implementation_scores.csv": core / "outputs" / "Figure_12_implementation_scores.csv",
    }
    supp_data = core / "figure_inputs_supplementary"
    for p in sorted(supp_data.glob("Figure_S8_*.csv")):
        data_map[p.name] = p
    for name, src in data_map.items():
        copy_required(src, data_out / name)

    pd.DataFrame(rows).to_csv(out / "PAPER_FIGURE_OUTPUTS.csv", index=False)
    map_rows = [
        ["Figure 4", "src/core/scripts/generate_historical_modal_validation.py", "Figure_04_modal_share_series.csv; Figure_04_validation_pooled.csv; Figure_04_validation_per_mode.csv"],
        ["Figure 5", "src/core/scripts/generate_main_trajectories.py", "Figure_05_06_timeseries.csv"],
        ["Figure 6", "src/core/scripts/generate_secondary_trajectories.py", "Figure_05_06_timeseries.csv"],
        ["Figure 7", "src/core/scripts/generate_main_scenario_deltas.py", "Figure_07_main_scenario_deltas.csv; Figure_07_delta_modes.csv"],
        ["Figure 8", "src/core/scripts/generate_secondary_scenario_deltas.py", "Figure_08_secondary_scenario_deltas.csv; Figure_08_delta_modes.csv"],
        ["Figure 9", "src/core/scripts/generate_cumulative_kpi_fingerprints.py", "Figure_09_auc_input.csv"],
        ["Figure 10", "src/core/scripts/generate_four_domain_outcome_lenses.py", "Figure_10_outcome_lens_scores.csv"],
        ["Figure 11", "src/core/scripts/generate_preference_conditioned_decision.py", "Figure_11_equal_weight_scores.csv; Figure_11_rank_acceptability.csv; Figure_11_priority_sweep_winners.csv"],
        ["Figure 12", "src/core/scripts/generate_implementation_screen.py", "Figure_12_implementation_scores.csv"],
        ["Figure S8", "src/supplementary_analyses/generate_figure_S8.py", "Figure_S8_*.csv"],
    ]
    pd.DataFrame(map_rows, columns=["paper_item", "generator_script", "source_data"]).to_csv(out / "FIGURE_DATA_MAP.csv", index=False)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    main(Path(parser.parse_args().repo))
