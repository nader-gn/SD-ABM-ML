"""Generate manuscript Figures 4–12 from refreshed workflow inputs."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import generate_main_trajectories
import generate_secondary_trajectories
import generate_main_scenario_deltas
import generate_secondary_scenario_deltas
import generate_cumulative_kpi_fingerprints
import generate_four_domain_outcome_lenses
import generate_preference_conditioned_decision
import generate_implementation_screen
import generate_historical_modal_validation
import generate_truth_validation

def main(root: Path):
    root = root.resolve()
    generate_main_trajectories.generate_figure(generate_main_trajectories.load_decision12_timeseries(root), root)
    generate_secondary_trajectories.generate_figure(generate_secondary_trajectories.load_timeseries(root), root)
    generate_main_scenario_deltas.generate_figure(generate_main_scenario_deltas.load_decision12_timeseries(root), root)
    generate_secondary_scenario_deltas.generate_figure(generate_secondary_scenario_deltas.load_timeseries(root), root)
    generate_cumulative_kpi_fingerprints.main()
    generate_four_domain_outcome_lenses.main(root)
    generate_preference_conditioned_decision.main(root)
    generate_implementation_screen.main(root)
    if not (root / 'verification' / 'Table_05_validation_per_mode.csv').exists():
        generate_truth_validation.main(root)
    generate_historical_modal_validation.main(root)
    print('DONE figures')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    main(Path(args.root))
