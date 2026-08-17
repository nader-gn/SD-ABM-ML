from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'supplementary_analyses'


def main() -> None:
    p = pd.read_csv(EXT / 'walk_forward_predictions.csv')
    p = p[(p.specification == 'deployed') & (p.year != 2020)].copy()
    eps = 1e-6
    truth = np.clip(p.truth.to_numpy(float), eps, 1 - eps)
    pred = np.clip(p.prediction.to_numpy(float), eps, 1 - eps)
    p['residual_logit'] = np.log(truth / (1-truth)) - np.log(pred / (1-pred))
    rows = []
    for agent, g in p.groupby('agent', sort=True):
        r = g.residual_logit.to_numpy(float)
        rows.append({
            'agent': agent, 'n': len(r), 'mean': float(np.mean(r)),
            'sd': float(np.std(r, ddof=1)), 'rmse': float(np.sqrt(np.mean(r*r))),
            'mae': float(np.mean(np.abs(r))),
        })
    out = pd.DataFrame(rows)
    out.to_csv(EXT / 'ml_logit_residual_scales.csv', index=False)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
