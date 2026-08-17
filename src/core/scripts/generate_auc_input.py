from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
from generate_cumulative_kpi_input import main as _main

def main(root: Path):
    return _main(root)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    main(Path(args.root))
