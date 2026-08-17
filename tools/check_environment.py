#!/usr/bin/env python3
"""Verify the package versions used for the tested reproduction environment."""
import importlib.metadata as metadata
import platform
import sys

EXPECTED = {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "matplotlib": "3.10.8",
    "PyYAML": "6.0.3",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.0",
    "joblib": "1.5.3",
    "Pillow": "12.3.0",
    "tabulate": "0.10.0",
    "psutil": "7.2.2",
}

print("Python", platform.python_version())
failures = []
for package, expected in EXPECTED.items():
    installed = metadata.version(package)
    print(f"{package:16s} expected={expected:10s} installed={installed}")
    if installed != expected:
        failures.append((package, expected, installed))
if failures:
    print("\nVersion mismatch:", failures, file=sys.stderr)
    sys.exit(2)
print("\nPinned environment matches.")
