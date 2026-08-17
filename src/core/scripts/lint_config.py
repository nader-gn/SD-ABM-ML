"""Lint BASE_CONFIG.yaml for behavior-neutral hygiene guarantees."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
from collections import defaultdict
import yaml
import pandas as pd

ALLOWED_REGIONS = {"city", "region 12"}
ALLOWED_CATEGORIES = {"time", "economic", "transport", "social", "environment", "urban", "impact", "policy", "policy_ref"}
REMOVED_WRAPPERS = {
    "fuel_price_gasoline_IRR_litre_base",
    "len_met_base",
    "len_bus_base",
    "len_met_r12_base",
    "len_bus_r12_base",
    "trp_r12_obs",
}


def add(rows: list[dict], check: str, ok: bool, detail: str, value=None) -> None:
    rows.append({"check": check, "status": "ok" if ok else "fail", "detail": detail, "value": value})


def main(root: Path) -> None:
    root = root.resolve()
    ver = root / "verification"
    ver.mkdir(exist_ok=True)
    cfg = yaml.safe_load((root / "config" / "BASE_CONFIG.yaml").read_text(encoding="utf-8"))
    decision_cfg = yaml.safe_load((root / "config" / "decision_architecture.yaml").read_text(encoding="utf-8"))
    data = pd.read_csv(root / "config" / "DATA_clean.csv", nrows=2)
    agents: dict = cfg["agents"]
    aliases: dict = cfg.get("data_aliases", {})

    inventory = pd.DataFrame([
        {
            "agent": n,
            "type": a.get("type", ""),
            "column": a.get("column", ""),
            "category": a.get("category", ""),
            "subcategory": a.get("subcategory", ""),
            "region": a.get("region", ""),
            "has_expression": bool(a.get("expression")),
            "has_inflows": bool(a.get("inflows")),
            "has_outflows": bool(a.get("outflows")),
        }
        for n, a in agents.items()
    ]).sort_values(["type", "agent"]).reset_index(drop=True)
    inventory.to_csv(ver / "config_agent_inventory.csv", index=False)

    rows: list[dict] = []
    # 1. no duplicate input-column mappings
    col_to_agents = defaultdict(list)
    for n, a in agents.items():
        if a.get("type") == "input" and a.get("column"):
            col_to_agents[str(a["column"])].append(n)
    dup_cols = {c: ns for c, ns in col_to_agents.items() if len(ns) > 1}
    add(rows, "duplicate_input_columns", len(dup_cols) == 0, str(dup_cols) if dup_cols else "none", len(dup_cols))

    # 2. input definitions should stay pure
    anomalies = []
    for n, a in agents.items():
        if a.get("type") == "input":
            for key in ("expression", "inflows", "outflows"):
                if a.get(key):
                    anomalies.append(f"{n}:{key}")
    add(rows, "input_definition_purity", len(anomalies) == 0, ", ".join(anomalies) if anomalies else "none", len(anomalies))

    # 3. metadata completeness
    missing = []
    for n, a in agents.items():
        for key in ("region", "category", "subcategory"):
            if a.get(key) in (None, ""):
                missing.append(f"{n}:{key}")
    add(rows, "metadata_completeness", len(missing) == 0, ", ".join(missing[:20]) if missing else "none", len(missing))

    # 4. metadata vocab
    bad_regions = sorted({a.get("region") for a in agents.values() if a.get("region") not in ALLOWED_REGIONS})
    bad_categories = sorted({a.get("category") for a in agents.values() if a.get("category") not in ALLOWED_CATEGORIES})
    add(rows, "region_vocabulary", len(bad_regions) == 0, str(bad_regions) if bad_regions else "none", len(bad_regions))
    add(rows, "category_vocabulary", len(bad_categories) == 0, str(bad_categories) if bad_categories else "none", len(bad_categories))

    # 5. aliases should resolve to a raw column, not to removed wrappers
    unresolved_aliases = {k: v for k, v in aliases.items() if v not in data.columns}
    removed_still_present = sorted([k for k in REMOVED_WRAPPERS if k in agents or k in aliases])
    add(rows, "alias_targets_resolve_to_data", len(unresolved_aliases) == 0, str(unresolved_aliases) if unresolved_aliases else "none", len(unresolved_aliases))
    add(rows, "noncanonical_wrapper_aliases_absent", len(removed_still_present) == 0, str(removed_still_present) if removed_still_present else "none", len(removed_still_present))

    # 6. dependencies should resolve
    names = set(agents)
    raw_cols = set(map(str, data.columns))
    def _resolve(dep: str) -> bool:
        if dep in names or dep in raw_cols:
            return True
        base = dep.split("__lag", 1)[0] if "__lag" in dep else dep
        if base in names or base in raw_cols:
            return True
        if base in aliases and aliases[base] in raw_cols:
            return True
        return False
    missing_deps = []
    for n, a in agents.items():
        for dep in a.get("dependencies", []) or []:
            if not _resolve(str(dep)):
                missing_deps.append(f"{n}->{dep}")
        for dep in a.get("inflows", []) or []:
            if not _resolve(str(dep)):
                missing_deps.append(f"{n}->{dep}")
        for dep in a.get("outflows", []) or []:
            if not _resolve(str(dep)):
                missing_deps.append(f"{n}->{dep}")
    add(rows, "dependency_resolution", len(missing_deps) == 0, ", ".join(missing_deps[:20]) if missing_deps else "none", len(missing_deps))

    # 7. decision-architecture labels should be canonical and OPEX-safe
    forbidden_label_fragments = [
        'PT service OPEX (operating cost)',
        'PT service operating cost',
        'Net public recurrent/OPEX burden',
        'Net public recurring burden',
        'OPEX/recurring',
        'Budget utilization',
        'PT cost / effective budget',
        'Net annual value after implementation',
    ]
    canonical_labels = decision_cfg.get('canonical_metric_labels', {}) or {}
    label_text = '\n'.join(str(v) for v in canonical_labels.values())
    for pillar_cfg in decision_cfg.get('core_outcome', {}).get('pillars', {}).values():
        for specs in pillar_cfg.get('subfamilies', {}).values():
            for spec in specs:
                label_text += '\n' + str(spec.get('metric', ''))
    for item in decision_cfg.get('implementation_screen', {}).get('metrics', []):
        label_text += '\n' + str(item.get('metric', '')) + '\n' + str(item.get('display', ''))
    found_forbidden = sorted([frag for frag in forbidden_label_fragments if frag in label_text])
    add(rows, 'decision_labels_canonical_opex_terms', len(found_forbidden) == 0, str(found_forbidden) if found_forbidden else 'none', len(found_forbidden))
    add(rows, 'decision_label_pt_opex_short', canonical_labels.get('pt_opex') == 'PT OPEX', f"pt_opex={canonical_labels.get('pt_opex')}")
    add(rows, 'decision_label_net_recurrent_short', canonical_labels.get('net_recurrent_public_burden') == 'Net recurrent public burden', f"net_recurrent_public_burden={canonical_labels.get('net_recurrent_public_burden')}")
    impl_key_missing = [m.get('metric') for m in decision_cfg.get('implementation_screen', {}).get('metrics', []) if not m.get('key')]
    add(rows, 'implementation_metrics_have_keys', len(impl_key_missing) == 0, str(impl_key_missing) if impl_key_missing else 'none', len(impl_key_missing))

    checks = pd.DataFrame(rows)
    checks.to_csv(ver / "config_lint_checks.csv", index=False)
    passed = int(checks["status"].eq("ok").sum())
    total = len(checks)
    summary = [
        "# Config lint summary",
        f"- Checks passing: **{passed}/{total}**.",
        f"- Agent inventory size: **{len(agents)}** agents.",
        f"- Allowed top-level categories: {sorted(ALLOWED_CATEGORIES)}.",
        f"- Allowed regions: {sorted(ALLOWED_REGIONS)}.",
        "",
    ]
    bad = checks.loc[checks["status"] != "ok"]
    if bad.empty:
        summary.append("No config-hygiene violations were detected.")
    else:
        summary += ["## Remaining config issues", bad.to_markdown(index=False)]
    (ver / "config_lint_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    args = ap.parse_args(); main(Path(args.root))
