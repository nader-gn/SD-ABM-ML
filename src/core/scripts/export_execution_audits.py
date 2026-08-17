#!/usr/bin/env python3
"""Export deterministic agent-role and execution-graph audit tables.

The structural inventory comes directly from BASE_CONFIG.yaml. Functional roles are
assigned by a small explicit ABM registry plus structural rules: input nodes are inputs,
ML nodes are modal-prior surrogates, registered mediation nodes are ABM, and all
remaining expressions/stocks are SD consequence/feedback roles.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml


def main(root: Path) -> None:
    root = root.resolve()
    cfg_path = root / "config" / "BASE_CONFIG.yaml"
    registry_path = root / "tables" / "functional_role_registry.csv"
    out_dir = root / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    agents = cfg_raw.get("agents", {}) or {}
    registry = pd.read_csv(registry_path)
    abm_ids = set(registry.loc[registry["declared_functional_role"].eq(
        "ABM behavioral mediation / competition / exposure"), "agent_id"].astype(str))

    unknown = sorted(abm_ids - set(agents))
    if unknown:
        raise ValueError(f"Functional-role registry contains unknown agent IDs: {unknown}")

    rows = []
    for agent_id, spec in agents.items():
        spec = spec or {}
        stype = str(spec.get("type", "input")).strip().lower()
        if stype == "input":
            role = "Input/exogenous or initialized state"
        elif stype == "ml":
            role = "ML modal-prior surrogate"
        elif agent_id in abm_ids:
            role = "ABM behavioral mediation / competition / exposure"
        else:
            role = "SD stock-flow / aggregate consequence / feedback"
        deps = list(spec.get("dependencies") or [])
        rows.append({
            "agent_id": agent_id,
            "structural_type": stype,
            "functional_role": role,
            "category": spec.get("category", ""),
            "subcategory": spec.get("subcategory", ""),
            "region": spec.get("region", ""),
            "dependencies_count": len(deps),
            "dependencies": " | ".join(map(str, deps)),
            "target_column": spec.get("target_column", ""),
            "expression": spec.get("expression", ""),
        })

    mapping = pd.DataFrame(rows)
    mapping.to_csv(out_dir / "agent_role_mapping.csv", index=False)

    structural = mapping["structural_type"].value_counts().to_dict()
    functional = mapping["functional_role"].value_counts().to_dict()
    summary_rows = [
        {"classification": "structural", "role": k, "count": int(v)}
        for k, v in sorted(structural.items())
    ] + [
        {"classification": "functional", "role": k, "count": int(v)}
        for k, v in sorted(functional.items())
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "agent_role_summary.csv", index=False)

    # Use the actual engine's normalized dependency semantics for graph statistics.
    sys.path.insert(0, str(root / "config"))
    import system_core as sc  # noqa: E402

    config = sc.load_config_from_yaml(cfg_path)
    runner = sc.HybridSimulationRunner(config)
    solver = runner.solver
    groups = solver.execution_groups
    delayed_count = sum(bool(runner.delayed_selector.is_delayed(name)) for name in runner.agents)
    stats = [
        {"metric": "total_agents", "value": len(runner.agents), "evidence": "BASE_CONFIG.yaml"},
        {"metric": "input_agents", "value": int(structural.get("input", 0)), "evidence": "BASE_CONFIG.yaml"},
        {"metric": "expression_agents", "value": int(structural.get("expression", 0)), "evidence": "BASE_CONFIG.yaml"},
        {"metric": "stock_agents", "value": int(structural.get("stock", 0)), "evidence": "BASE_CONFIG.yaml"},
        {"metric": "ml_agents", "value": int(structural.get("ml", 0)), "evidence": "BASE_CONFIG.yaml"},
        {"metric": "functional_ABM_roles", "value": int(functional.get("ABM behavioral mediation / competition / exposure", 0)), "evidence": "functional_role_registry.csv"},
        {"metric": "functional_SD_roles", "value": int(functional.get("SD stock-flow / aggregate consequence / feedback", 0)), "evidence": "functional_role_registry.csv + structural rule"},
        {"metric": "execution_group_count", "value": len(groups), "evidence": "engine execution graph"},
        {"metric": "multi_node_execution_groups", "value": sum(len(g) > 1 for g in groups), "evidence": "engine execution graph"},
        {"metric": "self_loops", "value": len(getattr(solver, "_self_loops", set())), "evidence": "engine execution graph"},
        {"metric": "delayed_nodes", "value": int(delayed_count), "evidence": "delayed_evaluation selector"},
    ]
    pd.DataFrame(stats).to_csv(out_dir / "execution_graph_audit.csv", index=False)

    # Connectivity audit after expression-dependency inference. A configured node is considered
    # potentially unused only when it has no engine dependency edge in either direction.
    # job_r12 is retained deliberately because its column mapping resolves job_r12_truth, which
    # is an explicit lagged feature for all six R12 ML surrogates. The separate TFP subgraph is
    # consumed by the decision-architecture postprocessor for Region 12 fiscal allocation.
    import re as _re
    names = set(runner.agents)
    incoming = {n: set() for n in names}
    outgoing = {n: set() for n in names}
    def _base_dep(dep):
        return _re.sub(r"(?:__lag|_lag)\d+$", "", str(dep))
    for node, agent in runner.agents.items():
        cfg = agent.config
        deps = (list(cfg.inflows or []) + list(cfg.outflows or [])) if cfg.type == "stock" else list(cfg.dependencies or [])
        for dep in deps:
            base = _base_dep(dep)
            if base in names:
                incoming[node].add(base)
                outgoing[base].add(node)
    training_support = {"job_r12"}
    downstream_support = {"trp_tfp", "trp_r12_tfp"}
    connectivity_rows = []
    for node in sorted(names):
        isolated = not incoming[node] and not outgoing[node]
        support = ""
        if node in training_support:
            support = "ML training/serving feature-column bridge"
        elif node in downstream_support:
            support = "Decision-architecture postprocessing"
        connectivity_rows.append({
            "agent_id": node,
            "in_degree": len(incoming[node]),
            "out_degree": len(outgoing[node]),
            "engine_isolated": bool(isolated),
            "external_support_role": support,
            "unused_configured_node": bool(isolated and not support),
        })
    connectivity = pd.DataFrame(connectivity_rows)
    connectivity.to_csv(out_dir / "agent_connectivity_audit.csv", index=False)
    unused_count = int(connectivity["unused_configured_node"].sum())
    isolated_count = int(connectivity["engine_isolated"].sum())
    pd.DataFrame([
        {"metric": "engine_isolated_nodes", "value": isolated_count, "note": "Includes explicit external-support bridge nodes."},
        {"metric": "unused_configured_nodes", "value": unused_count, "note": "Zero required for the configured model."},
    ]).to_csv(out_dir / "agent_connectivity_summary.csv", index=False)
    if unused_count != 0:
        raise RuntimeError(f"Connectivity audit found unused configured nodes: {connectivity.loc[connectivity.unused_configured_node, 'agent_id'].tolist()}")

    expected = {
        "total_agents": 706,
        "input_agents": 237,
        "expression_agents": 441,
        "stock_agents": 16,
        "ml_agents": 12,
        "functional_ABM_roles": 57,
        "functional_SD_roles": 400,
        "execution_group_count": 706,
        "multi_node_execution_groups": 0,
        "self_loops": 0,
        "delayed_nodes": 44,
    }
    actual = {r["metric"]: int(r["value"]) for r in stats}
    failures = {k: (actual.get(k), v) for k, v in expected.items() if actual.get(k) != v}
    if failures:
        raise RuntimeError(f"Execution/role audit mismatch: {failures}")
    print("Execution/role audit PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
