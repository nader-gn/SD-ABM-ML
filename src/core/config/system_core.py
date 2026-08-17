"""
Hybrid SD–ABM–ML simulation engine.

The module provides the central shared-state runtime used by the Tehran study, including:
- configurable input, expression, stock, and ML agents;
- safe expression evaluation and numerical guards;
- fixed offline ML fitting with closed-loop serving;
- time-aware validation utilities;
- scenario-overlay support;
- configurable constraints and delayed evaluation; and
- YAML configuration loading and simulation entry points.

`SimulationConfig.end_year` is treated as an exclusive upper bound. For example,
start_year=2012, end_year=2030, and timestep=1.0 produce annual steps for 2012–2029.
"""

from __future__ import annotations

import ast
import json
import dataclasses
import datetime
import keyword
import logging
import inspect
import math
import re

def _sanitize_filename(s: str) -> str:
    """Return a filesystem-safe filename token (Windows-safe)."""
    s = str(s or "").strip()
    if not s:
        return "sim"
    # allow alnum and a few safe separators; replace everything else with underscore
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "sim"

import time
import warnings
import yaml
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union, Iterable

import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import copy

# Suppress noisy sklearn warnings locally (do not silence globally in libraries by default).
warnings.filterwarnings("ignore", category=UserWarning)

# Dependencies that should not be lagged even when an ML agent has lag>0
NON_LAG_DEPS: Set[str] = {"YEAR_GRG", "time", "time_idx", "__step", "step"}

# Keys in a hyperparameters dict that are *not* passed to sklearn models
# (used to filter user-provided hyperparameters safely).
NON_MODEL_KEYS: set[str] = {
    'model_type',
    'random_state',
    'target_transform',
    'target_clip',
    'cv_folds',
    'hpo_trials',
    'hpo_timeout_s',
    'hpo_sampler',
    'hpo_pruner',
    'min_samples',
    'max_train_window',
}


# =============================================================================
# NUMERICAL SAFETY HELPERS
# =============================================================================
def safe_div(
    numerator: float,
    denominator: float,
    eps: float = 1e-12,
    cap: float = 1e6,
) -> float:
    """Numerically safe division for expressions."""
    try:
        n = float(numerator)
        d = float(denominator)
    except Exception:
        return 0.0

    if not np.isfinite(n) or not np.isfinite(d):
        return 0.0
    if abs(d) <= eps:
        return 0.0

    v = n / d
    if not np.isfinite(v):
        return 0.0
    return float(np.clip(v, -cap, cap)) if cap is not None else float(v)


def safe_exp(x: Any, clip: float = 5.0) -> Any:
    """Exponent with clipping to avoid overflow."""
    try:
        return float(np.exp(np.clip(float(x), -clip, clip)))
    except Exception:
        arr = np.asarray(x, dtype=float)
        return np.exp(np.clip(arr, -clip, clip))


def safe_log(x: Any, eps: float = 1e-12) -> Any:
    """Log with floor to avoid -inf / NaN."""
    try:
        return float(np.log(max(float(x), eps)))
    except Exception:
        arr = np.asarray(x, dtype=float)
        return np.log(np.maximum(arr, eps))


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class AgentConfig:
    """Unified configuration for all agent types (input/stock/expression/ml)."""

    name: str
    type: str  # 'input', 'stock', 'expression', 'ml'

    # Documentation metadata (optional)
    description: str = ""
    units: str = ""
    category: str = ""
    subcategory: str = ""
    region: str = ""
    subregion: str = ""
    usage: str = ""

    initial_value: float = 0.0
    bounds: Optional[Tuple[float, float]] = None

    # Input agent
    column: Optional[str] = None  # raw data column mapping (optional)

    # Stock agent
    inflows: List[str] = field(default_factory=list)
    outflows: List[str] = field(default_factory=list)

    # Expression agent
    expression: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    # Features that must always be included during feature selection for this ML agent.
    mandatory_features: List[str] = field(default_factory=list)
    lag: int = 0  # expression lag support

    # ML agent
    model_type: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    target_column: Optional[str] = None
    # Feature lag: if dep doesn't declare __lagK explicitly, uses agent.lag

    # Per-agent feature-selection overrides (optional; defaults inherit global feature_selection)
    fs_mode: str = "inherit"  # inherit | manual | auto | hybrid
    fs_enabled: Optional[bool] = None
    fs_pool: Optional[str] = None  # deps_only | all_lagged
    fs_top_k: Optional[int] = None
    fs_min_k: Optional[int] = None
    # NOTE: expression also uses .lag; for ML it is feature lag.
    # (kept as same field name for config simplicity)
    # lag: int already defined above


@dataclass
class HyperparamConfig:
    """Metadata for fixed ML serving and walk-forward stress-year reporting."""
    enabled: bool = False
    exclude_test_years: List[int] = field(default_factory=list)


@dataclass
class FeatureSelectionConfig:
    enabled: bool = False
    # If True, feature selection is fit on each TRAIN fold only (recommended to prevent leakage).
    foldwise: bool = True
    strategy: str = "hybrid"  # filter | embedded | hybrid
    top_k: int = 10
    min_k: int = 0
    # Optional method for filter strategy: "mi" (mutual information) or "corr" (abs Pearson corr)
    method: str = "mi"
    # Candidate pool for ML feature selection:
    #   - "deps_only": use explicit dependencies only
    #   - "all_lagged": use all numeric lagged columns available in the dataset
    pool: str = "deps_only"
    # Which lags to include when pool == "all_lagged" (e.g., [1] for lag-1 only)
    lags: List[int] = field(default_factory=lambda: [1])
    # Mandatory features that must be kept when feature selection is enabled.
    # Useful to ensure policy levers are always present.
    mandatory_features: List[str] = field(default_factory=list)


@dataclass
class DelayedEvaluationConfig:
    """Rules for delaying evaluation of selected nodes until the end of the timestep."""
    enabled: bool = False
    nodes: List[str] = field(default_factory=list)            # explicit node names
    prefixes: List[str] = field(default_factory=list)         # name startswith any
    regex: List[str] = field(default_factory=list)            # regex patterns (full search)


@dataclass
class ConstraintGroupConfig:
    """A generic group constraint (e.g., shares that should sum to 1 or 100)."""
    name: str
    members: List[str]

    enabled: bool = True

    # Normalization target. Common: 1.0 for shares, 100.0 for percentages.
    target_sum: float = 1.0

    # Optional additive smoothing applied before normalization (Dirichlet-like).
    smoothing_alpha: float = 0.0

    # Optional clipping for each member before normalization.
    clip_min: Optional[float] = 0.0
    clip_max: Optional[float] = 1.0

    # Optional inertia for updates:
    # new = prev + update_rate * (new - prev)
    update_rate: float = 1.0

    # When to apply this constraint
    apply_in_history: bool = True
    apply_in_forecast: bool = True


@dataclass
class ConstraintEngineConfig:
    """Domain-agnostic constraint engine."""
    enabled: bool = False
    groups: List[ConstraintGroupConfig] = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Main simulation configuration."""

    name: str
    data_file: str
    start_year: int
    end_year: int  # exclusive end bound
    timestep: float

    # Optional boundary between history (hindcast) and projection (forecast).
    # If provided, it is used as a default for ML training cutoffs and for clarity in logs/exports.
    # Core history detection still relies on presence/absence of aligned rows in the input data.
    projection_start_year: Optional[int] = None

    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    data_aliases: Dict[str, str] = field(default_factory=dict)

    # History alignment column. If None, history rows are matched by row index.
    history_align_col: Optional[str] = "YEAR_GRG"

    # Which raw columns are allowed to be injected into env during history
    exogenous_columns: List[str] = field(default_factory=list)

    # Exogenous forecast locks: var -> {time_key(int): value}; applied only outside history
    exogenous_forecast: Dict[str, Dict[int, float]] = field(default_factory=dict)

    # Optional user-defined lag variables retained for compatibility
    lagged_state_vars: List[str] = field(default_factory=list)

    # Debug/tracing
    trace_agents: List[str] = field(default_factory=list)
    trace_every_n: int = 1
    trace_to_log: bool = False

    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"
    auto_log_file: bool = True
    log_dir: Optional[str] = None
    log_file: Optional[str] = None
    log_file_mode: str = "w"
    log_to_console: bool = True
    log_rotate_mb: int = 20
    log_backups: int = 3

    # Features & engines
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    conservation_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Domain-agnostic constraints and delayed evaluation
    constraints: ConstraintEngineConfig = field(default_factory=ConstraintEngineConfig)
    delayed_evaluation: DelayedEvaluationConfig = field(default_factory=DelayedEvaluationConfig)

    # Optimization
    hyperparams: HyperparamConfig = field(default_factory=HyperparamConfig)

    # Runtime
    max_memory_mb: int = 4096
    random_seed: int = 42

    # Validation (post-hoc)
    validation_targets: Dict[str, str] = field(default_factory=dict)
    validation_align_col: Optional[str] = "YEAR_GRG"
    validation_max_shift: int = 0
    validation_drop_warmup: bool = True
    # If set, validation compares only rows with YEAR_GRG >= validation_min_year
    validation_min_year: Optional[int] = None

    # History assimilation (optional)
    assimilate_history_observations: bool = False
    assimilate_history_exclude_suffixes: Tuple[str, ...] = ("_truth",)
    assimilate_history_targets: bool = False



    # Hindcast/system-validation controls (optional; ignored unless used by runner/pipelines)
    hindcast_clamp_ml_to_observed: bool = False
    hindcast_clamp_years: Optional[List[int]] = None  # inclusive years to clamp; if None, clamp all history rows

    # Runner output control: if set, core writes outputs directly into this folder (no auto run-id)
    force_run_dir: Optional[str] = None
# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(config: SimulationConfig) -> None:
    """Configure root logging with optional console and rotating file handlers.

    Notes:
      - Duplicate handlers are avoided by tagging handlers created here.
      - User-provided handlers are not removed.
    """
    if not config.enable_logging:
        return

    level = getattr(logging, str(config.log_level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    def is_ours(h: logging.Handler) -> bool:
        return bool(getattr(h, "_hybrid_sim_handler", False))

    def has_console_handler() -> bool:
        return any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)

    def has_same_file_handler(path_abs: str) -> bool:
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if str(Path(getattr(h, "baseFilename", "")).resolve()) == path_abs:
                        return True
                except Exception:
                    continue
        return False

    # Auto-generate log file if needed
    desired_file = config.log_file
    if not desired_file and config.auto_log_file:
        try:
            log_dir = config.log_dir or "logs"
            sim_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(config.name).strip())[:80]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            desired_file = str(Path(log_dir) / f"{sim_name}_{ts}.log")
            config.log_file = desired_file
        except Exception:
            desired_file = None

    # Console handler
    if config.log_to_console and not has_console_handler():
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        setattr(ch, "_hybrid_sim_handler", True)
        root.addHandler(ch)

    # File handler
    if desired_file:
        path_abs = str(Path(desired_file).resolve())
        if not has_same_file_handler(path_abs):
            log_path = Path(path_abs)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                filename=str(log_path),
                mode=str(config.log_file_mode),
                maxBytes=int(config.log_rotate_mb) * 1024 * 1024,
                backupCount=int(config.log_backups),
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(fmt)
            setattr(fh, "_hybrid_sim_handler", True)
            root.addHandler(fh)

    # Optional: clean up old hybrid handlers if user re-inits many times
    # Keep only the most recent tagged console and file handlers.
    # If more than six tagged handlers exist, remove the oldest ones.
    ours = [h for h in root.handlers if is_ours(h)]
    if len(ours) > 6:
        for h in ours[:-6]:
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                pass


# =============================================================================
# EXPRESSION DEPENDENCY INFERENCE
# =============================================================================
def infer_expression_dependencies(
    expression: str,
    *,
    known_vars: Optional[Set[str]] = None,
    extra_reserved: Optional[Set[str]] = None,
) -> List[str]:
    """Infer dependencies from an expression string (AST-based, safe and deterministic)."""
    if not expression:
        return []
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError:
        return []

    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(str(node.id))

    reserved = {
        "current",
        "dt",
        "True",
        "False",
        "None",
        "__builtins__",
        "np",
        "numpy",
        "math",
    }
    helper_funcs = {
        "min",
        "max",
        "abs",
        "round",
        "pow",
        "len",
        "sqrt",
        "log",
        "exp",
        "safe_div",
        "int",
        "float",
    }
    if extra_reserved:
        reserved |= set(map(str, extra_reserved))

    filtered = [
        n
        for n in names
        if n not in reserved and n not in helper_funcs and not keyword.iskeyword(n)
    ]

    if known_vars is not None:
        kv = set(map(str, known_vars))
        filtered = [n for n in filtered if n in kv]

    return sorted(set(filtered))


# =============================================================================
# FEATURE SELECTION
# =============================================================================
class FeatureSelector:
    """Simple automated feature selection for ML agents."""

    def __init__(
        self,
        strategy: str = "hybrid",
        top_k: int = 10,
        random_state: int = 42,
        method: str = "mi",
        min_k: int = 0,
    ):
        self.strategy = str(strategy or "hybrid").lower()
        self.top_k = int(top_k)
        self.min_k = int(min_k or 0)
        self.random_state = int(random_state)
        self.method = str(method or "mi").lower()

    def _enforce_k_bounds(self, ranked_names: List[str], mandatory: Optional[List[str]] = None) -> List[str]:
        """Return up to top_k, but at least min_k when possible, while preserving mandatory features."""
        ranked_names = ranked_names or []
        mandatory = [m for m in (mandatory or []) if m is not None]

        ranked = list(dict.fromkeys([n for n in ranked_names if n is not None]))

        ordered: List[str] = []
        for mname in mandatory:
            if mname not in ordered:
                ordered.append(mname)
        for n in ranked:
            if n not in ordered:
                ordered.append(n)

        if not ordered:
            return []

        max_k = max(1, int(self.top_k or len(ordered)))
        min_k = max(0, int(self.min_k or 0))
        if max_k < min_k:
            max_k = min_k

        if len(ordered) <= max_k:
            picked = ordered
        else:
            picked: List[str] = []
            for mname in mandatory:
                if mname in ordered and mname not in picked:
                    picked.append(mname)
                # If mandatory features exceed the budget, truncate deterministically.
                if len(picked) >= max_k:
                    break
            for n in ordered:
                if n not in picked:
                    picked.append(n)
                if len(picked) >= max_k:
                    break

        if len(picked) < min_k:
            for n in ordered[len(picked):]:
                if n not in picked:
                    picked.append(n)
                if len(picked) >= min_k:
                    break

        return picked

    def select_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        mandatory: Optional[List[str]] = None,
    ) -> List[str]:
        if X is None or X.empty or y is None:
            return list(X.columns) if X is not None else []
        if self.strategy == "filter":
            return self._filter_selection(X, y, mandatory=mandatory)
        if self.strategy == "embedded":
            return self._embedded_selection(X, y, mandatory=mandatory)
        if self.strategy == "hybrid":
            f1 = self._filter_selection(X, y, mandatory=mandatory)
            return self._embedded_selection(X[f1], y, mandatory=mandatory) if f1 else []
        return list(X.columns)

    def _filter_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        mandatory: Optional[List[str]] = None,
    ) -> List[str]:
        vt = VarianceThreshold(threshold=1e-5)
        _ = vt.fit_transform(X)
        kept = list(X.columns[vt.get_support()])
        if not kept:
            return []

        m = str(getattr(self, "method", "mi") or "mi").lower()
        if m == "corr":
            yv = pd.to_numeric(pd.Series(y), errors="coerce")
            scores = []
            for c in kept:
                xv = pd.to_numeric(X[c], errors="coerce")
                mask = xv.notna() & yv.notna()
                if mask.sum() < 3:
                    scores.append((c, 0.0))
                    continue
                try:
                    r = float(xv[mask].corr(yv[mask]))
                    if not (r == r):
                        r = 0.0
                except Exception:
                    r = 0.0
                scores.append((c, abs(r)))
            ranked_names = [c for c, _ in sorted(scores, key=lambda t: t[1], reverse=True)]
            return self._enforce_k_bounds(ranked_names, mandatory=mandatory)

        # mutual information
        try:
            mi = mutual_info_regression(X[kept], y, random_state=self.random_state)
            ranked = sorted(zip(kept, mi), key=lambda x: x[1], reverse=True)
            ranked_names = [n for n, _ in ranked]
            return self._enforce_k_bounds(ranked_names, mandatory=mandatory)
        except Exception:
            return self._enforce_k_bounds(kept, mandatory=mandatory)

    def _embedded_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        mandatory: Optional[List[str]] = None,
    ) -> List[str]:
        # Try L1/ElasticNet first (fast), then RF fallback
        try:
            en = ElasticNet(alpha=0.001, l1_ratio=0.8, random_state=self.random_state, max_iter=2000)
            en.fit(X, y)
            coefs = np.abs(en.coef_)
            ranked = sorted(zip(X.columns, coefs), key=lambda x: x[1], reverse=True)
            ranked_names = [n for n, _ in ranked]
            return self._enforce_k_bounds(ranked_names, mandatory=mandatory)
        except Exception:
            try:
                from sklearn.ensemble import RandomForestRegressor

                rf = RandomForestRegressor(n_estimators=120, random_state=self.random_state)
                rf.fit(X, y)
                imps = getattr(rf, "feature_importances_", None)
                if imps is None:
                    return self._enforce_k_bounds(list(X.columns), mandatory=mandatory)
                ranked = sorted(zip(X.columns, imps), key=lambda x: x[1], reverse=True)
                ranked_names = [n for n, _ in ranked]
                return self._enforce_k_bounds(ranked_names, mandatory=mandatory)
            except Exception:
                return self._enforce_k_bounds(list(X.columns), mandatory=mandatory)

class DataManager:
    """Centralized data I/O and lag generation (train/serve parity support)."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._raw_df: Optional[pd.DataFrame] = None
        self._lagged_df: Optional[pd.DataFrame] = None

    def load_data(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Load from provided DataFrame or CSV specified in config."""
        if df is not None:
            self._raw_df = self._postprocess_raw_df(df.copy())
            return self._raw_df

        if self.config.data_file:
            try:
                self._raw_df = self._postprocess_raw_df(pd.read_csv(self.config.data_file))
                return self._raw_df
            except Exception as e:
                logging.warning("DataManager: failed to load %s: %s", self.config.data_file, e)

        self._raw_df = pd.DataFrame()
        return self._raw_df

    def _postprocess_raw_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw dataframe: time columns and aliases."""
        if df is None or df.empty:
            return df
        df = df.copy()

        # Normalize time index columns
        if "time" in df.columns and "time_idx" not in df.columns:
            df["time_idx"] = df["time"]

        # Infer YEAR_GRG if missing (backward compatibility)
        if "YEAR_GRG" not in df.columns:
            start_year = int(self.config.start_year) if self.config.start_year is not None else 0
            if "time" in df.columns:
                df["YEAR_GRG"] = start_year + pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)
            elif "time_idx" in df.columns:
                df["YEAR_GRG"] = start_year + pd.to_numeric(df["time_idx"], errors="coerce").fillna(0).astype(int)

        # Apply configured aliases: new_col <- src_col
        aliases = dict(self.config.data_aliases or {})
        for new_col, src_col in aliases.items():
            if new_col and src_col and new_col not in df.columns and src_col in df.columns:
                df[new_col] = df[src_col]

        # --- Bike share merge: fold bicycle into "other" truth so ML does not model bicycle separately.
        # Keep a copy of the pre-merge other truth for diagnostics.
        if "modal_share_other_truth" in df.columns and "modal_share_bik_truth" in df.columns:
            if "modal_share_other_wo_bik_truth" not in df.columns:
                df["modal_share_other_wo_bik_truth"] = df["modal_share_other_truth"]
            df["modal_share_other_truth"] = (
                pd.to_numeric(df["modal_share_other_truth"], errors="coerce").fillna(0.0)
                + pd.to_numeric(df["modal_share_bik_truth"], errors="coerce").fillna(0.0)
            )

        return df

    def _needed_lag_columns(self) -> Tuple[Set[str], int]:
        """Return (needed_base_columns, max_lag) for lag feature building."""
        needed: Set[str] = set()
        max_lag = 0

        for a in self.config.agents.values():
            if a.type != "ml":
                continue

            base_lag = int(a.lag or 0)
            max_lag = max(max_lag, base_lag)

            # Include target column (raw or truth), but final resolution happens post-load
            if a.target_column:
                needed.add(str(a.target_column))

            for dep_raw in (a.dependencies or []):
                ds = str(dep_raw)
                # Auto-derived previous-delta feature: <base>__dy_prev (requires lag1 & lag2)
                m_dy = re.match(r"^(.*?)(?:_truth)?__dy_prev$", ds)
                if m_dy:
                    base, lk = m_dy.group(1), 2
                else:
                    m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", ds)
                    if m:
                        base, lk = m.group(1), int(m.group(2))
                    else:
                        base, lk = ds, 0
                if base in NON_LAG_DEPS:
                    lk = 0
                else:
                    lk = lk if lk > 0 else base_lag
                max_lag = max(max_lag, lk)
                needed.add(base)

        return needed, max_lag

    def build_lagged_features(self, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build lagged columns required by ML agents. Columns: <col>__lagK.

        Improvements:
          - Only lag columns actually needed by ML deps/targets.
          - Only lag columns that can be safely coerced to numeric.
        """
        if raw_df is None:
            raw_df = getattr(self, "_raw_df", None)
        if raw_df is None or raw_df.empty:
            self._lagged_df = pd.DataFrame()
            return self._lagged_df

        needed_bases, max_lag = self._needed_lag_columns()

        # Ensure lag generation when using a global lagged feature pool
        try:
            fs_cfg = getattr(self.config, 'feature_selection', None)
            pool = str(getattr(fs_cfg, 'pool', 'deps_only') or 'deps_only').lower() if fs_cfg else 'deps_only'
            if pool == 'all_lagged':
                lags_fs = list(getattr(fs_cfg, 'lags', [1]) or [1]) if fs_cfg else [1]
                try:
                    max_lag = max(int(max_lag), max(int(x) for x in lags_fs))
                except Exception:
                    max_lag = max(int(max_lag), 1)
        except Exception:
            pass

        if max_lag <= 0:
            self._lagged_df = raw_df.copy()
            return self._lagged_df

        df = raw_df.copy()

        # If feature_selection.pool == 'all_lagged', build lags for all **configured agents** (not arbitrary CSV columns).
        # This enforces "the feature pool must be declared in the hybrid model config".
        try:
            fs_cfg = getattr(self.config, 'feature_selection', None)
            pool = str(getattr(fs_cfg, 'pool', 'deps_only') or 'deps_only').lower() if fs_cfg else 'deps_only'
            lags_fs = list(getattr(fs_cfg, 'lags', [1]) or [1]) if fs_cfg else [1]
            if pool == 'all_lagged':
                try:
                    max_lag = max(int(max_lag), max(int(x) for x in lags_fs))
                except Exception:
                    max_lag = int(max_lag)

                # Add *only* configured agent names as potential bases. Input agents may map to raw columns.
                for ag_name, ag_cfg in (self.config.agents or {}).items():
                    needed_bases.add(str(ag_name))
                    if getattr(ag_cfg, 'type', None) == 'input' and getattr(ag_cfg, 'column', None):
                        needed_bases.add(str(ag_cfg.column))
        except Exception:
            pass

        # Expand needed set with possible resolved forms (truth + mapped input columns)
        cols_to_consider: Set[str] = set()
        raw_cols = set(map(str, df.columns))

        for base in needed_bases:
            cols_to_consider.add(base)
            cols_to_consider.add(f"{base}_truth")

            ag = self.config.agents.get(base)
            if ag and ag.type == "input" and ag.column:
                cols_to_consider.add(str(ag.column))

        # Keep only columns that exist
        cols_to_lag = [c for c in cols_to_consider if c in raw_cols]

        # Coerce to numeric safely without overwriting the original columns; lag a numeric view.
        numeric_view: Dict[str, pd.Series] = {}
        for col in cols_to_lag:
            # Coerce to numeric but **do not** fill NaNs.
            # Lags should remain NaN where data is unavailable to avoid leakage.
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() < 2:
                # If mostly non-numeric, skip lagging this column
                continue
            numeric_view[col] = s

        if not numeric_view:

            self._lagged_df = df
            return self._lagged_df

        new_cols = {}
        
        for k in range(1, int(max_lag) + 1):
            for col, s in numeric_view.items():
                lagged = s.shift(k)
        
                new_cols[f"{col}__lag{k}"] = lagged
        
        if new_cols:
            lag_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, lag_df], axis=1)
        
        df = df.copy()

        # Auto-derived previous-delta features for ML dependencies: <base>__dy_prev = <base>__lag1 - <base>__lag2
        try:
            dy_bases: Set[str] = set()
            for a in self.config.agents.values():
                if a.type != "ml":
                    continue
                for dep_raw in (a.dependencies or []):
                    ds = str(dep_raw)
                    m_dy = re.match(r"^(.*?)(?:_truth)?__dy_prev$", ds)
                    if m_dy:
                        dy_bases.add(m_dy.group(1))

            dy_new = {}
            for base in dy_bases:
                # Prefer truth lags if present; otherwise use non-truth lags
                if f"{base}__lag1" in df.columns and f"{base}__lag2" in df.columns:
                    dy_new[f"{base}__dy_prev"] = df[f"{base}__lag1"] - df[f"{base}__lag2"]
                if f"{base}_truth__lag1" in df.columns and f"{base}_truth__lag2" in df.columns:
                    dy_new[f"{base}_truth__dy_prev"] = df[f"{base}_truth__lag1"] - df[f"{base}_truth__lag2"]
            if dy_new:
                df = pd.concat([df, pd.DataFrame(dy_new, index=df.index)], axis=1)
        except Exception:
            pass

        self._lagged_df = df
        return self._lagged_df

    def warmup_lags(self, agents: Dict[str, "Agent"], raw_df: pd.DataFrame) -> None:
        """Warm-up MLBehavior internal lag buffers from early history (kept for compatibility).

        Note: In this core, online features are read from runner-injected env lag keys.
        This warmup is harmless and can help if MLBehavior is later extended to use internal buffers.
        """
        if raw_df is None or raw_df.empty:
            return

        for agent in agents.values():
            if agent.config.type != "ml":
                continue
            mlb = next((b for b in agent.behaviors if isinstance(b, MLBehavior)), None)
            if mlb is None:
                continue

            base_lag = int(agent.config.lag or 0)
            dep_lag = 0
            for d in (agent.config.dependencies or []):
                m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", str(d))
                if m:
                    dep_lag = max(dep_lag, int(m.group(2)))
            max_lag = max(base_lag, dep_lag)
            if max_lag <= 0:
                continue

            for dep_raw in (agent.config.dependencies or []):
                dep_s = str(dep_raw)
                m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", dep_s)
                dep = m.group(1) if m else dep_s
                lag_k = int(m.group(2)) if m else base_lag
                if lag_k <= 0:
                    continue

                if dep in raw_df.columns:
                    series = pd.to_numeric(raw_df[dep], errors="coerce")
                elif f"{dep}_truth" in raw_df.columns:
                    series = pd.to_numeric(raw_df[f"{dep}_truth"], errors="coerce")
                else:
                    continue

                series = series.dropna()
                if series.empty:
                    continue

                hist_vals = list(series.iloc[:lag_k])
                if len(hist_vals) < lag_k:
                    pad_val = float(hist_vals[-1]) if hist_vals else 0.0
                    hist_vals = hist_vals + [pad_val] * (lag_k - len(hist_vals))

                for v in hist_vals:
                    vv = float(v) if np.isfinite(float(v)) else 0.0
                    mlb._lag_buffers.setdefault(dep, deque(maxlen=mlb._max_lag + 1)).append(vv)
                    mlb._truth_lag_buffers.setdefault(dep, deque(maxlen=mlb._max_lag + 1)).append(vv)

    def _resolve_training_column(self, dep: str, df_columns: List[str]) -> str:
        """Resolve which dataframe column should be used for a dependency during training.

        Priority:
          1) Endogenous deps: prefer '<dep>_truth' if present
          2) Input deps: prefer agent.column mapping if present
          3) '<dep>' if present
          4) '<dep>_truth' fallback (if present)
        """
        ag = self.config.agents.get(dep)
        is_input = bool(ag and ag.type == "input")

        # Training can exclude truth columns for endogenous dependencies
        # to reduce train/serve mismatch in closed-loop validation.
        prefer_truth = bool(getattr(self.config, "prefer_truth_for_endogenous_deps", True))

        if prefer_truth and (not is_input) and f"{dep}_truth" in df_columns:
            return f"{dep}_truth"

        if ag and ag.column and ag.column in df_columns:
            return ag.column

        if dep in df_columns:
            return dep
        if f"{dep}_truth" in df_columns:
            return f"{dep}_truth"

        raise ValueError(
            f"Training column for dependency '{dep}' not found "
            f"(tried '{dep}', '{dep}_truth', and agent.column mapping)."
        )

    def _resolve_target_column(self, target: str, df_columns: List[str]) -> str:
        """Resolve ML target column.

        If target_column is an agent name, '<name>_truth' is preferred when present.
        Otherwise treat it as a raw column name.
        """
        t = str(target)
        # If it looks like endogenous agent name, prefer truth
        if t in self.config.agents and f"{t}_truth" in df_columns:
            return f"{t}_truth"
        # Or if user already specified truth explicitly
        if t in df_columns:
            return t
        # Try mapped input column
        ag = self.config.agents.get(t)
        if ag and ag.type == "input" and ag.column and ag.column in df_columns:
            return str(ag.column)
        # Try truth fallback
        if f"{t}_truth" in df_columns:
            return f"{t}_truth"
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    def prepare_training_data(
        self,
        agent_cfg: AgentConfig,
        feature_selector: Optional[FeatureSelector] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare (X, y, feature_names) for an ML agent with strict train/serve parity."""
        if self._raw_df is None:
            raise ValueError("DataManager.prepare_training_data called before load_data().")
        if agent_cfg.type != "ml":
            raise ValueError(f"prepare_training_data supports only ML agents (got {agent_cfg.type}).")
        if not agent_cfg.target_column:
            raise ValueError(f"ML agent '{agent_cfg.name}' missing target_column.")

        df = self._lagged_df if self._lagged_df is not None and not self._lagged_df.empty else self._raw_df
        # Holdout control: optionally restrict offline training to years <= offline_train_end_year
        # Filter on df (not `sub`) because `sub` may not include YEAR_GRG.
        try:
            end_y = getattr(self.config, 'offline_train_end_year', None)
            if end_y is not None and df is not None and (not df.empty) and ('YEAR_GRG' in df.columns):
                df = df[pd.to_numeric(df['YEAR_GRG'], errors='coerce') <= float(end_y)].copy()
        except Exception:
            pass
        df_cols = list(df.columns)

        y_col = self._resolve_target_column(agent_cfg.target_column, df_cols)

        # Optional: use a global lagged feature pool (system-wide) instead of explicit deps
        fs_cfg = getattr(self.config, "feature_selection", None)
        pool = str(getattr(fs_cfg, "pool", "deps_only") or "deps_only").strip().lower() if fs_cfg else "deps_only"
        method = str(getattr(fs_cfg, "method", "mi") or "mi").strip().lower() if fs_cfg else "mi"
        lags_req = list(getattr(fs_cfg, "lags", [1]) or [1]) if fs_cfg else [1]
        # Per-agent FS overrides (optional)
        agent_fs_mode = str(getattr(agent_cfg, "fs_mode", "inherit") or "inherit").strip().lower()
        agent_fs_enabled = getattr(agent_cfg, "fs_enabled", None)

        # In auto/hybrid, default candidate pool is all_lagged (agent-centric automatic feature discovery)
        if agent_fs_mode in {"auto", "hybrid"}:
            pool = "all_lagged"
            if agent_fs_enabled is None:
                agent_fs_enabled = True

        # Optional per-agent override for pool
        if getattr(agent_cfg, "fs_pool", None) is not None:
            pool_override = str(getattr(agent_cfg, "fs_pool") or "").strip().lower()
            if pool_override:
                pool = pool_override

        resolved_feat_cols: List[str] = []
        resolved_to_canonical: Dict[str, str] = {}

        if pool == "all_lagged":
            # Candidate features: **only** agents declared in the config (plus their input column mappings),
            # lagged by the requested lags. This enforces that the feature pool is part of the hybrid model,
            # not an arbitrary extra column in the CSV.
            y_base = str(y_col).replace("_truth", "")
            lset = [int(x) for x in lags_req if int(x) > 0] or [1]

            for base_agent, ag_cfg in (self.config.agents or {}).items():
                base_agent = str(base_agent)
                if base_agent in NON_LAG_DEPS:
                    continue
                if base_agent == y_base:
                    continue

                try:
                    resolved_base = self._resolve_training_column(base_agent, df_cols)
                except Exception:
                    # If the base doesn't exist, skip it
                    continue

                for lk in lset:
                    col = f"{resolved_base}__lag{lk}"
                    if col not in df.columns:
                        continue
                    resolved_feat_cols.append(col)
                    resolved_to_canonical[col] = f"{base_agent}__lag{lk}"
        else:
            for dep_raw in (agent_cfg.dependencies or []):
                dep_raw_s = str(dep_raw)
                m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", dep_raw_s)
                if m:
                    base = m.group(1)
                    explicit_lag = int(m.group(2))
                else:
                    base = dep_raw_s
                    explicit_lag = 0

                resolved_base = self._resolve_training_column(base, df_cols)

                effective_lag = explicit_lag if explicit_lag > 0 else int(agent_cfg.lag or 0)
                if base in NON_LAG_DEPS:
                    effective_lag = 0

                if effective_lag > 0:
                    resolved_col = f"{resolved_base}__lag{effective_lag}"
                    canonical = f"{base}__lag{effective_lag}"
                else:
                    resolved_col = resolved_base
                    canonical = base

                resolved_feat_cols.append(resolved_col)
                resolved_to_canonical[resolved_col] = canonical

        # De-duplicate features while preserving order (YAML deps can contain repeats).
        if resolved_feat_cols:
            resolved_feat_cols = list(dict.fromkeys(resolved_feat_cols))
            # Keep only mappings for the retained resolved columns
            resolved_to_canonical = {k: resolved_to_canonical[k] for k in resolved_feat_cols if k in resolved_to_canonical}

        needed_cols = list(resolved_feat_cols) + [y_col]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for training '{agent_cfg.name}': {missing}")

        # Drop rows with NaN in required columns; lag NaNs remain unfilled.
        sub = df[needed_cols].dropna().copy()
        # After holdout filtering, sub can become empty; fail fast with a clear message.
        if sub.empty:
            raise ValueError(f"No training rows after holdout/dropna for '{agent_cfg.name}'.")
        if sub.empty:
            raise ValueError(f"No training rows after dropna for '{agent_cfg.name}'.")

        # Prepare selector with possible per-agent top_k override
        fs_local = feature_selector
        try:
            if fs_local is not None and getattr(agent_cfg, 'fs_top_k', None) is not None:
                # Clone selector with overridden top_k
                fs_local = FeatureSelector(
                    getattr(fs_local, 'strategy', 'hybrid'),
                    int(getattr(agent_cfg, 'fs_top_k')),
                    int(getattr(self.config, 'random_seed', 42) or 42),
                    method=str(getattr(self.config.feature_selection, 'method', 'mi') or 'mi'),
                    min_k=int(getattr(self.config.feature_selection, 'min_k', 0) or 0),
                )
        except Exception:
            fs_local = feature_selector

        # Apply per-agent FS mode
        selected_resolved = list(resolved_feat_cols)
        if agent_fs_mode == 'manual' or agent_fs_enabled is False:
            # manual: use all resolved deps (no FS)
            selected_resolved = list(resolved_feat_cols)
        else:
            # auto/hybrid/inherit: run FS if available
            if fs_local is not None:
                try:
                    # Collect mandatory features (global + per-agent)
                    mand: List[str] = []
                    try:
                        mand.extend(list(getattr(self.config.feature_selection, 'mandatory_features', []) or []))
                    except Exception:
                        pass
                    try:
                        mand.extend(list(getattr(agent_cfg, 'mandatory_features', []) or []))
                    except Exception:
                        pass
                    mand = [str(m) for m in mand if m not in (None, '', 'none')]
                    mand = list(dict.fromkeys(mand))
                    # map mandatory to resolved columns if needed
                    mand_resolved = []
                    for m in mand:
                        # allow mandatory to be specified either as resolved or canonical name
                        if m in resolved_to_canonical.values():
                            # find resolved key(s) that map to this canonical
                            mand_resolved.extend([k for k, v in resolved_to_canonical.items() if v == m])
                        else:
                            mand_resolved.append(m)
                    mand_resolved = [m for m in list(dict.fromkeys(mand_resolved)) if m in resolved_feat_cols]
                    # In hybrid mode, treat explicit deps (deps_only construction) as mandatory.
                    # In auto mode with pool==all_lagged, resolved_feat_cols are already the candidate pool.
                    sel = fs_local.select_features(sub[resolved_feat_cols], sub[y_col], mandatory=mand_resolved)
                    if sel:
                        sel_set = set(sel)
                        selected_resolved = [c for c in resolved_feat_cols if c in sel_set]
                except Exception as e:
                    logging.warning("Feature selection failed for %s: %s", agent_cfg.name, e)

        # Build X robustly even if dataframe has duplicate column names (selecting by label can expand columns)
        X_df = sub[selected_resolved]
        if isinstance(X_df, pd.DataFrame) and X_df.shape[1] != len(selected_resolved):
            cols = []
            series_list = []
            for c in selected_resolved:
                col_obj = sub[c]
                if isinstance(col_obj, pd.DataFrame):
                    col_obj = col_obj.iloc[:, 0]
                series_list.append(pd.to_numeric(col_obj, errors='coerce'))
                cols.append(c)
            X_df = pd.concat(series_list, axis=1)
            X_df.columns = cols
        X = X_df.to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        feat_cols = [resolved_to_canonical[c] for c in selected_resolved]
        return X, y, feat_cols

    def prepare_training_data_with_meta(
        self,
        agent_cfg: AgentConfig,
        feature_selector: Optional[FeatureSelector] = None,
        *,
        meta_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
        """Prepare (X, y, feature_names, meta) for an ML agent.

        This is identical to `prepare_training_data`, but also returns requested meta columns
        aligned to the returned rows (after dropna and optional feature selection).

        Example:
            X, y, feats, meta = dm.prepare_training_data_with_meta(cfg, meta_cols=["YEAR_GRG"])
            years = meta["YEAR_GRG"]
        """
        meta_cols = list(meta_cols or [])

        if self._raw_df is None:
            raise ValueError("DataManager.prepare_training_data_with_meta called before load_data().")
        if agent_cfg.type != "ml":
            raise ValueError(f"prepare_training_data_with_meta supports only ML agents (got {agent_cfg.type}).")
        if not agent_cfg.target_column:
            raise ValueError(f"ML agent '{agent_cfg.name}' missing target_column.")

        df = self._lagged_df if self._lagged_df is not None and not self._lagged_df.empty else self._raw_df
        # Holdout control: optionally restrict offline training to years <= offline_train_end_year
        # Filter on df (not `sub`) because `sub` may not include YEAR_GRG.
        try:
            end_y = getattr(self.config, 'offline_train_end_year', None)
            if end_y is not None and df is not None and (not df.empty) and ('YEAR_GRG' in df.columns):
                df = df[pd.to_numeric(df['YEAR_GRG'], errors='coerce') <= float(end_y)].copy()
        except Exception:
            pass
        df_cols = list(df.columns)

        # Reuse the same resolution logic as prepare_training_data
        y_col = self._resolve_target_column(agent_cfg.target_column, df_cols)

        # Optional: use a global lagged feature pool (system-wide) instead of explicit deps
        fs_cfg = getattr(self.config, "feature_selection", None)
        pool = str(getattr(fs_cfg, "pool", "deps_only") or "deps_only").strip().lower() if fs_cfg else "deps_only"
        method = str(getattr(fs_cfg, "method", "mi") or "mi").strip().lower() if fs_cfg else "mi"
        lags_req = list(getattr(fs_cfg, "lags", [1]) or [1]) if fs_cfg else [1]

        # Per-agent FS overrides (optional)
        agent_fs_mode = str(getattr(agent_cfg, "fs_mode", "inherit") or "inherit").strip().lower()
        agent_fs_enabled = getattr(agent_cfg, "fs_enabled", None)

        # In auto/hybrid, default candidate pool is all_lagged (agent-centric automatic feature discovery)
        if agent_fs_mode in {"auto", "hybrid"}:
            pool = "all_lagged"
            if agent_fs_enabled is None:
                agent_fs_enabled = True

        # Optional per-agent override for pool
        if getattr(agent_cfg, "fs_pool", None) is not None:
            pool_override = str(getattr(agent_cfg, "fs_pool") or "").strip().lower()
            if pool_override:
                pool = pool_override

        resolved_feat_cols: List[str] = []
        resolved_to_canonical: Dict[str, str] = {}

        if pool == 'all_lagged':
            # Candidate features: **only** agents declared in the config (plus their input column mappings),
            # lagged by the requested lags. This enforces that the feature pool is part of the hybrid model,
            # not an arbitrary extra column in the CSV.
            y_base = str(y_col).replace('_truth', '')
            lset = [int(x) for x in lags_req if int(x) > 0] or [1]

            for base_agent, ag_cfg in (self.config.agents or {}).items():
                base_agent = str(base_agent)
                if base_agent in NON_LAG_DEPS:
                    continue
                if base_agent == y_base:
                    continue
                try:
                    resolved_base = self._resolve_training_column(base_agent, df_cols)
                except Exception:
                    continue
                for lk in lset:
                    col = f"{resolved_base}__lag{lk}"
                    if col not in df.columns:
                        continue
                    resolved_feat_cols.append(col)
                    resolved_to_canonical[col] = f"{base_agent}__lag{lk}"
        else:
            for dep_raw in (agent_cfg.dependencies or []):
                dep_raw_s = str(dep_raw)
                m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", dep_raw_s)
                if m:
                    base = m.group(1)
                    explicit_lag = int(m.group(2))
                else:
                    base = dep_raw_s
                    explicit_lag = 0

                resolved_base = self._resolve_training_column(base, df_cols)

                effective_lag = explicit_lag if explicit_lag > 0 else int(agent_cfg.lag or 0)
                if base in NON_LAG_DEPS:
                    effective_lag = 0

                if effective_lag > 0:
                    resolved_col = f"{resolved_base}__lag{effective_lag}"
                    canonical = f"{base}__lag{effective_lag}"
                else:
                    resolved_col = resolved_base
                    canonical = base

                resolved_feat_cols.append(resolved_col)
                resolved_to_canonical[resolved_col] = canonical

        needed = list(resolved_feat_cols) + [y_col] + [c for c in meta_cols if c in df.columns]
        missing = [c for c in (list(resolved_feat_cols) + [y_col]) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for training '{agent_cfg.name}': {missing}")

        sub = df[needed].dropna().copy()
        if sub.empty:
            raise ValueError(f"No training rows after dropna for '{agent_cfg.name}'.")
        # Prepare selector with possible per-agent top_k override
        fs_local = feature_selector
        try:
            if fs_local is not None and getattr(agent_cfg, 'fs_top_k', None) is not None:
                fs_local = FeatureSelector(
                    getattr(fs_local, 'strategy', 'hybrid'),
                    int(getattr(agent_cfg, 'fs_top_k')),
                    int(getattr(self.config, 'random_seed', 42) or 42),
                    method=str(getattr(self.config.feature_selection, 'method', 'mi') or 'mi'),
                )
        except Exception:
            fs_local = feature_selector

        selected_resolved = list(resolved_feat_cols)
        if agent_fs_mode == 'manual' or agent_fs_enabled is False:
            selected_resolved = list(resolved_feat_cols)
        else:
            if fs_local is not None:
                try:
                    sel = fs_local.select_features(sub[resolved_feat_cols], sub[y_col])
                    if sel:
                        sel_set = set(sel)
                        selected_resolved = [c for c in resolved_feat_cols if c in sel_set]
                except Exception as e:
                    logging.warning("Feature selection failed for %s: %s", agent_cfg.name, e)

        # Build X robustly even if dataframe has duplicate column names (selecting by label can expand columns)
        X_df = sub[selected_resolved]
        if isinstance(X_df, pd.DataFrame) and X_df.shape[1] != len(selected_resolved):
            cols = []
            series_list = []
            for c in selected_resolved:
                col_obj = sub[c]
                if isinstance(col_obj, pd.DataFrame):
                    # Duplicate columns with the same name: take the first occurrence
                    col_obj = col_obj.iloc[:, 0]
                series_list.append(pd.to_numeric(col_obj, errors="coerce"))
                cols.append(c)
            X_df = pd.concat(series_list, axis=1)
            X_df.columns = cols
        X = X_df.to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        feat_cols = [resolved_to_canonical[c] for c in selected_resolved]

        meta: Dict[str, np.ndarray] = {}
        for c in meta_cols:
            if c in sub.columns:
                col_obj = sub[c]
                # If the dataframe has duplicate column names, sub[c] becomes a DataFrame.
                # Metadata alignment requires a 1D array; use the first column and emit a warning.
                if isinstance(col_obj, pd.DataFrame):
                    logging.debug(
                        "Meta column '%s' for agent '%s' has duplicate columns; using the first occurrence.",
                        c,
                        agent_cfg.name,
                    )
                    col_obj = col_obj.iloc[:, 0]
                meta[c] = pd.to_numeric(col_obj, errors="coerce").to_numpy()

        return X, y, feat_cols, meta

    def ensure_columns_in_environment(self, env: Dict[str, float]) -> Dict[str, float]:
        """Ensure every raw column exists in env (defaults to 0)."""
        if self._raw_df is None or self._raw_df.empty:
            return env
        for col in self._raw_df.columns:
            env.setdefault(str(col), 0.0)
        return env


# =============================================================================
# FIXED ML SUPPORT UTILITIES
# =============================================================================


# =============================================================================
# BEHAVIOR SYSTEM
# =============================================================================
class Behavior(ABC):
    """Behavior interface used by agents."""

    @abstractmethod
    def apply(self, current_value: float, environment: Dict[str, float], dt: float) -> float:
        raise NotImplementedError


class InputBehavior(Behavior):
    """Reads values for an exogenous/input agent from the environment.

    - Prefer agent name as key (policy-friendly).
    - Fallback to mapped raw column name for compatibility.
    """

    def __init__(self, column: str, default: float = 0.0, *, agent_name: Optional[str] = None):
        self.column = str(column or "")
        self.agent_name = str(agent_name or self.column)
        self.default = float(default)

    def apply(self, current_value: float, environment: Dict[str, float], dt: float) -> float:
        if not environment:
            return self.default

        if self.agent_name and self.agent_name in environment:
            try:
                return float(environment.get(self.agent_name, self.default))
            except Exception:
                return self.default

        if self.column:
            try:
                return float(environment.get(self.column, self.default))
            except Exception:
                return self.default

        return self.default


class StockBehavior(Behavior):
    """Integrates inflows/outflows over time."""

    def __init__(self, inflows: List[str], outflows: List[str]):
        self.inflows = list(inflows or [])
        self.outflows = list(outflows or [])

    def apply(self, current_value: float, environment: Dict[str, float], dt: float) -> float:
        inflow = float(sum(float(environment.get(n, 0.0) or 0.0) for n in self.inflows))
        outflow = float(sum(float(environment.get(n, 0.0) or 0.0) for n in self.outflows))
        return float(current_value) + (inflow - outflow) * float(dt)


class ExpressionBehavior(Behavior):
    """Evaluates mathematical expressions using a restricted eval context."""

    _PROTECTED_NAMES = {
        "min",
        "max",
        "abs",
        "round",
        "pow",
        "sqrt",
        "log",
        "exp",
        "safe_div",
        "int",
        "float",
        "__builtins__",
    }

    def __init__(self, expression: str, dependencies: List[str], agent_name: str = "", lag: int = 0):
        self.expression = str(expression or "")
        self.dependencies = list(dependencies or infer_expression_dependencies(self.expression))
        self.agent_name = str(agent_name or "")
        self.lag = int(lag or 0)

        try:
            self.compiled_expr = compile(self.expression, "<expression>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression for '{self.agent_name}': {e}") from e

    def apply(self, current_value: float, environment: Dict[str, float], dt: float) -> float:
        try:
            env = environment or {}

            class DefaultingDict(dict):
                def __missing__(self, key: str) -> float:
                    if key in ExpressionBehavior._PROTECTED_NAMES:
                        raise KeyError(key)
                    return 0.0

            def dep_value(dep: str) -> float:
                """Get dependency value, optionally lagged."""
                try:
                    if self.lag > 0:
                        k = int(self.lag)
                        v = env.get(f"{dep}__lag{k}", None)
                        if v is None:
                            v = env.get(f"{dep}_lag{k}", None)
                        if v is not None and np.isfinite(float(v)):
                            return float(v)
                    v0 = env.get(dep, 0.0)
                    return float(v0) if np.isfinite(float(v0)) else 0.0
                except Exception:
                    return 0.0

            ctx = DefaultingDict()
            ctx.update(env)

            for name in self.dependencies:
                ctx[name] = dep_value(name)

            ctx["current"] = float(current_value)
            ctx["dt"] = float(dt)

            safe_builtins = {
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow,
                "sqrt": np.sqrt,
                "log": safe_log,
                "exp": safe_exp,
                "safe_div": safe_div,
                "int": int,
                "float": float,
            }

            result = eval(self.compiled_expr, {"__builtins__": safe_builtins}, ctx)

            if isinstance(result, (int, float, np.floating)) and not np.isfinite(float(result)):
                logging.warning("Expression produced non-finite for %s; returning current.", self.agent_name)
                return float(current_value)

            return float(result)
        except Exception as e:
            logging.warning(
                "Expression failed for %s: %s. Expr=%r",
                self.agent_name or "[unknown]",
                e,
                self.expression[:120],
            )
            return float(current_value)


# =============================================================================
# ML BEHAVIOR
# =============================================================================
class MLBehavior(Behavior):
    """ML behavior with fixed offline training and lag-aware closed-loop feature reading."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model: Any = None


        # Fit state
        self._is_fitted = False
        self._warned_not_fitted = False
        self._n_features_fit: Optional[int] = None

        # Parse dependency specs (base_dep, explicit_lag)
        self._lag = int(config.lag or 0)
        self._dep_specs: List[Tuple[str, int]] = []
        self._base_deps: List[str] = []
        self._max_lag = self._lag

        for d in (config.dependencies or []):
            ds = str(d)
            m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", ds)
            if m:
                base, lk = m.group(1), int(m.group(2))
            else:
                base, lk = ds, 0
            self._dep_specs.append((base, lk))
            self._max_lag = max(self._max_lag, lk)

        self._base_deps = sorted(set(b for b, _ in self._dep_specs))
        self._lag_buffers: Dict[str, Deque[float]] = {b: deque(maxlen=self._max_lag + 1) for b in self._base_deps}
        self._truth_lag_buffers: Dict[str, Deque[float]] = {b: deque(maxlen=self._max_lag + 1) for b in self._base_deps}

        # Feature selection memory (optional)
        self.selected_features: Optional[List[str]] = None
        self._active_dep_specs_for_online: Optional[List[Tuple[str, int]]] = None

        # Target transform (optional)
        self._target_transform = self._parse_target_transform()

        self._initialize_model()

    @staticmethod
    def _create_basic_model(model_type: str, params: Dict[str, Any]) -> Any:
        """Create a regression model wrapped in an imputer pipeline for robustness."""
        NON_MODEL_KEYS = {
            "optimize",
            "n_trials",
            "target_transform",
            "logit_eps",
            "smoothing_alpha",
            "smoothing",
            "eps",
            "activation_agent",
            "nonnegative",
            "service_on_col",
            "pre_service_weight",
            "zero_when_col",
            "zero_when_value",
            "zero_output",
            "clip_min",
            "clip_max",
            "update_rate",
            "inertia",
            "prediction_target_scale_shift",
        }
        clean = {k: v for k, v in (params or {}).items() if k not in NON_MODEL_KEYS}

        # Avoid passing duplicate random_state
        rs = int(clean.pop("random_state", 42) or 42)

        def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
            """Filter kwargs to those accepted by cls.__init__ (sklearn safety)."""
            try:
                sig = inspect.signature(cls.__init__)
                allowed = set(sig.parameters.keys())
                allowed.discard('self')
                return {k: v for k, v in kwargs.items() if k in allowed}
            except Exception:
                return kwargs

        def with_imputer(est: Any) -> Pipeline:
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", est)])

        def with_imputer_scaler(est: Any) -> Pipeline:
            from sklearn.preprocessing import StandardScaler
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", est)])

        model_type = str(model_type or "").strip().lower()

        try:
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                return with_imputer(RandomForestRegressor(random_state=rs, **_filter_kwargs(RandomForestRegressor, clean)))
            if model_type == "xgboost":
                import xgboost as xgb  # type: ignore
                return with_imputer(xgb.XGBRegressor(random_state=rs, **clean))
            if model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                return with_imputer(GradientBoostingRegressor(random_state=rs, **_filter_kwargs(GradientBoostingRegressor, clean)))
            if model_type in {"linear", "ridge"}:
                from sklearn.linear_model import Ridge
                alpha = float(clean.pop("alpha", 10.0))
                return with_imputer_scaler(Ridge(alpha=alpha, **clean))
            if model_type == "neural_network":
                from sklearn.neural_network import MLPRegressor
                return with_imputer_scaler(MLPRegressor(random_state=rs, max_iter=500, **_filter_kwargs(MLPRegressor, clean)))
            if model_type == "huber":
                from sklearn.linear_model import HuberRegressor
                epsilon = float(clean.pop("epsilon", 1.35))
                alpha = float(clean.pop("alpha", 1e-4))
                return with_imputer_scaler(HuberRegressor(epsilon=epsilon, alpha=alpha, **clean))
            if model_type == "poisson":
                from sklearn.linear_model import PoissonRegressor
                alpha = float(clean.pop("alpha", 1.0))
                max_iter = int(clean.pop("max_iter", 300))
                return with_imputer_scaler(PoissonRegressor(alpha=alpha, max_iter=max_iter, **_filter_kwargs(PoissonRegressor, clean)))
            if model_type == "tweedie":
                from sklearn.linear_model import TweedieRegressor
                alpha = float(clean.pop("alpha", 1.0))
                power = float(clean.pop("power", 1.5))
                max_iter = int(clean.pop("max_iter", 300))
                return with_imputer_scaler(TweedieRegressor(alpha=alpha, power=power, max_iter=max_iter, **_filter_kwargs(TweedieRegressor, clean)))

            # Fallback
            from sklearn.linear_model import LinearRegression
            logging.warning("Unknown model type '%s'. Using LinearRegression.", model_type)
            return with_imputer(LinearRegression())
        except ImportError as e:
            logging.error("ML backend not available (%s). Using a dummy predictor.", e)

            class DummyModel:
                def fit(self, X: Any, y: Any, **kwargs: Any) -> "DummyModel":
                    return self

                def partial_fit(self, X: Any, y: Any, **kwargs: Any) -> "DummyModel":
                    return self

                def predict(self, X: Any) -> np.ndarray:
                    X = np.asarray(X)
                    return np.zeros((X.shape[0],), dtype=float)

            return DummyModel()

    def _parse_target_transform(self) -> Optional[str]:
        tt = (self.config.hyperparameters or {}).get("target_transform", None)
        tts = str(tt).strip().lower() if tt not in (None, "", "none") else ""
        return tts or None

    def _initialize_model(self) -> None:
        if self.config.model_type:
            self.model = self._create_basic_model(self.config.model_type, self.config.hyperparameters or {})
            self._is_fitted = False
            self._warned_not_fitted = False

    def apply_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        self.config.hyperparameters.update(hyperparams or {})
        self._target_transform = self._parse_target_transform()
        self._initialize_model()

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if not self._target_transform:
            return y
        y = np.asarray(y, dtype=float)
        tt = self._target_transform

        if tt == "log1p":
            if y.size and float(np.nanmin(y)) <= -1.0:
                logging.warning("%s: target_transform=log1p invalid for y<=-1. Disabled.", self.config.name)
                self._target_transform = None
                return y
            return np.log1p(y)

        if tt == "log":
            if y.size and float(np.nanmin(y)) <= 0.0:
                logging.warning("%s: target_transform=log invalid for y<=0. Disabled.", self.config.name)
                self._target_transform = None
                return y
            return np.log(y)

        if tt == "logit":
            eps = float((self.config.hyperparameters or {}).get("logit_eps", 1e-6) or 1e-6)
            eps = max(1e-12, min(1e-3, eps))
            y_clip = np.clip(y, eps, 1.0 - eps)
            return np.log(y_clip / (1.0 - y_clip))

        logging.warning("%s: unknown target_transform=%r. Disabled.", self.config.name, tt)
        self._target_transform = None
        return y

    def _inverse_transform_y(self, y_hat: np.ndarray) -> np.ndarray:
        if not self._target_transform:
            return y_hat
        y_hat = np.asarray(y_hat, dtype=float)
        tt = self._target_transform

        if tt == "log1p":
            return np.expm1(y_hat)
        if tt == "log":
            return np.exp(y_hat)
        if tt == "logit":
            y_hat = np.clip(y_hat, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-y_hat))
        return y_hat

    def fit_from_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feat_cols: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit model on X/y arrays (respects target transform)."""
        if self.model is None:
            self._initialize_model()
        if self.model is None:
            return

        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float)
        y_t = self._transform_y(yv)

        try:
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=float)
                try:
                    self.model.fit(Xv, y_t, model__sample_weight=sw)  # pipeline convention
                except TypeError:
                    self.model.fit(Xv, y_t, sample_weight=sw)
            else:
                self.model.fit(Xv, y_t)
        except TypeError:
            self.model.fit(Xv, y_t)

        self._n_features_fit = int(Xv.shape[1])
        self._is_fitted = True
        self._warned_not_fitted = False
        if feat_cols is not None:
            self.selected_features = list(feat_cols)
            self._active_dep_specs_for_online = [self._to_dep_spec(c) for c in feat_cols]

    def predict_from_arrays(self, X: np.ndarray) -> np.ndarray:
        """Predict y in original target space (inverse-transformed)."""
        Xv = np.asarray(X, dtype=float)
        if Xv.ndim == 1:
            Xv = Xv.reshape(1, -1)

        expected = self._n_features_fit
        if expected is None:
            expected = getattr(self.model, "n_features_in_", None)

        if expected is not None and Xv.shape[1] != int(expected):
            exp = int(expected)
            if Xv.shape[1] > exp:
                Xv = Xv[:, :exp]
            else:
                pad = np.full((Xv.shape[0], exp - Xv.shape[1]), np.nan, dtype=float)
                Xv = np.hstack([Xv, pad])

        y_hat_t = np.asarray(self.model.predict(Xv), dtype=float)
        # Ensemble uncertainty hook: a deterministic additive shift on the
        # model target scale (logit for modal-share surrogates). The default is
        # exactly zero, so the central reference run is unchanged.
        try:
            shift = float((self.config.hyperparameters or {}).get("prediction_target_scale_shift", 0.0) or 0.0)
        except Exception:
            shift = 0.0
        if shift != 0.0:
            y_hat_t = y_hat_t + shift
        return self._inverse_transform_y(y_hat_t)

    @staticmethod
    def _to_dep_spec(colname: str) -> Tuple[str, int]:
        """Convert '<dep>__lagK' / '<dep>_truth__lagK' to (dep, K)."""
        m = re.search(r"__lag(\d+)$", str(colname))
        lag_k = int(m.group(1)) if m else 0
        base = str(colname).split("__lag")[0]
        if base.endswith("_truth"):
            base = base[:-6]
        return base, lag_k

    def _build_feature_vector_online(self, environment: Dict[str, float]) -> List[float]:
        """Build serving features from the runner-injected committed-state lag keys."""
        env = environment or {}
        dep_specs = self._active_dep_specs_for_online or list(self._dep_specs)

        feats: List[float] = []
        for base_dep, explicit_lag in dep_specs:
            lag_k = explicit_lag if explicit_lag > 0 else int(self._lag or 0)
            if base_dep in NON_LAG_DEPS:
                lag_k = 0

            if lag_k <= 0:
                feats.append(float(env.get(base_dep, 0.0) or 0.0))
                continue

            k1 = f"{base_dep}__lag{lag_k}"
            k2 = f"{base_dep}_lag{lag_k}"
            if k1 in env:
                feats.append(float(env.get(k1, 0.0) or 0.0))
            elif k2 in env:
                feats.append(float(env.get(k2, 0.0) or 0.0))
            else:
                feats.append(float(env.get(base_dep, 0.0) or 0.0))

        return feats

    def apply(self, current_value: float, environment: Dict[str, float], dt: float) -> float:
        """Predict the fixed agent output from the current admissible serving state."""
        if self.model is None:
            return float(current_value)

        hp = dict(self.config.hyperparameters or {})

        # Optional hard gating rule: "zero output when a column equals a value"
        zwc = hp.get("zero_when_col", None)
        if zwc is not None:
            try:
                zval = float(hp.get("zero_when_value", 0.0))
                if float(environment.get(str(zwc), 0.0) or 0.0) == zval:
                    return float(hp.get("zero_output", 0.0) or 0.0)
            except Exception:
                pass

        X = self._build_feature_vector_online(environment)
        if not X:
            return float(current_value)

        in_history = bool(environment.get("__in_history", False))

        # Hindcast: optionally clamp ML output to observed (teacher-forcing of ML targets)
        if in_history and bool(environment.get("__clamp_ml_to_observed", False)):
            # Prefer '<agent>_truth' if runner injected it, else fall back to raw column if present
            truth_key = f"{self.config.name}_truth"
            if truth_key in environment and environment.get(truth_key) is not None:
                try:
                    return float(environment.get(truth_key) or 0.0)
                except Exception:
                    pass
            # If target_column itself is present in env (rare), try that too
            if self.config.target_column:
                tcol = str(self.config.target_column)
                if tcol in environment and environment.get(tcol) is not None:
                    try:
                        return float(environment.get(tcol) or 0.0)
                    except Exception:
                        pass

        # Predict
        try:
            y_pred = float(self.predict_from_arrays(np.asarray([X], dtype=float))[0])
            self._is_fitted = True
        except NotFittedError:
            if not self._warned_not_fitted:
                logging.warning("ML model not fitted for %s; retaining current value.", self.config.name)
                self._warned_not_fitted = True
            return float(current_value)
        except Exception as e:
            logging.warning("ML prediction failed for %s: %s", self.config.name, e)
            return float(current_value)

        # Optional activation scaling
        act_agent = hp.get("activation_agent", None)
        if act_agent:
            try:
                act = float(environment.get(str(act_agent), 1.0) or 0.0)
            except Exception:
                act = 1.0
            y_pred = 0.0 if act <= 0.0 else y_pred * act

        # Optional clipping
        try:
            cmin = hp.get("clip_min", None)
            cmax = hp.get("clip_max", None)
            if cmin is not None:
                y_pred = max(float(cmin), y_pred)
            if cmax is not None:
                y_pred = min(float(cmax), y_pred)
        except Exception:
            pass

        # Config bounds
        if self.config.bounds is not None:
            try:
                lo, hi = self.config.bounds
                y_pred = max(float(lo), y_pred)
                y_pred = min(float(hi), y_pred)
            except Exception:
                pass

        # Count-like safety for log/log1p targets
        if (self._target_transform or "").lower() in {"log", "log1p"}:
            y_pred = max(0.0, y_pred)

        # Optional inertia / partial adjustment (generic)
        rate = hp.get("update_rate", hp.get("inertia", None))
        if rate is not None:
            try:
                r = max(0.0, min(1.0, float(rate)))
                y_pred = float(current_value) + r * (y_pred - float(current_value))
            except Exception:
                pass


        return float(y_pred)


# =============================================================================
# AGENT SYSTEM
# =============================================================================
class Agent:
    """Agent composed of behaviors.

    Important:
      - `compute()` is PURE (does not mutate self.value)
      - `update()` calls compute and commits the new value
    """

    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.config = config
        self.value = float(config.initial_value)
        self.bounds = config.bounds
        self.behaviors: List[Behavior] = []
        self._setup_behaviors()

    def _setup_behaviors(self) -> None:
        if self.config.type == "input":
            col = self.config.column or self.name
            self.behaviors.append(InputBehavior(col, self.config.initial_value, agent_name=self.name))

        elif self.config.type == "stock":
            if self.config.inflows or self.config.outflows:
                self.behaviors.append(StockBehavior(self.config.inflows, self.config.outflows))

        elif self.config.type == "expression":
            if self.config.expression:
                deps = self.config.dependencies or infer_expression_dependencies(self.config.expression)
                self.config.dependencies = deps
                self.behaviors.append(
                    ExpressionBehavior(
                        self.config.expression,
                        deps,
                        agent_name=self.name,
                        lag=int(self.config.lag or 0),
                    )
                )

        elif self.config.type == "ml":
            self.behaviors.append(MLBehavior(self.config))

        else:
            raise ValueError(f"Unknown agent type: {self.config.type}")

    def compute(self, current_value: float, environment: Dict[str, float], dt: float) -> float:
        """Compute next value WITHOUT mutating agent state."""
        val = float(current_value)
        for b in self.behaviors:
            val = float(b.apply(val, environment, dt))

        if self.bounds is not None:
            lo, hi = float(self.bounds[0]), float(self.bounds[1])
            val = max(lo, min(hi, val))

        return float(val)

    def update(self, environment: Dict[str, float], dt: float) -> float:
        """Compute and commit agent state."""
        current = float(environment.get(self.name, self.value))
        self.value = self.compute(current, environment, dt)
        return self.value

    def get_dependencies(self) -> List[str]:
        """Return dependencies relevant for within-timestep ordering."""
        cfg = self.config

        if cfg.type == "expression":
            return list(cfg.dependencies or [])

        if cfg.type == "ml":
            deps = list(cfg.dependencies or [])
            lag_k = int(cfg.lag or 0)
            has_explicit_lag = any(re.search(r"(?:__lag|_lag)\d+$", str(d)) for d in deps)
            # ML with lag is treated as exogenous within timestep to avoid SCC blowups
            return [] if (lag_k > 0 or has_explicit_lag) else deps

        if cfg.type == "stock":
            return list(cfg.inflows or []) + list(cfg.outflows or [])

        return []


# =============================================================================
# GRAPH UTILITIES (SCC)
# =============================================================================
def analyze_strongly_connected_components(agents: Dict[str, Agent]) -> List[List[str]]:
    """Tarjan SCC analysis on agent dependency graph (stocks treated as exogenous within timestep)."""
    stock_nodes = {n for n, a in agents.items() if a.config.type == "stock"}

    graph: Dict[str, List[str]] = {}
    for name, agent in agents.items():
        deps = [d for d in agent.get_dependencies() if d in agents]
        if name in stock_nodes:
            graph[name] = []
        else:
            graph[name] = [d for d in deps if d not in stock_nodes]

    index = 0
    stack: List[str] = []
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    on_stack: Set[str] = set()
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in graph:
        if v not in indices:
            strongconnect(v)

    return sccs


# =============================================================================
# DOMAIN-AGNOSTIC DELAYED EVALUATION
# =============================================================================
class DelayedNodeSelector:
    """Select nodes that should be evaluated last in a timestep (domain-agnostic)."""

    def __init__(self, cfg: Optional[DelayedEvaluationConfig] = None):
        self.cfg = cfg or DelayedEvaluationConfig(enabled=False)
        self._nodes = set(map(str, self.cfg.nodes or []))
        self._prefixes = [str(p) for p in (self.cfg.prefixes or []) if p]
        self._regex = []
        for pat in (self.cfg.regex or []):
            try:
                self._regex.append(re.compile(str(pat)))
            except Exception:
                continue

    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def is_delayed(self, name: str) -> bool:
        if not self.enabled():
            return False
        n = str(name)
        if n in self._nodes:
            return True
        if any(n.startswith(p) for p in self._prefixes):
            return True
        return any(r.search(n) for r in self._regex)


# =============================================================================
# DOMAIN-AGNOSTIC CONSTRAINT ENGINE
# =============================================================================
class ConstraintEngine:
    """Applies generic, config-driven constraints post-step."""

    def __init__(self, cfg: ConstraintEngineConfig):
        self.cfg = cfg or ConstraintEngineConfig(enabled=False)

    @staticmethod
    def _clip(v: float, lo: Optional[float], hi: Optional[float]) -> float:
        if not np.isfinite(v):
            v = 0.0
        if lo is not None:
            v = max(float(lo), v)
        if hi is not None:
            v = min(float(hi), v)
        return float(v)

    def apply(
        self,
        env: Dict[str, float],
        *,
        prev_env: Optional[Dict[str, float]] = None,
        in_history: bool = False,
    ) -> Dict[str, float]:
        """Apply constraints to env and return env (mutates in place for performance)."""
        if not (self.cfg and self.cfg.enabled):
            return env

        prev = prev_env or {}

        for g in (self.cfg.groups or []):
            if not g.enabled:
                continue
            if in_history and not g.apply_in_history:
                continue
            if (not in_history) and not g.apply_in_forecast:
                continue

            members = [str(m) for m in (g.members or []) if m]
            if not members:
                continue
            if not any(m in env for m in members):
                continue

            # Optional inertia: move towards new values from previous timestep values
            ur = float(g.update_rate if g.update_rate is not None else 1.0)
            ur = 1.0 if (not np.isfinite(ur)) else float(np.clip(ur, 0.0, 1.0))

            raw_vals: List[float] = []
            for m in members:
                v = float(env.get(m, 0.0) or 0.0)
                v = self._clip(v, g.clip_min, g.clip_max)
                if ur < 1.0:
                    pv = float(prev.get(m, v) or v)
                    pv = self._clip(pv, g.clip_min, g.clip_max)
                    v = pv + ur * (v - pv)
                raw_vals.append(v)

            # Normalize to target sum with optional smoothing
            alpha = float(g.smoothing_alpha or 0.0)
            alpha = 0.0 if (not np.isfinite(alpha) or alpha < 0.0) else alpha

            total = float(np.sum(raw_vals)) + alpha * float(len(raw_vals))
            if total <= 1e-12:
                # If degenerate, keep values as-is (already clipped/inertial)
                for m, v in zip(members, raw_vals):
                    env[m] = float(v)
                continue

            target = float(g.target_sum if g.target_sum is not None else 1.0)
            target = 1.0 if (not np.isfinite(target) or target <= 0.0) else target

            for m, v in zip(members, raw_vals):
                nv = (v + alpha) / total
                nv = target * nv
                env[m] = float(nv)

        return env


# =============================================================================
# SD SOLVER (PURE; DOES NOT MUTATE AGENTS)
# =============================================================================
class SimpleSDSolver:
    """System dynamics solver with SCC grouping and safe handling of stocks.

    IMPORTANT:
      - This solver is PURE w.r.t agent states (uses Agent.compute, not Agent.update).
      - Runner commits agent.value after the step is finalized.
    """

    def __init__(self, agents: Dict[str, Agent], delayed_selector: Optional[DelayedNodeSelector] = None):
        self.agents = agents
        self.delayed_selector = delayed_selector or DelayedNodeSelector(DelayedEvaluationConfig(enabled=False))

        self.solver_stats: Dict[str, Any] = {
            "total_steps": 0,
            "avg_step_time": 0.0,
            "circular_dependencies": [],
            "circular_dependency_groups": [],
        }

        try:
            self.execution_groups, self._self_loops = self._compute_execution_groups()
            self.dependency_order = [n for grp in self.execution_groups for n in grp]
        except Exception as e:
            logging.warning("Execution group build failed, using flat order: %s", e)
            self.dependency_order = self._compute_execution_order()
            self.execution_groups = [list(self.dependency_order)]
            self._self_loops = set()

    def solve_timestep(self, environment: Dict[str, float], dt: float) -> Dict[str, float]:
        """Solve one timestep using a staged approach (stocks integrated once)."""
        start = time.time()
        results = dict(environment or {})

        delayed_nodes = [n for n in self.agents if self.delayed_selector.is_delayed(n)]
        stock_names = {n for n, a in self.agents.items() if a.config.type == "stock"}

        # Seed delayed nodes (freeze during earlier phases)
        delayed_seed = {n: float(results.get(n, 0.0)) for n in delayed_nodes}

        # Phase 1: solve algebraic nodes (no stocks, no delayed)
        for grp in self.execution_groups:
            algebraic = [n for n in grp if (n not in stock_names and n not in delayed_nodes)]
            if not algebraic:
                continue

            if len(algebraic) == 1 and algebraic[0] not in self._self_loops:
                name = algebraic[0]
                try:
                    cur = float(results.get(name, 0.0))
                    results[name] = self.agents[name].compute(cur, results, dt)
                except Exception as e:
                    logging.warning("Update failed for %s: %s", name, e)
            else:
                results = self._solve_cycles(results, algebraic, dt, integrate_stocks=False)

        # Keep delayed nodes held fixed while integrating stocks
        for n, v in delayed_seed.items():
            results[n] = v

        # Phase 2: integrate stocks once
        for name in self.dependency_order:
            if name in stock_names:
                try:
                    cur = float(results.get(name, 0.0))
                    results[name] = self.agents[name].compute(cur, results, dt)
                except Exception as e:
                    logging.warning("Stock integration failed for %s: %s", name, e)

        # Phase 3: refresh expressions (post-stock) for consistency
        for name in self.dependency_order:
            if name in delayed_nodes or name in stock_names:
                continue
            if self.agents[name].config.type == "expression":
                try:
                    cur = float(results.get(name, 0.0))
                    results[name] = self.agents[name].compute(cur, results, dt)
                except Exception as e:
                    logging.warning("Post-stock refresh failed for %s: %s", name, e)

        # Phase 4: evaluate delayed nodes once
        for n in delayed_nodes:
            try:
                cur = float(results.get(n, delayed_seed.get(n, 0.0)))
                results[n] = self.agents[n].compute(cur, results, dt)
            except Exception as e:
                logging.warning("Delayed update failed for %s: %s", n, e)
                results[n] = float(results.get(n, delayed_seed.get(n, 0.0)))

        self._update_stats(time.time() - start)
        return results

    def _solve_cycles(self, env: Dict[str, float], nodes: List[str], dt: float, *, integrate_stocks: bool) -> Dict[str, float]:
        """Fixed-point iteration on an SCC (stocks held fixed)."""
        results = dict(env)
        nodes = [n for n in nodes if n in self.agents]

        stock_names = [n for n in nodes if self.agents[n].config.type == "stock"]
        non_stock = [n for n in nodes if n not in stock_names]

        delayed = [n for n in non_stock if self.delayed_selector.is_delayed(n)]
        ml_nodes = [n for n in non_stock if self.agents[n].config.type == "ml"]
        iter_nodes = [n for n in non_stock if n not in delayed and n not in ml_nodes]

        stock_seed = {n: float(results.get(n, 0.0)) for n in stock_names}
        delayed_seed = {n: float(results.get(n, 0.0)) for n in delayed}

        # Freeze ML nodes (evaluate once)
        ml_seed: Dict[str, float] = {}
        for n in ml_nodes:
            try:
                cur = float(results.get(n, 0.0))
                ml_seed[n] = float(self.agents[n].compute(cur, results, dt))
            except Exception:
                ml_seed[n] = float(results.get(n, 0.0))
            results[n] = ml_seed[n]

        # Picard iteration with damping
        tol_abs, tol_rel = 1e-7, 1e-6
        alpha, alpha_min = 0.5, 0.15
        max_iter = 40
        prev_resid = float("inf")

        for _ in range(max_iter):
            # Freeze stock/delayed/ML nodes during iteration
            for n, v in stock_seed.items():
                results[n] = v
            for n, v in delayed_seed.items():
                results[n] = v
            for n, v in ml_seed.items():
                results[n] = v

            max_resid = 0.0
            for n in iter_nodes:
                old = float(results.get(n, 0.0))
                try:
                    new = float(self.agents[n].compute(old, results, dt))
                    damped = alpha * new + (1.0 - alpha) * old
                    results[n] = damped
                    max_resid = max(max_resid, abs(damped - old))
                except Exception:
                    results[n] = old

            if max_resid <= (tol_abs + tol_rel):
                break

            if max_resid > prev_resid * 1.05:
                alpha = max(alpha * 0.7, alpha_min)
            elif max_resid < prev_resid * 0.7:
                alpha = min(alpha * 1.1, 0.7)
            prev_resid = max_resid

        # Integrate stocks exactly once (if requested)
        if integrate_stocks:
            for n in stock_names:
                try:
                    results[n] = stock_seed[n]
                    results[n] = float(self.agents[n].compute(results[n], results, dt))
                except Exception:
                    results[n] = float(results.get(n, stock_seed[n]))

        # Evaluate delayed nodes once post-iteration
        for n in delayed:
            try:
                results[n] = delayed_seed[n]
                results[n] = float(self.agents[n].compute(results[n], results, dt))
            except Exception:
                results[n] = float(results.get(n, delayed_seed[n]))

        return results

    def newton_correction_global(self, env: Dict[str, float], dt: float) -> Dict[str, float]:
        """Global Newton-like correction (safe: skips stocks and delayed nodes)."""
        results = dict(env)

        for name, agent in self.agents.items():
            if agent.config.type == "stock":
                continue
            if self.delayed_selector.is_delayed(name):
                continue

            old_val = float(results.get(name, 0.0))
            eps = 1e-4
            try:
                f_x = float(agent.compute(old_val, results, dt)) - old_val
                perturbed = dict(results)
                perturbed[name] = old_val + eps
                f_x_eps = float(agent.compute(old_val + eps, perturbed, dt)) - (old_val + eps)
            except Exception:
                continue

            deriv = (f_x_eps - f_x) / eps if abs(eps) > 1e-12 else 1.0
            corr = -f_x / (deriv + 1e-8)
            results[name] = old_val + corr

        return results

    def _compute_execution_order(self) -> List[str]:
        """Topological sort (stocks are included but edges into stocks are removed by Agent.get_dependencies logic)."""
        graph = defaultdict(list)
        indeg = defaultdict(int)
        nodes = set(self.agents.keys())

        for name, agent in self.agents.items():
            for dep in agent.get_dependencies():
                if dep in nodes:
                    graph[dep].append(name)
                    indeg[name] += 1

        for n in nodes:
            indeg.setdefault(n, 0)

        q = deque([n for n in nodes if indeg[n] == 0])
        order: List[str] = []

        while q:
            v = q.popleft()
            order.append(v)
            for w in graph.get(v, []):
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

        if len(order) != len(nodes):
            remaining = list(nodes - set(order))
            self.solver_stats["circular_dependencies"] = remaining
            order.extend(remaining)

        return order

    def _compute_execution_groups(self) -> Tuple[List[List[str]], Set[str]]:
        """Condense SCCs and topologically sort SCC graph to get execution groups."""
        nodes = list(self.agents.keys())
        node_set = set(nodes)

        deps_map: Dict[str, List[str]] = {}
        self_loops: Set[str] = set()

        for name, agent in self.agents.items():
            deps = [d for d in agent.get_dependencies() if d in node_set]
            if name in deps:
                self_loops.add(name)
            deps_map[name] = deps

        sccs = analyze_strongly_connected_components(self.agents)
        scc_id: Dict[str, int] = {}
        for i, comp in enumerate(sccs):
            for n in comp:
                scc_id[n] = i

        scc_graph: Dict[int, Set[int]] = defaultdict(set)
        indeg: Dict[int, int] = defaultdict(int)

        for name, deps in deps_map.items():
            b = scc_id.get(name)
            if b is None:
                continue
            for dep in deps:
                a = scc_id.get(dep)
                if a is None or a == b:
                    continue
                if b not in scc_graph[a]:
                    scc_graph[a].add(b)
                    indeg[b] += 1

        for i in range(len(sccs)):
            indeg.setdefault(i, 0)

        q = deque([i for i in range(len(sccs)) if indeg[i] == 0])
        topo: List[int] = []
        while q:
            i = q.popleft()
            topo.append(i)
            for j in scc_graph.get(i, set()):
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)

        if len(topo) != len(sccs):
            topo = list(range(len(sccs)))

        groups = [list(sccs[i]) for i in topo]
        cyc_groups = [g for g in groups if len(g) > 1]
        if cyc_groups:
            self.solver_stats["circular_dependency_groups"] = cyc_groups

        return groups, self_loops

    def _update_stats(self, step_time: float) -> None:
        self.solver_stats["total_steps"] += 1
        n = self.solver_stats["total_steps"]
        prev = float(self.solver_stats.get("avg_step_time", 0.0))
        self.solver_stats["avg_step_time"] = (prev * (n - 1) + float(step_time)) / n


# =============================================================================
# POLICY SYSTEM
# =============================================================================




# =============================================================================
# VALIDATION
# =============================================================================
class ValidationSuite:
    """Post-hoc validation comparing simulated results to ground truth columns."""

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _apply_shift(yt: np.ndarray, yp: np.ndarray, shift: int) -> Tuple[np.ndarray, np.ndarray]:
        if shift == 0:
            return yt, yp
        if shift > 0:
            if len(yt) <= shift or len(yp) <= shift:
                return np.array([]), np.array([])
            return yt[:-shift], yp[shift:]
        k = -shift
        if len(yt) <= k or len(yp) <= k:
            return np.array([]), np.array([])
        return yt[k:], yp[:-k]

    def compute(
        self,
        sim_results: List[Dict[str, Any]],
        raw_df: pd.DataFrame,
        mapping: Dict[str, str],
        *,
        align_col: Optional[str] = "YEAR_GRG",
        max_shift: int = 0,
        warmup: int = 0,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if not sim_results or raw_df is None or raw_df.empty or not mapping:
            return metrics

        sim_df = pd.DataFrame(sim_results)
        use_merge = bool(align_col) and (align_col in sim_df.columns) and (align_col in raw_df.columns)

        if use_merge:
            try:
                sim_df = sim_df.copy()
                raw_df = raw_df.copy()
                sim_df[align_col] = pd.to_numeric(sim_df[align_col], errors="coerce")
                raw_df[align_col] = pd.to_numeric(raw_df[align_col], errors="coerce")
                sim_df = sim_df.dropna(subset=[align_col])
                raw_df = raw_df.dropna(subset=[align_col])
                # Keep merge stable for annual cases; for fractional, keep float merge.
            except Exception:
                use_merge = False

        for agent, truth_col in (mapping or {}).items():
            if agent not in sim_df.columns or truth_col not in raw_df.columns:
                continue

            if use_merge:
                dfp = sim_df[[align_col, agent]].rename(columns={agent: "__y_pred"})
                dft = raw_df[[align_col, truth_col]].rename(columns={truth_col: "__y_true"})
                joined = dfp.merge(dft, on=align_col, how="inner").sort_values(by=align_col)
                # Optional: evaluate only years >= validation_min_year
                try:
                    min_y = getattr(self, '_validation_min_year', None)
                    if min_y is not None and align_col in joined.columns:
                        joined = joined[pd.to_numeric(joined[align_col], errors='coerce') >= float(min_y)].copy()
                except Exception:
                    pass
                if warmup > 0 and len(joined) > warmup:
                    joined = joined.iloc[warmup:].copy()
                y_pred = pd.to_numeric(joined["__y_pred"], errors="coerce").to_numpy(dtype=float)
                y_true = pd.to_numeric(joined["__y_true"], errors="coerce").to_numpy(dtype=float)
            else:
                n = min(len(sim_df), len(raw_df))
                y_pred = pd.to_numeric(sim_df[agent].to_numpy()[:n], errors="coerce").astype(float)
                y_true = pd.to_numeric(raw_df[truth_col].to_numpy()[:n], errors="coerce").astype(float)
                if warmup > 0 and n > warmup:
                    y_pred = y_pred[warmup:]
                    y_true = y_true[warmup:]

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 2:
                continue

            yt0 = y_true[mask]
            yp0 = y_pred[mask]

            best = {
                "alignment_shift": 0,
                "r2": self._r2(yt0, yp0),
                "rmse": self._rmse(yt0, yp0),
                "mae": self._mae(yt0, yp0),
                "mse": self._mse(yt0, yp0),
                "n": int(len(yt0)),
            }

            if int(max_shift or 0) > 0:
                rmse_unshifted = best["rmse"]
                r2_unshifted = best["r2"]
                for s in range(-int(max_shift), int(max_shift) + 1):
                    yt_s, yp_s = self._apply_shift(yt0, yp0, s)
                    if yt_s.size < 2:
                        continue
                    rmse_s = self._rmse(yt_s, yp_s)
                    if rmse_s < float(best["rmse"]):
                        best = {
                            "alignment_shift": int(s),
                            "r2": self._r2(yt_s, yp_s),
                            "rmse": rmse_s,
                            "mae": self._mae(yt_s, yp_s),
                            "mse": self._mse(yt_s, yp_s),
                            "n": int(len(yt_s)),
                        }
                best["rmse_unshifted"] = float(rmse_unshifted)
                best["r2_unshifted"] = float(r2_unshifted)

            metrics[agent] = best

        return metrics


# =============================================================================
# SENSITIVITY / UNCERTAINTY (OPTIONAL)
# =============================================================================




# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================
class SimpleMemoryManager:
    """Lightweight memory monitor (psutil optional)."""

    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = int(max_memory_mb)
        self.memory_history: List[float] = []

    def check_memory(self) -> Dict[str, Any]:
        try:
            import psutil  # type: ignore
            process = psutil.Process()
            memory_mb = float(process.memory_info().rss) / 1024 / 1024
        except ImportError:
            import resource  # Unix fallback
            # ru_maxrss is KB on Linux, bytes on macOS; treat as KB for most servers
            memory_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024

        stats = {
            "current_mb": memory_mb,
            "max_mb": float(self.max_memory_mb),
            "usage_percent": (memory_mb / max(1.0, float(self.max_memory_mb))) * 100.0,
            "warning": memory_mb > float(self.max_memory_mb) * 0.8,
            "critical": memory_mb > float(self.max_memory_mb) * 0.95,
        }

        self.memory_history.append(memory_mb)
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)

        return stats


# =============================================================================
# MAIN RUNNER
# =============================================================================
class HybridSimulationRunner:
    """Main simulation runner orchestrating agents, solver, ML training, constraints, and validation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        np.random.seed(int(self.config.random_seed))
        self.rng = np.random.default_rng(int(self.config.random_seed))

        setup_logging(config)

        self.agents: Dict[str, Agent] = {}
        self.solver: Optional[SimpleSDSolver] = None
        self.memory_manager = SimpleMemoryManager(config.max_memory_mb)
        self.data_manager = DataManager(config)
        self.validator = ValidationSuite()

        self.delayed_selector = DelayedNodeSelector(self.config.delayed_evaluation)
        self.constraint_engine = ConstraintEngine(self.config.constraints)

        self.results: List[Dict[str, Any]] = []
        self.simulation_stats: Dict[str, Any] = {"start_time": None, "end_time": None, "total_steps": 0, "errors": []}
        self._ml_offline_trained = False

        self._initialize_system()

    def _initialize_system(self) -> None:
        for name, cfg in self.config.agents.items():
            try:
                self.agents[name] = Agent(cfg)
            except Exception as e:
                msg = f"Failed to create agent '{name}': {e}"
                logging.error(msg)
                self.simulation_stats["errors"].append(msg)

        self._normalize_agent_dependencies()

        self.solver = SimpleSDSolver(self.agents, delayed_selector=self.delayed_selector)
        logging.info("Solver initialized with %d agents.", len(self.agents))

    def _normalize_agent_dependencies(self) -> None:
        """Ensure expression dependencies refer only to agent names and exclude self-deps."""
        agent_names = set(self.agents.keys())
        for name, agent in self.agents.items():
            cfg = agent.config
            if cfg.type == "expression" and cfg.expression:
                deps_raw = list(cfg.dependencies or []) or infer_expression_dependencies(cfg.expression)
                deps = [d for d in deps_raw if d in agent_names and d != name]
                cfg.dependencies = deps
                # Sync behavior
                for b in agent.behaviors:
                    if isinstance(b, ExpressionBehavior):
                        b.dependencies = deps

    def _validate_step(self, env: Dict[str, float]) -> bool:
        """Validate bounds and conservation rules (soft-clips small numerical violations)."""
        for name, agent in self.agents.items():
            if agent.bounds is None:
                continue
            lo, hi = float(agent.bounds[0]), float(agent.bounds[1])
            v = float(env.get(name, 0.0) or 0.0)
            span = hi - lo
            eps = max(1e-9, 1e-8 * max(1.0, abs(span)))
            if v < lo - eps or v > hi + eps:
                logging.warning("Validation failed: %s=%s out of bounds %s", name, v, agent.bounds)
                return False
            if v < lo:
                env[name] = lo
            elif v > hi:
                env[name] = hi

        for rule in (self.config.conservation_rules or []):
            lhs = float(sum(float(env.get(var, 0.0) or 0.0) for var in rule.get("lhs", [])))
            rhs = float(sum(float(env.get(var, 0.0) or 0.0) for var in rule.get("rhs", [])))
            tol = float(rule.get("tolerance", 1e-6))
            if abs(lhs - rhs) > tol:
                logging.warning("Conservation rule failed: %s (lhs=%s rhs=%s)", rule, lhs, rhs)
                return False

        return True

    def _enforce_guardrails(self, env: Dict[str, float]) -> Dict[str, float]:
        """Hard guardrails to prevent non-physical blow-ups (domain-agnostic).

        Non-finite values for *_truth keys are not coerced because they are placeholders outside history
        and should remain NaN to avoid contaminating validation/debugging.
        """
        for k, v in list(env.items()):
            try:
                ks = str(k)
                if ks.endswith("_truth"):
                    continue
                if isinstance(v, (int, float, np.floating)) and not np.isfinite(float(v)):
                    env[k] = 0.0
            except Exception:
                # If anything goes wrong, keep previous value
                pass
        return env

    def _build_history_index(self, raw_df: pd.DataFrame) -> Tuple[Optional[str], Dict[int, Dict[str, Any]]]:
        """Build a time_key->row mapping using config.history_align_col, else return empty mapping.

        Notes:
          - Integer keys are retained for compatibility with the exogenous_forecast mapping type.
          - Align values are rounded to nearest int.
        """
        if raw_df is None or raw_df.empty:
            return self.config.history_align_col, {}

        align_col = self.config.history_align_col
        if not align_col or align_col not in raw_df.columns:
            return None, {}

        tmp = raw_df.copy()
        tmp["_key"] = pd.to_numeric(tmp[align_col], errors="coerce").round().astype("Int64")
        tmp = tmp.dropna(subset=["_key"]).sort_values("_key")
        mapping: Dict[int, Dict[str, Any]] = {}
        for _, r in tmp.iterrows():
            k = int(r["_key"])
            mapping.setdefault(k, r.to_dict())
        return align_col, mapping

    def _step_time_key(self, step: int, year_val: float) -> int:
        """Integer key used for history/forecast mapping when align_col exists."""
        return int(round(float(year_val)))

    def _commit_agent_states(self, env: Dict[str, float]) -> None:
        """Commit final env values into Agent.value (single commit per step)."""
        for name, agent in self.agents.items():
            if name in env:
                try:
                    agent.value = float(env[name])
                except Exception:
                    # If non-numeric in env, keep prior state
                    pass



    def run_simulation(self, data: Optional[pd.DataFrame] = None, *, skip_offline_train: bool = False) -> Dict[str, Any]:
        """Run full simulation (offline train + step loop + post-hoc validation)."""
        self.results = []
        self.simulation_stats = {"start_time": time.time(), "end_time": None, "total_steps": 0, "errors": []}
        self._ml_offline_trained = False

        # Load simulation dataframe (used for the actual simulation loop)
        raw_df = self.data_manager.load_data(data)

        # If a separate training dataframe was provided (e.g., reconstructed history),
        # use it to build lagged features and perform offline ML training/HPO with
        # consistent feature trajectories.
        train_df_cfg = getattr(self.config, "_training_df", None)
        if train_df_cfg is not None:
            try:
                # Build lagged features from training df
                _train_raw = self.data_manager.load_data(train_df_cfg)
                lagged_df = self.data_manager.build_lagged_features(_train_raw) if not _train_raw.empty else pd.DataFrame()
                # Reload simulation df for the simulation loop (keep lagged_df from training)
                raw_df = self.data_manager.load_data(data)
            except Exception:
                lagged_df = self.data_manager.build_lagged_features(raw_df) if not raw_df.empty else pd.DataFrame()
        else:
            lagged_df = self.data_manager.build_lagged_features(raw_df) if not raw_df.empty else pd.DataFrame()

        # Warm-up ML lag buffers (optional)
        self.data_manager.warmup_lags(self.agents, raw_df)

        # Determine which columns are allowed to be injected during history
        allowed_hist_cols: Set[str] = set(map(str, self.config.exogenous_columns or []))
        agent_names = set(self.agents.keys())
        raw_cols = set(map(str, raw_df.columns)) if not raw_df.empty else set()

        # Always allow inputs and their mapped columns
        for a_name, a in self.agents.items():
            if a.config.type == "input":
                allowed_hist_cols.add(str(a_name))
                if a.config.column:
                    allowed_hist_cols.add(str(a.config.column))

        # Allow raw columns referenced by ML deps when they are not agent names (parity support)
        for a in self.agents.values():
            if a.config.type != "ml":
                continue
            for dep in (a.config.dependencies or []):
                base = re.sub(r"(__lag|_lag)\d+$", "", str(dep))
                if base.endswith("_truth"):
                    base = base[:-6]
                if base in raw_cols and base not in agent_names:
                    allowed_hist_cols.add(base)

        # Offline ML training once
        # Default: assume dependencies are unchanged unless feature selection explicitly updates them.
        deps_updated = False
        if (not skip_offline_train) and (not self._ml_offline_trained) and (not lagged_df.empty):
            fs_cfg = self.config.feature_selection
            fs = FeatureSelector(
                fs_cfg.strategy,
                fs_cfg.top_k,
                self.config.random_seed,
                method=getattr(fs_cfg, "method", "mi"),
                min_k=getattr(fs_cfg, "min_k", 0),
            ) if fs_cfg.enabled else None

            for agent in self.agents.values():
                mlb = next((b for b in agent.behaviors if isinstance(b, MLBehavior)), None)
                if not mlb:
                    continue
                try:
                    X, y, feat_cols = self.data_manager.prepare_training_data(agent.config, feature_selector=fs if (fs_cfg.enabled and getattr(agent.config,'fs_mode','inherit')!='manual' and getattr(agent.config,'fs_enabled',None) is not False) else None)
                    mlb.fit_from_arrays(X, y, feat_cols)
                    # Keep runner train/serve parity: update agent dependencies to match selected features
                    try:
                        agent.config.dependencies = list(feat_cols)
                        deps_updated = True
                    except Exception:
                        pass
                    logging.info("Offline trained %s (%d samples, %d features).", agent.name, len(y), len(feat_cols))
                except Exception as e:
                    logging.warning("Offline training failed for %s: %s", agent.name, e)

            self._ml_offline_trained = True

        # Initialize environment with committed agent states
        env: Dict[str, float] = {name: float(agent.value) for name, agent in self.agents.items()}
        env = self.data_manager.ensure_columns_in_environment(env)
        # If feature selection updated dependencies, rebuild the dependency graph and solver.
        # Otherwise, the solver may use stale SCC/execution groups and produce incorrect results/order.
        if (not skip_offline_train) and deps_updated:
            try:
                self._normalize_agent_dependencies()
            except Exception:
                # Normalization is best-effort; failure should not crash the run.
                pass
            try:
                self.solver = SimpleSDSolver(self.agents, delayed_selector=self.delayed_selector)
                logging.info("Rebuilt solver after offline training to reflect updated dependencies.")
            except Exception as e:
                logging.warning("Could not rebuild solver after dependency updates: %s", e)



        # Required lag keys for runner-managed injection

        required_lags: Dict[str, int] = {}
        unlagged = set(NON_LAG_DEPS)

        for a in self.agents.values():
            cfg = a.config
            base_lag = int(cfg.lag or 0)
            for dep_raw in (cfg.dependencies or []):
                ds = str(dep_raw)
                # Auto-derived previous-delta feature: <base>__dy_prev (requires lag1 & lag2)
                m_dy = re.match(r"^(.*?)(?:_truth)?__dy_prev$", ds)
                if m_dy:
                    base = m_dy.group(1)
                    required_lags[base] = max(required_lags.get(base, 0), 2)
                    continue
                m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", ds)
                if m:
                    base, lk = m.group(1), int(m.group(2))
                else:
                    base = ds
                    lk = 0 if (base in unlagged) else base_lag
                if lk > 0:
                    required_lags[base] = max(required_lags.get(base, 0), lk)
        # If feature_selection.pool == all_lagged, inject lags for all agents to preserve train/serve parity
        try:
            fs_cfg = getattr(self.config, "feature_selection", None)
            pool = str(getattr(fs_cfg, "pool", "deps_only") or "deps_only").lower() if fs_cfg else "deps_only"
            lags_fs = list(getattr(fs_cfg, "lags", [1]) or [1]) if fs_cfg else [1]
            if pool == "all_lagged":
                max_lag_fs = max(int(x) for x in lags_fs if int(x) > 0) if lags_fs else 1
                for ag_name in (self.config.agents or {}).keys():
                    if str(ag_name) in NON_LAG_DEPS:
                        continue
                    required_lags[str(ag_name)] = max(required_lags.get(str(ag_name), 0), max_lag_fs)
        except Exception:
            pass

        lag_buffers: Dict[str, Deque[float]] = {b: deque(maxlen=required_lags[b] + 1) for b in required_lags}
        truth_buffers: Dict[str, Deque[float]] = {b: deque(maxlen=required_lags[b] + 1) for b in required_lags}

        prev_env_final = dict(env)

        # History mapping (align_col->row)
        align_col, key_to_row = self._build_history_index(raw_df)

        # Map raw input columns -> input agent names (for forecast locks)
        input_col_to_agent: Dict[str, str] = {}
        for a_name, a in self.agents.items():
            if a.config.type == "input" and a.config.column:
                input_col_to_agent[str(a.config.column)] = str(a_name)

        # Steps (end_year is exclusive)
        total_steps_float = (float(self.config.end_year) - float(self.config.start_year)) / float(self.config.timestep)
        total_steps = int(round(total_steps_float))
        if total_steps <= 0:
            raise ValueError("Invalid time range: end_year must be greater than start_year (exclusive end).")
        if abs(total_steps_float - total_steps) > 1e-6:
            logging.warning("Time range not divisible by timestep. Using rounded steps=%d.", total_steps)


        for step in range(total_steps):
            try:
                year_val = float(self.config.start_year) + float(step) * float(self.config.timestep)
                time_key = self._step_time_key(step, year_val)

                env["__step"] = int(step)
                env["time"] = float(step)
                env["time_idx"] = float(step)
                env["YEAR_GRG"] = float(year_val)

                # Determine whether the current step is within the historical record
                if align_col and key_to_row:
                    in_hist = (time_key in key_to_row)
                    row = key_to_row.get(time_key, {})
                else:
                    in_hist = (not raw_df.empty and step < len(raw_df))
                    row = raw_df.iloc[step].to_dict() if in_hist else {}

                # Make __in_history reflect observed-history years (not just "row exists in df")
                try:
                    hist_years = getattr(self.config, "hindcast_clamp_years", None)
                    if hist_years:
                        in_hist = int(year_val) in set(int(x) for x in hist_years)
                except Exception:
                    pass
                
                env["__in_history"] = bool(in_hist)
                # Optional: inject observed ML-target keys during historical reconstruction.
                # These keys support offline supervised fitting/diagnostics and do not clamp outputs unless the clamp flag is enabled.
                if in_hist and row and bool(getattr(self.config, 'inject_ml_truth_in_history', False)):
                    try:
                        for _an, _acfg in (self.config.agents or {}).items():
                            if getattr(_acfg, 'type', None) != 'ml':
                                continue
                            cand_cols = []
                            if getattr(_acfg, 'target_column', None):
                                cand_cols.append(str(_acfg.target_column))
                            cand_cols.append(str(_an))
                            if getattr(_acfg, 'column', None):
                                cand_cols.append(str(_acfg.column))
                            obs_val = None
                            for _c in cand_cols:
                                if _c in row and row.get(_c) is not None:
                                    obs_val = row.get(_c)
                                    break
                            if obs_val is not None:
                                env[f"{_an}_truth"] = obs_val
                    except Exception:
                        pass

                # Hindcast: clamp ML-agent outputs to observed values during history if configured
                clamp_enabled = bool(getattr(self.config, "hindcast_clamp_ml_to_observed", False))
                clamp_years = getattr(self.config, "hindcast_clamp_years", None)
                clamp_this_step = False
                if clamp_enabled and in_hist:
                    if clamp_years is None:
                        clamp_this_step = True
                    else:
                        try:
                            clamp_this_step = int(year_val) in set(int(x) for x in clamp_years)
                        except Exception:
                            clamp_this_step = True
                env["__clamp_ml_to_observed"] = bool(clamp_this_step)

                if clamp_this_step and row:
                    # Inject observed ("truth") values for ML agents so MLBehavior can clamp deterministically.
                    for _an, _acfg in self.config.agents.items():
                        if _acfg.type != "ml":
                            continue
                        # Candidate raw columns that might contain observed values
                        cand_cols = []
                        if _acfg.target_column:
                            cand_cols.append(str(_acfg.target_column))
                        cand_cols.append(_an)
                        if _acfg.column:
                            cand_cols.append(str(_acfg.column))
                        obs_val = None
                        for _c in cand_cols:
                            if _c in row and row.get(_c) is not None:
                                obs_val = row.get(_c)
                                break
                        if obs_val is not None:
                            env[f"{_an}_truth"] = obs_val
                            # bookkeeping for reporting
                            cmc = self.simulation_stats.setdefault("clamp_ml_counts", {})
                            cmc[_an] = int(cmc.get(_an, 0)) + 1
                            cmy = self.simulation_stats.setdefault("clamp_ml_years", {})
                            ys = cmy.get(_an, [])
                            if int(year_val) not in ys:
                                ys.append(int(year_val))
                                ys.sort()
                                cmy[_an] = ys


                # Runner-managed lag injection (both __lagK and _lagK aliases)
                if step == 0:
                    prev_env_final = dict(env)

                for base, max_lag in required_lags.items():
                    for k in range(1, int(max_lag) + 1):
                        if len(lag_buffers[base]) >= k:
                            val = float(list(lag_buffers[base])[-k])
                        else:
                            val = float(prev_env_final.get(base, env.get(base, 0.0)) or 0.0)
                        env[f"{base}__lag{k}"] = val
                        env[f"{base}_lag{k}"] = val

                        if len(truth_buffers[base]) >= k:
                            tval = float(list(truth_buffers[base])[-k])
                        else:
                            tval = float(env.get(f"{base}_truth", val) or val)
                        env[f"{base}_truth__lag{k}"] = tval
                        env[f"{base}_truth_lag{k}"] = tval

                # Runner-managed derived features: previous delta (lag1 - lag2)
                for base, max_lag in required_lags.items():
                    if int(max_lag) >= 2:
                        try:
                            env[f"{base}__dy_prev"] = float(env.get(f"{base}__lag1", 0.0) - env.get(f"{base}__lag2", 0.0))
                            env[f"{base}_truth__dy_prev"] = float(env.get(f"{base}_truth__lag1", env.get(f"{base}__lag1", 0.0)) - env.get(f"{base}_truth__lag2", env.get(f"{base}__lag2", 0.0)))
                        except Exception:
                            env[f"{base}__dy_prev"] = float(env.get(f"{base}__lag1", 0.0) - env.get(f"{base}__lag2", 0.0))
                            env[f"{base}_truth__dy_prev"] = float(env.get(f"{base}_truth__lag1", 0.0) - env.get(f"{base}_truth__lag2", 0.0))

                # Compatibility lag variables (lag1)
                for base in (self.config.lagged_state_vars or []):
                    env[f"{base}_lag1"] = float(prev_env_final.get(base, env.get(base, 0.0)) or 0.0)

                # Forecast locks (pre-history injection and pre-policy)
                if not in_hist:
                    series_map = self.config.exogenous_forecast or {}
                    for vname, series in series_map.items():
                        if isinstance(series, dict) and int(time_key) in series:
                            val = float(series[int(time_key)])
                            env[str(vname)] = val
                            if str(vname) in input_col_to_agent:
                                env[input_col_to_agent[str(vname)]] = val

                # Inject historical raw columns (only whitelisted)
                if in_hist and row:
                    reserved = {"time", "time_idx", "YEAR_GRG", "__step"}
                    for col, val in row.items():
                        if col in reserved:
                            continue
                        if str(col) not in allowed_hist_cols and not str(col).endswith("_truth"):
                            continue
                        if pd.notna(val):
                            # Keep numeric values as floats; retain non-numeric whitelisted metadata as-is
                            try:
                                env[str(col)] = float(val) if isinstance(val, (int, float, np.floating)) else val
                            except Exception:
                                env[str(col)] = val

                    # Optional assimilation of observed columns into agent internal states (non-ML)
                    if self.config.assimilate_history_observations:
                        exclude_suffixes = self.config.assimilate_history_exclude_suffixes or ("_truth",)
                        for a_name, a in self.agents.items():
                            if a.config.type == "ml":
                                continue
                            col = a.config.column or (a_name if a_name in row else None)
                            if not col:
                                continue
                            if any(str(col).endswith(s) for s in exclude_suffixes):
                                continue
                            if col in row and pd.notna(row[col]):
                                try:
                                    env[a_name] = float(row[col])
                                except Exception:
                                    pass

                    # Inputs: hard-set using their mapped columns
                    for a_name, a in self.agents.items():
                        if a.config.type != "input":
                            continue
                        col = a.config.column or a_name
                        if col in row and pd.notna(row[col]):
                            try:
                                v = float(row[col])
                                env[a_name] = v
                                env[str(col)] = v
                            except Exception:
                                pass
                else:
                    # Outside history: remove truth leakage (keep keys but set NaN)
                    for col in list(env.keys()):
                        if str(col).endswith("_truth"):
                            env[col] = np.nan
                    env["__truth_is_placeholder"] = True

                # Reapply forecast locks after history/exogenous state assembly
                if not in_hist:
                    series_map = self.config.exogenous_forecast or {}
                    for vname, series in series_map.items():
                        if isinstance(series, dict) and int(time_key) in series:
                            val = float(series[int(time_key)])
                            env[str(vname)] = val
                            if str(vname) in input_col_to_agent:
                                env[input_col_to_agent[str(vname)]] = val

                # Solve timestep (PURE solver)
                if not self.solver:
                    raise RuntimeError("Solver not initialized.")
                env_before = dict(env)
                env = self.solver.solve_timestep(env_before, float(self.config.timestep))

                # Optional reconstruction: overwrite ML targets from observed truth in history
                if in_hist and self.config.assimilate_history_targets and row:
                    for a_name, a in self.agents.items():
                        if a.config.type != "ml":
                            continue
                        tcol = a.config.target_column
                        fallback = f"{a_name}_truth"
                        try:
                            if tcol and tcol in row and pd.notna(row[tcol]):
                                env[a_name] = float(row[tcol])
                            elif fallback in row and pd.notna(row[fallback]):
                                env[a_name] = float(row[fallback])
                        except Exception:
                            pass

                # Post-step constraints (generalized)
                env = self.constraint_engine.apply(env, prev_env=prev_env_final, in_history=in_hist)
                env = self._enforce_guardrails(env)

                # Forecast locks post-solve (hard override)
                if not in_hist:
                    series_map = self.config.exogenous_forecast or {}
                    for vname, series in series_map.items():
                        if isinstance(series, dict) and int(time_key) in series:
                            val = float(series[int(time_key)])
                            env[str(vname)] = val
                            if str(vname) in input_col_to_agent:
                                env[input_col_to_agent[str(vname)]] = val

                # Validate, with fallback (safe because solver is pure)
                if not self._validate_step(env):
                    logging.warning("Validation failed -> retry with smaller dt.")
                    env = self.solver.solve_timestep(env_before, float(self.config.timestep) * 0.5)
                    env = self.constraint_engine.apply(env, prev_env=prev_env_final, in_history=in_hist)
                    env = self._enforce_guardrails(env)
                    if not self._validate_step(env):
                        logging.warning("Still failing -> applying global Newton correction.")
                        env = self.solver.newton_correction_global(env, float(self.config.timestep) * 0.5)
                        env = self.constraint_engine.apply(env, prev_env=prev_env_final, in_history=in_hist)
                        env = self._enforce_guardrails(env)
                        self._validate_step(env)  # final soft clip

                # Commit agent internal state ONCE per finalized step
                self._commit_agent_states(env)

                # Store output row (remove internal markers)
                out_row = dict(env)
                out_row["step"] = int(step)
                out_row.pop("__step", None)
                out_row.pop("__in_history", None)
                self.results.append(out_row)

                # Update buffers once per timestep (after env finalized)
                for base in required_lags:
                    lag_buffers[base].append(float(env.get(base, 0.0) or 0.0))

                if in_hist and row:
                    for base in required_lags:
                        tcol = f"{base}_truth"
                        if tcol in row and pd.notna(row[tcol]):
                            truth_buffers[base].append(float(row[tcol]))
                        elif base in row and pd.notna(row[base]):
                            truth_buffers[base].append(float(row[base]))
                        else:
                            truth_buffers[base].append(float(env.get(base, 0.0) or 0.0))

                prev_env_final = dict(env)

                mem = self.memory_manager.check_memory()
                if mem["critical"]:
                    logging.error("Critical memory usage reached. Stopping simulation.")
                    break

                if self.config.trace_to_log:
                    every = max(1, int(self.config.trace_every_n or 1))
                    if (step % every) == 0 and self.config.trace_agents:
                        payload = {n: out_row.get(n) for n in self.config.trace_agents}
                        logging.info("TRACE step=%d time=%s values=%s", step, out_row.get("YEAR_GRG", year_val), payload)

                self.simulation_stats["total_steps"] += 1

            except Exception as e:
                err = f"Error at step {step}: {e}"
                logging.error(err)
                self.simulation_stats["errors"].append(err)
                if self.results:
                    last = dict(self.results[-1])
                    last["step"] = int(step)
                    self.results.append(last)

        self.simulation_stats["end_time"] = time.time()
        execution_time = float(self.simulation_stats["end_time"] - float(self.simulation_stats["start_time"]))


        # Post-hoc validation
        validation_metrics: Dict[str, Any] = {}
        if self.config.validation_targets and not raw_df.empty:
            warmup = 0
            if self.config.validation_drop_warmup:
                ml_lags = []
                for ac in self.config.agents.values():
                    if ac.type != "ml":
                        continue
                    base_lag = int(ac.lag or 0)
                    dep_lag = 0
                    for d in (ac.dependencies or []):
                        m = re.match(r"^(.*?)(?:__lag|_lag)(\d+)$", str(d))
                        if m:
                            dep_lag = max(dep_lag, int(m.group(2)))
                    ml_lags.append(max(base_lag, dep_lag))
                warmup = max(ml_lags) if ml_lags else 0

            try:
                # Pass optional validation_min_year to validator (kept as attribute for backward compatibility)
                try:
                    setattr(self.validator, '_validation_min_year', getattr(self.config, 'validation_min_year', None))
                except Exception:
                    pass
                validation_metrics = self.validator.compute(
                    self.results,
                    raw_df,
                    self.config.validation_targets,
                    align_col=self.config.validation_align_col or "YEAR_GRG",
                    max_shift=int(self.config.validation_max_shift or 0),
                    warmup=int(warmup or 0),
                )
            except Exception as e:
                logging.warning("Validation failed: %s", e)

        out: Dict[str, Any] = {
            "config": self.config,
            "results": self.results,
            "execution_time": execution_time,
            "stats": self.simulation_stats,
            "solver_stats": self.solver.solver_stats if self.solver else {},
            "memory_stats": self.memory_manager.check_memory(),
            "success": len(self.simulation_stats["errors"]) == 0,
        }
        try:
            out["timeseries"] = pd.DataFrame(self.results)
        except Exception:
            out["timeseries"] = self.results

        if validation_metrics:
            out["validation"] = validation_metrics

        return out


# =============================================================================
# REPORTING
# =============================================================================
class SimpleReporter:
    """Write JSON/CSV reports to disk (no plotting; keep core lightweight)."""

    def __init__(self):
        self.logger = logging.getLogger("reporter")

    def generate_report(self, simulation_results: Dict[str, Any], output_dir: Union[str, Path]) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reports: List[str] = []

        summary = self._create_executive_summary(simulation_results)
        (output_dir / "executive_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        reports.append("executive_summary.json")

        if simulation_results.get("results"):
            df_full = pd.DataFrame(simulation_results["results"])
            # Keep the full environment dump for debugging (raw columns, lags, truth, etc.)
            df_full.to_csv(output_dir / "simulation_data_full.csv", index=False)
            reports.append("simulation_data_full.csv")

            # But the main CSV should contain ONLY simulation outputs (agents) + time index columns.
            cfg: SimulationConfig = simulation_results["config"]
            agent_cols = set(getattr(cfg, "agents", {}).keys())
            time_cols = [str(getattr(cfg, "validation_align_col", "YEAR_GRG") or "YEAR_GRG"), "YEAR_GRG", "time", "time_idx", "__step", "step"]
            keep = []
            for c in time_cols:
                if c in df_full.columns and c not in keep:
                    keep.append(c)
            for c in df_full.columns:
                if c in agent_cols and c not in keep:
                    keep.append(c)

            df_sim = _filter_timeseries_for_export(df_full, cfg)
            df_sim.to_csv(output_dir / "simulation_data.csv", index=False)
            reports.append("simulation_data.csv")

        perf = self._create_performance_report(simulation_results)
        (output_dir / "performance_metrics.json").write_text(json.dumps(perf, indent=2, default=str), encoding="utf-8")
        reports.append("performance_metrics.json")

        diag = self._create_system_diagnostics(simulation_results)
        (output_dir / "system_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
        reports.append("system_diagnostics.json")


        if simulation_results.get("validation"):
            (output_dir / "validation_metrics.json").write_text(
                json.dumps(simulation_results["validation"], indent=2, default=str), encoding="utf-8"
            )
            reports.append("validation_metrics.json")

        text_summary = self._create_text_summary(simulation_results)
        (output_dir / "summary.txt").write_text(text_summary, encoding="utf-8")
        reports.append("summary.txt")

        self.logger.info("Generated %d reports in %s", len(reports), str(output_dir))
        return reports

    def _create_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        cfg: SimulationConfig = results["config"]
        solver_stats = results.get("solver_stats", {})
        mem = results.get("memory_stats", {})

        agent_types: Dict[str, int] = {}
        for a in cfg.agents.values():
            agent_types[a.type] = agent_types.get(a.type, 0) + 1

        return {
            "simulation_name": cfg.name,
            "execution_time_seconds": float(results.get("execution_time", 0.0)),
            "total_agents": len(cfg.agents),
            "timesteps_completed": int(results.get("stats", {}).get("total_steps", 0)),
            "success": bool(results.get("success", False)),
            "errors_count": len(results.get("stats", {}).get("errors", []) or []),
            "memory_usage_mb": float(mem.get("current_mb", 0.0) or 0.0),
            "agent_types": agent_types,
            "performance_summary": {
                "avg_step_time_ms": float(solver_stats.get("avg_step_time", 0.0) or 0.0) * 1000.0,
                "steps_per_second": 1.0 / max(float(solver_stats.get("avg_step_time", 1e-6) or 1e-6), 1e-6),
                "circular_dependencies": solver_stats.get("circular_dependencies", []),
            },
        }

    def _create_performance_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        cfg: SimulationConfig = results["config"]
        solver_stats = results.get("solver_stats", {})
        mem = results.get("memory_stats", {})
        avg_step = float(solver_stats.get("avg_step_time", 0.0) or 0.0)

        return {
            "execution_metrics": {
                "total_time_seconds": float(results.get("execution_time", 0.0)),
                "steps_completed": int(results.get("stats", {}).get("total_steps", 0)),
                "avg_step_time_ms": avg_step * 1000.0,
                "throughput_steps_per_second": 1.0 / max(avg_step, 1e-6),
            },
            "memory_metrics": mem,
            "scalability_assessment": {
                "current_agents": len(cfg.agents),
                "memory_per_agent_mb": float(mem.get("current_mb", 0.0) or 0.0) / max(1, len(cfg.agents)),
            },
        }

    def _create_system_diagnostics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        cfg: SimulationConfig = results["config"]
        solver_stats = results.get("solver_stats", {})
        return {
            "configuration": {
                "total_agents": len(cfg.agents),
                "circular_dependencies": solver_stats.get("circular_dependencies", []),
                "circular_dependency_groups": solver_stats.get("circular_dependency_groups", []),
            },
            "runtime": {
                "errors": results.get("stats", {}).get("errors", []),
                "success": bool(results.get("success", False)),
            },
        }

    def _create_text_summary(self, results: Dict[str, Any]) -> str:
        cfg: SimulationConfig = results["config"]
        lines = [
            "HYBRID SIMULATION REPORT",
            "=" * 60,
            f"Simulation: {cfg.name}",
            f"Execution Time: {float(results.get('execution_time', 0.0)):.2f} seconds",
            f"Total Agents: {len(cfg.agents)}",
            f"Steps Completed: {int(results.get('stats', {}).get('total_steps', 0))}",
            f"Success: {'Yes' if results.get('success') else 'No'}",
            "",
        ]

        if results.get("validation"):
            lines.append("VALIDATION:")
            for agent, m in results["validation"].items():
                lines.append(
                    f"  {agent}: SC3={m['r2']:.3f}, RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, n={m['n']}"
                )

        errs = results.get("stats", {}).get("errors", []) or []
        if errs:
            lines.extend(["", "ERRORS:"])
            for e in errs[:10]:
                lines.append(f"  - {e}")
            if len(errs) > 10:
                lines.append(f"  ... and {len(errs) - 10} more")

        return "\n".join(lines)


# =============================================================================
# SCENARIOS
# =============================================================================



def _filter_timeseries_for_export(df: pd.DataFrame, cfg: "SimulationConfig") -> pd.DataFrame:
    """Return a compact dataframe containing only time columns + configured agent outputs (+ *_truth columns).

    This keeps exported CSVs scenario-proof and avoids mixing raw input columns into model outputs.
    """
    if df is None or df.empty:
        return df
    agent_cols = list(getattr(cfg, "agents", {}).keys()) if cfg is not None else []
    # keep truth columns too (useful for metrics)
    truth_cols = [c for c in df.columns if c.endswith("_truth")]
    time_hint = str(getattr(cfg, "history_align_col", None) or getattr(cfg, "validation_align_col", None) or "YEAR_GRG")
    time_cols = [time_hint, "YEAR_GRG", "time", "time_idx", "__step", "step"]
    keep = []
    for c in time_cols:
        if c in df.columns and c not in keep:
            keep.append(c)
    for c in agent_cols:
        if c in df.columns and c not in keep:
            keep.append(c)
    for c in truth_cols:
        if c in df.columns and c not in keep:
            keep.append(c)
    return df[keep].copy() if keep else df.copy()


# =============================================================================
# YAML LOADER + CONFIG VALIDATION
# =============================================================================
def load_config_from_yaml(yaml_file: Union[str, Path]) -> SimulationConfig:
    """Load SimulationConfig from a YAML file."""
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

    yaml_file = Path(yaml_file)
    cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}


    
    # Backward-compatible aliases for cleanliness:
    # - allow top-level `name:` to populate simulation.name
    # - allow top-level assimilate_history_* flags to be read even if user keeps them at top-level
    try:
        if isinstance(cfg, dict):
            if "name" in cfg:
                sim_blk = cfg.get("simulation", {}) or {}
                if isinstance(sim_blk, dict) and not sim_blk.get("name"):
                    sim_blk["name"] = cfg.get("name")
                    cfg["simulation"] = sim_blk
    except Exception:
        pass
# Optional: warn about unknown top-level keys to keep configs clean/transparent.
    try:
        cv = cfg.get("config_validation", {}) or {}
        warn_unknown = bool(cv.get("warn_unknown_keys", True))
        if warn_unknown:
            known_top = {
                "simulation",
                "runtime",
                "resources",
                "constraints",
                "conservation_rules",
                "delayed_evaluation",
                "agents",
                "hindcast",
                "training",
                "history_align_col",
                "hyperparams",
                "feature_selection",
                "config_validation",
                "time",
                "ml_defaults",
                "ml_training_end_year",
                "history_truth_mode",
                "ml",
                "validation",
                "exogenous_columns",
                "exogenous_forecast",
                "data_aliases",
                "name",
                "assimilate_history_targets",
                "assimilate_history_observations",
                "runner"
            }
            unknown = [k for k in cfg.keys() if k not in known_top]
            if unknown:
                logging.warning("Unknown top-level config keys (safe to remove if unintended): %s", unknown)
    except Exception:
        pass



    # -------------------------------------------------------------------------
    # Friendly/clean config aliases (optional)
    #   - time: {start_year, projection_start_year, end_year, timestep}
    #   - ml_defaults: shared defaults applied to all agents with type: ml
    #   - history_truth_mode: none | features_only | features_and_labels | clamp_outputs
    #   - ml_training_end_year: int | "auto" (defaults to projection_start_year-1 if available)
    # -------------------------------------------------------------------------
    time_block = cfg.get("time", {}) or {}
    # Allow time.* to override simulation.* for readability
    sim = cfg.get("simulation", {}) or {}
    if time_block:
        sim = {**sim, **time_block}


    config = SimulationConfig(
        name=str(sim.get("name", "Unnamed Simulation")),
        data_file=str(sim.get("data_file", "")),
        start_year=int(sim.get("start_year", 2020)),
        end_year=int(sim.get("end_year", int(sim.get("start_year", 2020)) + int(sim.get("years", 10)))),
        timestep=float(sim.get("time_step", sim.get("timestep", 1.0))),
    )

    # Optional projection boundary (for readability and as default ML cutoff)
    try:
        psy = sim.get("projection_start_year", sim.get("projection_year", None))
        config.projection_start_year = None if psy in (None, "", "none") else int(psy)
    except Exception:
        config.projection_start_year = None


    # History alignment
    config.history_align_col = cfg.get("history_align_col", config.history_align_col)

    runtime = cfg.get("runtime", {})
    config.random_seed = int(runtime.get("random_seed", 42))
    # Backward-compatible top-level flags for history assimilation (optional)
    try:
        if "assimilate_history_targets" in cfg:
            config.assimilate_history_targets = bool(cfg.get("assimilate_history_targets"))
        if "assimilate_history_observations" in cfg:
            config.assimilate_history_observations = bool(cfg.get("assimilate_history_observations"))
    except Exception:
        pass




    # Optional hindcast controls (used by paper/pipeline runner) (accept both root.hindcast and simulation.hindcast)
    hind_root = cfg.get("hindcast", {}) or {}
    hind_sim  = sim.get("hindcast", {}) or {}
    hind = {**hind_sim, **hind_root}  # root wins if both set
    
    try:
        config.hindcast_clamp_ml_to_observed = bool(hind.get("clamp_ml_to_observed", False))
    except Exception:
        config.hindcast_clamp_ml_to_observed = False
    
    years = hind.get("clamp_years", None)
    if years is not None:
        try:
            config.hindcast_clamp_years = [int(x) for x in years]
        except Exception:
            config.hindcast_clamp_years = None


    # Training-time feature resolution controls
    # If set, offline ML training uses only rows with YEAR_GRG <= offline_train_end_year.
    offline_train_end_year: Optional[int] = None
    # If True, inject '<ml_agent>_truth' during historical reconstruction for offline supervised fitting/diagnostics.
    # This does NOT clamp outputs unless hindcast_clamp_ml_to_observed is enabled.
    inject_ml_truth_in_history: bool = False
    training_root = cfg.get("training", {}) or {}
    training_sim  = sim.get("training", {}) or {}
    training = {**training_sim, **training_root}  # root wins if both set
    try:
        config.prefer_truth_for_endogenous_deps = bool(training.get("prefer_truth_for_endogenous_deps", True))
    except Exception:
        config.prefer_truth_for_endogenous_deps = True

    # Optional: offline training cutoff (holdout)
    # Optional: inject ML target keys during historical reconstruction, without output clamping
    try:
        imt = training.get('inject_ml_truth_in_history', None)
        config.inject_ml_truth_in_history = bool(imt) if imt is not None else False
    except Exception:
        config.inject_ml_truth_in_history = False

    try:
        otey = training.get('offline_train_end_year', None)
        config.offline_train_end_year = (None if otey in (None, '', 'none') else int(otey))
    except Exception:
        config.offline_train_end_year = None

    
    # ---------------------------------------------------------------------
    # Simplified, user-friendly training controls (optional)
    # ---------------------------------------------------------------------
    # ml_training_end_year: int | "auto" -> defaults to projection_start_year-1 (if provided)
    mtey = cfg.get("ml_training_end_year", None)
    if mtey is not None:
        try:
            if str(mtey).strip().lower() in ("auto", "default"):
                if config.projection_start_year is not None:
                    config.offline_train_end_year = int(config.projection_start_year) - 1
            elif str(mtey).strip().lower() in ("none", ""):
                config.offline_train_end_year = None
            else:
                config.offline_train_end_year = int(mtey)
        except Exception:
            pass
    else:
        # If user did not set offline_train_end_year explicitly, use projection_start_year-1 as a sensible default.
        if config.offline_train_end_year is None and config.projection_start_year is not None:
            config.offline_train_end_year = int(config.projection_start_year) - 1

    # history_truth_mode controls how truth is used in history for ML:
    #   none              -> no truth injection; features rely on simulated env
    #   features_only     -> truth may be used for feature resolution (prefer_truth_for_endogenous_deps)
    #   features_and_labels -> also inject '<agent>_truth' for supervision (no output clamp)
    #   clamp_outputs     -> additionally clamp ML outputs to observed in history (optionally by clamp_years)
    htm = cfg.get("history_truth_mode", None)
    if htm is not None:
        mode = str(htm).strip().lower()
        if mode in ("none", "off", "false"):
            config.prefer_truth_for_endogenous_deps = False
            config.inject_ml_truth_in_history = False
            config.hindcast_clamp_ml_to_observed = False
        elif mode in ("features_only", "features"):
            config.prefer_truth_for_endogenous_deps = True
            config.inject_ml_truth_in_history = False
            config.hindcast_clamp_ml_to_observed = False
        elif mode in ("features_and_labels", "labels", "supervised"):
            config.prefer_truth_for_endogenous_deps = True
            config.inject_ml_truth_in_history = True
            config.hindcast_clamp_ml_to_observed = False
        elif mode in ("clamp_outputs", "clamp", "clamped"):
            config.prefer_truth_for_endogenous_deps = True
            config.inject_ml_truth_in_history = True
            config.hindcast_clamp_ml_to_observed = True
        else:
            logging.warning("Unknown history_truth_mode=%r (ignored).", htm)

# Optional runner output control
    rrun = sim.get("runner", {}) or {}
    frd = rrun.get("force_run_dir", None)
    if frd:
        config.force_run_dir = str(frd)
    resources = cfg.get("resources", {})
    config.max_memory_mb = int(resources.get("max_memory_mb", 4096))
    config.enable_logging = bool(resources.get("enable_logging", True))
    config.log_level = str(resources.get("log_level", "INFO"))



    config.conservation_rules = list(cfg.get("conservation_rules") or [])

    # Delayed evaluation
    de = cfg.get("delayed_evaluation", {}) or {}
    config.delayed_evaluation = DelayedEvaluationConfig(
        enabled=bool(de.get("enabled", False)),
        nodes=list(de.get("nodes") or []),
        prefixes=list(de.get("prefixes") or []),
        regex=list(de.get("regex") or []),
    )

    # Constraints
    cons = cfg.get("constraints", {}) or {}
    groups_cfg: List[ConstraintGroupConfig] = []
    for g in (cons.get("groups", []) or []):
        try:
            groups_cfg.append(
                ConstraintGroupConfig(
                    name=str(g.get("name", "group")),
                    members=list(g.get("members") or []),
                    enabled=bool(g.get("enabled", True)),
                    target_sum=float(g.get("target_sum", 1.0)),
                    smoothing_alpha=float(g.get("smoothing_alpha", 0.0)),
                    clip_min=g.get("clip_min", 0.0),
                    clip_max=g.get("clip_max", 1.0),
                    update_rate=float(g.get("update_rate", 1.0)),
                    apply_in_history=bool(g.get("apply_in_history", True)),
                    apply_in_forecast=bool(g.get("apply_in_forecast", True)),
                )
            )
        except Exception:
            continue
    config.constraints = ConstraintEngineConfig(
        enabled=bool(cons.get("enabled", False)),
        groups=groups_cfg,
    )


    hpo = cfg.get("hyperparams", {}) or {}
    config.hyperparams = HyperparamConfig(
        enabled=bool(hpo.get("enabled", False)),
        exclude_test_years=[int(x) for x in (hpo.get("exclude_test_years") or []) if x is not None],
    )

    val = cfg.get("validation", {})
    config.validation_targets = dict(val.get("targets", {}))
    try:
        vmin = val.get('min_year', None)
        config.validation_min_year = (None if vmin in (None,'','none') else int(vmin))
    except Exception:
        config.validation_min_year = None
    config.validation_align_col = val.get("align_col", config.validation_align_col)
    config.validation_max_shift = int(val.get("max_shift", config.validation_max_shift))
    config.validation_drop_warmup = bool(val.get("drop_warmup", config.validation_drop_warmup))

    fs = cfg.get("feature_selection", {})
    config.feature_selection = FeatureSelectionConfig(
        enabled=bool(fs.get("enabled", True)),
        strategy=str(fs.get("strategy", "hybrid") or "hybrid").strip().lower(),
        top_k=int(fs.get("top_k", 10)),
        min_k=int(fs.get("min_k", 0)),
        method=str(fs.get("method", "mi") or "mi").strip().lower(),
        pool=str(fs.get("pool", "deps_only") or "deps_only").strip().lower(),
        lags=list(fs.get("lags", [1]) or [1]),
        mandatory_features=list(fs.get("mandatory_features") or []),
    )

    config.exogenous_columns = list(cfg.get("exogenous_columns") or [])
    config.exogenous_forecast = dict(cfg.get("exogenous_forecast", {}))
    config.data_aliases = dict(cfg.get("data_aliases", {}))

    # Agents
    agents_block = cfg.get("agents", {}) or {}

    # Shared defaults for all ML agents (optional, to keep configs compact).
    # 'ml' is accepted as an alias of 'ml_defaults'.
    ml_defaults = cfg.get("ml_defaults", None)
    if ml_defaults is None:
        ml_defaults = cfg.get("ml", {}) or {}
    else:
        ml_defaults = ml_defaults or {}
    for name, a in agents_block.items():
        a_use = dict(a or {})
        if str(a_use.get('type','input')).strip().lower() == 'ml':
            for _k, _v in (ml_defaults or {}).items():
                if _k not in a_use:
                    a_use[_k] = _v
        agent_cfg = AgentConfig(
            name=str(name),
            type=str(a_use.get("type", "input")),
            description=str(a_use.get("description", "")),
            units=str(a_use.get("units", "")),
            category=str(a_use.get("category", "")),
            subcategory=str(a_use.get("subcategory", "")),
            region=str(a_use.get("region", "")),
            subregion=str(a_use.get("subregion", "")),
            usage=str(a_use.get("usage", "")),
            initial_value=float(a_use.get("initial_value", 0.0)),
            bounds=tuple(a["bounds"]) if ("bounds" in a and a["bounds"] is not None) else None,
            column=a_use.get("column"),
            inflows=list(a_use.get("inflows") or []),
            outflows=list(a_use.get("outflows") or []),
            expression=a_use.get("expression"),
            dependencies=list(a_use.get("dependencies") or []),
            mandatory_features=list(a_use.get("mandatory_features") or []),
            model_type=a_use.get("model_type"),
            hyperparameters=dict(a_use.get("hyperparameters", {})),
            target_column=a_use.get("target_column"),
            # per-agent FS overrides
            fs_mode=str(a_use.get("fs_mode", a_use.get("feature_selection_mode", "inherit")) or "inherit").strip().lower(),
            fs_enabled=(None if (a_use.get("fs_enabled", None) is None) else bool(a_use.get("fs_enabled"))),
            fs_pool=(None if a_use.get("fs_pool", None) is None else str(a_use.get("fs_pool")).strip().lower()),
            fs_top_k=(None if a_use.get("fs_top_k", None) is None else int(a_use.get("fs_top_k"))),
            lag=int(a_use.get("lag", 0)),
        )
        config.agents[str(name)] = agent_cfg

    return config


def validate_config(config: SimulationConfig) -> None:
    """Validate configuration and raise ValueError on failures."""
    errors: List[str] = []

    if config.start_year >= config.end_year:
        errors.append("end_year must be greater than start_year (end_year is exclusive).")
    if config.timestep <= 0:
        errors.append("timestep must be positive.")

    supported_ml = {
        "random_forest",
        "xgboost",
        "gradient_boosting",
        "linear",
        "ridge",
        "neural_network",
        "huber",
        "poisson",
        "tweedie",
    }
    allowed_tt = {"", "log", "log1p", "logit"}

    for name, a in config.agents.items():
        if a.type not in {"input", "stock", "expression", "ml"}:
            errors.append(f"Agent '{name}' has invalid type '{a.type}'.")

        if a.bounds and float(a.bounds[0]) > float(a.bounds[1]):
            errors.append(f"Agent '{name}' has invalid bounds {a.bounds}.")

        if a.type == "ml":
            if not a.model_type:
                errors.append(f"Agent '{name}' is ML but model_type is missing.")
            else:
                mt = str(a.model_type).strip().lower()
                if mt not in supported_ml:
                    errors.append(
                        f"Agent '{name}' has unsupported model_type='{a.model_type}'. Supported: {sorted(supported_ml)}"
                    )

            if not a.dependencies:
                errors.append(f"Agent '{name}' is ML but dependencies are empty.")
            if not a.target_column:
                errors.append(f"Agent '{name}' is ML but target_column is missing.")
            if int(a.lag or 0) < 0:
                errors.append(f"Agent '{name}' has negative lag.")

            # per-agent FS overrides
            fsm = str(getattr(a, 'fs_mode', 'inherit') or 'inherit').strip().lower()
            if fsm not in {'inherit','manual','auto','hybrid'}:
                errors.append(f"Agent '{name}' has invalid fs_mode='{getattr(a,'fs_mode',None)}'. Allowed: inherit/manual/auto/hybrid")
            fsp = getattr(a, 'fs_pool', None)
            if fsp is not None:
                fsp_s = str(fsp).strip().lower()
                if fsp_s not in {'deps_only','all_lagged'}:
                    errors.append(f"Agent '{name}' has invalid fs_pool='{fsp}'. Allowed: deps_only/all_lagged")
            fst = getattr(a, 'fs_top_k', None)
            if fst is not None and int(fst) <= 0:
                errors.append(f"Agent '{name}' has invalid fs_top_k={fst}. Must be positive.")

            tt = (a.hyperparameters or {}).get("target_transform", None)
            tt_s = str(tt).strip().lower() if tt not in (None, "", "none") else ""
            if tt_s not in allowed_tt:
                errors.append(f"Agent '{name}' has invalid target_transform='{tt}'. Allowed: {sorted(allowed_tt)}")
            if str(a.model_type or "").strip().lower() in {"poisson", "tweedie"} and tt_s:
                errors.append(f"Agent '{name}': target_transform='{tt}' is not allowed for model_type='{a.model_type}'.")


    for agent, col in (config.validation_targets or {}).items():
        if agent not in config.agents:
            errors.append(f"Validation target agent '{agent}' not found in agents.")
        if not isinstance(col, str) or not col:
            errors.append(f"Validation mapping for '{agent}' must be a non-empty string column name.")


    fs = config.feature_selection
    if fs.enabled:
        strat = str(fs.strategy or "hybrid").strip().lower()
        if strat not in {"filter", "embedded", "hybrid"}:
            errors.append("feature_selection.strategy must be one of {'filter','embedded','hybrid'}.")
        if int(fs.top_k) <= 0:
            errors.append("feature_selection.top_k must be positive.")
        meth = str(getattr(fs, 'method', 'mi') or 'mi').strip().lower()
        if meth not in {"mi", "corr"}:
            errors.append("feature_selection.method must be one of {'mi','corr'}.")
        pool = str(getattr(fs, 'pool', 'deps_only') or 'deps_only').strip().lower()
        if pool not in {"deps_only", "all_lagged"}:
            errors.append("feature_selection.pool must be one of {'deps_only','all_lagged'}.")
        try:
            lags = list(getattr(fs, 'lags', [1]) or [1])
            lags_int = [int(x) for x in lags]
            if not lags_int or any(x <= 0 for x in lags_int):
                errors.append("feature_selection.lags must be a list of positive integers.")
        except Exception:
            errors.append("feature_selection.lags must be a list of positive integers.")

    # Constraints validation (generic)
    if config.constraints and config.constraints.enabled:
        for g in (config.constraints.groups or []):
            if not g.members:
                errors.append(f"Constraint group '{g.name}' has empty members.")
            if g.target_sum <= 0 or not np.isfinite(float(g.target_sum)):
                errors.append(f"Constraint group '{g.name}' has invalid target_sum={g.target_sum}.")
            if g.update_rate is not None:
                ur = float(g.update_rate)
                if not np.isfinite(ur) or ur < 0.0 or ur > 1.0:
                    errors.append(f"Constraint group '{g.name}' update_rate must be in [0,1].")

    if errors:
        raise ValueError("Config validation failed:\n- " + "\n- ".join(errors))



def run_simulation_from_config(
    config_file: Union[str, Path],
    *,
    data_file: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run a simulation from YAML config.

    Enhancements (maintainable behavior):
      - Builds a reconstructed *training* dataframe for offline ML training,
        so expression-derived features are available in training even if they don't
        exist as raw CSV columns.
      - Trains ML agents on the reconstructed history up to `offline_train_end_year`
        (typically projection_start_year-1), then runs the simulation on the original
        dataframe (no leakage into projection).
    """
    config_file = Path(config_file)
    if config_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Only YAML configuration files are supported.")

    config = load_config_from_yaml(config_file)
    if data_file:
        config.data_file = str(data_file)

    validate_config(config)

    # Load base dataset
    df: Optional[pd.DataFrame] = None
    if config.data_file:
        try:
            df = pd.read_csv(config.data_file)
            logging.info("Loaded dataset: %d rows from %s", len(df), config.data_file)
        except Exception as e:
            logging.warning("Could not load dataset %s: %s", config.data_file, e)
            df = None

    # Nothing to do without data
    if df is None or df.empty:
        runner = HybridSimulationRunner(config)
        return runner.run_simulation(df)

    # ------------------------------------------------------------------
    # Reconstruct training dataframe (history-only) so expression features
    # are available to ML training.
    # ------------------------------------------------------------------
    # Determine year column and observed range
    year_col = str(config.history_align_col or "YEAR_GRG")
    if year_col not in df.columns and "YEAR_GRG" in df.columns:
        year_col = "YEAR_GRG"

    df_train = df
    try:
        years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        if not years.empty:
            y_min, y_max = int(years.min()), int(years.max())

            # History-only config for reconstruction (clamp ML outputs to observed)
            cfg_h = copy.deepcopy(config)
            cfg_h.start_year = int(min(cfg_h.start_year, y_min))
            cfg_h.end_year = int(y_max + 1)  # exclusive
            cfg_h.hindcast_clamp_ml_to_observed = True
            cfg_h.hindcast_clamp_years = list(range(y_min, y_max + 1))
            cfg_h.inject_ml_truth_in_history = True
            cfg_h.prefer_truth_for_endogenous_deps = True


            runner_h = HybridSimulationRunner(cfg_h)
            res_h = runner_h.run_simulation(df, skip_offline_train=True)
            ts = res_h.get("timeseries", None)
            ts_df = pd.DataFrame(ts) if isinstance(ts, list) else (ts.copy() if isinstance(ts, pd.DataFrame) else pd.DataFrame())
            ts_df = ts_df.loc[:, ~ts_df.columns.duplicated()].copy()

            if not ts_df.empty and year_col in ts_df.columns:
                # Overwrite agent columns in history with reconstructed trajectories
                df_train = df.copy()
                ts_df = ts_df.copy()
                ts_df[year_col] = pd.to_numeric(ts_df[year_col], errors="coerce").astype("Int64")
                df_train[year_col] = pd.to_numeric(df_train[year_col], errors="coerce").astype("Int64")

                # Select only agent columns (exclude *_truth)
                agent_cols = [a.name for a in config.agents.values() if a.name in ts_df.columns]
                agent_cols = [c for c in agent_cols if not str(c).endswith("_truth")]
                agent_cols = [c for c in agent_cols if c != year_col]

                merged = df_train[[year_col]].merge(
                    ts_df[[year_col] + agent_cols],
                    on=year_col,
                    how="left",
                    suffixes=("", "__recon"),
                )
                if agent_cols:
                    # Overwrite all at once to avoid pandas DataFrame fragmentation warnings/perf hits
                    df_train.loc[:, agent_cols] = merged[agent_cols].to_numpy()

            # Save reconstructed df to output_dir if requested
            if output_dir is not None:
                try:
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _filter_timeseries_for_export(df_train, config).to_csv(out_dir / "reconstructed_training_data.csv", index=False)
                except Exception:
                    pass
    except Exception as e:
        logging.warning("Training reconstruction failed; falling back to raw training data: %s", e)
        df_train = df

    # Attach training df to config for the main runner (offline training uses this)
    try:
        setattr(config, "_training_df", df_train)
    except Exception:
        pass

    # Main run
    runner = HybridSimulationRunner(config)

    # The repository configuration uses fixed per-agent hyperparameters; runtime HPO is not part of the execution path.

    result = runner.run_simulation(df)

    # Write outputs if requested
    if output_dir is not None:
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save main timeseries
            ts = result.get("timeseries", None)
            if ts is not None:
                _filter_timeseries_for_export(pd.DataFrame(ts), config).to_csv(out_dir / "simulation_data.csv", index=False)
        except Exception:
            pass

    return result

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file safely and return a dict (empty dict if file is empty)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def _make_standard_output_paths(run_root: Union[str, Path]) -> Dict[str, Path]:
    """Create the standard output directory structure for a run/stage and return paths."""
    rr = Path(run_root)
    rr.mkdir(parents=True, exist_ok=True)

    meta = rr / "00_meta"
    hpo = rr / "10_hpo"
    hpo_trials = hpo / "trials"
    simulation = rr / "20_simulation"
    validation = rr / "30_validation"
    reports = rr / "40_reports"

    for d in (meta, hpo, hpo_trials, simulation, validation, reports):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": rr,
        "meta": meta,
        "hpo": hpo,
        "hpo_trials": hpo_trials,
        "simulation": simulation,
        "validation": validation,
        "reports": reports,
    }

def to_jsonable(obj: Any) -> Any:
    """Convert common python/numpy/pandas/custom objects into JSON-serializable structures."""
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # paths / datetimes
    if isinstance(obj, Path):
        return str(obj)
    # datetime/date handling (robust to shadowed 'datetime' names)
    import datetime as _dt
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    # numpy / pandas containers
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict(orient="records")]

    # python containers
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    # dataclasses / custom objects (e.g., SimulationConfig)
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return to_jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return to_jsonable({k: v for k, v in vars(obj).items() if not k.startswith("_")})

    return obj

def dump_json(path: Union[str, Path], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)


_ALLOWED_MODEL_KEYS: Dict[str, Optional[set]] = {
    "ridge": {"alpha", "fit_intercept", "solver", "tol", "max_iter", "random_state"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "random_state"},
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight", "gamma", "random_state"},
    "gradient_boosting": {"n_estimators", "learning_rate", "max_depth", "min_samples_split", "min_samples_leaf", "subsample", "random_state"},
    "mlp": {"hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter", "random_state"},
    "poisson": {"alpha", "fit_intercept", "max_iter", "tol"},
    "tweedie": {"power", "alpha", "fit_intercept", "max_iter", "tol"},
    "linear": {"fit_intercept"},
    "naive_last": set(),
    "mean": set(),
    "median": set(),
    "custom": None,  # allow anything
}


def write_timeseries_outputs(results: Dict[str, Any], sim_dir: Union[str, Path], config: Optional[Any] = None) -> None:
    """Save simulation time series in wide and/or long CSV formats.

    Defaults (if config.outputs is missing):
      - write_timeseries_wide = False
      - write_timeseries_long = True
    """
    try:
        import pandas as _pd
        sim_dir = Path(sim_dir)
        sim_dir.mkdir(parents=True, exist_ok=True)

        ts = results.get("timeseries", None)
        if ts is None:
            return

        if isinstance(ts, list):
            ts_df = _pd.DataFrame(ts)
        elif isinstance(ts, _pd.DataFrame):
            ts_df = ts.copy()
        else:
            # unknown type
            ts_df = _pd.DataFrame(results.get("results", []))

        if ts_df.empty:
            return

        outputs = getattr(config, "outputs", None) if config is not None else None
        write_wide = getattr(outputs, "write_timeseries_wide", False) if outputs is not None else False
        write_long = getattr(outputs, "write_timeseries_long", True) if outputs is not None else True

        # Wide
        if write_wide:
            wide_path = sim_dir / "timeseries.csv"
            try:
                ts_df.to_csv(wide_path, index=False)
            except Exception:
                # fallback: stringify
                ts_df.astype(str).to_csv(wide_path, index=False)

        # Long
        if write_long:
            id_vars = []
            for c in ["YEAR_GRG", "year", "t", "step", "region", "ZONE", "zone"]:
                if c in ts_df.columns:
                    id_vars.append(c)
            if not id_vars:
                ts_df = ts_df.reset_index().rename(columns={"index": "t"})
                id_vars = ["t"]

            value_vars = [c for c in ts_df.columns if c not in id_vars]
            long_df = ts_df.melt(id_vars=id_vars, value_vars=value_vars, var_name="agent", value_name="value")
            long_path = sim_dir / "timeseries_long.csv"
            long_df.to_csv(long_path, index=False)
    except Exception:
        # keep pipeline robust
        return

def write_validation_outputs(results: Dict[str, Any], reports_dir: Union[str, Path]) -> None:
    """Save validation metrics (JSON + flat CSV summary) if available."""
    try:
        import csv as _csv
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        validation = results.get("validation", None)
        if not validation:
            return

        # Raw JSON
        dump_json(reports_dir / "validation_metrics.json", validation)

        # Flat CSV
        csv_path = reports_dir / "metrics_by_agent.csv"
        keys = [
            "alignment_shift",
            "r2",
            "rmse",
            "mae",
            "mse",
            "n",
            "rmse_unshifted",
            "r2_unshifted",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["agent"] + keys)
            for agent, met in sorted(validation.items(), key=lambda kv: kv[0]):
                row = [agent]
                for k in keys:
                    row.append(met.get(k, ""))
                w.writerow(row)

        # Summary JSON
        try:
            import numpy as _np
            rmse_vals = _np.array([float(v.get("rmse")) for v in validation.values() if v.get("rmse") is not None], dtype=float)
            mae_vals = _np.array([float(v.get("mae")) for v in validation.values() if v.get("mae") is not None], dtype=float)
            r2_vals = _np.array([float(v.get("r2")) for v in validation.values() if v.get("r2") is not None], dtype=float)

            summary = {
                "n_agents": int(len(validation)),
                "rmse_mean": float(_np.nanmean(rmse_vals)) if rmse_vals.size else None,
                "rmse_median": float(_np.nanmedian(rmse_vals)) if rmse_vals.size else None,
                "mae_mean": float(_np.nanmean(mae_vals)) if mae_vals.size else None,
                "mae_median": float(_np.nanmedian(mae_vals)) if mae_vals.size else None,
                "r2_mean": float(_np.nanmean(r2_vals)) if r2_vals.size else None,
                "r2_median": float(_np.nanmedian(r2_vals)) if r2_vals.size else None,
            }
            dump_json(reports_dir / "metrics_summary.json", summary)
        except Exception:
            pass
    except Exception:
        return



def sanitize_hyperparameters_in_config(config: SimulationConfig) -> None:
    """Drop unknown hyperparameters so sklearn estimators won't crash on init."""
    for a in getattr(config, "agents", []) or []:
        if getattr(a, "type", None) != "ml":
            continue
        params = getattr(a, "hyperparameters", None)
        if not isinstance(params, dict) or not params:
            continue
        model_type = str(getattr(a, "model_type", "") or "").strip().lower()
        allowed = _ALLOWED_MODEL_KEYS.get(model_type, None)
        if allowed is None:
            # custom or unknown -> keep as-is
            continue
        cleaned = {k: v for k, v in params.items() if k in allowed and k not in NON_MODEL_KEYS}
        # keep common metadata keys if present
        for k in ("target_transform", "features", "feature_selector"):
            if k in params:
                cleaned[k] = params[k]
        a.hyperparameters = cleaned


def preflight_check_ml_dependencies(
    config: SimulationConfig,
    df: pd.DataFrame,
    *,
    year_col: str = "YEAR_GRG",
    log_missing: bool = True,
) -> Dict[str, Any]:
    """Validate that ML agents can find their target/dependency columns in the provided df.

    Returns a dict with `ok`, `missing_by_agent`, and `missing_overall`.
    """
    cols = set(map(str, df.columns))
    missing_by_agent: Dict[str, List[str]] = {}

    for a in getattr(config, "agents", []) or []:
        if getattr(a, "type", None) != "ml":
            continue
        need: List[str] = []
        if a.target_column:
            need.append(str(a.target_column))
        for dep in (a.dependencies or []):
            need.append(str(dep))
        # year col is optional but helpful for WF/HPO
        if year_col:
            pass

        miss = []
        for c in need:
            # allow truth fallback conventions
            if c in cols:
                continue
            if f"{c}_truth" in cols:
                continue
            if f"{c}__truth" in cols:
                continue
            miss.append(c)

        if miss:
            missing_by_agent[str(a.name)] = sorted(set(miss))

    missing_overall = sorted({c for v in missing_by_agent.values() for c in v})
    ok = len(missing_overall) == 0
    if (not ok) and log_missing:
        logging.warning("Preflight: missing columns for ML agents: %s", missing_by_agent)

    return {"ok": ok, "missing_by_agent": missing_by_agent, "missing_overall": missing_overall}


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 1:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 1:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def run_walk_forward_one_step(
    config: SimulationConfig,
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    *,
    year_col: str = "YEAR_GRG",
    min_train_years: int = 6,
) -> Dict[str, Any]:
    """One-step-ahead walk-forward validation for each ML agent.

    For each ML agent and each test year y:
        - train on years < y
        - predict the last available row in year y
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------------------------------
    # Paper-ready structured outputs (no cfg_path dependency)
    # -------------------------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if year_col not in df.columns:
        raise ValueError(f"Walk-forward requires year_col='{year_col}' in df.")

    years_all = pd.to_numeric(df[year_col], errors="coerce")
    if years_all.isna().all():
        raise ValueError(f"Walk-forward: year_col '{year_col}' could not be parsed to numeric.")

    # Make sure model params won't crash at init
    sanitize_hyperparameters_in_config(config)

    dm = DataManager(config)
    dm.load_data(df)
    dm.build_lagged_features(df)

    summary: Dict[str, Any] = {"year_col": year_col, "min_train_years": int(min_train_years), "agents": {}}

    unique_years = np.array(sorted(set(int(y) for y in years_all.dropna().unique())))
    if unique_years.size < int(min_train_years) + 1:
        logging.warning("Walk-forward: not enough unique years (%s) for min_train_years=%s", unique_years.size, min_train_years)

    agents_iter = (config.agents.values() if isinstance(config.agents, dict) else (config.agents or []))
    for a in agents_iter:
        if a.type != "ml":
            continue

        agent_name = str(a.name)
        agent_cfg = a
        # Global feature selection config (fold-wise selection is applied per test-year below)
        fs_cfg = config.feature_selection
        selector = (
            FeatureSelector(
                fs_cfg.strategy,
                fs_cfg.top_k,
                int(getattr(config, 'random_seed', 42) or 42),
                method=getattr(fs_cfg, 'method', 'mi'),
                min_k=getattr(fs_cfg, 'min_k', 0),
            )
            if getattr(fs_cfg, 'enabled', False)
            else None
        )

        try:
            X_full, y_full, feat_cols, meta = dm.prepare_training_data_with_meta(agent_cfg, feature_selector=None, meta_cols=[year_col])
            years_feat = meta.get(year_col)
            if years_feat is None:
                raise ValueError(f"'{year_col}' missing after preprocessing for {agent_name}")

            years_feat = np.asarray(pd.to_numeric(years_feat, errors="coerce"))
        except Exception as e:
            summary["agents"][agent_name] = {"ok": False, "error": str(e)}
            continue

        preds = []
        trues = []
        test_years = []
        selected_by_year: Dict[int, List[str]] = {}

        for ty in unique_years:
            tr_mask = years_feat < ty
            te_mask = years_feat == ty
            if tr_mask.sum() < int(min_train_years) or te_mask.sum() < 1:
                continue

            # Build and train a fresh model each step
            beh = MLBehavior(agent_cfg)
            model = beh.model

            tt = getattr(agent_cfg, "target_transform", None) or (agent_cfg.hyperparameters or {}).get("target_transform")
            y_tr = y_full[tr_mask]
            X_tr = X_full[tr_mask]

            # last row in test-year
            te_local_idx = np.where(te_mask)[0][-1]
            X_te = X_full[te_local_idx : te_local_idx + 1]
            y_te = y_full[te_local_idx : te_local_idx + 1]

            # Fold-wise feature selection (fit on training fold only to avoid leakage)
            if selector is not None and X_tr.shape[1] > 1:
                try:
                    import pandas as _pd
                    X_tr_df = _pd.DataFrame(X_tr, columns=feat_cols)
                    mand = []
                    try:
                        mand.extend(list(getattr(config.feature_selection, "mandatory_features", []) or []))
                    except Exception:
                        pass
                    try:
                        mand.extend(list(getattr(agent_cfg, "mandatory_features", []) or []))
                    except Exception:
                        pass
                    mand = [m for m in list(dict.fromkeys([str(x) for x in mand if x not in (None, "", "none")])) if m in feat_cols]
                    sel_cols = selector.select_features(X_tr_df, y_tr, mandatory=mand) or list(feat_cols)
                    sel_cols = [c for c in sel_cols if c in feat_cols]
                    if sel_cols and len(sel_cols) < len(feat_cols):
                        idx = [feat_cols.index(c) for c in sel_cols]
                        X_tr = X_tr[:, idx]
                        X_te = X_te[:, idx]
                    else:
                        # selection returned all features
                        sel_cols = list(feat_cols)
                except Exception as _e:
                    sel_cols = list(feat_cols)
            else:
                sel_cols = list(feat_cols)

            # Record selected features for this test year
            try:
                selected_by_year[int(ty)] = list(sel_cols)
            except Exception:
                pass

            if str(agent_cfg.model_type).lower() in {"poisson", "tweedie"}:
                if np.nanmin(y_tr) < 0 or np.nanmin(y_te) < 0:
                    continue

            try:
                y_tr_t = beh._transform_y(y_tr)
                model.fit(X_tr, y_tr_t)
                y_hat_t = model.predict(X_te)
                y_hat = beh._inverse_transform_y(y_hat_t)
            except Exception as e:
                summary["agents"].setdefault(agent_name, {}).setdefault("errors", []).append({"year": int(ty), "error": str(e)})
                continue

            preds.append(float(y_hat[0]))
            trues.append(float(y_te[0]))
            test_years.append(int(ty))

        y_pred = np.array(preds, dtype=float)
        y_true = np.array(trues, dtype=float)

        # Dual-report split years (e.g., COVID shock years)
        covid_years = set()
        try:
            covid_years = set(int(x) for x in (getattr(getattr(config, 'hyperparams', None), 'exclude_test_years', []) or []) if x is not None)
        except Exception:
            covid_years = set()

        years_arr = np.array(test_years, dtype=int) if test_years else np.array([], dtype=int)
        main_mask = np.array([y not in covid_years for y in years_arr], dtype=bool) if years_arr.size else np.array([], dtype=bool)
        stress_mask = np.array([y in covid_years for y in years_arr], dtype=bool) if years_arr.size else np.array([], dtype=bool)

        rmse_all, mae_all, r2_all = _rmse(y_true, y_pred), _mae(y_true, y_pred), _r2(y_true, y_pred)
        rmse_main = _rmse(y_true[main_mask], y_pred[main_mask]) if main_mask.size else float('nan')
        mae_main  = _mae(y_true[main_mask], y_pred[main_mask]) if main_mask.size else float('nan')
        r2_main   = _r2(y_true[main_mask], y_pred[main_mask]) if main_mask.size else float('nan')
        rmse_stress = _rmse(y_true[stress_mask], y_pred[stress_mask]) if stress_mask.size else float('nan')
        mae_stress  = _mae(y_true[stress_mask], y_pred[stress_mask]) if stress_mask.size else float('nan')
        r2_stress   = _r2(y_true[stress_mask], y_pred[stress_mask]) if stress_mask.size else float('nan')

        # Keep backward-compatible top-level rmse/mae/r2 as MAIN (non-covid) metrics
        agent_res = {
            "ok": True,
            "n_tests": int(len(test_years)),
            "test_years": test_years,
            "covid_years": sorted(list(covid_years)),
            "rmse": rmse_main,
            "mae": mae_main,
            "r2": r2_main,
            "rmse_all": rmse_all,
            "mae_all": mae_all,
            "r2_all": r2_all,
            "rmse_stress": rmse_stress,
            "mae_stress": mae_stress,
            "r2_stress": r2_stress,
        }
        summary["agents"][agent_name] = agent_res

        # Save per-agent predictions (+ selected features only, paper-ready)
        try:
            union_feats = sorted({f for lst in selected_by_year.values() for f in (lst or [])})
        except Exception:
            union_feats = []

        dump_json(
            out_dir / f"wf_{agent_name}.json",
            {
                "year": test_years,
                "y_true": trues,
                "y_pred": preds,
                "metrics": agent_res,
                "selected_features_by_year": selected_by_year,
                "selected_features_union": union_feats,
            },
        )

    dump_json(out_dir / "walk_forward_summary.json", summary)

    # Write convenience summaries split by covid_years (dual report)
    try:
        covid_years = set(int(x) for x in (getattr(getattr(config, 'hyperparams', None), 'exclude_test_years', []) or []) if x is not None)
    except Exception:
        covid_years = set()

    try:
        import copy
        main_summary = copy.deepcopy(summary)
        main_summary['split'] = 'main_non_covid'
        main_summary['covid_years'] = sorted(list(covid_years))
        stress_summary = copy.deepcopy(summary)
        stress_summary['split'] = 'stress_covid'
        stress_summary['covid_years'] = sorted(list(covid_years))
        # For each agent, keep only the relevant metrics keys
        for ag, res in (summary.get('agents', {}) or {}).items():
            if not isinstance(res, dict):
                continue
            main_summary['agents'][ag] = {k: v for k, v in res.items() if k not in {'rmse_stress','mae_stress','r2_stress'}}
            stress_summary['agents'][ag] = {
                'ok': res.get('ok', False),
                'n_tests': res.get('n_tests', 0),
                'test_years': res.get('test_years', []),
                'covid_years': res.get('covid_years', []),
                'rmse': res.get('rmse_stress', float('nan')),
                'mae': res.get('mae_stress', float('nan')),
                'r2': res.get('r2_stress', float('nan')),
                'rmse_all': res.get('rmse_all', float('nan')),
                'mae_all': res.get('mae_all', float('nan')),
                'r2_all': res.get('r2_all', float('nan')),
            }
        dump_json(out_dir / 'walk_forward_summary_main.json', main_summary)
        dump_json(out_dir / 'walk_forward_summary_stress.json', stress_summary)
    except Exception:
        pass

    return summary


def prepare_simulation_from_yaml(
    cfg_path: Union[str, Path],
    *,
    data_file_override: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    log_level: Optional[str] = None,
) -> Tuple[SimulationConfig, pd.DataFrame, Dict[str, Any]]:
    """Load YAML, return (config, df, raw_yaml). Runner-only keys are preserved in raw_yaml."""
    cfg_path = Path(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = load_config_from_yaml(cfg_path)

    if data_file_override:
        config.data_file = str(data_file_override)

    if log_level:
        config.log_level = str(log_level)

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config.log_file = str(out_dir / "run.log")

    sanitize_hyperparameters_in_config(config)
    validate_config(config)

    df = pd.read_csv(config.data_file)
    preflight_check_ml_dependencies(config, df)

    return config, df, raw






if __name__ == "__main__":
    # Minimal CLI usage:
    #   python hybrid_sim_core.py path/to/config.yaml --output_dir out
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid SD–ABM–ML Simulation Framework (Core)")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--data_file", type=str, default=None, help="Override config data_file")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional report directory")
    args = parser.parse_args()

    res = run_simulation_from_config(
        args.config,
        data_file=args.data_file,
        output_dir=args.output_dir,
    )
    print(f"Done. success={res.get('success')} steps={res.get('stats', {}).get('total_steps')} errors={len(res.get('stats', {}).get('errors', []) or [])}")