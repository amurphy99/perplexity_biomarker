"""
Compact stats/plot helper for two-group comparisons with optional outlier handling.

"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy       import stats
from dataclasses import dataclass
from typing      import Iterable, Tuple, Dict, List

from .log_helpers import BOLD, RESET, format_time

# ======================================================================= ===================================
# Pretty + ANSI styling
# ======================================================================= ===================================
# Human-friendly labels
metrics_str: Dict[str, str] = {"ppl": "Perplexity", "avg_nll": "Average Negative Log-Likelihood"}
tables_str : Dict[str, str] = {"ppl": "PPL",        "avg_nll": "aNLL"}

# Default table metadata if the caller doesn't provide some fields
DEFAULT_TABLE_META: Dict[str, object] = dict(
    model_id   = "—",
    stride     = "—",
    max_length = "—",
    runtime_s  = None,   # seconds (float or int)
    aggr       = "winsor",
    min_tokens = "—",
)


# ======================================================================= ===================================
# Helpers
# ======================================================================= ===================================
def filter_outliers(x: Iterable[float], method: str = "none", q: Tuple[float, float] = (0.05, 0.95)) -> np.ndarray:
    """Filter outlier data points using different methods."""
    # Validity checks
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0: return x

    # Different outlier filtering methods
    if method == "none": return x
    
    if method == "trim":
        lo, hi = np.quantile(x, q[0]), np.quantile(x, q[1])
        return x[(x >= lo) & (x <= hi)]
    
    if method == "winsor":
        lo, hi = np.quantile(x, q[0]), np.quantile(x, q[1])
        return np.clip(x, lo, hi)
    
    if method == "iqr":
        q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = q3 - q1
        lo, hi = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        return x[(x >= lo) & (x <= hi)]

    return x


# =======================================================================
# Effect size 
# =======================================================================
def hedges_g(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    
    if na < 2 or nb < 2: return np.nan
    
    sp = np.sqrt(((na-1) * np.var(a, ddof=1) + (nb-1) * np.var(b, ddof=1)) / (na+nb-2))
    if sp == 0: return np.nan
    
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)  # small-sample correction
    
    return d * J


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    """O(n*m). Fine for ~hundreds per group."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    
    return (gt - lt) / (len(a) * len(b))


def bootstrap_mean_diff(a: Iterable[float], b: Iterable[float], n_boot: int = 5000, seed: int = 0) -> Tuple[float, float]:
    """
    Mean difference between groups
    """
    rng = np.random.default_rng(seed)
    a,  b  = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    
    if na == 0 or nb == 0: return (np.nan, np.nan)
    
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, na, replace=True).mean() - rng.choice(b, nb, replace=True).mean()
        
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


# ======================================================================= ===================================
# Stats core
# ======================================================================= ===================================
@dataclass
class StatsResult:
    n0: int; m0: float; s0: float
    n1: int; m1: float; s1: float

    diff  : float
    t_stat: float; p_welch: float
    u_stat: float; p_mwu  : float
    g     : float; delta  : float
    ci_lo : float; ci_hi  : float


def comp_calculations(g0: Iterable[float], g1: Iterable[float], *, n_boot: int = 5000, seed: int = 0) -> StatsResult:
    g0 = np.asarray(g0); g1 = np.asarray(g1)

    n0, n1 = len(g0), len(g1)
    m0, s0 = (np.mean(g0), np.std(g0, ddof=1)) if n0 > 1 else (np.nan, np.nan)
    m1, s1 = (np.mean(g1), np.std(g1, ddof=1)) if n1 > 1 else (np.nan, np.nan)
    diff   = m0 - m1

    # Welch t-test (two-sided)
    t_stat, p_welch = stats.ttest_ind(g0, g1, equal_var=False, alternative="two-sided")

    # Mann-Whitney U (two-sided)
    try:               u_stat, p_mwu = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    except ValueError: u_stat, p_mwu = np.nan, np.nan

    g_val        = hedges_g(g0, g1)
    delta_val    = cliffs_delta(g0, g1)
    ci_lo, ci_hi = bootstrap_mean_diff(g0, g1, n_boot=n_boot, seed=seed)

    return StatsResult(n0, m0, s0, n1, m1, s1, diff, t_stat, p_welch, u_stat, p_mwu, g_val, delta_val, ci_lo, ci_hi)


# ======================================================================= ===================================
# Plot helpers
# ======================================================================= ===================================
def _violin_box(ax: plt.Axes, a: np.ndarray, b: np.ndarray, groups: Tuple[str, str]) -> None:
    ax.violinplot([a, b], showmeans=True, showmedians=True, showextrema=False)
    ax.boxplot   ([a, b], positions=[1, 2], widths=0.15)
    ax.set_xticks([1, 2]); ax.set_xticklabels(groups)

def _annotate_stats(ax: plt.Axes, st: StatsResult) -> None:
    text_args = dict(ha="center", va="bottom", transform=ax.transAxes, fontname="monospace")
    ax.text(0.5, 0.90, f"Welch        p = {st.p_welch:7.4f}", **text_args)
    ax.text(0.5, 0.85, f"Mann-Whitney p = {st.p_mwu  :7.4f}", **text_args)
    ax.text(0.5, 0.80, f"Hedges       g = {st.g      :7.4f}", **text_args)
    ax.text(0.5, 0.75, f"Cliff's      δ = {st.delta  :7.4f}", **text_args)


# ======================================================================= ===================================
# Markdown row builder (replaces the old global-dependent get_table_string)
# ======================================================================= ===================================
def get_table_string(metric: str, outliers: str, p_welch: float, p_mwu: float, g: float, delta: float,
                     *, table_meta: Dict[str, object] | None = None) -> str:
    """Build a Markdown row for the results table."""
    meta = {**DEFAULT_TABLE_META, **(table_meta or {})}

    metric_short = tables_str.get(metric, metric)
    col_sep = "\\|"

    col_setup = f"| {meta['model_id']} | {meta['max_length']} | {meta['stride']} | "
    col_str   = f"{format_time(meta['runtime_s'])}  | {meta['aggr']} | {meta['min_tokens']} | {col_sep} | "
    details   = f"{metric_short} | {outliers} | {col_sep} | "
    stats_str = f"{p_welch:7.4f} | {p_mwu:7.4f} | {g:7.4f} | {delta:7.4f} | "

    return col_setup + col_str + details + stats_str


# ======================================================================= ===================================
# Compare & Plot Function
# ======================================================================= ===================================
def compare_and_plot(
    df           : pd.DataFrame,
    metric       : str = "avg_nll",                 # or "ppl"
    group_col    : str = "dx",
    groups       : Tuple[str, str] = ("Control", "ProbableAD"),
    outliers     : str = "none",                    # 'none' | 'trim' | 'winsor' | 'iqr'
    trim_q       : Tuple[float, float] = (0.05, 0.95),
    title_suffix : str = "",
    *,
    table_meta   : Dict[str, object] | None = None,
    n_boot       : int = 5000,
    seed         : int = 0,
) -> List[str]:
    """
    Compare two groups on a metric, print stats, plot two panels, and return Markdown row strings.
    Returns [row_with_outlier_handling, row_with_outliers_included]
    """
    # --------------------------------------------------------------------
    # Split groups and filter NaNs
    # --------------------------------------------------------------------
    g0 = df.loc[df[group_col] == groups[0], metric].dropna().to_numpy()
    g1 = df.loc[df[group_col] == groups[1], metric].dropna().to_numpy()

    # Filtered copies (outlier handling)
    g0f = filter_outliers(g0, outliers, trim_q)
    g1f = filter_outliers(g1, outliers, trim_q)

    # Stats
    st_f = comp_calculations(g0f, g1f, n_boot=n_boot, seed=seed)  # filtered
    st_r = comp_calculations(g0,  g1,  n_boot=n_boot, seed=seed)  # raw

    # Pretty metric label
    metric_label = metrics_str.get(metric, metric)

    # --------------------------------------------------------------------
    # Print block
    # --------------------------------------------------------------------
    print("─" * 100)
    print(f"Metric: {BOLD}{metric_label}{RESET}  |  Outliers: {BOLD}{outliers}{RESET}")
    print("─" * 100)

    print(f"{BOLD}{(groups[0] + ':'):<11}{RESET} {st_f.m0  :>8.4f} (+/- {st_f.s0:6.4f}) | n = {st_f.n0:3} ({st_r.n0:3})")
    print(f"{BOLD}{(groups[1] + ':'):<11}{RESET} {st_f.m1  :>8.4f} (+/- {st_f.s1:6.4f}) | n = {st_f.n1:3} ({st_r.n1:3})")
    print(f"{BOLD}{'Diff:'          :<11}{RESET} {st_f.diff:>8.4f} | 95% CI [{st_f.ci_lo:6.4f}, {st_f.ci_hi:6.4f}]\n")

    print(f"Welch t p = {st_f.p_welch:8.5f}  |  Mann-Whitney p = {st_f.p_mwu:8.5f}")
    print(f"Hedges  g = {st_f.g      :8.5f}  |  Cliff's      δ = {st_f.delta:8.5f}")

    # --------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # Left: Including outliers
    ax = axes[0]
    _violin_box(ax, g0, g1, groups)
    _annotate_stats(ax, st_r)
    ax.set_ylabel(metric_label)
    ax.set_title("Including Outliers")

    # Right: After outlier handling
    ax = axes[1]
    _violin_box(ax, g0f, g1f, groups)
    _annotate_stats(ax, st_f)
    ax.set_title(f"Outliers Removed {title_suffix}".strip())

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------
    # Build Markdown table rows
    # --------------------------------------------------------------------
    row_filtered = get_table_string(metric, outliers, st_f.p_welch, st_f.p_mwu, st_f.g, st_f.delta, table_meta=table_meta)
    row_raw      = get_table_string(metric, "none",   st_r.p_welch, st_r.p_mwu, st_r.g, st_r.delta, table_meta=table_meta)

    return [row_filtered, row_raw]




# ======================================================================= ===================================
# Plotting
# ======================================================================= ===================================
# Small helpers
def _split_groups(df: pd.DataFrame, group_col: str = "dx", groups: Tuple[str, str] = ("Control", "ProbableAD")):
    cn = df[df[group_col] == groups[0]].copy()
    ad = df[df[group_col] == groups[1]].copy()
    return cn, ad

def _scatter_by_group(ax, df, x_col, y_col, alpha=0.5, groups=("Control", "ProbableAD")):
    cn, ad = _split_groups(df, groups=groups)
    ax.scatter(cn[x_col], cn[y_col], alpha=alpha, color="tab:blue",   label=groups[0])
    ax.scatter(ad[x_col], ad[y_col], alpha=alpha, color="tab:orange", label=groups[1])
    return cn, ad

def _fit_line(ax, x, y, *, color="black", ls="--", label_prefix="Both"):
    """Add a linear fit line; return r-value or np.nan if not enough points."""
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 2 or y.size < 2: return np.nan
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    xs = np.sort(x)
    ax.plot(xs, slope * xs + intercept, color=color, linestyle=ls, label=f"{label_prefix} (r = {r_value:.4f})")
    return r_value


# =======================================================================
# Main plotting function
# =======================================================================
def plot_mmse_panels(
    d0              : pd.DataFrame,
    *,
    alpha           : float = 0.5,
    token_min       : int   = 50,
    ppl_max         : float = 150.0,
    anll_range      : Tuple[float, float] = (3.0, 5.5),
    groups          : Tuple[str, str] = ("Control", "ProbableAD"),
):
    """
    Three panels:
      (1) MMSE vs Perplexity (filtered: mmse notna, tokens > token_min, ppl < ppl_max)
      (2) MMSE vs Avg NLL    (filtered: mmse notna, tokens > token_min, anll in range)
      (3) Perplexity vs Tokens (full df)
    """
    # Print header
    print("─" * 100)
    print(f"{BOLD}Comparison Plots{RESET}")
    print("─" * 100)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Base filtered frame for panels 1 & 2
    base = d0[d0["mmse"].notna() & (d0["tokens"] > token_min)].copy()

    # =======================================================================
    # 1) Perplexity
    # =======================================================================
    ax    = axes[0]
    y_col = "ppl"

    d1 = base[base[y_col].notna() & (base[y_col] < ppl_max)].copy()
    ad = d1[d1["dx"] == groups[1]]
    
    # Scatter
    _scatter_by_group(ax, d1, x_col=y_col, y_col="mmse", alpha=alpha, groups=groups)

    # Fits: all + ProbableAD only
    _fit_line(ax, d1[y_col].values, d1["mmse"].values, color="black",     ls="--", label_prefix="Both")
    _fit_line(ax, ad[y_col].values, ad["mmse"].values, color="indianred", ls=":",  label_prefix=f"ONLY {groups[1]}")

    # Labels
    ax.set_xlabel(y_col); ax.set_ylabel("MMSE")
    ax.set_title("Perplexity")
    ax.legend(); ax.grid()

    # =======================================================================
    # 2) Average Negative Log-Likelihood
    # =======================================================================
    ax    = axes[1]
    y_col = "avg_nll"
    lo, hi = anll_range

    d2 = base[base[y_col].notna() & (base[y_col] > lo) & (base[y_col] < hi)].copy()
    ad = d2[d2["dx"] == groups[1]]
    
    # Scatter
    _scatter_by_group(ax, d2, x_col=y_col, y_col="mmse", alpha=alpha, groups=groups)

    # Fits: all + ProbableAD-only
    _fit_line(ax, d2[y_col].values, d2["mmse"].values, color="black",     ls="--", label_prefix="Both")
    _fit_line(ax, ad[y_col].values, ad["mmse"].values, color="indianred", ls=":",  label_prefix=f"ONLY {groups[1]}")

    # Labels
    ax.set_xlabel(y_col); ax.set_ylabel("MMSE")
    ax.set_title("Average Negative Log-Likelihood")
    ax.legend(); ax.grid()

    # =======================================================================
    # 3) Tokens & Perplexity (possible confound)
    # =======================================================================
    ax    = axes[2]
    y_col = "ppl"

    cn, ad = _split_groups(d0, groups=groups)
    ax.scatter(cn["tokens"], cn[y_col], alpha=alpha, color="tab:blue",   label=groups[0])
    ax.scatter(ad["tokens"], ad[y_col], alpha=alpha, color="tab:orange", label=groups[1])

    ax.set_xlim([0, 300])
    ax.set_xlabel("Tokens"); ax.set_ylabel(y_col)
    ax.set_title("Perplexity")
    ax.legend(); ax.grid()

    # =======================================================================
    plt.tight_layout()
    plt.show()
    return fig, axes

