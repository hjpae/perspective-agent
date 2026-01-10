# cear_pilot/analysis/figure_gate.py
# -*- coding: utf-8 -*-
"""
Compare a simple "gate" built from g vs a gate built from policy signals.

Goal:
- Show that g can be used as a slow, stateful regime sensor (meta-variable)
  by turning it into a stable gate / alarm with hysteresis.
- Compare against a policy-based gate (entropy or margin-based), using the same gate logic.

Usage:
  python -m cear_pilot.analysis.figure_gate --run_dir outputs/runs/<RUN_ID> --episode 0

Notes:
- Works with traj.parquet or traj.csv.
- If d_g is not logged, it will be computed from g_* columns as:
    d_g(t) = || g(t) - mean(g[0:pre_steps]) ||
- Policy signals expected (any subset is ok):
    pi_entropy, pi_max, margin_top1_top2 (or margin)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_traj(run_dir: Path) -> pd.DataFrame:
    p_parq = run_dir / "traj.parquet"
    p_csv = run_dir / "traj.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"traj.parquet/csv not found in: {run_dir}")


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _g_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("g_")], key=lambda s: int(s.split("_")[1]))


def _compute_dg(df_ep: pd.DataFrame, pre_steps: int) -> np.ndarray:
    if "d_g" in df_ep.columns:
        return df_ep["d_g"].to_numpy(dtype=np.float64)

    g_cols = _g_columns(df_ep)
    if not g_cols:
        raise ValueError("No d_g column and no g_* columns found. Cannot compute g distance.")
    G = df_ep[g_cols].to_numpy(dtype=np.float64)

    pre_steps = int(pre_steps)
    pre_steps = max(1, min(pre_steps, len(G)))
    g0 = G[:pre_steps].mean(axis=0)
    d = np.linalg.norm(G - g0[None, :], axis=1)
    return d


def _get_policy_signal(df_ep: pd.DataFrame, which: str) -> Tuple[np.ndarray, str]:
    """
    Returns (signal, label).
    signal should be "instability-like": larger => more unstable / more alarm-worthy.
    """
    which = which.lower().strip()

    # Prefer explicit entropy if present
    if which in ("entropy", "pi_entropy"):
        if "pi_entropy" in df_ep.columns:
            return df_ep["pi_entropy"].to_numpy(dtype=np.float64), "pi_entropy"

    # Margin: smaller margin => more uncertain => convert to instability = (1 - margin)
    if which in ("margin", "top_margin"):
        for c in ("margin_top1_top2", "margin"):
            if c in df_ep.columns:
                m = df_ep[c].to_numpy(dtype=np.float64)
                return (1.0 - m), f"1 - {c}"

    # pi_max: smaller pi_max => more uncertain => convert to instability = (1 - pi_max)
    if which in ("pi_max", "max", "pimax"):
        if "pi_max" in df_ep.columns:
            p = df_ep["pi_max"].to_numpy(dtype=np.float64)
            return (1.0 - p), "1 - pi_max"

    # Fallback: try any known column
    for c, label in (("pi_entropy", "pi_entropy"), ("margin_top1_top2", "1 - margin_top1_top2"), ("pi_max", "1 - pi_max")):
        if c in df_ep.columns:
            x = df_ep[c].to_numpy(dtype=np.float64)
            if c == "pi_entropy":
                return x, label
            return (1.0 - x), label

    raise ValueError("No usable policy columns found (pi_entropy, margin_top1_top2/margin, pi_max).")


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def _hysteresis_gate(x: np.ndarray, on_thr: float, off_thr: float) -> np.ndarray:
    """
    x: higher => more alarm-worthy
    Gate turns ON when x >= on_thr
    Gate turns OFF when x <= off_thr
    """
    on_thr = float(on_thr)
    off_thr = float(off_thr)
    if off_thr > on_thr:
        raise ValueError("off_thr must be <= on_thr for hysteresis.")

    g = np.zeros(len(x), dtype=np.int32)
    state = 0
    for i, v in enumerate(x):
        if state == 0 and v >= on_thr:
            state = 1
        elif state == 1 and v <= off_thr:
            state = 0
        g[i] = state
    return g


def _robust_thresholds(x_pre: np.ndarray, k_on: float, k_off: float) -> Tuple[float, float]:
    """
    Robust thresholds from pre-window using MAD.
    on_thr = median + k_on * MAD
    off_thr = median + k_off * MAD
    """
    med = float(np.median(x_pre))
    mad = float(np.median(np.abs(x_pre - med)) + 1e-8)
    on_thr = med + float(k_on) * mad
    off_thr = med + float(k_off) * mad
    return on_thr, off_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--pre_steps", type=int, default=30, help="Window for pre-baseline (distance + thresholds).")
    ap.add_argument("--ema_alpha", type=float, default=0.12, help="EMA alpha. Smaller => slower gate.")
    ap.add_argument("--policy_signal", type=str, default="entropy",
                    choices=["entropy", "pi_entropy", "margin", "top_margin", "pi_max", "max", "pimax"],
                    help="Which policy signal to use for the policy gate.")

    # Threshold strategy: either provide absolute thresholds, or use robust pre-window stats.
    ap.add_argument("--use_robust_thr", action="store_true",
                    help="If set, compute thresholds from pre-window with MAD (recommended).")
    ap.add_argument("--k_on", type=float, default=6.0, help="Robust on-threshold multiplier (median + k_on*MAD).")
    ap.add_argument("--k_off", type=float, default=3.0, help="Robust off-threshold multiplier (median + k_off*MAD).")

    ap.add_argument("--g_on", type=float, default=None, help="Absolute on-threshold for g EMA (override robust).")
    ap.add_argument("--g_off", type=float, default=None, help="Absolute off-threshold for g EMA (override robust).")
    ap.add_argument("--p_on", type=float, default=None, help="Absolute on-threshold for policy EMA (override robust).")
    ap.add_argument("--p_off", type=float, default=None, help="Absolute off-threshold for policy EMA (override robust).")

    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--outname", type=str, default="fig_gate.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = _load_traj(run_dir)
    meta = _load_meta(run_dir)

    if "episode" not in df.columns:
        raise ValueError("traj must contain an 'episode' column.")
    df_ep = df[df["episode"] == args.episode].copy()
    if len(df_ep) == 0:
        raise ValueError(f"No rows found for episode={args.episode} in {run_dir}")

    # Ensure sorted by time
    t_col = "t" if "t" in df_ep.columns else None
    if t_col is not None:
        df_ep = df_ep.sort_values(t_col).reset_index(drop=True)
        t = df_ep[t_col].to_numpy(dtype=np.int64)
    else:
        t = np.arange(len(df_ep), dtype=np.int64)

    # Switch info (optional)
    t_switch = None
    if "t_switch" in meta:
        try:
            t_switch = int(meta["t_switch"])
        except Exception:
            t_switch = None

    # ---- signals
    d_g = _compute_dg(df_ep, pre_steps=args.pre_steps)
    p_sig, p_label = _get_policy_signal(df_ep, which=args.policy_signal)

    # Normalize policy signal to be comparable-ish visually (optional but helpful)
    # We keep it simple: robust scale based on pre-window.
    pre_n = max(5, min(int(args.pre_steps), len(t)))
    p_pre = p_sig[:pre_n]
    p_med = np.median(p_pre)
    p_mad = np.median(np.abs(p_pre - p_med)) + 1e-8
    p_sig_n = (p_sig - p_med) / p_mad  # dimensionless "instability units"

    g_pre = d_g[:pre_n]
    g_med = np.median(g_pre)
    g_mad = np.median(np.abs(g_pre - g_med)) + 1e-8
    d_g_n = (d_g - g_med) / g_mad

    # ---- EMA
    g_ema = _ema(d_g_n, alpha=args.ema_alpha)
    p_ema = _ema(p_sig_n, alpha=args.ema_alpha)

    # ---- thresholds + gates
    if args.use_robust_thr:
        g_on, g_off = _robust_thresholds(g_ema[:pre_n], k_on=args.k_on, k_off=args.k_off)
        p_on, p_off = _robust_thresholds(p_ema[:pre_n], k_on=args.k_on, k_off=args.k_off)
    else:
        # If user doesn't want robust thresholds, default to something reasonable in z-MAD units
        g_on, g_off = 6.0, 3.0
        p_on, p_off = 6.0, 3.0

    # absolute overrides
    if args.g_on is not None: g_on = float(args.g_on)
    if args.g_off is not None: g_off = float(args.g_off)
    if args.p_on is not None: p_on = float(args.p_on)
    if args.p_off is not None: p_off = float(args.p_off)

    g_gate = _hysteresis_gate(g_ema, on_thr=g_on, off_thr=g_off)
    p_gate = _hysteresis_gate(p_ema, on_thr=p_on, off_thr=p_off)

    # ---- cumulative "alarm time" after first switch (if any) else whole episode
    start_idx = 0
    if t_switch is not None:
        # find first index where t >= t_switch
        start_idx = int(np.searchsorted(t, t_switch, side="left"))
    g_alarm_cum = np.cumsum(g_gate[start_idx:])
    p_alarm_cum = np.cumsum(p_gate[start_idx:])
    t_cum = t[start_idx:]

    # ---- plotting
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)

    # Panel 1: raw-ish normalized signals
    ax1.plot(t, d_g_n, label="g distance (robust z)")
    ax1.plot(t, p_sig_n, label=f"policy signal (robust z): {p_label}")
    ax1.set_ylabel("signal (robust z)")
    ax1.legend(loc="upper right")

    # Panel 2: EMA + thresholds + gate state
    ax2.plot(t, g_ema, label="EMA(g)")
    ax2.plot(t, p_ema, label="EMA(policy)")
    ax2.axhline(g_on, linestyle="--", linewidth=1.0)
    ax2.axhline(g_off, linestyle="--", linewidth=1.0)
    ax2.axhline(p_on, linestyle="--", linewidth=1.0)
    ax2.axhline(p_off, linestyle="--", linewidth=1.0)

    # Gate state (scaled to sit nicely at bottom)
    y0 = min(np.min(g_ema), np.min(p_ema)) - 0.8
    ax2.plot(t, y0 + 0.35 * g_gate, label="g_gate (hysteresis)")
    ax2.plot(t, y0 + 0.35 * p_gate, label="pi_gate (hysteresis)")
    ax2.set_ylabel("EMA + gate")
    ax2.legend(loc="upper right")

    # Panel 3: cumulative alarm time after switch (or whole episode)
    ax3.plot(t_cum, g_alarm_cum, label="cum alarm (g_gate)")
    ax3.plot(t_cum, p_alarm_cum, label="cum alarm (pi_gate)")
    ax3.set_ylabel("cumulative")
    ax3.set_xlabel("t")
    ax3.legend(loc="upper left")

    # Mark switch if known
    if t_switch is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvline(t_switch, linestyle="--", linewidth=2.0)
            ax.axvspan(t_switch, t[-1], alpha=0.08)

    title = args.title.strip()
    if not title:
        title = f"Gate demo (ep={args.episode}, alpha={args.ema_alpha}, pre_steps={args.pre_steps})"
        if t_switch is not None:
            title += f", t_switch={t_switch}"
    fig.suptitle(title)

    out_path = run_dir / "figs" / args.outname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()