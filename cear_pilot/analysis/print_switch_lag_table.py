# cear_pilot/analysis/print_switch_lag_table.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from cear_pilot.analysis.metrics import transition_lag_half_rise


def _find_traj(run_dir: Path) -> Path:
    for ext in [".parquet", ".csv"]:
        p = run_dir / f"traj{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No traj.parquet/csv in {run_dir}")


def _load_table(traj_path: Path):
    import pandas as pd
    if traj_path.suffix == ".parquet":
        return pd.read_parquet(traj_path)
    return pd.read_csv(traj_path)


def _safe_mean(xs: List[int]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return float(np.mean(xs))


def _safe_median(xs: List[int]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return float(np.median(xs))


def _fmt(x, width=8, prec=3) -> str:
    if x is None:
        return " " * (width - 1) + "-"
    if isinstance(x, int):
        return f"{x:>{width}d}"
    return f"{x:>{width}.{prec}f}"


def _extract_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _score_g(df, warmup: int, signed: bool = True) -> np.ndarray:
    # Default to signed projection; if you don't have enough A/B samples it falls back.
    g_cols = [c for c in df.columns if c.startswith("g_")]
    if len(g_cols) == 0:
        raise ValueError("No g_* columns found.")

    G = df[g_cols].to_numpy(dtype=np.float32)
    T = G.shape[0]
    w = int(warmup)

    if "regime" not in df.columns:
        signed = False

    if not signed:
        w0 = max(10, min(w, T))
        mu = G[:w0].mean(axis=0)
        return np.linalg.norm(G - mu[None, :], axis=-1).astype(np.float32)

    regime = df["regime"].to_numpy(dtype=int)

    idx = np.arange(T)
    post = idx >= w
    A = post & (regime == 0)
    B = post & (regime == 1)

    # If one side is empty, fall back to unsigned distance
    if A.sum() < 10 or B.sum() < 10:
        w0 = max(10, min(w, T))
        mu = G[:w0].mean(axis=0)
        return np.linalg.norm(G - mu[None, :], axis=-1).astype(np.float32)

    muA = G[A].mean(axis=0)
    muB = G[B].mean(axis=0)
    v = (muB - muA).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)

    # Signed projection centered at muA
    s = (G - muA[None, :]) @ v
    return s.astype(np.float32)


def _score_pi_entropy(df, zscore: bool = False) -> np.ndarray:
    if "entropy" not in df.columns:
        raise KeyError("Missing 'entropy' column in traj.")
    x = df["entropy"].to_numpy(dtype=np.float32)
    if not zscore:
        return x
    return (x - x.mean()) / (x.std() + 1e-6)


def _compute_lag(score: np.ndarray, df, L: int) -> Dict[str, Any]:
    regime = df["regime"].to_numpy(dtype=int)
    switches = df["switch"].to_numpy(dtype=int)
    return transition_lag_half_rise(score, regime, switches, L=L)


def _pick_L(P: int, L_user: int) -> int:
    # Critical: ensure L < P for toggle experiments.
    if P <= 0:
        return int(L_user)
    return int(min(L_user, max(2, P - 1)))


def _collect_run_dirs(root_dir: Path) -> List[Path]:
    if root_dir.is_dir() and (root_dir / "meta.json").exists():
        return [root_dir]
    if not root_dir.is_dir():
        raise FileNotFoundError(root_dir)
    # Search immediate children for run dirs
    return sorted([p for p in root_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True,
                    help="Either a single run_dir, or a parent dir containing multiple run dirs.")
    ap.add_argument("--periods", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--L", type=int, default=60, help="Will be clipped to <= P-1 automatically.")
    ap.add_argument("--signed_g", action="store_true", help="Use signed projection for g-score.")
    ap.add_argument("--entropy_z", action="store_true", help="Use z-scored entropy (default: raw entropy).")
    args = ap.parse_args()

    root = Path(args.root_dir)
    run_dirs = _collect_run_dirs(root)

    # Build index by period from meta.json
    byP: Dict[int, Path] = {}
    for rd in run_dirs:
        meta = _extract_meta(rd)
        P = int(meta.get("period", -1))
        if P in args.periods:
            byP[P] = rd

    # Header
    print("\n=== Switch-sweep lag table (console only) ===")
    print(f"root_dir: {root.resolve()}")
    print(f"periods:  {args.periods}")
    print(f"warmup:   {args.warmup}")
    print(f"L_user:   {args.L} (effective L = min(L_user, P-1))")
    print(f"g_score:  {'signed' if args.signed_g else 'unsigned'}")
    print(f"pi_score: {'entropy_z' if args.entropy_z else 'entropy_raw'}")
    print("")

    cols = [
        "P",
        "L",
        "n_g_up", "lag_g_up",
        "n_g_dn", "lag_g_dn",
        "lag_g_mean", "lag_g_mean/P",
        "n_pi_up", "lag_pi_up",
        "n_pi_dn", "lag_pi_dn",
        "lag_pi_mean", "lag_pi_mean/P",
    ]
    print(" ".join([f"{c:>12s}" for c in cols]))

    # Rows
    for P in args.periods:
        rd = byP.get(P, None)
        if rd is None:
            print(f"{P:>12d} {'-':>12s}  (missing run_dir for this period)")
            continue

        df = _load_table(_find_traj(rd))
        meta = _extract_meta(rd)
        Pm = int(meta.get("period", P))
        L_eff = _pick_L(Pm, args.L)

        s_g = _score_g(df, warmup=args.warmup, signed=args.signed_g)
        s_pi = _score_pi_entropy(df, zscore=args.entropy_z)

        lag_g = _compute_lag(s_g, df, L=L_eff)
        lag_pi = _compute_lag(s_pi, df, L=L_eff)

        # Extract stats
        g_up = lag_g["lag_up"]
        g_dn = lag_g["lag_dn"]
        pi_up = lag_pi["lag_up"]
        pi_dn = lag_pi["lag_dn"]

        lag_g_up_mean = None if g_up is None else float(g_up["mean"])
        lag_g_dn_mean = None if g_dn is None else float(g_dn["mean"])
        lag_pi_up_mean = None if pi_up is None else float(pi_up["mean"])
        lag_pi_dn_mean = None if pi_dn is None else float(pi_dn["mean"])

        # Mean of up/dn means (ignore None)
        g_means = [x for x in [lag_g_up_mean, lag_g_dn_mean] if x is not None]
        pi_means = [x for x in [lag_pi_up_mean, lag_pi_dn_mean] if x is not None]
        lag_g_mean = None if len(g_means) == 0 else float(np.mean(g_means))
        lag_pi_mean = None if len(pi_means) == 0 else float(np.mean(pi_means))

        # Normalized lag
        lag_g_norm = None if lag_g_mean is None else float(lag_g_mean / max(1, Pm))
        lag_pi_norm = None if lag_pi_mean is None else float(lag_pi_mean / max(1, Pm))

        row = [
            Pm, L_eff,
            (0 if g_up is None else int(g_up["n"])), lag_g_up_mean,
            (0 if g_dn is None else int(g_dn["n"])), lag_g_dn_mean,
            lag_g_mean, lag_g_norm,
            (0 if pi_up is None else int(pi_up["n"])), lag_pi_up_mean,
            (0 if pi_dn is None else int(pi_dn["n"])), lag_pi_dn_mean,
            lag_pi_mean, lag_pi_norm,
        ]

        # Print row with formatting
        print(
            f"{row[0]:>12d}"
            f"{row[1]:>12d}"
            f"{row[2]:>12d}{_fmt(row[3], 12)}"
            f"{row[4]:>12d}{_fmt(row[5], 12)}"
            f"{_fmt(row[6], 12)}{_fmt(row[7], 12)}"
            f"{row[8]:>12d}{_fmt(row[9], 12)}"
            f"{row[10]:>12d}{_fmt(row[11], 12)}"
            f"{_fmt(row[12], 12)}{_fmt(row[13], 12)}"
        )

    print("\nNotes:")
    print("- lag_* are half-rise times (steps) estimated around each switch.")
    print("- lag_mean is the average of (A->B mean) and (B->A mean), ignoring missing.")
    print("- lag_mean/P is the normalized lag (scale-free), good for showing time-scale dependence.")


if __name__ == "__main__":
    main()
