# script_switch_sweep_eval_spyder.py
# -*- coding: utf-8 -*-

from pathlib import Path
import os, sys, subprocess, time


# -----------------------
# 0) Make execution robust in Spyder
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CWD:", Path.cwd())

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def newest_run_dir() -> Path:
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found in {RUNS_DIR}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_sleep():
    time.sleep(0.7)


# -----------------------
# 1) Checkpoint
# -----------------------
TRAIN_ID = "20260109_144355"   # <-- change if needed
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"
if not CKPT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

print("\nCKPT:", CKPT)


# -----------------------
# 2) Experiment settings
# -----------------------
T_TOTAL  = 400
WARMUP   = 150
PERIODS  = [10, 20, 40, 80]

SIGMA_A = (0.60, 0.30, 0.05)
SIGMA_B = (0.05, 0.30, 0.60)

DEVICE = "cpu"
SEED   = "0"
GREEDY = True

# figure_switch_eval params
PRE_WINDOW = 80
ALPHA = 0.05
CONSEC = 3
L = 60                 # will be clipped internally to min(L, P-1)
POLICY_SIGNAL = "entropy"


# -----------------------
# 3) Run sweep + generate figures
# -----------------------
results = []

for P in PERIODS:
    print("\n" + "=" * 80)
    print(f"=== Switch-sweep: period={P} (T={T_TOTAL}, warmup={WARMUP}) ===")

    before = set(p.name for p in RUNS_DIR.iterdir() if p.is_dir())

    args_collect = [
        "--ckpt", str(CKPT),
        "--device", DEVICE,
        "--seed", SEED,
        "--T", str(T_TOTAL),
        "--warmup", str(WARMUP),
        "--period", str(P),
        "--sigma_A", str(SIGMA_A[0]), str(SIGMA_A[1]), str(SIGMA_A[2]),
        "--sigma_B", str(SIGMA_B[0]), str(SIGMA_B[1]), str(SIGMA_B[2]),
        "--max_steps", str(T_TOTAL),
    ]
    if GREEDY:
        args_collect.append("--greedy")

    run_module("cear_pilot.analysis.figure_switch_eval", [
        "--run_dir", str(run_dir),
        "--warmup", str(WARMUP),
        "--pre_window", str(PRE_WINDOW),
        "--alpha", str(ALPHA),
        "--consec", str(CONSEC),
        "--L", "60",
        "--policy_signal", "entropy",
    ])

    print(f"[DONE] period={P}")
    print(f"  fig : {run_dir / 'figs' / f'fig_switch_eval_{POLICY_SIGNAL}.png'}")
    print(f"  json: {run_dir / 'switch_eval.json'}")

    results.append((P, run_dir))


# -----------------------
# 4) Console lag table (summary)
# -----------------------
print("\n" + "=" * 80)
print("FINAL LAG SUMMARY TABLE")

args_table = [
    "--root_dir", str(RUNS_DIR),
    "--periods", *[str(p) for p in PERIODS],
    "--warmup", str(WARMUP),
    "--L", str(L),
    "--signed_g",
]
run_module("cear_pilot.analysis.print_switch_lag_table", args_table)

print("\nALL DONE.")




---

# cear_pilot/experiments/run_switch_sweep.py
# -*- coding: utf-8 -*-
"""
Run one long rollout with regime switching AFTER warmup.
Saves traj.(parquet|csv) with columns:
  t, regime, switch, g_*, pi_max, entropy, margin, (optional logits_*)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_save_table(rows, out_path: Path) -> Path:
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        p = out_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[idx] = 1.0
    return v


def build_agent_from_meta(meta: Dict[str, Any], device: str, max_steps_override=None):
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if max_steps_override is not None:
        env_cfg.max_steps = int(max_steps_override)
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)
    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    agent_cfg.encoder.__dict__.update(enc)
    agent_cfg.world.__dict__.update(world)
    agent_cfg.state.__dict__.update(state)
    agent_cfg.policy.__dict__.update(pol)

    agent = CEARAgent(agent_cfg)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder, env


def set_env_zone_sigma(env: NZoneGridEnv, sigma_triplet: Tuple[float, float, float]) -> None:
    env._zone_sigma = np.array(list(sigma_triplet), dtype=np.float32)


def policy_stats_from_logits(logits: np.ndarray) -> Tuple[float, float, float]:
    # logits: (A,)
    ex = np.exp(logits - np.max(logits))
    p = ex / (np.sum(ex) + 1e-12)
    p_sorted = np.sort(p)[::-1]
    pi_max = float(p_sorted[0])
    margin = float(p_sorted[0] - (p_sorted[1] if len(p_sorted) > 1 else 0.0))
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    return pi_max, ent, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")

    # timeline
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--period", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=400)

    # regimes: sigma only (A/B)
    ap.add_argument("--sigma_A", type=float, nargs=3, default=(0.60, 0.30, 0.05))
    ap.add_argument("--sigma_B", type=float, nargs=3, default=(0.05, 0.30, 0.60))

    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    device = args.device
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=device, max_steps_override=args.T)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(device).eval()
    decoder.to(device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    run_meta = {
        "mode": "switch_sweep",
        "ckpt": str(Path(args.ckpt).resolve()),
        "seed": args.seed,
        "device": args.device,
        "greedy": bool(args.greedy),
        "T": int(args.T),
        "warmup": int(args.warmup),
        "period": int(args.period),
        "sigma_A": list(map(float, args.sigma_A)),
        "sigma_B": list(map(float, args.sigma_B)),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    # reset
    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    agent.reset(batch_size=1)
    last_action = 4
    n_actions = int(env.action_space.n)

    rows: List[Dict[str, Any]] = []

    # start in regime A
    regime = 0  # 0=A, 1=B
    set_env_zone_sigma(env, tuple(args.sigma_A))

    for t_global in range(int(args.T)):
        # schedule switching after warmup
        switched = 0
        if t_global >= int(args.warmup):
            k = (t_global - int(args.warmup)) // max(1, int(args.period))
            new_regime = int(k % 2)  # 0,1,0,1,...
            if new_regime != regime:
                regime = new_regime
                switched = 1
                set_env_zone_sigma(env, tuple(args.sigma_A) if regime == 0 else tuple(args.sigma_B))

        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=False)

        a_int = int(action.item())
        obs_next, _, terminated, truncated, info2 = env.step(a_int)

        g = out["g"].squeeze(0).detach().cpu().numpy()
        logits = out["logits"].squeeze(0).detach().cpu().numpy()  # policy logits (from s)

        pi_max, ent, margin = policy_stats_from_logits(logits)

        row = {
            "t": int(info2["t"]),
            "t_global": int(t_global),
            "regime": int(regime),
            "switch": int(switched),
            "a": int(a_int),
            "pi_max": float(pi_max),
            "entropy": float(ent),
            "margin": float(margin),
            "zone_id": int(info2.get("zone_id", -1)),
            "x": int(info2.get("x", -1)),
            "y": int(info2.get("y", -1)),
        }
        for i, v in enumerate(g.tolist()):
            row[f"g_{i}"] = float(v)

        rows.append(row)

        obs = obs_next
        last_action = a_int

        if terminated or truncated:
            break

    out_path = try_save_table(rows, run_dir / "traj")
    print(f"[OK] Saved traj: {out_path}")
    print(f"[OK] Run dir: {run_dir}")


if __name__ == "__main__":
    main()


--- 

# cear_pilot/analysis/figure_switch_eval.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import detect_delay_quantile, hysteresis_area, transition_lag_half_rise


def load_table(traj_path: Path):
    import pandas as pd
    if traj_path.suffix == ".parquet":
        return pd.read_parquet(traj_path)
    return pd.read_csv(traj_path)


def find_traj(run_dir: Path) -> Path:
    for ext in [".parquet", ".csv"]:
        p = run_dir / f"traj{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No traj.parquet/csv in {run_dir}")


# def g_score_from_df(df, warmup: int) -> np.ndarray:
#     g_cols = [c for c in df.columns if c.startswith("g_")]
#     if len(g_cols) == 0:
#         raise ValueError("No g_* columns found in traj.")
#     G = df[g_cols].to_numpy(dtype=np.float32)

#     # Use warmup mean as baseline (pre-mean)
#     w = max(10, int(warmup))
#     w = min(w, G.shape[0])
#     mu = G[:w].mean(axis=0)
#     return np.linalg.norm(G - mu[None, :], axis=-1).astype(np.float32)


def g_signed_score_from_df(df, warmup: int, regime: np.ndarray, buffer: int = 2) -> np.ndarray:
    g_cols = [c for c in df.columns if c.startswith("g_")]
    G = df[g_cols].to_numpy(dtype=np.float32)
    T = G.shape[0]

    # Use only post-warmup points, and exclude a small buffer around switches if desired
    idx = np.arange(T)
    post = idx >= int(warmup)

    # Regime masks in post-warmup
    A = post & (regime == 0)
    B = post & (regime == 1)

    # Fallback if one side is empty
    if A.sum() < 10 or B.sum() < 10:
        mu = G[:max(10, min(int(warmup), T))].mean(axis=0)
        return (G - mu[None, :]).sum(axis=-1).astype(np.float32)

    muA = G[A].mean(axis=0)
    muB = G[B].mean(axis=0)

    w = (muB - muA).astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-8)

    # Signed projection (centered at muA)
    s = (G - muA[None, :]) @ w
    return s.astype(np.float32)


def _segment_boundaries(t: np.ndarray, switch_times_idx: np.ndarray, warmup_t: int) -> List[int]:
    # Build boundaries in "time" coordinates, not indices
    t_arr = t.astype(int)

    b: List[int] = [int(warmup_t)]
    for idx in switch_times_idx.tolist():
        b.append(int(t_arr[idx]))

    b = sorted(set(b))
    b.append(int(t_arr[-1]) + 1)
    return b


def _shade_regimes(ax, t: np.ndarray, regime: np.ndarray, switch_times_idx: np.ndarray, warmup_t: int) -> None:
    # Shade warmup region
    t_arr = t.astype(int)
    ax.axvspan(int(t_arr[0]), int(warmup_t), alpha=0.08)

    # Shade post-warmup regimes as alternating bands
    boundaries = _segment_boundaries(t_arr, switch_times_idx, warmup_t)

    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        if b <= a:
            continue

        # Get regime label at the start boundary a
        idx_a = int(np.searchsorted(t_arr, a, side="left"))
        idx_a = min(max(idx_a, 0), len(regime) - 1)
        r = int(regime[idx_a])

        # Slightly different opacity for A vs B
        band_alpha = 0.06 if r == 0 else 0.10
        ax.axvspan(a, b, alpha=band_alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--pre_window", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--consec", type=int, default=3)
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--policy_signal", type=str, default="entropy", choices=["entropy", "pi_max", "margin"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_table(find_traj(run_dir))

    # Prefer t_global if present (collector should store it)
    t = df["t_global"].to_numpy(dtype=int) if "t_global" in df.columns else df["t"].to_numpy(dtype=int)

    if "regime" not in df.columns or "switch" not in df.columns:
        raise KeyError("traj must contain 'regime' and 'switch' columns. Re-run the collector.")

    regime = df["regime"].to_numpy(dtype=int)
    switches = df["switch"].to_numpy(dtype=int)

    # Read run meta (optional)
    meta_path = run_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        P = int(meta.get("period", -1))
        W = int(meta.get("warmup", args.warmup))
        T = int(meta.get("T", -1))
    else:
        P, W, T = -1, int(args.warmup), -1
    
    # After reading P from meta
    if P > 0:
        args.L = min(int(args.L), max(2, P - 1))

    # --- scores
    s_g = g_signed_score_from_df(df, warmup=args.warmup, regime=regime)

    if args.policy_signal not in df.columns:
        raise KeyError(f"Missing policy signal column: {args.policy_signal}")
    s_pi_raw = df[args.policy_signal].to_numpy(dtype=np.float32)

    # Normalize policy signal to a comparable scale (z-score)
    s_pi = (s_pi_raw - s_pi_raw.mean()) / (s_pi_raw.std() + 1e-6)

    # Switch indices (after warmup)
    switch_times = np.where((switches == 1) & (t >= int(args.warmup)))[0]

    # --- A) detection delay per switch
    delays_g: List[int] = []
    delays_pi: List[int] = []

    for idx in switch_times:
        sw_t = int(idx)

        dg = detect_delay_quantile(
            score=s_g,
            switch_t=sw_t,
            pre_window=args.pre_window,
            alpha=args.alpha,
            consec=args.consec,
        )
        dp = detect_delay_quantile(
            score=s_pi,
            switch_t=sw_t,
            pre_window=args.pre_window,
            alpha=args.alpha,
            consec=args.consec,
        )

        if dg is not None:
            delays_g.append(int(dg))
        if dp is not None:
            delays_pi.append(int(dp))

    # --- B) hysteresis area
    hyst_g = hysteresis_area(s_g, regime, switches, L=args.L)
    hyst_pi = hysteresis_area(s_pi, regime, switches, L=args.L)
    
    # --- B2) transition lag (half-rise time)
    lag_g = transition_lag_half_rise(s_g, regime, switches, L=args.L)
    lag_pi = transition_lag_half_rise(s_pi, regime, switches, L=args.L)

    # --- summary
    def summarize(x: List[int]) -> Optional[Dict[str, float]]:
        if len(x) == 0:
            return None
        return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x))}

    out: Dict[str, Any] = {
        "delay_g": summarize(delays_g),
        "delay_pi": summarize(delays_pi),
        "hysteresis_g": {"area": hyst_g["area"], "n_up": hyst_g["n_up"], "n_dn": hyst_g["n_dn"]},
        "hysteresis_pi": {"area": hyst_pi["area"], "n_up": hyst_pi["n_up"], "n_dn": hyst_pi["n_dn"]},
        "lag_g": {"up": lag_g["lag_up"], "dn": lag_g["lag_dn"], "L": lag_g["L"]},
        "lag_pi": {"up": lag_pi["lag_up"], "dn": lag_pi["lag_dn"], "L": lag_pi["L"]},
        "policy_signal": args.policy_signal,
        "params": vars(args),
        "meta": {"period": P, "warmup": W, "T": T},
    }
    print(out)

    # --- plot
    figdir = run_dir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 7))

    # (1) score time series + regime shading + switch markers
    ax1 = plt.subplot(2, 1, 1)

    _shade_regimes(ax1, t=t, regime=regime, switch_times_idx=switch_times, warmup_t=int(args.warmup))

    ax1.plot(t, s_g, label="g_score")
    ax1.plot(t, s_pi, label=f"{args.policy_signal}_z")

    for sw in switch_times:
        ax1.axvline(int(t[sw]), linewidth=0.7, alpha=0.35)

    # Period label
    ax1.text(
        0.01, 0.90,
        #f"lag_g(up/dn)={lag_g['lag_up']} / {lag_g['lag_dn']}\n"
        #f"lag_pi(up/dn)={lag_pi['lag_up']} / {lag_pi['lag_dn']}",
        transform=ax1.transAxes,
        ha="left", va="top", fontsize=9, alpha=0.9
    )

    ax1.set_title(f"Scores + regime shading | P={P}  warmup={W}  T={T}  (policy={args.policy_signal})")
    ax1.set_xlabel("t")
    ax1.legend()

    # (2) hysteresis mean curves (g)
    ax2 = plt.subplot(2, 2, 3)
    if hyst_g["m_up"] is not None:
        ax2.plot(hyst_g["m_up"], label="A->B")
    if hyst_g["m_dn"] is not None:
        ax2.plot(hyst_g["m_dn"], label="B->A")
    ax2.set_title(f"g hysteresis (area={hyst_g['area']})")
    ax2.set_xlabel("tau")
    ax2.legend()

    # (3) hysteresis mean curves (policy)
    ax3 = plt.subplot(2, 2, 4)
    if hyst_pi["m_up"] is not None:
        ax3.plot(hyst_pi["m_up"], label="A->B")
    if hyst_pi["m_dn"] is not None:
        ax3.plot(hyst_pi["m_dn"], label="B->A")
    ax3.set_title(f"{args.policy_signal} hysteresis (area={hyst_pi['area']})")
    ax3.set_xlabel("tau")
    ax3.legend()

    out_png = figdir / f"fig_switch_eval_{args.policy_signal}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}")

    # Save json summary
    (run_dir / "switch_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



    ----# cear_pilot/analysis/print_switch_lag_table.py
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
