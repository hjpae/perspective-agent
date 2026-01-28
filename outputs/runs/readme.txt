// training script: script_spyder.py line 48, 

#%% Phase 2 - initial training (WITHOUT pygame viewer)
## 1. Slip only: zone0 volatile, zone2 stable
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from cear_pilot.training.train import main

if __name__ == "__main__":
    sys.argv = [
      str(Path(__file__).name),
      "--device","cpu",
      "--steps","40000",
      
      "--w_entropy","0.001",
      "--w_actor","0.25",
      "--actor_b","0.98",
      
      # "--use_slip",
      # "--p_slip","0.60","0.30","0.0",

      # "--view",
      # "--view_every", "2",
      # "--view_fps", "20",
      # "--view_cell_px", "42",
    ]
    main()
    
// ckpts | github demo: 20260109_144355 | AAAI paper: 20260127_215133 ... but used github demo anyways


// figure script: 
#%%
# script_switch_sweep_eval_spyder.py
# -*- coding: utf-8 -*-

from pathlib import Path
import os, sys, subprocess, time

# -----------------------
# 0) Spyder-safe setup
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def newest_run_dir() -> Path:
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError("No run dirs found")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_sleep():
    time.sleep(0.5)

# -----------------------
# 1) Checkpoint
# -----------------------
TRAIN_ID = "20260109_144355"   # <-- change if needed
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"
assert CKPT.exists(), f"Missing ckpt: {CKPT}"

# -----------------------
# 2) Experiment settings
# -----------------------
T_TOTAL = 400
WARMUP  = 150
PERIODS = [10, 20, 40, 80]

SIGMA_A = (0.60, 0.30, 0.05)
SIGMA_B = (0.05, 0.30, 0.60)

DEVICE = "cpu"
SEED   = "0"
GREEDY = True

# figure params
PRE_WINDOW = 80
ALPHA = 0.05
CONSEC = 3
L = 60
POLICY_SIGNAL = "entropy"

# -----------------------
# 3) Run sweep + figures
# -----------------------
for P in PERIODS:
    print("\n" + "=" * 80)
    print(f"=== period = {P} ===")

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
    
    # 1) Collect
    run_module("cear_pilot.experiments.run_switch_sweep", args_collect)
    safe_sleep()
    
    # 2) Detect run_dir
    after = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name not in before]
    run_dir = max(after, key=lambda p: p.stat().st_mtime) if after else newest_run_dir()
    
    # 3) Make figure
    run_module("cear_pilot.analysis.figure_switch_eval", [
        "--run_dir", str(run_dir),
        "--warmup", str(WARMUP),
        "--pre_window", str(PRE_WINDOW),
        "--alpha", str(ALPHA),
        "--consec", str(CONSEC),
        "--L", str(L),
        "--policy_signal", POLICY_SIGNAL,
    ])
    
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

    print(f"[OK] figure saved:")
    print(run_dir / "figs" / f"fig_switch_eval_{POLICY_SIGNAL}.png")


print("\nALL DONE.")
