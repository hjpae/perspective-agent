#%% Phase 1 - initial training (NO pygame viewer - quick training) 
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
        "--device", "cpu",          # or "cuda"
        "--steps", "80000",
        # "--view"  # NOT set
    ]
    main()

#%% Phase 1 - initial training (WITH pygame viewer)
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
        "--device", "cpu",          # or "cuda"
        "--steps", "20000",
        
        "--w_entropy", "0.0", 
        "--w_actor", "0.5",

        "--view",
        "--view_every", "2",
        "--view_fps", "20",
        "--view_cell_px", "42",
    ]
    main()

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

#%% testing/zone comparison for demo purpose (g for eval index)
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from cear_pilot.testing import main

if __name__ == "__main__":
    sys.argv = [
      str(Path(__file__).name),
      "--device","cpu",
      "--steps","40000",
      
      "--ckpt", "outputs/runs/20260109_144355/ckpt.pt",
      "--seed", "0",
      "--steps", "240",
      "--sigmas", "0.60,0.30,0.05", "0.05,0.30,0.60", "0.30,0.30,0.30",
      
    ]
    main()

#%% Phase 2 - initial training (WITHOUT pygame viewer)
## 2. Drift only 
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
      "--device", "cpu",
      "--steps", "20000",
    
      "--use_drift",
      "--p_drift", "0.0", "0.0", "0.40",
      "--drift_vec", "0","0",  "0","0",  "1","0",   # only zone2 wind to +x
    ]
    main()

#%% Phase 2 - initial training (WITHOUT pygame viewer)
## 3. volatility with drift: zone0 volatile, zone2 stable
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
      "--device", "cpu",
      "--steps", "20000",
    
      "--use_drift",
      "--p_drift", "0.0", "0.0", "0.40",
      "--drift_vec", "0","0",  "0","0",  "1","0",
    
      "--use_volatility",
      "--volatile_zone", "0",
      "--volatile_period", "40",
      "--volatile_strength", "0.5",   # flip/rotate probability 
    ]
    main()

    
#%% Phase 2 - initial training (WITHOUT pygame viewer)
## 4. Hazard only 
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
      "--device", "cpu",
      "--steps", "20000",
    
      "--use_hazard",
      "--hazard_mode", "sensor_blackout",
      "--p_hazard", "0.0", "0.0", "0.05",
      "--hazard_blackout_steps", "8",
    ]
    main()


#%% Phase 2 - initial training (WITH pygame viewer)
## 2. Slip only: zone0 volatile, zone2 stable
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
      "--device", "cpu",
      "--steps", "20000",
    
      "--use_slip",
      "--p_slip", "0.0", "0.0", "0.35",   # z0,z1,z2
      "--volatile_zone", "0",
      "--volatile_period", "40",
      "--volatile_strength", "0.0",        # volatility off
      
    "--view",
    "--view_every", "2",
    "--view_fps", "20",
    "--view_cell_px", "42",
      
    ]
    main()
#%% Analysis: one script for all (root) | one-on-one sanity check, with B) Ablation 
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
    time.sleep(0.6)

# -----------------------
# 1) Point to your trained checkpoint
# -----------------------
TRAIN_ID = "20260109_144355"   # <-- ckpt run id
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"
if not CKPT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

# -----------------------
# A) Collect (g ON) + Figure A
# -----------------------
print("\n=== A) Collect (g ON) ===")
run_module("cear_pilot.experiments.run_collect", [
    "--ckpt", str(CKPT),
    "--episodes", "30",
    "--greedy",
])
safe_sleep()
collect_run = newest_run_dir()
print("Detected collect run:", collect_run)

print("\n=== A) Embed + Figure A ===")
run_module("cear_pilot.analysis.embed_latents", [
    "--run_dir", str(collect_run),
])
run_module("cear_pilot.analysis.figure_attractor", [
    "--run_dir", str(collect_run),
    "--lines",
])
print("Figure A done for:", collect_run)


# -----------------------
# B) Ablation + Figure B
# -----------------------
print("\n=== B) Ablation (g ON vs g OFF) ===")
run_module("cear_pilot.experiments.run_ablation", [
    "--ckpt", str(CKPT),
    "--episodes", "30",
    "--greedy",
])
safe_sleep()
ablation_run = newest_run_dir()
print("Detected ablation run:", ablation_run)
#print([p.name for p in (Path("outputs/runs").iterdir()) if p.is_dir()][-10:])

# Many implementations store subruns like:
#   outputs/runs/<ablation_run>/g_on/
#   outputs/runs/<ablation_run>/g_off/
g_on_dir = ablation_run / "g_on"
g_off_dir = ablation_run / "g_off"

if g_on_dir.exists() and g_off_dir.exists():
    print("Found g_on/g_off subdirs inside ablation run.")
    run_module("cear_pilot.analysis.embed_latents", ["--run_dir", str(g_on_dir)])
    run_module("cear_pilot.analysis.embed_latents", ["--run_dir", str(g_off_dir)])
    run_module("cear_pilot.analysis.figure_ablation", ["--root_dir", str(ablation_run)])
else:
    # Fallback: some implementations create two separate runs.
    # In that case, you can manually set the directories here or adjust detection logic.
    print("Warning! Did not find g_on/g_off subdirs.")
    print("Contents of ablation_run:", [p.name for p in ablation_run.iterdir()])
    print("If your ablation implementation creates separate runs, tell me the folder layout and I'll patch this.")

print("Figure B (if generated) should be under:", ablation_run / "figs")


# -----------------------
# C) Perturbation + Figure C
# -----------------------
print("\n=== C) Perturbation run ===")
run_module("cear_pilot.experiments.run_perturb", [
    "--ckpt", str(CKPT),
    "--t_perturb", "80",
    "--kind", "shock",
    "--scale", "1.0",
    "--greedy",
])
safe_sleep()
perturb_run = newest_run_dir()
print("Detected perturb run:", perturb_run)

print("\n=== C) Figure C ===")
run_module("cear_pilot.analysis.figure_perturb", [
    "--run_dir", str(perturb_run),
])

print("\n ALL DONE.")
print("Figure A:", collect_run / "figs")
print("Figure B:", ablation_run / "figs")
print("Figure C:", perturb_run / "figs")
#%% Analysis: sigma demo script (root) | ONE trajectory per env + shared PCA + overlay
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
    time.sleep(0.6)

# -----------------------
# 1) Point to trained checkpoint
# -----------------------
TRAIN_ID = "20260109_144355"   # <-- ckpt run id
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"
if not CKPT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

# -----------------------
# 2) Sigma conditions (A / flip / flat)
# -----------------------
SIGMA_A    = (0.60, 0.30, 0.05)
SIGMA_FLIP = (0.05, 0.30, 0.60)
SIGMA_FLAT = (0.30, 0.30, 0.30)

CONDITIONS = [
    ("A",    SIGMA_A,    False),
    ("flip", SIGMA_FLIP, True),
    ("flat", SIGMA_FLAT, True),
]

# where we save the reference action list
ACTIONS_REF = PROJECT_ROOT / "outputs" / "tests_sigma_demo" / "actions_ref.json"
ACTIONS_REF.parent.mkdir(parents=True, exist_ok=True)

# -----------------------
# 3) Make actions_ref once (baseline greedy rollout)
# -----------------------
print("\n=== (0) Build actions_ref from baseline (A sigma) ===")
run_module("cear_pilot.testing", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--seed", "0",
    "--steps", "240",
    "--sigmas",
    f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",
    f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",  # dummy second (not used)
    "--outdir", str(ACTIONS_REF.parent),
])

if not ACTIONS_REF.exists():
    raise FileNotFoundError(f"actions_ref not found at: {ACTIONS_REF}")
print("actions_ref:", ACTIONS_REF)
safe_sleep()

# -----------------------
# 4) Collect 30 episodes + shared-PCA embed + per-run attractor plot
# -----------------------
collect_runs = {}

for name, sigma, use_replay in CONDITIONS:
    print(f"\n=== A) Collect ONE episode ({name}) sigma={sigma} replay={use_replay} ===")

    args = [
        "--ckpt", str(CKPT),
        "--episodes", "30",
        "--greedy",
        "--zone_sigma", str(sigma[0]), str(sigma[1]), str(sigma[2]),
    ]
    if use_replay:
        args += ["--replay_actions", str(ACTIONS_REF)]

    run_module("cear_pilot.experiments.run_collect", args)
    safe_sleep()

    run_dir = newest_run_dir()
    collect_runs[name] = run_dir
    print(f"Detected collect run ({name}):", run_dir)

    print(f"\n=== A) Embed (shared PCA) ({name}) ===")
    if name == "A":
        # Fit PCA on baseline A
        run_module("cear_pilot.analysis.embed_latents", ["--run_dir", str(run_dir)])
    else:
        # Reuse PCA from A
        run_module("cear_pilot.analysis.embed_latents", [
            "--run_dir", str(run_dir),
            "--pca_fit_dir", str(collect_runs["A"]),
        ])

    # Optional: save per-run attractor figure
    run_module("cear_pilot.analysis.figure_attractor", ["--run_dir", str(run_dir), "--lines"])
    print(f"Attractor figure done ({name}):", run_dir / "figs")

# -----------------------
# 5) Overlay the ONE-episode trajectories from 3 envs into ONE figure
# -----------------------
print("\n=== B) Overlay ONE-episode PCA(g) trajectories (A vs flip vs flat) ===")
run_module("cear_pilot.analysis.overlay_one_traj", [
    "--runs",
    f"A={collect_runs['A']}",
    f"flip={collect_runs['flip']}",
    f"flat={collect_runs['flat']}",
    "--episode", "29",
    "--connect",
])
print("Overlay figure saved under:", collect_runs["A"] / "figs")

# -----------------------
# 6) Perturb + recovery per condition (optional: keep as-is)
# -----------------------
perturb_runs = {}

for name, sigma, use_replay in CONDITIONS:
    print(f"\n=== C) Perturb ({name}) sigma={sigma} replay={use_replay} ===")
    args = [
        "--ckpt", str(CKPT),
        "--t_perturb", "80",
        "--kind", "shock",
        "--scale", "1.0",
        "--greedy",
        "--zone_sigma", str(sigma[0]), str(sigma[1]), str(sigma[2]),
    ]
    if use_replay:
        args += ["--replay_actions", str(ACTIONS_REF)]

    run_module("cear_pilot.experiments.run_perturb", args)
    safe_sleep()

    run_dir = newest_run_dir()
    perturb_runs[name] = run_dir
    print(f"Detected perturb run ({name}):", run_dir)

    print(f"\n=== C) Figure perturb ({name}) ===")
    run_module("cear_pilot.analysis.figure_perturb", ["--run_dir", str(run_dir)])
    print(f"Perturb figure done ({name}):", run_dir / "figs")

# -----------------------
# 7) Summary
# -----------------------
print("\n=== ALL DONE ===")
print("Actions ref:", ACTIONS_REF)

print("\nCollect runs:")
for k, v in collect_runs.items():
    print(" ", k, "->", v / "figs")

print("\nPerturb runs:")
for k, v in perturb_runs.items():
    print(" ", k, "->", v / "figs")

#%% Demo: regime-switch (A -> flip) with action replay
from pathlib import Path
import os, sys, subprocess, time

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def newest_run_dir() -> Path:
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime)

def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def safe_sleep(): time.sleep(0.6)

TRAIN_ID = "20260109_144355"
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"

SIGMA_A    = (0.60, 0.30, 0.05)
SIGMA_FLIP = (0.05, 0.30, 0.60)

ACTIONS_REF = PROJECT_ROOT / "outputs" / "tests_sigma_demo" / "actions_ref.json"
ACTIONS_REF.parent.mkdir(parents=True, exist_ok=True)

# 1) Build one action sequence in baseline A (greedy)
run_module("cear_pilot.testing", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--seed", "0",
    "--steps", "240",
    "--sigmas", f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",
               f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",
    "--outdir", str(ACTIONS_REF.parent),
])
assert ACTIONS_REF.exists()

safe_sleep()

# 2) Collect ONE episode with action replay, but switch sigma at t_switch
T_SWITCH = 80

run_module("cear_pilot.experiments.run_collect", [
    "--ckpt", str(CKPT),
    "--episodes", "1",
    "--greedy",
    "--replay_actions", str(ACTIONS_REF),
    "--zone_sigma", str(SIGMA_A[0]), str(SIGMA_A[1]), str(SIGMA_A[2]),
    "--t_switch", str(T_SWITCH),
    "--zone_sigma2", str(SIGMA_FLIP[0]), str(SIGMA_FLIP[1]), str(SIGMA_FLIP[2]),
])

safe_sleep()
run_dir = newest_run_dir()
print("Run:", run_dir)

# # 3) Optional: embed latents (not required for switch figure)
# run_module("cear_pilot.analysis.embed_latents", ["--run_dir", str(run_dir)])

# 4) make the switch figure
run_module("cear_pilot.analysis.figure_switch", [
    "--run_dir", str(run_dir),
    "--episode", "0",
    "--t_switch", str(T_SWITCH),
])
print("Saved:", run_dir / "figs" / "fig_switch.png")

# 5) make the switch-perturb figure
run_module("cear_pilot.experiments.run_switch_perturb", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--steps", "240",
    "--greedy",
    "--replay_actions", str(ACTIONS_REF),
    "--zone_sigma", str(SIGMA_A[0]), str(SIGMA_A[1]), str(SIGMA_A[2]),
    "--t_switch", str(T_SWITCH),
    "--zone_sigma2", str(SIGMA_FLIP[0]), str(SIGMA_FLIP[1]), str(SIGMA_FLIP[2]),
    "--t_perturb", "120",
    "--scale", "1.0",
])

safe_sleep()
run_dir = newest_run_dir()
print("Run:", run_dir)


run_module("cear_pilot.analysis.figure_switch_perturb", [
    "--run_dir", str(run_dir),
    "--shade_after_switch",
])
print("Saved:", run_dir / "figs" / "fig_switch.png")

#%% Demo: switch sweep + hysteresis (root)
from pathlib import Path
import os, sys, subprocess, time

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def newest_run_dir() -> Path:
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime)

def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def safe_sleep(): time.sleep(0.6)

TRAIN_ID = "20260109_144355"
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"

SIGMA_A    = (0.60, 0.30, 0.05)
SIGMA_FLIP = (0.05, 0.30, 0.60)

ACTIONS_REF = PROJECT_ROOT / "outputs" / "tests_sigma_demo" / "actions_ref.json"
ACTIONS_REF.parent.mkdir(parents=True, exist_ok=True)

# 0) Build one action sequence in baseline A (greedy)
run_module("cear_pilot.testing", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--seed", "0",
    "--steps", "240",
    "--sigmas", f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}", f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",
    "--outdir", str(ACTIONS_REF.parent),
])
assert ACTIONS_REF.exists()
safe_sleep()

# 1) Frequency sweep: A<->B toggle with different periods
periods = [5, 10, 20, 80]   # fast -> slow
for P in periods:
    run_module("cear_pilot.experiments.run_switch_sweep", [
        "--ckpt", str(CKPT),
        "--device", "cpu",
        "--seed", "0",
        "--steps", "240",
        "--greedy",
        "--replay_actions", str(ACTIONS_REF),
        "--zone_sigma", str(SIGMA_A[0]), str(SIGMA_A[1]), str(SIGMA_A[2]),
        "--zone_sigma2", str(SIGMA_FLIP[0]), str(SIGMA_FLIP[1]), str(SIGMA_FLIP[2]),
        "--pattern", "toggle",
        "--t0", "0",
        "--period", str(P),
    ])
    safe_sleep()
    run_dir = newest_run_dir()
    run_module("cear_pilot.analysis.figure_switch_sweep", ["--run_dir", str(run_dir), "--save_name", f"fig_toggle_P{P}.png"])
    print("Saved figs:", run_dir / "figs")
    safe_sleep()

# 2) Hysteresis: A -> B -> A
run_module("cear_pilot.experiments.run_switch_sweep", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--seed", "0",
    "--steps", "240",
    "--greedy",
    "--replay_actions", str(ACTIONS_REF),
    "--zone_sigma", str(SIGMA_A[0]), str(SIGMA_A[1]), str(SIGMA_A[2]),
    "--zone_sigma2", str(SIGMA_FLIP[0]), str(SIGMA_FLIP[1]), str(SIGMA_FLIP[2]),
    "--pattern", "hysteresis",
    "--t_switch", "80",
    "--t_back", "160",
])
safe_sleep()
run_dir = newest_run_dir()
run_module("cear_pilot.analysis.figure_switch_sweep", ["--run_dir", str(run_dir), "--save_name", "fig_hysteresis.png"])
print("Saved figs:", run_dir / "figs")


#%%
# run_gate_demo.py
from pathlib import Path
import os, sys, subprocess, time

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def newest_run_dir() -> Path:
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not dirs:
        raise RuntimeError(f"No run dirs found under: {RUNS_DIR}")
    return max(dirs, key=lambda p: p.stat().st_mtime)

def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def safe_sleep(): time.sleep(0.6)

# -----------------------------
# Config
# -----------------------------
TRAIN_ID = "20260109_144355"
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"

SIGMA_A    = (0.60, 0.30, 0.05)
SIGMA_FLIP = (0.05, 0.30, 0.60)

ACTIONS_REF = PROJECT_ROOT / "outputs" / "tests_sigma_demo" / "actions_ref.json"
ACTIONS_REF.parent.mkdir(parents=True, exist_ok=True)

# Gate figure config
GATE_EPISODE = 0
GATE_ALPHA   = 0.05
GATE_K_ON    = 8
GATE_K_OFF   = 4
GATE_POLICY  = "entropy"   # entropy / margin / pi_max ...

# -----------------------------
# 0) Build one action sequence in baseline A (greedy)
# -----------------------------
run_module("cear_pilot.testing", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--seed", "0",
    "--steps", "240",
    "--sigmas",
        f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",
        f"{SIGMA_A[0]},{SIGMA_A[1]},{SIGMA_A[2]}",
    "--outdir", str(ACTIONS_REF.parent),
])
assert ACTIONS_REF.exists(), f"Missing: {ACTIONS_REF}"
safe_sleep()

# -----------------------------
# 1) Frequency sweep: A<->B toggle with different periods
# -----------------------------
periods = [5, 10, 20, 80]   # fast -> slow
for P in periods:
    run_module("cear_pilot.experiments.run_switch_sweep", [
        "--ckpt", str(CKPT),
        "--device", "cpu",
        "--seed", "0",
        "--steps", "240",
        "--greedy",
        "--replay_actions", str(ACTIONS_REF),
        "--zone_sigma",  str(SIGMA_A[0]),   str(SIGMA_A[1]),   str(SIGMA_A[2]),
        "--zone_sigma2", str(SIGMA_FLIP[0]),str(SIGMA_FLIP[1]),str(SIGMA_FLIP[2]),
        "--pattern", "toggle",
        "--t0", "0",
        "--period", str(P),
    ])
    safe_sleep()

    run_dir = newest_run_dir()

    run_module("cear_pilot.analysis.figure_switch_sweep", [
        "--run_dir", str(run_dir),
        "--save_name", f"fig_toggle_P{P}.png",
    ])

    run_module("cear_pilot.analysis.figure_gate", [
        "--run_dir", str(run_dir),
        "--episode", str(GATE_EPISODE),
        "--use_robust_thr",
        "--ema_alpha", str(GATE_ALPHA),
        "--k_on", str(GATE_K_ON),
        "--k_off", str(GATE_K_OFF),
        "--policy_signal", str(GATE_POLICY),
        "--outname", f"fig_gate_toggle_P{P}.png",
        "--title", f"Gate demo (toggle, P={P})",
    ])

    print("Saved figs:", run_dir / "figs")
    safe_sleep()

# -----------------------------
# 2) Hysteresis: A -> B -> A
# -----------------------------
run_module("cear_pilot.experiments.run_switch_sweep", [
    "--ckpt", str(CKPT),
    "--device", "cpu",
    "--seed", "0",
    "--steps", "240",
    "--greedy",
    "--replay_actions", str(ACTIONS_REF),
    "--zone_sigma",  str(SIGMA_A[0]),   str(SIGMA_A[1]),   str(SIGMA_A[2]),
    "--zone_sigma2", str(SIGMA_FLIP[0]),str(SIGMA_FLIP[1]),str(SIGMA_FLIP[2]),
    "--pattern", "hysteresis",
    "--t_switch", "80",
    "--t_back", "160",
])
safe_sleep()

run_dir = newest_run_dir()

run_module("cear_pilot.analysis.figure_switch_sweep", [
    "--run_dir", str(run_dir),
    "--save_name", "fig_hysteresis.png",
])

run_module("cear_pilot.analysis.figure_gate", [
    "--run_dir", str(run_dir),
    "--episode", str(GATE_EPISODE),
    "--use_robust_thr",
    "--ema_alpha", str(GATE_ALPHA),
    "--k_on", str(GATE_K_ON),
    "--k_off", str(GATE_K_OFF),
    "--policy_signal", str(GATE_POLICY),
    "--outname", "fig_gate_hysteresis.png",
    "--title", "Gate demo (hysteresis)",
])

print("Saved figs:", run_dir / "figs")

#%%
# script_switch_sweep_eval_spyder.py
# -*- coding: utf-8 -*-
"""
Spyder-friendly script:
- Run switch-sweep (T=400, warmup=150) for periods: 10/20/40/80
- For each run: compute (A) detection delay and (B) hysteresis area via figure_switch_eval.py
- Saves outputs under: outputs/runs/<timestamp>/

Usage in Spyder:
- Set TRAIN_ID to your ckpt run folder name
- Run this file
"""

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
    time.sleep(0.7)  # filesystem timestamp separation


# -----------------------
# 1) Point to trained checkpoint
# -----------------------
TRAIN_ID = "20260109_144355"   # <-- change this
CKPT = PROJECT_ROOT / "outputs" / "runs" / TRAIN_ID / "ckpt.pt"
if not CKPT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

print("\nCKPT:", CKPT)


# -----------------------
# 2) Experiment settings
# -----------------------
T_TOTAL  = 400
WARMUP  = 150
PERIODS = [10, 20, 40, 80]

SIGMA_A = (0.60, 0.30, 0.05)
SIGMA_B = (0.05, 0.30, 0.60)

DEVICE = "cpu"
SEED   = "0"
GREEDY = True

# detection params
PRE_WINDOW = 80
ALPHA = 0.05
CONSEC = 3

# hysteresis window length after each switch
L = 60

# which policy signal to compare against g
# options: "entropy", "pi_max", "margin"
POLICY_SIGNAL = "entropy"


# -----------------------
# 3) Loop over periods
# -----------------------
results = []

for P in PERIODS:
    print("\n" + "=" * 80)
    print(f"=== Switch-sweep: period={P} (T={T_TOTAL}, warmup={WARMUP}) ===")

    # --- (A) Run switch sweep collector
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
        "--max_steps", "400",
    ]
    if GREEDY:
        args_collect.append("--greedy")

    run_module("cear_pilot.experiments.run_switch_sweep", args_collect)
    safe_sleep()

    # detect new run dir
    after = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name not in before]
    if len(after) == 0:
        # fallback
        run_dir = newest_run_dir()
    else:
        run_dir = max(after, key=lambda p: p.stat().st_mtime)

    print("Detected run_dir:", run_dir)

    # --- (B) Run evaluation + figure
    args_eval = [
        "--run_dir", str(run_dir),
        "--warmup", str(WARMUP),
        "--pre_window", str(PRE_WINDOW),
        "--alpha", str(ALPHA),
        "--consec", str(CONSEC),
        "--L", str(L),
        "--policy_signal", POLICY_SIGNAL,
    ]
    run_module("cear_pilot.analysis.figure_switch_eval", args_eval)

    print(f"[DONE] period={P} -> {run_dir / 'switch_eval.json'}")
    results.append((P, run_dir))

print("\n" + "=" * 80)
print("ALL DONE. Summary:")
for P, run_dir in results:
    print(f"  period={P}: {run_dir}")
    print(f"    fig:  {run_dir / 'figs' / f'fig_switch_eval_{POLICY_SIGNAL}.png'}")
    print(f"    json: {run_dir / 'switch_eval.json'}")
