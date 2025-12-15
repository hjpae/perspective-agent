#%% initial training 
import sys
sys.path.append(".")

from cear_pilot.training.train import main

if __name__ == "__main__":
    main()

# %autoreload 2

#%% one script for all (root)
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
TRAIN_ID = "20251215_090437"   # <-- ckpt run id
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
