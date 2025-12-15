# cear_pilot/experiments/run_ablation.py
# -*- coding: utf-8 -*-
"""
Run collection twice: g on vs g ablated, into the same parent run dir.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import subprocess
import sys


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    root = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    root.mkdir(parents=True, exist_ok=True)
    (root / "figs").mkdir(parents=True, exist_ok=True)

    cmd_base = [
        sys.executable, "-m", "cear_pilot.experiments.run_collect",
        "--ckpt", args.ckpt,
        "--episodes", str(args.episodes),
        "--device", args.device,
    ]
    if args.greedy:
        cmd_base.append("--greedy")

    # g on
    subprocess.check_call(cmd_base + ["--outdir", str(root / "g_on")])
    # g off
    subprocess.check_call(cmd_base + ["--outdir", str(root / "g_off"), "--ablate_g"])

    meta = {
        "mode": "ablation",
        "ckpt": args.ckpt,
        "episodes": args.episodes,
        "device": args.device,
        "greedy": bool(args.greedy),
        "runs": {"g_on": str(root / "g_on"), "g_off": str(root / "g_off")},
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Ablation run dir: {root}")


if __name__ == "__main__":
    main()
