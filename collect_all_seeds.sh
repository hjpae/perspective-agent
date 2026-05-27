#!/usr/bin/env bash
# collect_all_seeds.sh
# -----------------------------------------------------------------
# Collect replay trajectories for 5 seeds × {clean, p20} conditions.
# Total: 10 runs × 10 episodes × 240 steps = 24,000 timesteps per condition.
#
# Usage:
#   bash collect_all_seeds.sh
#
# Expects ckpts at: runs - AAAI/seed{1..5}/ckpt.pt
# Outputs to:       outputs/replay_seed{1..5}_{clean,p20}/traj.parquet
# -----------------------------------------------------------------

set -e  # exit on first error

EPISODES=10
T_SWITCH=120
SIGMA_BASE="0.6 0.3 0.05"
SIGMA_POST="0.3 0.05 0.6"   # p20-style permutation

echo "================================================================"
echo "Batch replay collection: 5 seeds × {clean, p20}"
echo "================================================================"

for seed in 1 2 3 4 5; do
    ckpt="runs - outputs/runs/seed${seed}/ckpt.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERR ] missing ckpt: $ckpt — skipping seed $seed"
        continue
    fi

    # --- clean condition ---
    outdir="outputs/replay_seed${seed}_clean"
    if [[ -f "$outdir/traj.parquet" ]]; then
        echo "[skip] seed $seed clean: already exists ($outdir/traj.parquet)"
    else
        echo "[run ] seed $seed clean → $outdir"
        PYTHONPATH=. python -m cear_pilot.experiments.run_collect \
            --ckpt "$ckpt" \
            --episodes $EPISODES \
            --seed $seed \
            --device cpu \
            --outdir "$outdir" \
            > "$outdir.log" 2>&1 \
            || { echo "[ERR ] seed $seed clean failed; check $outdir.log"; continue; }
        mv "$outdir.log" "$outdir/run.log" 2>/dev/null || true
    fi

    # --- p20 (regime switch) condition ---
    outdir="outputs/replay_seed${seed}_p20"
    if [[ -f "$outdir/traj.parquet" ]]; then
        echo "[skip] seed $seed p20: already exists ($outdir/traj.parquet)"
    else
        echo "[run ] seed $seed p20 → $outdir"
        PYTHONPATH=. python -m cear_pilot.experiments.run_collect \
            --ckpt "$ckpt" \
            --episodes $EPISODES \
            --seed $seed \
            --device cpu \
            --zone_sigma $SIGMA_BASE \
            --t_switch $T_SWITCH \
            --zone_sigma2 $SIGMA_POST \
            --outdir "$outdir" \
            > "$outdir.log" 2>&1 \
            || { echo "[ERR ] seed $seed p20 failed; check $outdir.log"; continue; }
        mv "$outdir.log" "$outdir/run.log" 2>/dev/null || true
    fi
done

echo ""
echo "================================================================"
echo "Done. Summary:"
echo "================================================================"
for seed in 1 2 3 4 5; do
    for cond in clean p20; do
        f="outputs/replay_seed${seed}_${cond}/traj.parquet"
        if [[ -f "$f" ]]; then
            sz=$(du -h "$f" | cut -f1)
            echo "  [YES] seed $seed $cond ($sz)"
        else
            echo "  [NO] seed $seed $cond  MISSING"
        fi
    done
done
