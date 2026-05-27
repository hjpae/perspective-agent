# Handoff: Φ_r estimator + AAAI replay data

## What you got

```
handoff/
├── phi/                      # Drop into cear_pilot/analysis/phi/
│   ├── __init__.py
│   ├── information.py        # Vendored from Pigozzi & Levin (with one path fix)
│   └── phi_r.py              # Our clean wrapper API
├── sanity_phi_r.py           # Run after placing the pickle
└── outputs/                  # AAAI seed1 replay data — already collected
    ├── replay_seed1_clean/   # 10 episodes, no perturbation
    │   ├── meta.json
    │   └── traj.parquet      # T×{episode, t, z_0..z_15, g_0..g_11, s_0..s_15, ...}
    └── replay_seed1_p20/     # 10 episodes, regime switch at t=120
        ├── meta.json
        └── traj.parquet
```

## One thing you must do locally (sandbox couldn't fetch it)

Download `phi_lattice_22.pickle` from:
  https://github.com/pigozzif/PhiRL/blob/master/phi_lattice_22.pickle
and place it in `cear_pilot/analysis/phi/` (next to information.py).

## Then

```bash
cd <your AAAI repo>
PYTHONPATH=. python sanity_phi_r.py outputs/replay_seed1_clean/traj.parquet
```

Expected output:
  [1] pure noise:    Φ_r ≈ 0
  [2] coupled:       Φ_r > 0
  [3] real replay:   per-episode Φ_r(z), Φ_r(g), Φ_r([z,g]), ΔΦ_r table

## What's also patched

- `cear_pilot/models/agent.py`: AgentConfig now uses field(default_factory=...)
  for Python 3.12 compatibility. (Was using mutable defaults.)
- `cear_pilot/experiments/run_collect.py`: your collect script, dropped in.

## What's NOT done yet

- Run sanity_phi_r.py to confirm estimator works on real data
- Scale to all 5 seeds × {clean, p20, p40, p80}
- Shuffled-g ablation (the architectural control for the abstract)
- PE-alignment analysis (target = −loss_pred or similar)
