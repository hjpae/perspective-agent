# MettaGrid MVP (Valence → Pro-sociality)
**Goal**: Show that without extrinsic rewards, agents with *valence updates* (from prediction error), *context gating* (crowding), and a *slow integrator* \(g_t\) can self-organize into **lower conflict**, **higher cooperative allocation**, and **lower joint-occupancy entropy**—i.e., emergent *pro-sociality*.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install mettagrid gymnasium pettingzoo numpy matplotlib pandas
# Optional for MP4/GIF export:
pip install pillow imageio imageio-ffmpeg
```

## Run
```bash
# Baseline (PettingZoo)
python run_metta_valence_mvp.py --adapter pettingzoo

# Ablations
python run_metta_valence_mvp.py --adapter pettingzoo --no-context
python run_metta_valence_mvp.py --adapter pettingzoo --no-longg

# Save animation
python run_metta_valence_mvp.py --adapter pettingzoo --save-mp4 demo.mp4
```

If you prefer Gym:
```bash
python run_metta_valence_mvp.py --adapter gym
```

## What to Expect
- **Conflict (↓)** and **Cooperative (↑)** curves over time.
- **Entropy (↓)**: proxy of joint occupancy diversity stabilizing.
- **Ablations**: removing context gating or long-term \(g_t\) should degrade pro-social convergence.

## Customize
Open `run_metta_valence_mvp.py` and edit the hyperparameters block:
```
EPISODES, STEPS, TEMP, ETA_V, G_ALPHA, W_CROWD, CROWD_RAD
```
You can also change `zone_of(x,y,grid_w,grid_h)` to tailor zone shapes/sizes.

## Notes
- Observation key auto-detection tries to guess coordinate fields; if it fails, print a sample `obs` to adjust keys.
- The animation is a simple scatter overlaid with zone partition lines; replace with a richer renderer if needed.

Good luck with your demo & fundraising!
