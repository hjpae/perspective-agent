# run_metta_valence_mvp.py
"""
MettaGrid MVP: Valence (contextual adjustment + long-term integration) -> pro-sociality emergence
- Adapter: PettingZoo (default) or Gym (flag)
- Two agents, three "zones" (by x,y partitioning) with different signal statistics
- No extrinsic reward; agents update valence from prediction error (PE), plus context gating (crowding) and slow state g_t
- Metrics: conflict rate, cooperative allocation, entropy proxy; ablations: --no-context, --no-longg
- Optional: save replay animation as MP4/GIF

USAGE
-----
python run_metta_valence_mvp.py --adapter pettingzoo             # baseline
python run_metta_valence_mvp.py --adapter pettingzoo --no-context
python run_metta_valence_mvp.py --adapter pettingzoo --no-longg
python run_metta_valence_mvp.py --adapter pettingzoo --save-mp4 demo.mp4
python run_metta_valence_mvp.py --adapter gym

REQUIREMENTS
------------
pip install mettagrid gymnasium pettingzoo numpy matplotlib pandas
(Optional for mp4): pip install pillow imageio imageio-ffmpeg
"""
import argparse, importlib, math, time, sys
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Hyperparameters ----------
EPISODES = 4
STEPS    = 800
TEMP     = 0.25
ETA_V    = 0.05
G_ALPHA  = 0.98
W_CROWD  = 0.8
CROWD_RAD = 1

# Zone signal stats (affects PE calculation only)
ZONE_MU  = {0:0.2, 1:0.8, 2:0.05}
ZONE_STD = {0:0.05,1:0.25,2:0.02}

# ---------- Helpers ----------
def softmax(x, temp=1.0):
    z = (x - np.max(x)) / max(temp, 1e-8)
    e = np.exp(z)
    return e / (e.sum() + 1e-8)

def cheby(a,b): 
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

def zone_of(x:int, y:int, grid_w:int, grid_h:int)->int:
    """Partition space by x,y to define 3 zones: left half=0, right-top=1, right-bottom=2"""
    if x < grid_w//2: return 0
    return 1 if y < grid_h//2 else 2

def rolling_mean(arr: np.ndarray, k:int=25)->np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    s=0.0
    for i,a in enumerate(arr):
        s += a
        if i>=k: s -= arr[i-k]
        out[i] = s / min(i+1, k)
    return out

# ---------- Agent ----------
class ValenceAgent:
    def __init__(self, aid:int, seed:int=0):
        self.id=aid
        self.rng=np.random.default_rng(seed+aid)
        # Start with opposite biases to induce early conflicts
        self.v=np.array([0.4,-0.1,0.0]) if aid==0 else np.array([-0.1,0.4,0.0])
        self.g=np.zeros(3)

    def update_valence(self, pe_z: Dict[int,float]):
        # Smaller PE -> increase liking; clip range
        for z, val in pe_z.items():
            self.v[z] += -ETA_V * val
        self.v = np.clip(self.v, -1.5, 1.5)

    def update_g(self, zid:int):
        oh=np.zeros(3); oh[zid]=1.0
        self.g = G_ALPHA*self.g + (1.0-G_ALPHA)*oh

    def pick_action(self, my_xy:Tuple[int,int], other_xys:List[Tuple[int,int]], grid_w:int, grid_h:int)->int:
        # 5 actions: stay/up/down/left/right  mapped as 0..4
        actions=[(0,0),(0,-1),(0,1),(-1,0),(1,0)]
        scores=[]
        for dx,dy in actions:
            nx,ny = my_xy[0]+dx, my_xy[1]+dy
            nx = int(np.clip(nx,0,grid_w-1))
            ny = int(np.clip(ny,0,grid_h-1))
            zid=zone_of(nx,ny,grid_w,grid_h)
            base = self.v[zid] + self.g[zid]

            crowd_pen=0.0
            for ox,oy in other_xys:
                d = cheby((nx,ny),(ox,oy))
                if d <= CROWD_RAD:
                    crowd_pen += (CROWD_RAD - d + 1) * 0.1

            scores.append(base - W_CROWD * crowd_pen)
        probs = softmax(np.asarray(scores), temp=TEMP)
        return self.rng.choice(len(actions), p=probs)

# ---------- Observation field auto-detection ----------
def autodetect_xy(obs) -> Tuple[str,str]:
    """Heuristically infer x/y keys from obs dict/array-like structure."""
    # common guesses
    candidates=[("x","y"),("pos_x","pos_y"),("px","py"),("row","col"),("i","j")]
    if isinstance(obs, dict):
        keys = set(obs.keys())
        for kx,ky in candidates:
            if kx in keys and ky in keys:
                return kx,ky
        # fallback: look for numeric pairs
        for k in keys:
            if isinstance(obs[k], (int,float)) and any(s in k.lower() for s in ["x","col","i"]):
                for k2 in keys:
                    if k2!=k and isinstance(obs[k2], (int,float)) and any(s in k2.lower() for s in ["y","row","j"]):
                        return k,k2
    # if array-like, assume separate per-agent views; handle in adapter-specific code
    return "x","y"

def extract_signal(obs, default_mu:float) -> float:
    # try common fields, else fallback to default_mu
    if isinstance(obs, dict):
        for k in ["signal","value","obs","o","s"]:
            if k in obs:
                try: return float(obs[k])
                except: pass
    return float(default_mu)

# ---------- Animation ----------
def save_animation_mp4(frames: List[np.ndarray], out_path:str, fps:int=20):
    try:
        import imageio
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"[saved] {out_path}")
    except Exception as e:
        print(f"[warn] Could not save MP4/GIF: {e}")

# ---------- Main runners ----------
def run_pettingzoo(no_context:bool, no_longg:bool, save_mp4:str=None):
    global W_CROWD, G_ALPHA
    if no_context: W_CROWD = 0.0
    if no_longg:   G_ALPHA = 0.0

    mod = importlib.import_module("mettagrid.adapters.pettingzoo_env")
    MakeEnv = getattr(mod, "MettaGridPettingZooEnv")
    env = MakeEnv()
    aec = env.aec_env
    grid_w, grid_h = 40, 40  # adjust if env exposes size
    agents=[ValenceAgent(0,0), ValenceAgent(1,0)]
    frames=[]

    logs=[]
    for ep in range(EPISODES):
        aec.reset(seed=ep)
        last_xy={}
        # detect xy keys from first observation
        first_agent = None
        first_obs = None
        for name in aec.agent_iter():
            obs, rew, term, trunc, info = aec.last()
            first_agent = name; first_obs = obs; break
        xk,yk = autodetect_xy(first_obs)
        aec.reset(seed=ep)  # restart after peeking

        for t in range(STEPS):
            for name in aec.agent_iter():
                obs, rew, term, trunc, info = aec.last()
                # coords
                try:
                    x = int(obs.get(xk, 0))
                    y = int(obs.get(yk, 0))
                except Exception:
                    # if obs not dict, try info or skip to 0,0
                    x,y = 0,0
                if name not in last_xy: last_xy[name]=(x,y)
                my_xy = (x,y)
                others=[xy for n,xy in last_xy.items() if n!=name]
                aid = 0 if name.endswith("0") else 1
                act_idx = agents[aid].pick_action(my_xy, others, grid_w, grid_h)
                aec.step(act_idx)

                zid = zone_of(x,y,grid_w,grid_h)
                mu = ZONE_MU[zid]
                v  = extract_signal(obs, mu)
                pe = abs(float(v)-mu)
                pe_z = {0:ZONE_STD[0],1:ZONE_STD[1],2:ZONE_STD[2]}; pe_z[zid]=pe
                if not no_longg: agents[aid].update_g(zid)
                agents[aid].update_valence(pe_z)
                last_xy[name]=(x,y)

            # logging once per step (after all moved)
            if len(last_xy)==2:
                names=list(last_xy.keys())
                p0,p1 = last_xy[names[0]], last_xy[names[1]]
                conflict = 1 if cheby(p0,p1)<=1 else 0
                coop     = 1 if zone_of(*p0,grid_w,grid_h) != zone_of(*p1,grid_w,grid_h) else 0
                logs.append({"ep":ep,"t":t,"conflict":conflict,"coop":coop})

            # simple frame: scatter agents
            if save_mp4:
                fig,ax=plt.subplots(figsize=(4,4))
                ax.set_xlim(0,grid_w); ax.set_ylim(0,grid_h)
                # zone boundaries
                ax.axvline(grid_w//2, linestyle="--")
                ax.axhline(grid_h//2, xmin=0.5, linestyle="--")
                xs=[xy[0] for xy in last_xy.values()]; ys=[xy[1] for xy in last_xy.values()]
                ax.scatter(xs,ys)
                ax.set_title(f"ep{ep} t{t}")
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)

    df=pd.DataFrame(logs)
    return df, frames

def run_gym(no_context:bool, no_longg:bool, save_mp4:str=None):
    global W_CROWD, G_ALPHA
    if no_context: W_CROWD = 0.0
    if no_longg:   G_ALPHA = 0.0

    mod = importlib.import_module("mettagrid.adapters.gym_env")
    MakeEnv = getattr(mod, "MettaGridGymEnv")
    env = MakeEnv()
    grid_w, grid_h = 40, 40
    agents=[ValenceAgent(0,0), ValenceAgent(1,0)]
    frames=[]
    logs=[]

    for ep in range(EPISODES):
        obs, info = env.reset(seed=ep)
        # Attempt to autodetect xy keys from obs[0]
        try:
            xk,yk = autodetect_xy(obs[0])
        except Exception:
            xk,yk = "x","y"

        last_xy=[(0,0),(1,0)]
        for t in range(STEPS):
            try:
                x0,y0 = int(obs[0].get(xk,0)), int(obs[0].get(yk,0))
                x1,y1 = int(obs[1].get(xk,1)), int(obs[1].get(yk,0))
            except Exception:
                x0,y0,x1,y1 = 0,0,1,0

            a0 = agents[0].pick_action((x0,y0), [(x1,y1)], grid_w, grid_h)
            a1 = agents[1].pick_action((x1,y1), [(x0,y0)], grid_w, grid_h)
            obs, rew, term, trunc, info = env.step([a0,a1])

            zid0,zid1 = zone_of(x0,y0,grid_w,grid_h), zone_of(x1,y1,grid_w,grid_h)
            v0 = extract_signal(obs[0], ZONE_MU[zid0])
            v1 = extract_signal(obs[1], ZONE_MU[zid1])
            pe0,pe1 = abs(float(v0)-ZONE_MU[zid0]), abs(float(v1)-ZONE_MU[zid1])
            for aid,(zid,pe) in enumerate([(zid0,pe0),(zid1,pe1)]):
                pe_z={0:ZONE_STD[0],1:ZONE_STD[1],2:ZONE_STD[2]}; pe_z[zid]=pe
                if not no_longg: agents[aid].update_g(zid)
                agents[aid].update_valence(pe_z)

            conflict = 1 if cheby((x0,y0),(x1,y1))<=1 else 0
            coop     = 1 if zid0!=zid1 else 0
            logs.append({"ep":ep,"t":t,"conflict":conflict,"coop":coop})

            if save_mp4:
                fig,ax=plt.subplots(figsize=(4,4))
                ax.set_xlim(0,grid_w); ax.set_ylim(0,grid_h)
                ax.axvline(grid_w//2, linestyle="--")
                ax.axhline(grid_h//2, xmin=0.5, linestyle="--")
                ax.scatter([x0,x1],[y0,y1])
                ax.set_title(f"ep{ep} t{t}")
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)

    df=pd.DataFrame(logs)
    return df, frames

def plot_metrics(df: pd.DataFrame, title: str):
    tvals = sorted(df["t"].unique())
    agg = df.groupby("t")[["conflict","coop"]].mean().reindex(tvals)
    t = np.arange(len(agg))
    conf = agg["conflict"].values
    coop = agg["coop"].values

    plt.figure(figsize=(9,3.2))
    plt.plot(rolling_mean(conf,25), label="Conflict (rolling)")
    plt.plot(rolling_mean(coop,25), label="Cooperative (rolling)")
    plt.legend(); plt.title(title + " — Conflict/Coop"); plt.xlabel("t"); plt.ylabel("rate"); plt.tight_layout(); plt.show()

    p = coop.clip(1e-6, 1-1e-6)
    H = -(p*np.log(p) + (1-p)*np.log(1-p))
    plt.figure(figsize=(9,3.2))
    plt.plot(rolling_mean(H,25), label="Joint occupancy entropy (proxy)")
    plt.legend(); plt.title(title + " — Entropy"); plt.xlabel("t"); plt.ylabel("entropy"); plt.tight_layout(); plt.show()

def summarize(df: pd.DataFrame, label:str)->Dict[str,float]:
    conf_all = df["conflict"].mean()
    coop_all = df["coop"].mean()
    tmax = df["t"].max()
    m_final = df["t"] > (2*tmax/3)
    conf_f = df.loc[m_final,"conflict"].mean()
    coop_f = df.loc[m_final,"coop"].mean()

    # stance shift under prior conflict
    changes = 0; conds=0
    for ep, sub in df.groupby("ep"):
        sub = sub.sort_values("t")
        prev_conf = 0
        prev_pair = None
        for _,row in sub.iterrows():
            if prev_conf==1 and prev_pair is not None:
                # naive proxy: change in coop state
                if row["coop"] != prev_pair:
                    changes += 1
                conds += 1
            prev_conf = int(row["conflict"])
            prev_pair = int(row["coop"])
    pshift = (changes/conds) if conds>0 else float("nan")

    return {
        "run": label,
        "conflict_overall": round(conf_all,3),
        "coop_overall": round(coop_all,3),
        "conflict_final_third": round(conf_f,3),
        "coop_final_third": round(coop_f,3),
        "P(change|prior_conflict)": round(pshift,3) if not math.isnan(pshift) else float("nan"),
        "n_conditions": int(conds),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["pettingzoo","gym"], default="pettingzoo")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-longg", action="store_true")
    ap.add_argument("--save-mp4", type=str, default=None)
    args=ap.parse_args()

    runner = run_pettingzoo if args.adapter=="pettingzoo" else run_gym

    # Baseline
    df_base, frames = runner(no_context=args.no_context, no_longg=args.no_longg, save_mp4=args.save_mp4)
    plot_metrics(df_base, "baseline" if not (args.no_context or args.no_longg) else "custom")
    s1 = summarize(df_base, "baseline/custom")

    # Ablations (only if baseline had both on)
    rows=[s1]
    if not args.no_context and not args.no_longg:
        df_noctx,_ = runner(no_context=True, no_longg=False, save_mp4=None)
        df_nog,_   = runner(no_context=False, no_longg=True, save_mp4=None)
        plot_metrics(df_noctx, "no_context")
        plot_metrics(df_nog,   "no_longG")
        rows += [summarize(df_noctx,"no_context"), summarize(df_nog,"no_longG")]

    summ = pd.DataFrame(rows)
    print("\n=== MVP summary ===")
    print(summ.to_string(index=False))

    if args.save_mp4 and len(frames)>0:
        save_animation_mp4(frames, args.save_mp4)

if __name__=="__main__":
    main()
