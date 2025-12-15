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

# ---- Robust importer for MettaGrid 0.16/0.2 path differences ----
def _resolve_env(adapter: str):
    """
    Resolve MettaGrid env constructor across versions.
    Returns (callable_make_env, where_str).
    """
    import importlib, inspect

    if adapter == "pettingzoo":
        # 0.16에는 pettingzoo가 없음: 곧바로 안내 에러
        raise ImportError(
            "[MettaGrid import resolver] This MettaGrid build has no PettingZoo adapter. "
            "Use --adapter gym or --adapter builtin."
        )

    if adapter == "gym":
        errs = []

        # ---- 1) 0.16 스타일: mettagrid.gym_wrapper 안에서 생성자/팩토리 찾기
        try:
            gw = importlib.import_module("mettagrid.gym_wrapper")
            # 후보 이름들: 버전마다 달라질 수 있어 전부 시도
            for name in ["MettaGridGymEnv", "GymWrapper", "make_env", "make", "create_env", "Env"]:
                if hasattr(gw, name):
                    obj = getattr(gw, name)
                    if inspect.isclass(obj):
                        # 클래스면 인자 없이 만들어보는 람다를 반환 (필요시 나중에 인자 추가)
                        return (lambda **kw: obj(**kw) if kw else obj()), f"mettagrid.gym_wrapper.{name}"
                    if callable(obj):
                        # 팩토리 함수면 그대로 반환
                        return (lambda **kw: obj(**kw)), f"mettagrid.gym_wrapper.{name}()"
            # 못 찾았으면, 모듈 내에서 이름에 'env'가 들어간 callables를 스캔
            for k, v in gw.__dict__.items():
                if ("env" in k.lower() or "gym" in k.lower()) and callable(v):
                    return (lambda **kw: v(**kw)), f"mettagrid.gym_wrapper.{k}()"
        except Exception as e:
            errs.append(f"mettagrid.gym_wrapper: {e.__class__.__name__}: {e}")

        # ---- 2) 마지막 폴백: mettagrid.mettagrid_env 안의 Env 클래스를 감싸기
        try:
            me = importlib.import_module("mettagrid.mettagrid_env")
            # Env 같은 이름을 찾아서 최소 래퍼를 씌움
            for k, v in me.__dict__.items():
                if inspect.isclass(v) and ("Env" in k or "Environment" in k):
                    BaseEnv = v

                    # 간단 래퍼 (reset/step 시그니처를 gymnasium 스타일로)
                    class _GymLike:
                        def __init__(self, **kw):
                            self._env = BaseEnv(**kw) if kw else BaseEnv()

                        def reset(self, seed=None):
                            if hasattr(self._env, "reset"):
                                out = self._env.reset(seed=seed) if seed is not None else self._env.reset()
                                # (obs, info) 형태로 정규화
                                if isinstance(out, tuple) and len(out)==2:
                                    return out
                                return out, {}
                            return {}, {}

                        def step(self, actions):
                            # actions: [a0, a1] 형태를 기대
                            if hasattr(self._env, "step"):
                                out = self._env.step(actions)
                                # (obs, rew, term, trunc, info) 형태로 정규화
                                if isinstance(out, tuple) and len(out) == 5:
                                    return out
                                # 못 맞추면 대충 빈 값이라도 반환
                                return out, 0.0, False, False, {}
                            raise RuntimeError("Underlying env has no step()")

                    return (lambda **kw: _GymLike(**kw)), f"mettagrid.mettagrid_env.{k} (wrapped)"
        except Exception as e:
            errs.append(f"mettagrid.mettagrid_env: {e.__class__.__name__}: {e}")

        tried = "\n  - " + "\n  - ".join(errs) if errs else ""
        raise ImportError(
            "[MettaGrid import resolver] Could not resolve a gym environment. "
            f"Tried:{tried}\n"
            "Tips:\n"
            "  • This build exposes gym via mettagrid.gym_wrapper in many cases.\n"
            "  • If API mismatches remain, use --adapter builtin (no external deps)."
        )

    raise ValueError(f"Unknown adapter: {adapter}")



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

    #mod = importlib.import_module("mettagrid.adapters.pettingzoo_env")
    #MakeEnv = getattr(mod, "MettaGridPettingZooEnv")

    # NEW (robust across 0.2/0.16):
    MakeEnv, where = _resolve_env("pettingzoo")
    print(f"[info] Using env: {where}")
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

def run_gym(no_context: bool, no_longg: bool, save_mp4: str = None):
    global W_CROWD, G_ALPHA
    if no_context: W_CROWD = 0.0
    if no_longg:   G_ALPHA = 0.0

    import importlib, inspect

    # 0.16용 진입점 탐색: mettagrid.gym_wrapper
    gw = importlib.import_module("mettagrid.gym_wrapper")

    MakeEnv = None
    tried = []

    # 1) 흔한 팩토리 이름들
    for name in ["make_env", "make", "create_env"]:
        if hasattr(gw, name) and callable(getattr(gw, name)):
            _f = getattr(gw, name)
            def MakeEnv(**kw):
                return _f(**kw)
            where = f"mettagrid.gym_wrapper.{name}()"
            break
        tried.append(name)

    # 2) 클래스 이름들
    if MakeEnv is None:
        for name in ["MettaGridGymEnv", "GymWrapper", "Env"]:
            if hasattr(gw, name) and inspect.isclass(getattr(gw, name)):
                Cls = getattr(gw, name)
                def MakeEnv(**kw):
                    return Cls(**kw)
                where = f"mettagrid.gym_wrapper.{name}"
                break
            tried.append(name)

    # 3) 이름에 env/gym가 들어간 callable 아무거나
    if MakeEnv is None:
        for k, v in gw.__dict__.items():
            if callable(v) and ("env" in k.lower() or "gym" in k.lower()):
                def _make(v_):
                    def _MakeEnv(**kw):
                        return v_(**kw)
                    return _MakeEnv
                MakeEnv = _make(v)
                where = f"mettagrid.gym_wrapper.{k}()"
                break

    if MakeEnv is None:
        raise ImportError(
            "[MettaGrid gym] Could not find a constructor in mettagrid.gym_wrapper. "
            f"Tried names: {tried}"
        )

    print(f"[info] Using env from {where}")
    env = MakeEnv()  # 인자 없는 기본 생성부터 시도

    grid_w, grid_h = 40, 40
    agents=[ValenceAgent(0,0), ValenceAgent(1,0)]
    frames=[]
    logs=[]

    # reset 표준화: (obs, info)
    def _reset(seed=None):
        try:
            if hasattr(env.reset, "__code__") and "seed" in env.reset.__code__.co_varnames:
                out = env.reset(seed=seed)
            else:
                out = env.reset()
        except TypeError:
            out = env.reset()
        return out if (isinstance(out, tuple) and len(out)==2) else (out, {})

    # step 표준화: (obs, rew, term, trunc, info)
    def _step(actions):
        out = env.step(actions)
        if isinstance(out, tuple):
            if len(out)==5:
                return out
            if len(out)==4:  # (obs, rew, done, info)
                obs, rew, done, info = out
                term = bool(done); trunc = False
                return obs, rew, term, trunc, info
        # 최후 폴백
        return out, 0.0, False, False, {}

    for ep in range(EPISODES):
        obs, info = _reset(seed=ep)

        # obs[0]/obs[1]이 dict라고 가정, 아니면 기본값
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

            obs, rew, term, trunc, info = _step([a0,a1])

            zid0,zid1 = zone_of(x0,y0,grid_w,grid_h), zone_of(x1,y1,grid_w,grid_h)
            v0 = extract_signal(obs[0], ZONE_MU[zid0]) if isinstance(obs, (list,tuple)) else ZONE_MU[zid0]
            v1 = extract_signal(obs[1], ZONE_MU[zid1]) if isinstance(obs, (list,tuple)) else ZONE_MU[zid1]
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

def run_builtin(no_context: bool, no_longg: bool, save_mp4: str = None):
    global W_CROWD, G_ALPHA
    if no_context: W_CROWD = 0.0
    if no_longg:   G_ALPHA = 0.0

    grid_w, grid_h = 40, 40
    agents = [ValenceAgent(0,0), ValenceAgent(1,0)]
    rng = np.random.default_rng(0)

    def clip_xy(x,y):
        return int(np.clip(x,0,grid_w-1)), int(np.clip(y,0,grid_h-1))

    frames=[]; logs=[]

    for ep in range(EPISODES):
        x0,y0 = int(rng.integers(0,grid_w//2)), int(rng.integers(0,grid_h))
        x1,y1 = int(rng.integers(grid_w//2,grid_w)), int(rng.integers(0,grid_h))

        for t in range(STEPS):
            a0 = agents[0].pick_action((x0,y0), [(x1,y1)], grid_w, grid_h)
            dx0,dy0 = [(0,0),(0,-1),(0,1),(-1,0),(1,0)][a0]
            nx0,ny0 = clip_xy(x0+dx0, y0+dy0)

            a1 = agents[1].pick_action((x1,y1), [(nx0,ny0)], grid_w, grid_h)
            dx1,dy1 = [(0,0),(0,-1),(0,1),(-1,0),(1,0)][a1]
            nx1,ny1 = clip_xy(x1+dx1, y1+dy1)

            zid0,zid1 = zone_of(nx0,ny0,grid_w,grid_h), zone_of(nx1,ny1,grid_w,grid_h)
            v0 = float(ZONE_MU[zid0] + rng.normal(0, ZONE_STD[zid0]))
            v1 = float(ZONE_MU[zid1] + rng.normal(0, ZONE_STD[zid1]))
            pe0,pe1 = abs(v0-ZONE_MU[zid0]), abs(v1-ZONE_MU[zid1])

            for aid,(zid,pe) in enumerate([(zid0,pe0),(zid1,pe1)]):
                pe_z = {0:ZONE_STD[0],1:ZONE_STD[1],2:ZONE_STD[2]}
                pe_z[zid] = pe
                if not no_longg:
                    agents[aid].update_g(zid)
                agents[aid].update_valence(pe_z)

            conflict = 1 if cheby((nx0,ny0),(nx1,ny1))<=1 else 0
            coop     = 1 if zid0!=zid1 else 0
            logs.append({"ep":ep,"t":t,"conflict":conflict,"coop":coop})

            x0,y0 = nx0,ny0
            x1,y1 = nx1,ny1

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

    df = pd.DataFrame(logs)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["pettingzoo","gym","builtin"], default="gym")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-longg", action="store_true")
    ap.add_argument("--save-mp4", type=str, default=None)
    args = ap.parse_args()

    # runner 선택
    if args.adapter == "pettingzoo":
        runner = run_pettingzoo
    elif args.adapter == "gym":
        runner = run_gym
    else:
        runner = run_builtin

    # Baseline (임포트 실패 시 builtin으로 자동 전환)
    try:
        df_base, frames = runner(no_context=args.no_context, no_longg=args.no_longg, save_mp4=args.save_mp4)
    except ImportError as e:
        print(f"[warn] {e}\n[info] Falling back to --adapter builtin")
        df_base, frames = run_builtin(no_context=args.no_context, no_longg=args.no_longg, save_mp4=args.save_mp4)

    # 베이스라인 플롯 & 요약
    plot_metrics(df_base, "baseline" if not (args.no_context or args.no_longg) else "custom")
    s1 = summarize(df_base, "baseline/custom")

    # Ablations (baseline이 둘 다 켜진 상태일 때만)
    rows = [s1]
    if not args.no_context and not args.no_longg:
        # no-context
        try:
            df_noctx, _ = runner(no_context=True, no_longg=False, save_mp4=None)
        except ImportError:
            df_noctx, _ = run_builtin(no_context=True, no_longg=False, save_mp4=None)
        plot_metrics(df_noctx, "no_context")

        # no-longG
        try:
            df_nog, _ = runner(no_context=False, no_longg=True, save_mp4=None)
        except ImportError:
            df_nog, _ = run_builtin(no_context=False, no_longg=True, save_mp4=None)
        plot_metrics(df_nog, "no_longG")

        rows += [summarize(df_noctx, "no_context"),
                 summarize(df_nog,   "no_longG")]

    # 요약 테이블 출력
    summ = pd.DataFrame(rows)
    print("\n=== MVP summary ===")
    print(summ.to_string(index=False))

    # 애니메이션 저장
    if args.save_mp4 and len(frames) > 0:
        save_animation_mp4(frames, args.save_mp4)

if __name__ == "__main__":
    main()
