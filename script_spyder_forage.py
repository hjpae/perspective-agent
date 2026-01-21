#%% forage - train

import sys
from cear_pilot.training import train_forage

sys.argv = [
    "train_forage",
    "--steps", "80000",
    "--seed", "0",
    "--device", "cpu",
    #"--view",
]

train_forage.main()


#%% forage - test
import sys
from cear_pilot.experiments import run_collect_forage

CKPT_PATH = r"outputs/runs/20260120_201325/ckpt.pt"

sys.argv = [
    "run_collect_forage",
    "--ckpt", CKPT_PATH,
    "--episodes", "5",
    "--seed", "0",
    "--device", "cpu",
    "--greedy",
    "--view",
    "--fps", "10",
    # do(g) experiment: 
    # "--do_g", "swap",
    # "--do_g_scale", "1.0",
]

run_collect_forage.main()
