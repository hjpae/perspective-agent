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
    
// ckpts | github demo: 20260109_144355 | AAAI paper: 20260127_215133
