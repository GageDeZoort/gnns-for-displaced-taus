import os
import sys
sys.path.append('../')

import itertools
import build_tau_graphs

# grab job ID
dm = 'OneProngTwoPi0'
infile = f'/tigress/jdezoort/displaced_tau_samples/{dm}.p'
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f"Running on {infile}")

eps = [1, 2, 3]
sim_min = [0.01, 0.025, 0.05, 0.1, 0.25]
ecal_min = [0.01, 0.025, 0.05, 0.1, 0.25]
hcal_min = [0.01, 0.025, 0.05, 0.1, 0.25]
params = list(itertools.product(eps, sim_min, ecal_min, hcal_min))
params = params[idx]
print(f"(eps, sim_min, ecal_min)={params}")

args = ['--infile', infile,
        '--n-workers', '4',
        '--n-events', '1000',
        '--eps', str(params[0]),
        '--sim-min', str(params[1]),
        '--ecal-min', str(params[2]),
        '--hcal-min', str(params[3]),
        '--statfile', f"{dm}_{idx}.csv"]
print(f"Running with args {args}")
build_tau_graphs.main(args)
