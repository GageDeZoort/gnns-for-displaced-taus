import os
import sys
sys.path.append('../')

import itertools
import build_tau_graphs

# grab job ID
indir = '/tigress/jdezoort/displaced_tau_samples/'
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
infiles = os.listdir(indir)
infiles = [f for f in infiles if 'ntuple' in f]
f = os.path.join(indir, infiles[idx])
print(f"Running on {f}")

args = ['--infile', f,
        '--n-workers', '3',
        '--n-events', '10000',
        '--eps', '2',
        '--sim-min', '0.025',
        '--ecal-min', '0.025',
        '--hcal-min', '0.025',
        '--save-output', 'True']
print(f"Running with args {args}")
build_tau_graphs.main(args)
