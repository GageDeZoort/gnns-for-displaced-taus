#!/bin/bash
#SBATCH --job-name=tcn-array-job    # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=10           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                   # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=6:02:00              # total run time limit (HH:MM:SS)
#SBATCH --output=slurm-%A.%a.out
#SBATCH --array=0-374

module purge
module load anaconda3
conda activate pyg2_env

echo $PWD
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Executing on machine: $(hostname)"

python scan_job.py
