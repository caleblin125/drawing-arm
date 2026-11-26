#!/bin/bash
# Monte-Carlo physics test of the drawing automaton, one Slurm array task per
# block of seeds.  Usage (from ~/drawing-arm on the cluster):
#   sbatch --array=0-19 hpc/physics_array.sbatch output/physics/scenario_heart.pkl 10
# -> output/physics/results_<jobid>/run_*.json ; then
#   ~/drawarm-venv/bin/python -m drawarm physics summarize output/physics/results_<jobid>
#SBATCH --job-name=drawarm-mc
#SBATCH --partition=no_dinos
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=output/physics/slurm-%A_%a.out
set -e
SCEN=${1:-output/physics/scenario_heart.pkl}
PER=${2:-10}
FIRST=$(( SLURM_ARRAY_TASK_ID * PER ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
~/drawarm-venv/bin/python -m drawarm physics run "$SCEN" --first "$FIRST" --count "$PER" \
    -o "output/physics/results_${SLURM_ARRAY_JOB_ID}"
