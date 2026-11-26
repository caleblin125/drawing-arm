# Running the physics test on a Slurm cluster (haydean)

```
# local machine: build the scenario (cams, followers, springs) and ship code + scenario
python -m drawarm physics prepare heart
git archive -o drawarm.tar HEAD && scp drawarm.tar haydean:~/
scp output/physics/scenario_heart.pkl haydean:~/drawing-arm/output/physics/

# cluster
ssh haydean
mkdir -p ~/drawing-arm && tar -xf ~/drawarm.tar -C ~/drawing-arm && cd ~/drawing-arm
bash hpc/setup_haydean.sh                       # once
mkdir -p output/physics
sbatch --array=0-19 hpc/physics_array.sbatch output/physics/scenario_heart.pkl 10   # 200 runs
squeue -u $USER
~/drawarm-venv/bin/python -m drawarm physics summarize output/physics/results_<jobid> -o mc_heart.md
```

Seed 0 is the nominal machine; every other seed draws manufacturing tolerances
(link lengths, cam profile print error, backlash, spring preload, stiffness,
pen friction, motor voltage and torque) from the ranges in `[physics]` of
`params.toml`.
