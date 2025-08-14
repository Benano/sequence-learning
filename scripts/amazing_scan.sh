#!/usr/bin/env sh

#!/bin/bash
#SBATCH --job-name="elise_array"
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=epyc2
#SBATCH --array=0-63  # 8 param_vals1 x 8 param_vals2 = 64 parameter combos
#SBATCH --output=slurm_logs/slurm-%A_A%a.out

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate elise

# --------------------
# Parameter definitions
# --------------------
param1='pattern_duration'
param_vals1=(10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0)
param2='num_lat'
param_vals2=(40 60 80 100 120 140 160 200)
seeds=(1 2 3 4 5)   # run these seeds serially for each param combo

num1=${#param_vals1[@]}
num2=${#param_vals2[@]}
total_combos=$((num1 * num2))

# --------------------
# Index mapping for param combos
# --------------------
IDX=$SLURM_ARRAY_TASK_ID
i=$((IDX % num1))
j=$((IDX / num1))

par1=${param_vals1[$i]}
par2=${param_vals2[$j]}

# --------------------
# Loop over seeds serially
# --------------------
for SEED in "${seeds[@]}"; do
    SLURM_JOB_LABEL="${par1}_${par2}_seed${SEED}"
    WORKDIR="${SLURM_SUBMIT_DIR}/runs/${SLURM_JOB_LABEL}"
    mkdir -p "$WORKDIR"

    # Copy required files
    cp -r "$SLURM_SUBMIT_DIR/"*.py \
          "$SLURM_SUBMIT_DIR/config.toml" \
          "$SLURM_SUBMIT_DIR/scan.sh" \
          "$SLURM_SUBMIT_DIR/patterns" \
          "$WORKDIR"

    # Update config for current parameters
    sed -i "s/^${param1} = .*/${param1} = ${par1}/" "$WORKDIR/config.toml"
    sed -i "s/^${param2} = .*/${param2} = ${par2}/" "$WORKDIR/config.toml"
    sed -i "s/^seed = .*/seed = ${SEED}/" "$WORKDIR/config.toml"

    # Run in the proper working directory
    echo "Running par1=$par1, par2=$par2, seed=$SEED"
    srun --exclusive --cpus-per-task=2 --chdir="$WORKDIR" python run.py --param_tag "$SLURM_JOB_LABEL" &

wait

done
