#!/usr/bin/env sh

#!/bin/bash
#SBATCH --job-name="elise_array"
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=epyc2
#SBATCH --array=0-7  # 8 param_vals1 x 8 param_vals2 = 64 parameter combos
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
seeds_count=${#seeds[@]}
total_combos=$((num1 * num2 * seeds_count))

# --------------------
# Index mapping for param combos
# --------------------
IDX=$SLURM_ARRAY_TASK_ID

# Calculate indices:
seed_idx=$((IDX % seeds_count))
i=$(( (IDX / seeds_count) % num1 ))
j=$(( IDX / (seeds_count * num1) ))

par1=${param_vals1[$i]}
par2=${param_vals2[$j]}
seed=${seeds[$seed_idx]}

# --------------------
# Create working directory and copy files
# --------------------

SLURM_JOB_LABEL="${par1}_${par2}_${seed}_${SLURM_ARRAY_JOB_ID}"

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
sed -i "s/^seed = .*/seed = ${seed}/" "$WORKDIR/config.toml"

# Run in the proper working directory
echo "Running par1=$par1, par2=$par2, seed=$seed"
TAG="${par1}_${par2}"
srun --exclusive --cpus-per-task=2 --chdir="$WORKDIR" python run.py --param_tag "$TAG"

done
