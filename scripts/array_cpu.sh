#!/bin/bash
#SBATCH --job-name="elise_array"
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=epyc2
#
Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate elise

NODE_ID=$SLURM_ARRAY_TASK_ID
seed=$NODE_ID

SLURM_JOB_LABEL="${seed}_${SLURM_ARRAY_JOB_ID}"
WORKDIR="${SLURM_SUBMIT_DIR}/runs/${SLURM_JOB_LABEL}"
mkdir -p "$WORKDIR"

# Copy required files
cp -r "$SLURM_SUBMIT_DIR/"*.py \
      "$SLURM_SUBMIT_DIR/config.toml" \
      "$SLURM_SUBMIT_DIR/scan.sh" \
      "$SLURM_SUBMIT_DIR/patterns" \
      "$WORKDIR"

# Update config for current parameters
sed -i "s/^seed = .*/seed = ${seed}/" "$WORKDIR/config.toml"

if [ $NODE_ID -eq 0 ]; then
  saving="--saving"
else
    saving=""
fi
  srun --exclusive --cpus-per-task=2 --chdir="$WORKDIR" python run.py --param_tag "$TAG" --saving $saving
  wait

rm -rf "$WORKDIR"
