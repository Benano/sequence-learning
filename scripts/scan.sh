#!/bin/bash
#SBATCH --job-name="elise_array"
#SBATCH --time=2:00:00
#SBATCH --ntasks=10
#SBATCH --array=0
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=epyc2

Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate elise

param1='pattern_duration'
param_vals1=(10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0)

param2='num_lat'
param_vals2=(40 60 80 100 120 140 160 200)

NODE_ID=$SLURM_ARRAY_TASK_ID
par2=${param_vals2[$NODE_ID]}

for par1 in "${param_vals1[@]}"; do

  # Use a unique directory based on SLURM_JOB_ID
  WORKDIR="${SLURM_SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
  mkdir -p "$WORKDIR"

  # Copy all necessary files to the new directory
  cp -r "$SLURM_SUBMIT_DIR/"*.py \
    "$SLURM_SUBMIT_DIR/config.toml" \
    "$SLURM_SUBMIT_DIR/scan.sh" \
    "$SLURM_SUBMIT_DIR/patterns" \
    "$WORKDIR"

  # CHANGE PARAM 1
  sed -i '' "s/^${param1} = .*/${param1} = ${par1}/" "$WORKDIR/config.toml"
  sed -i '' "s/^${param2} = .*/${param2} = ${par2}/" "$WORKDIR/config.toml"

  parameter_tag="${param1}_${par1}_${param2}_${par2}"

  echo "Running in directory: $WORKDIR"
  eval "cd $WORKDIR"
  eval "python run.py --seed 42 --param_tag ${parameter_tag}" & # TODO ADD SEED HERE WHEN NEEDED
  eval "cd $SLURM_SUBMIT_DIR"


  rm -rf "$WORKDIR"  # Clean up the working directory after the job is done

done

wait
