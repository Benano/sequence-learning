#!/bin/bash
#SBATCH --job-name="elise_array"
#SBATCH --time=7:00:00
#SBATCH --ntasks=10
#SBATCH --array=0-2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=12

Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate elise

NODE_ID=$SLURM_ARRAY_TASK_ID

SLURM_NTASKS=${SLURM_NTASKS:-10}  # Default to 10 tasks if not set

# using the SLURM_NTASKS env var
for i in $( seq 0 $((SLURM_NTASKS-1))); do

  TASK_ID=$((NODE_ID * SLURM_NTASKS + i))

  if [ $TASK_ID -eq 0 ]; then
    saving="--saving"
  else
      saving=""
  fi
    eval "python run.py --seed $TASK_ID $saving" &
    echo "Running task with ID: $TASK_ID"

done

wait
