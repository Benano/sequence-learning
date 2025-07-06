#!/bin/bash

for seed in {0..10}; do

  if [ $seed -eq 0 ]; then
    saving="--saving"
  else
      saving=""
  fi
    eval "python run.py --seed $seed $saving"

done

wait
