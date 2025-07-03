#!/bin/bash

for seed in {0..10}; do
  python run.py --seed $seed &
done

wait

