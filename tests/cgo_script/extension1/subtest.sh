#!/bin/bash 

if [ -n "$1" ]; then
  folder=("$1")
else
  echo "No folder provided: $1"
  exit 1
fi

echo "Processing folder: $folder"
cd $folder
rm perf*.txt 
for level in 1 2 3 4 5 9; do
  echo "Running with VORTEX_DIVERGENCE_OPT_LEVEL=$level for app $folder"
  VORTEX_DIVERGENCE_OPT_LEVEL=$level make run-simx > ../log/log_opt_level_${folder}_${level}.txt 2>&1
done