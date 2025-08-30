#!/bin/bash 

if [ -n "$1" ]; then
  folders=("$1")
else
  folders=("vecadd")
fi

for folder in "${folders[@]}"; do
  echo "Processing folder: $folder"
  cd $folder
  rm *.txt 
  for level in 1 2 3 5 8 9; do
    export VORTEX_DIVERGENCE_OPT_LEVEL=$level
    echo "Running with VORTEX_DIVERGENCE_OPT_LEVEL=$VORTEX_DIVERGENCE_OPT_LEVEL $TEST"
    make run-simx
  done
  cd ../
done 

for folder in "${folders[@]}"; do
  echo "Processing folder: $folder"
  python parser.py $folder
done
