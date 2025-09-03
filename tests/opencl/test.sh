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
    echo "Running with VORTEX_DIVERGENCE_OPT_LEVEL=$VORTEX_DIVERGENCE_OPT_LEVEL $TEST"
    VORTEX_DIVERGENCE_OPT_LEVEL=$level make run-simx &
  done
  cd ../
done 

#for folder in "${folders[@]}"; do
#  echo "Processing folder: $folder"
#  python parser.py $folder
#done
