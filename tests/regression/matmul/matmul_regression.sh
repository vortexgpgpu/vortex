#!/bin/bash 

# README:
# This script launches a sweep of TC_SIZE, TC_NUM and MATRIX SIZES
# default values of NUM_WARPS=32, NUM_THREADS=32, NUM_CORES=4, DATA_SIZE=1
# Edit matrix_sizes, tcsizes & tcnums variables to vary the sweep limits

# Define arrays for tc_size,tc_num and matrix sizes
matrix_sizes=(16 32 64 128 256 512)
tcsizes=(8 16 32)
tcnums=(4 8 16 32)

cd ../../../build/

# Loop through each combination of above configs
for size in "${matrix_sizes[@]}"; do
    for tcsize in "${tcsizes[@]}"; do
        for tcnum in "${tcnums[@]}"; do
            mkdir -p sim_final/mat${size}
            log_name="sim_final/mat${size}/tcsize${tcsize}_tcnum${tcnum}_32w32t"
            cmd="CONFIGS=\"-DTC_NUM=${tcnum} -DTC_SIZE=${tcsize}\" ./ci/blackbox.sh --cores=4 --app=matmul --driver=simx --threads=32 --warps=32 --args=\"-n${size} -d1\" --rebuild=1 --perf=1  > ${log_name} 2>&1"
            echo $cmd
            eval $cmd
        done
    done    
done
