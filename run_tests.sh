#!/bin/bash
set -e

mkdir -p test_outputs

output_dir="$(pwd)/test_outputs"

(cd rtl ; python3 gen_synth_configs.py ; ls -l configs)

config_location=rtl/configs

declare -a test_names=("sgemm" "saxpy" "bfs" "guassian" "vecadd" "nearn" "sfilter")

for test_name in ${test_names[@]}; do
 	if [ ! -d "benchmarks/new_opencl/$test_name" ]; then
		echo "Unknown benchmark $test_name"
		exit 1
	fi
done


for filename in "$config_location"/*.sh; do

name=${filename##*/}
base=${name%.*}

. "$filename"

make -C rtl build_config
make -C runtime build_config
make -C driver/sw/rtlsim

for test_name in ${test_names[@]}; do

(

echo "Running $base-$test_name..."

cd "benchmarks/new_opencl/$test_name"
make clean
make
make run-rtlsim 2>&1 | tee "$output_dir/$base-$test_name.log"
) &

done # test_name

wait

done # config


