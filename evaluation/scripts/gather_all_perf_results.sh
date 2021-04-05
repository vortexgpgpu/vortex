#!/bin/bash

cd ../../hw/syn/opae/

date=$(date +%Y_%m_%d)
results_dir="../../../evaluation/perf_${date}"
mkdir -p ${results_dir}

for ((i=1; i <= 16; i=i*2)); do
  mkdir -p "${results_dir}/${i}c"
done

for ((i=1; i <= 16; i=i*2)); do
  cp "./build_fpga_${i}c/build.log" "${results_dir}/${i}c/build.log"
  cp "./build_fpga_${i}c/build/output_files/afu_default.syn.summary" "${results_dir}/${i}c/afu_default.syn.summary"
  cp "./build_fpga_${i}c/build/output_files/afu_default.fit.summary" "${results_dir}/${i}c/afu_default.fit.summary"
  cp "./build_fpga_${i}c/build/output_files/afu_default.sta.summary" "${results_dir}/${i}c/afu_default.sta.summary"
  cp "./build_fpga_${i}c/build/output_files/user_clock_freq.txt" "${results_dir}/${i}c/user_clock_freq.txt"
done

cd ../../../evaluation/scripts
results_dir="../perf_${date}"

for ((i=1; i <= 16; i=i*2)); do
  echo "Programming fpga for ${i} core build..."
  ./program_fpga.sh -c ${i}
  echo "Running tests for ${i} core build..."
  ../../ci/blackbox.sh --driver=fpga --app=sgemm --perf > "${results_dir}/${i}c/sgemm.result"
  ../../ci/blackbox.sh --driver=fpga --app=vecadd --perf > "${results_dir}/${i}c/vecadd.result"
  ../../ci/blackbox.sh --driver=fpga --app=saxpy --perf > "${results_dir}/${i}c/saxpy.result"
  ../../ci/blackbox.sh --driver=fpga --app=sfilter --perf > "${results_dir}/${i}c/sfilter.result"
  ../../ci/blackbox.sh --driver=fpga --app=nearn --perf > "${results_dir}/${i}c/nearn.result"
  ../../ci/blackbox.sh --driver=fpga --app=guassian --perf > "${results_dir}/${i}c/guassian.result"
  echo "Done ${i} core build."
done
