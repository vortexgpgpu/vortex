#!/bin/bash

cd ../../hw/syn/opae/

while getopts c: flag
do
  case "${flag}" in
    c) i=${OPTARG};; #cores: 1, 2, 4, 8, 16
  esac
done

if [[ ! "$i" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'Invalid parameter for argument -c (1, 2, 4, 8, or 16 expected)'
  exit 1
fi

date=$(date +%Y_%m_%d)
results_dir="../../../evaluation/perf_${date}"
mkdir -p ${results_dir}

mkdir -p "${results_dir}/${i}c"

cp "./build_fpga_${i}c/build.log" "${results_dir}/${i}c/build.log"
cp "./build_fpga_${i}c/build/output_files/afu_default.syn.summary" "${results_dir}/${i}c/afu_default.syn.summary"
cp "./build_fpga_${i}c/build/output_files/afu_default.fit.summary" "${results_dir}/${i}c/afu_default.fit.summary"
cp "./build_fpga_${i}c/build/output_files/afu_default.sta.summary" "${results_dir}/${i}c/afu_default.sta.summary"
cp "./build_fpga_${i}c/build/output_files/user_clock_freq.txt" "${results_dir}/${i}c/user_clock_freq.txt"

../../../ci/blackbox.sh --driver=fpga --app=sgemm --perf > "${results_dir}/${i}c/sgemm.result"
../../../ci/blackbox.sh --driver=fpga --app=vecadd --perf > "${results_dir}/${i}c/vecadd.result"
../../../ci/blackbox.sh --driver=fpga --app=saxpy --perf > "${results_dir}/${i}c/saxpy.result"
../../../ci/blackbox.sh --driver=fpga --app=sfilter --perf > "${results_dir}/${i}c/sfilter.result"
../../../ci/blackbox.sh --driver=fpga --app=nearn --perf > "${results_dir}/${i}c/nearn.result"
../../../ci/blackbox.sh --driver=fpga --app=guassian --perf > "${results_dir}/${i}c/guassian.result"
