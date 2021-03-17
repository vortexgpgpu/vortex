#!/bin/bash

while getopts c:p: flag
do
  case "${flag}" in
    c) cores=${OPTARG};; #1, 2, 4, 8, 16
    p) perf=${OPTARG};; #perf counters enable (y/n)
  esac
done

if [[ ! "$cores" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'Invalid parameter for argument -c (1, 2, 4, 8, or 16 expected)'
  exit 1
fi

cd ../../hw/syn/opae

sources_file="./sources_${cores}c.txt"

if [ ${perf:0:1} = "n" ]; then
  if grep -v '^ *#' ${sources_file} | grep -Fxq '+define+SYNTHESIS'; then
    sed -i 's/+define+PERF_ENABLE/#+define+PERF_ENABLE/' ${sources_file}
  elif ! grep -Fxq '#+define+PERF_ENABLE' ${sources_file}; then
    sed -i '1s/^/#+define+PERF_ENABLE\n/' ${sources_file}
  fi
elif [ ${perf:0:1} = "y" ]; then
  if grep -Fxq '#+define+PERF_ENABLE' ${sources_file}; then
    sed -i 's/+define+PERF_ENABLE/#+define+PERF_ENABLE/' ${sources_file}
  elif ! grep -Fxq '+define+PERF_ENABLE' ${sources_file}; then
    sed -i '1s/^/+define+PERF_ENABLE\n/' ${sources_file}
  fi
else
  echo 'Invalid parameter for argument -p (y/n expected)'
  exit 1
fi

if [ -d "./build_fpga_{$cores}c" ]; then
  make "clean-fpga-${cores}c"
fi
make "fpga-${cores}c"

sleep 30

pids=($(pgrep -f "${OPAE_PLATFORM_ROOT}|quartus"))
for pid in ${pids[@]}; do
  while kill -0 ${pid} 2> /dev/null; do
    sleep 30
  done
done

