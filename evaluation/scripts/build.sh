#!/bin/bash

BUILD_DIR=../../hw/syn/opae

perf=0
wait=0

while getopts c:pwh flag
do
  case "${flag}" in
    c) cores=${OPTARG};; #1, 2, 4, 8, 16
    p) perf=1;; #perf counters enable
    w) wait=1;; # wait for build to complete
    h) echo "Usage: -c <cores> [-p perf] [-w wait] [-h help]"
       exit 0
    ;;
  \?)
    echo "Invalid option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done

if [[ ! "$cores" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'Invalid parameter for argument -c (1, 2, 4, 8, or 16 expected)'
  exit 1
fi

cd ${BUILD_DIR}

sources_file="./sources_${cores}c.txt"

if [ ${perf} = 1 ]; then
  if grep -Fxq '#+define+PERF_ENABLE' ${sources_file}; then
    sed -i 's/+define+PERF_ENABLE/#+define+PERF_ENABLE/' ${sources_file}
  elif ! grep -Fxq '+define+PERF_ENABLE' ${sources_file}; then
    sed -i '1s/^/+define+PERF_ENABLE\n/' ${sources_file}
  fi
else
  if grep -v '^ *#' ${sources_file} | grep -Fxq '+define+SYNTHESIS'; then
    sed -i 's/+define+PERF_ENABLE/#+define+PERF_ENABLE/' ${sources_file}
  elif ! grep -Fxq '#+define+PERF_ENABLE' ${sources_file}; then
    sed -i '1s/^/#+define+PERF_ENABLE\n/' ${sources_file}
  fi
fi

if [ -d "./build_fpga_{$cores}c" ]; then
  make "clean-fpga-${cores}c"
fi
make "fpga-${cores}c"

if [ ${wait} = 1 ]; then
  sleep 30
  pids=($(pgrep -f "${OPAE_PLATFORM_ROOT}|quartus"))
  for pid in ${pids[@]}; do
    while kill -0 ${pid} 2> /dev/null; do
      sleep 30
    done
  done
fi
