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

if [ -d "./build_fpga_{$cores}c" ]; then
  make "clean-fpga-${cores}c"
fi

if [ ${perf} = 1 ]; then
  PERF=1 make "fpga-${cores}c"
else
  make "fpga-${cores}c"
fi

if [ ${wait} = 1 ]; then
  sleep 30
  pids=($(pgrep -f "${OPAE_PLATFORM_ROOT}|quartus"))
  for pid in ${pids[@]}; do
    while kill -0 ${pid} 2> /dev/null; do
      sleep 30
    done
  done
fi
