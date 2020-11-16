#!/bin/sh

# exit when any command fails
set -e

make -C benchmarks/opencl run
