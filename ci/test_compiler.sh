#!/bin/bash

# exit when any command fails
set -e

# rebuild runtime
make -C runtime clean
make -C runtime

# rebuild native kernel
make -C driver/tests/dogfood clean-all
make -C driver/tests/dogfood
./ci/blackbox.sh --driver=vlsim --cores=1 --app=dogfood

# rebuild opencl kernel
make -C benchmarks/opencl/sgemm clean-all
make -C benchmarks/opencl/sgemm
./ci/blackbox.sh --driver=vlsim --cores=1 --app=sgemm