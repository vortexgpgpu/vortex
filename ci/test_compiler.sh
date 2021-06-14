#!/bin/bash

# exit when any command fails
set -e

# rebuild runtime
make -C runtime clean
make -C runtime

# clear POCL cache
rm -rf ~/.cache/pocl

# rebuild native kernel
make -C tests/regression/dogfood clean-all
make -C tests/regression/dogfood
./ci/blackbox.sh --driver=vlsim --cores=1 --app=dogfood

# rebuild opencl kernel
make -C tests/opencl/sgemm clean-all
make -C tests/opencl/sgemm
./ci/blackbox.sh --driver=vlsim --cores=1 --app=sgemm