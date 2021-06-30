#!/bin/bash

# exit when any command fails
set -e

# clear POCL cache
rm -rf ~/.cache/pocl

# rebuild runtime
make -C runtime clean
make -C runtime

# rebuild runtime test
make -C tests/runtime/simple clean
make -C tests/runtime/simple
make -C tests/runtime/simple run-simx

# rebuild native kernel
make -C tests/regression/dogfood clean-all
make -C tests/regression/dogfood
make -C tests/regression/dogfood run-simx

# rebuild opencl kernel
make -C tests/opencl/sgemm clean-all
make -C tests/opencl/sgemm
make -C tests/opencl/sgemm run-simx