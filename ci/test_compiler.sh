#!/bin/bash

# exit when any command fails
set -e

# rebuild runtime
make -C runtime clean
make -C runtime

# clear POCL cache
rm -rf ~/.cache/pocl

# rebuild runtime test
make -C tests/runtime/simple clean
make -C tests/runtime/simple
make -C tests/runtime/simple run

# rebuild native kernel
make -C tests/regression/dogfood clean-all
make -C tests/regression/dogfood
make -C tests/regression/dogfood run-rtlsim

# rebuild opencl kernel
make -C tests/opencl/sgemm clean-all
make -C tests/opencl/sgemm
make -C tests/opencl/sgemm run-rtlsim