#!/bin/bash

# exit when any command fails
set -e

# clear POCL cache
rm -rf ~/.cache/pocl

# rebuild runtime
make -C runtime clean
make -C runtime

# rebuild drivers
make -C driver clean
make -C driver

# rebuild runtime tests
make -C tests/runtime clean
make -C tests/runtime

# rebuild regression tests
make -C tests/regression clean-all
make -C tests/regression

# rebuild opencl tests
make -C tests/opencl clean-all
make -C tests/opencl