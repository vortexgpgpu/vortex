#!/bin/bash

# exit when any command fails
set -e

make -s

# Dogfood tests
./ci/test_runtime.sh
./ci/test_riscv_isa.sh  
./ci/test_opencl.sh
./ci/test_driver.sh  
./ci/test_simx.sh
./ci/test_compiler.sh

# Blackbox tests
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --perf --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --debug --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --scope --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=demo --args="-n1"

# Build tests disabling extensions
CONFIGS=-DEXT_M_DISABLE make -C hw/simulate
CONFIGS=-DEXT_F_DISABLE make -C hw/simulate

# disable shared memory
CONFIGS=-DSM_ENABLE=0 make -C hw/simulate

# test 128-bit DRAM bus
CONFIGS=-DPLATFORM_PARAM_LOCAL_MEMORY_DATA_SIZE_BITS=4 ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test 256-bit DRAM bus
CONFIGS=-DPLATFORM_PARAM_LOCAL_MEMORY_DATA_SIZE_BITS=4 ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo