#!/bin/bash

# exit when any command fails
set -e

# Dogfood tests
./ci/test_runtime.sh
./ci/test_riscv_isa.sh  
./ci/test_opencl.sh
./ci/test_driver.sh  

# Build tests disabling extensions
CONFIGS=-DEXT_M_DISABLE make -C hw/simulate
CONFIGS=-DEXT_F_DISABLE make -C hw/simulate

# Blackbox tests
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --perf --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --debug --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --scope --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=demo --args="-n1"