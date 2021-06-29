#!/bin/bash

# exit when any command fails
set -e

# build sources
make -s

# coverage tests
make -C tests/runtime run-rtlsim
make -C tests/riscv/isa run-rtlsim
make -C tests/regression run-vlsim
make -C tests/opencl run-vlsim
make -C tests/runtime run-simx
make -C tests/riscv/isa run-simx
make -C tests/regression run-simx
make -C tests/opencl run-simx

# warp/threads configurations
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=2 --threads=8 --app=demo
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=2 --app=demo

# cores clustering
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=1 --clusters=1 --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="-n1"

# L2/L3
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l3cache --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="-n1"

# build flags
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --perf --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --debug --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --scope --app=basic --args="-t0 -n1"

# disabling M extension
CONFIGS=-DEXT_M_DISABLE make -C hw/simulate

# disabling F extension
CONFIGS=-DEXT_F_DISABLE make -C hw/simulate

# disable shared memory
CONFIGS=-DSM_ENABLE=0 make -C hw/simulate

# using Default FPU core
FPU_CORE=FPU_DEFAULT ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood

# using FPNEW FPU core
FPU_CORE=FPU_FPNEW ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood

# test 128-bit MEM block
CONFIGS=-DMEM_BLOCK_SIZE=16 ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test 128-bit MEM and DRAM block
CONFIGS="-DMEM_BLOCK_SIZE=16 -DPLATFORM_PARAM_LOCAL_MEMORY_DATA_WIDTH=128 -DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=28 -DPLATFORM_PARAM_LOCAL_MEMORY_BANKS=1" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test 27-bit DRAM address
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=27" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test 128-bit DRAM block
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_DATA_WIDTH=128 -DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=28 -DPLATFORM_PARAM_LOCAL_MEMORY_BANKS=1" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test verilator reset values
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=4 --app=sgemm
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=4 --app=sgemm

# test long memory latency
CONFIGS="-DMEM_LATENCY=100 -DMEM_RQ_SIZE=4 -DMEM_STALLS_MODULO=4" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=sgemm

# test pipeline stress
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm --args="-n128"