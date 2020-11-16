#!/bin/sh

# test single core
make -C driver/opae/vlsim clean
CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=1" make -C driver/opae/vlsim > /dev/null 2>&1
make -C driver/tests/dogfood run-vlsim
make -C benchmarks/opencl/sgemm run-vlsim

# test 2 cores
make -C driver/opae/vlsim clean
CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0" make -C driver/opae/vlsim > /dev/null 2>&1
make -C driver/tests/dogfood run-vlsim
make -C benchmarks/opencl/sgemm run-vlsim

# test 4 cores with L2
make -C driver/opae/vlsim clean
CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
make -C driver/tests/dogfood run-vlsim
make -C benchmarks/opencl/sgemm run-vlsim

# test 8 cores with L2
make -C driver/opae/vlsim clean
CONFIGS="-DNUM_CLUSTERS=2 -DNUM_CORES=4 -DL2_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
make -C driver/tests/dogfood run-vlsim
make -C benchmarks/opencl/sgemm run-vlsim

# test 16 cores with L2 and L3
make -C driver/opae/vlsim clean
CONFIGS="-DNUM_CLUSTERS=4 -DNUM_CORES=4 -DL2_ENABLE=1 -DL3_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
make -C driver/tests/dogfood run-vlsim
make -C benchmarks/opencl/sgemm run-vlsim

# test debug build
make -C driver/opae/vlsim clean
DEBUG=1 CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=1" make -C driver/opae/vlsim > /dev/null 2>&1
make -C driver/tests/demo run-vlsim

# test build with scope analyzer
make -C driver/opae clean
SCOPE=1 CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=1" make -C driver/opae > /dev/null 2>&1
make -C driver/tests/demo run-vlsim