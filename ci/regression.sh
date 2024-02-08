#!/bin/bash

# Copyright © 2019-2023
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# exit when any command fails
set -e

# clear blackbox cache
rm -f blackbox.*.cache

unittest() 
{
make -C tests/unittest run
make -C hw/unittest > /dev/null
}

isa() 
{
echo "begin isa tests..."

make -C tests/riscv/isa run-simx
make -C tests/riscv/isa run-rtlsim

make -C sim/rtlsim clean && CONFIGS="-DDPI_DISABLE" make -C sim/rtlsim > /dev/null
make -C tests/riscv/isa run-rtlsim

make -C sim/rtlsim clean && CONFIGS="-DFPU_FPNEW" make -C sim/rtlsim > /dev/null
make -C tests/riscv/isa run-rtlsim-32f

make -C sim/rtlsim clean && CONFIGS="-DFPU_DPI" make -C sim/rtlsim > /dev/null
make -C tests/riscv/isa run-rtlsim-32f

make -C sim/rtlsim clean && CONFIGS="-DFPU_DSP" make -C sim/rtlsim > /dev/null
make -C tests/riscv/isa run-rtlsim-32f

if [ "$XLEN" == "64" ]
then
        make -C sim/rtlsim clean && CONFIGS="-DFPU_FPNEW" make -C sim/rtlsim > /dev/null
        make -C tests/riscv/isa run-rtlsim-64f

        make -C sim/rtlsim clean && CONFIGS="-DEXT_D_ENABLE -DFPU_FPNEW" make -C sim/rtlsim > /dev/null
        make -C tests/riscv/isa run-rtlsim-64d || true

        make -C sim/rtlsim clean && CONFIGS="-DFPU_DPI" make -C sim/rtlsim > /dev/null
        make -C tests/riscv/isa run-rtlsim-64f

        make -C sim/rtlsim clean && CONFIGS="-DFPU_DSP" make -C sim/rtlsim > /dev/null
        make -C tests/riscv/isa run-rtlsim-64fx
fi

# restore default prebuilt configuration
make -C sim/rtlsim clean && make -C sim/rtlsim > /dev/null

echo "isa tests done!"
}

regression() 
{
echo "begin regression tests..."

make -C tests/kernel run-simx
make -C tests/kernel run-rtlsim

make -C tests/regression run-simx
make -C tests/regression run-rtlsim

# test FPU hardware implementations
CONFIGS="-DFPU_DPI" ./ci/blackbox.sh --driver=rtlsim --app=dogfood
CONFIGS="-DFPU_DSP" ./ci/blackbox.sh --driver=rtlsim --app=dogfood
CONFIGS="-DFPU_FPNEW" ./ci/blackbox.sh --driver=rtlsim --app=dogfood

# test local barrier
./ci/blackbox.sh --driver=simx --app=dogfood --args="-n1 -t19"
./ci/blackbox.sh --driver=rtlsim --app=dogfood --args="-n1 -t19"

# test global barrier
CONFIGS="-DGBAR_ENABLE" ./ci/blackbox.sh --driver=simx --app=dogfood --args="-n1 -t20" --cores=2
CONFIGS="-DGBAR_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=dogfood --args="-n1 -t20" --cores=2

# test FPU core

echo "regression tests done!"
}

opencl() 
{
echo "begin opencl tests..."

make -C tests/opencl run-simx
make -C tests/opencl run-rtlsim

echo "opencl tests done!"
}

cluster() 
{
echo "begin clustering tests..."

# warp/threads configurations
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=1 --threads=1 --app=diverge
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=2 --threads=2 --app=diverge
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=2 --threads=8 --app=diverge
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=2 --app=diverge
./ci/blackbox.sh --driver=simx --cores=1 --warps=1 --threads=1 --app=diverge
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=16 --app=diverge

# cores clustering
./ci/blackbox.sh --driver=rtlsim --cores=1 --clusters=1 --app=diverge --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=4 --clusters=1 --app=diverge --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=diverge --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=1 --app=diverge --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --app=diverge --args="-n1"

# L2/L3
./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=diverge --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l3cache --app=diverge --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --l2cache --app=diverge --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=4 --l2cache --l3cache --app=diverge --args="-n1"

echo "clustering tests done!"
}

debug()
{
echo "begin debugging tests..."

# test CSV trace generation
make -C sim/simx clean && DEBUG=3 make -C sim/simx > /dev/null
make -C sim/rtlsim clean && DEBUG=3 CONFIGS="-DGPR_RESET" make -C sim/rtlsim > /dev/null
make -C tests/riscv/isa run-simx-32im > run_simx.log
make -C tests/riscv/isa run-rtlsim-32im > run_rtlsim.log
./ci/trace_csv.py -trtlsim run_rtlsim.log -otrace_rtlsim.csv
./ci/trace_csv.py -tsimx run_simx.log -otrace_simx.csv
diff trace_rtlsim.csv trace_simx.csv
# restore default prebuilt configuration
make -C sim/simx clean && make -C sim/simx > /dev/null
make -C sim/rtlsim clean && make -C sim/rtlsim > /dev/null

./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --perf=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --perf=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --debug=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --debug=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=opae --cores=1 --scope --app=basic --args="-t0 -n1"

echo "debugging tests done!"
}

config() 
{
echo "begin configuration tests..."

# disable DPI
CONFIGS="-DDPI_DISABLE -DFPU_FPNEW" ./ci/blackbox.sh --driver=rtlsim --app=dogfood
CONFIGS="-DDPI_DISABLE -DFPU_FPNEW" ./ci/blackbox.sh --driver=opae --app=dogfood

# issue width
CONFIGS="-DISSUE_WIDTH=1" ./ci/blackbox.sh --driver=rtlsim --app=diverge
CONFIGS="-DISSUE_WIDTH=2" ./ci/blackbox.sh --driver=rtlsim --app=diverge
CONFIGS="-DISSUE_WIDTH=1" ./ci/blackbox.sh --driver=simx --app=diverge
CONFIGS="-DISSUE_WIDTH=2" ./ci/blackbox.sh --driver=simx --app=diverge

# dispatch size
CONFIGS="-DNUM_ALU_BLOCK=1 -DNUM_ALU_LANES=1" ./ci/blackbox.sh --driver=rtlsim --app=diverge
CONFIGS="-DNUM_ALU_BLOCK=2 -DNUM_ALU_LANES=2" ./ci/blackbox.sh --driver=rtlsim --app=diverge
CONFIGS="-DNUM_ALU_BLOCK=1 -DNUM_ALU_LANES=1" ./ci/blackbox.sh --driver=simx --app=diverge
CONFIGS="-DNUM_ALU_BLOCK=2 -DNUM_ALU_LANES=2" ./ci/blackbox.sh --driver=simx --app=diverge

# FPU scaling
CONFIGS="-DNUM_ALU_BLOCK=4 -DNUM_FPU_LANES=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DNUM_ALU_BLOCK=2 -DNUM_FPU_LANES=4" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DNUM_ALU_BLOCK=4 -DNUM_FPU_LANES=4" ./ci/blackbox.sh --driver=rtlsim --app=sgemm

# custom program startup address
make -C tests/regression/dogfood clean-all
STARTUP_ADDR=0x40000000 make -C tests/regression/dogfood
CONFIGS="-DSTARTUP_ADDR=0x40000000" ./ci/blackbox.sh --driver=simx --app=dogfood
CONFIGS="-DSTARTUP_ADDR=0x40000000" ./ci/blackbox.sh --driver=rtlsim --app=dogfood
make -C tests/regression/dogfood clean-all
make -C tests/regression/dogfood

# disabling M extension
CONFIGS="-DEXT_M_DISABLE" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext

# disabling F extension
CONFIGS="-DEXT_F_DISABLE" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext
CONFIGS="-DEXT_F_DISABLE" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext --perf=1
CONFIGS="-DEXT_F_DISABLE" ./ci/blackbox.sh --driver=simx --cores=1 --app=no_mf_ext --perf=1

# disable shared memory
CONFIGS="-DSM_DISABLE" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem
CONFIGS="-DSM_DISABLE" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem --perf=1
CONFIGS="-DSM_DISABLE" ./ci/blackbox.sh --driver=simx --cores=1 --app=no_smem --perf=1

# disable L1 cache
CONFIGS="-DL1_DISABLE -DSM_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DDCACHE_DISABLE" ./ci/blackbox.sh --driver=rtlsim --app=sgemm

# multiple L1 caches per cluster
CONFIGS="-DNUM_DCACHES=2 -DNUM_ICACHES=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --cores=8 --warps=1 --threads=2

# test AXI bus
AXI_BUS=1 ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=demo

# adjust l1 block size to match l2
CONFIGS="-DL1_LINE_SIZE=64" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=io_addr --args="-n1"

# test cache banking
CONFIGS="-DSMEM_NUM_BANKS=4 -DDCACHE_NUM_BANKS=1" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DSMEM_NUM_BANKS=2 -DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm
CONFIGS="-DSMEM_NUM_BANKS=2 -DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=simx --app=sgemm
CONFIGS="-DDCACHE_NUM_BANKS=1" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm
CONFIGS="-DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=sgemm
CONFIGS="-DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=simx --cores=1 --app=sgemm

# test 128-bit MEM block
CONFIGS="-DMEM_BLOCK_SIZE=16" ./ci/blackbox.sh --driver=opae --cores=1 --app=demo

# test single-bank DRAM
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_BANKS=1" ./ci/blackbox.sh --driver=opae --cores=1 --app=demo

# test 27-bit DRAM address
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=27" ./ci/blackbox.sh --driver=opae --cores=1 --app=demo

echo "configuration tests done!"
}

stress0() 
{
echo "begin stress0 tests..."

# test verilator reset values
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=opae --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=opae --app=printf

echo "stress0 tests done!"
}

stress1() 
{
echo "begin stress1 tests..."

./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n128" --l2cache

echo "stress1 tests done!"
}

synthesis()
{
echo "begin synthesis tests..."

PREFIX=build_base make -C hw/syn/yosys clean
PREFIX=build_base CONFIGS="-DDPI_DISABLE -DEXT_F_DISABLE" make -C hw/syn/yosys elaborate

echo "synthesis tests done!"
}

show_usage()
{
    echo "Vortex Regression Test" 
    echo "Usage: $0 [--unittest] [--isa] [--regression] [--opencl] [--cluster] [--debug] [--config] [--stress[#n]] [--synthesis] [--all] [--h|--help]"
}

start=$SECONDS

while [ "$1" != "" ]; do
    case $1 in
        --unittest ) unittest
                ;;
        --isa ) isa
                ;;
        --regression ) regression
                ;;
        --opencl ) opencl
                ;;
        --cluster ) cluster
                ;;
        --debug ) debug
                ;;
        --config ) config
                ;;
        --stress0 ) stress0
                ;;
        --stress1 ) stress1
                ;;
        --stress ) stress0
                  stress1
                ;;
        --synthesis ) synthesis
                ;;
        --all ) unittest               
               isa
               regression
               opencl
               cluster
               debug
               config
               stress0
               stress1
               synthesis
                ;;
        -h | --help ) show_usage
                      exit
                ;;
        * )           show_usage
                      exit 1
    esac
    shift
done

echo "Regression completed!"

duration=$(( SECONDS - start ))
awk -v t=$duration 'BEGIN{t=int(t*1000); printf "Elapsed Time: %d:%02d:%02d\n", t/3600000, t/60000%60, t/1000%60}'
