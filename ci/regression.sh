#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

# clear blackbox cache
rm -f blackbox.*.cache

smoke() 
{
echo "begin smoke tests..."

make -C tests/riscv/isa run-simx
make -C tests/riscv/isa run-rtlsim
make -C tests/runtime run-simx
make -C tests/runtime run-rtlsim
make -C tests/regression run-simx
make -C tests/regression run-rtlsim
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=vlsim --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim  --cores=2 --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g1.png -g1"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim  --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g1.png -g1" --debug=1
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-ttriangle.cgltrace -rtriangle_ref.png"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=rop --args="-rwhitebox.png"
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=immad
CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=immad

echo "smoke tests done!"
}

unittest() 
{
make -C tests/unittest run
}

coverage() 
{
echo "begin coverage tests..."

make -C tests/runtime run-rtlsim
make -C tests/riscv/isa run-rtlsim
make -C tests/regression run-vlsim
make -C tests/opencl run-vlsim
make -C tests/runtime run-simx
make -C tests/riscv/isa run-simx
make -C tests/regression run-simx
make -C tests/opencl run-simx

echo "coverage tests done!"
}

tex()
{
echo "begin texture tests..."

CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=vlsim --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g0.png -g0"
CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g0.png -g0"
CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g0.png -g0"
CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="-isoccer.png -osoccer_result.png -rsoccerref_g1.png -g1" --perf
CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -osoccer_result.png -rsoccer_ref_g1.png -g1" --perf
CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=simx --app=tex --args="-isoccer.png -orsoccer_result.png -rsoccer_ref_g2.png -g2"

echo "coverage texture done!"
}

cluster() 
{
echo "begin clustering tests..."

# warp/threads configurations
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=2 --threads=8 --app=demo
./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=2 --app=demo
./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=16 --app=demo

# cores clustering
./ci/blackbox.sh --driver=rtlsim --cores=1 --clusters=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=4 --clusters=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --app=demo --args="-n1"

# L2/L3
./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l3cache --app=demo --args="-n1"
./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --l2cache --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=4 --clusters=4 --l2cache --l3cache --app=demo --args="-n1"

echo "clustering tests done!"
}

debug()
{
echo "begin debugging tests..."

./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --perf --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --perf --app=demo --args="-n1"
./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --debug=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --debug=1 --app=demo --args="-n1"
./ci/blackbox.sh --driver=vlsim --cores=1 --scope --app=basic --args="-t0 -n1"

echo "debugging tests done!"
}

config() 
{
echo "begin configuration tests..."

# disabling M extension
CONFIGS=-DEXT_M_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext

# disabling F extension
CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext
CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext --perf
CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=simx --cores=1 --app=no_mf_ext --perf

# disable shared memory
CONFIGS=-DSM_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem
CONFIGS=-DSM_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem --perf
CONFIGS=-DSM_DISABLE ./ci/blackbox.sh --driver=simx --cores=1 --app=no_smem --perf

# using Default FPU core
FPU_CORE=FPU_DEFAULT ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood

# using FPNEW FPU core
FPU_CORE=FPU_FPNEW ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood

# using AXI bus
AXI_BUS=1 ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=demo

# adjust l1 block size to match l2
CONFIGS="-DL1_BLOCK_SIZE=64" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=io_addr --args="-n1"

# test cache banking
CONFIGS="-DDCACHE_NUM_BANKS=1" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=io_addr
CONFIGS="-DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=io_addr
CONFIGS="-DDCACHE_NUM_BANKS=2" ./ci/blackbox.sh --driver=simx --cores=1 --app=io_addr

# test cache multi-porting
CONFIGS="-DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=io_addr
CONFIGS="-DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=demo --debug=1 --args="-n1"
CONFIGS="-DL2_NUM_PORTS=2 -DDCACHE_NUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=io_addr
CONFIGS="-DL2_NUM_PORTS=4 -DDCACHE_NUM_PORTS=4" ./ci/blackbox.sh --driver=rtlsim --cores=4 --l2cache --app=io_addr
CONFIGS="-DL2_NUM_PORTS=4 -DDCACHE_NUM_PORTS=4" ./ci/blackbox.sh --driver=simx --cores=4 --l2cache --app=io_addr

# test 128-bit MEM block
CONFIGS=-DMEM_BLOCK_SIZE=16 ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test single-bank DRAM
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_BANKS=1" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

# test 27-bit DRAM address
CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=27" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo

echo "configuration tests done!"
}

stress0() 
{
echo "begin stress0 tests..."

# test verilator reset values
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=sgemm
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=sgemm
FPU_CORE=FPU_DEFAULT CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood
FPU_CORE=FPU_DEFAULT CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr
CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --app=printf
CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --app=printf

echo "stress0 tests done!"
}

stress1() 
{
echo "begin stress1 tests..."

./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --clusters=2 --l3cache --app=sgemm --args="-n256"

echo "stress1 tests done!"
}

usage()
{
    echo "usage: regression [-smoke] [-unittest] [-coverage] [-tex] [-cluster] [-debug] [-config] [-stress[#n]] [-all] [-h|--help]"
}

while [ "$1" != "" ]; do
    case $1 in
        -smoke ) smoke
                ;;
        -unittest ) unittest
                ;;
        -coverage ) coverage
                ;;
        -tex ) tex
                ;;
        -cluster ) cluster
                ;;
        -debug ) debug
                ;;
        -config ) config
                ;;
        -stress0 ) stress0
                ;;
        -stress1 ) stress1
                ;;
        -stress ) stress0
                  stress1
                ;;
        -all ) unittest
               coverage
               tex
               cluster
               debug
               config
               stress0
               stress1
                ;;
        -h | --help ) usage
                      exit
                ;;
        * )           usage
                      exit 1
    esac
    shift
done