#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

unittest() 
{
    make -C tests/unittest run
}

coverage() 
{
    echo "begin coverage tests..."

    #make -C sim/simx clean
    #XLEN=64 make -C sim/simx
    #XLEN=64 make -C tests/runtime run-simx
    #XLEN=64 make -C tests/riscv/isa run-simx
    #XLEN=64 make -C tests/regression run-simx
    #XLEN=64 make -C tests/opencl run-simx
    #make -C . clean
    #XLEN=64 make -C .
    XLEN=64 make -C tests/riscv/isa run-rtlsim
    # XLEN=64 make -C tests/runtime run-rtlsim
    # XLEN=64 make -C tests/regression run-rtlsim

    echo "coverage tests done!"
}

tex()
{
    echo "begin texture tests..."

    CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=vlsim --app=tex --args="XLEN=64 -isoccer.png -osoccer_result.png -g0"
    CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=simx --app=tex --args="XLNE=64 -isoccer.png -osoccer_result.png -g0"
    CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="XLEN=64 -itoad.png -otoad_result.png -g1"
    CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=simx --app=tex --args="XLEN=64 -irainbow.png -orainbow_result.png -g2"
    CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=rtlsim --app=tex --args="XLEN=64 -itoad.png -otoad_result.png -g1" --perf
    CONFIGS="-DEXT_TEX_ENABLE=1" ./ci/blackbox.sh --driver=simx --app=tex --args="XLEN=64 -itoad.png -otoad_result.png -g1" --perf

    echo "coverage texture done!"
}

cluster() 
{
    echo "begin clustering tests..."

    # warp/threads configurations
    ./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=2 --threads=8 --app=demo --args="XLEN=64"
    ./ci/blackbox.sh --driver=rtlsim --cores=1 --warps=8 --threads=2 --app=demo --args="XLEN=64"
    ./ci/blackbox.sh --driver=simx --cores=1 --warps=8 --threads=16 --app=demo --args="XLEN=64"

    # cores clustering
    ./ci/blackbox.sh --driver=rtlsim --cores=1 --clusters=1 --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=rtlsim --cores=4 --clusters=1 --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=simx --cores=4 --clusters=1 --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --app=demo --args="XLEN=64 -n1"

    # L2/L3
    ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l3cache --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=simx --cores=4 --clusters=2 --l2cache --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=simx --cores=4 --clusters=4 --l2cache --l3cache --app=demo --args="XLEN=64 -n1"

    echo "clustering tests done!"
}

debug()
{
    echo "begin debugging tests..."

    ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --perf --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --perf --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --debug --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=simx --cores=2 --clusters=2 --l2cache --debug --app=demo --args="XLEN=64 -n1"
    ./ci/blackbox.sh --driver=vlsim --cores=1 --scope --app=basic --args="XLEN=64 -t0 -n1"

    echo "debugging tests done!"
}

config() 
{
    echo "begin configuration tests..."

    # disabling M extension
    CONFIGS=-DEXT_M_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext --args="XLEN=64"

    # disabling F extension
    CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext --args="XLEN=64"
    CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_mf_ext --perf --args="XLEN=64"
    CONFIGS=-DEXT_F_DISABLE ./ci/blackbox.sh --driver=simx --cores=1 --app=no_mf_ext --perf --args="XLEN=64"

    # disable shared memory
    CONFIGS=-DSM_ENABLE=0 ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem --args="XLEN=64"
    CONFIGS=-DSM_ENABLE=0 ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=no_smem --perf --args="XLEN=64"
    CONFIGS=-DSM_ENABLE=0 ./ci/blackbox.sh --driver=simx --cores=1 --app=no_smem --perf --args="XLEN=64"

    # using Default FPU core
    FPU_CORE=FPU_DEFAULT ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood --args="XLEN=64"

    # using FPNEW FPU core
    FPU_CORE=FPU_FPNEW ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=dogfood --args="XLEN=64"

    # using AXI bus
    AXI_BUS=1 ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=demo --args="XLEN=64"

    # adjust l1 block size to match l2
    CONFIGS="-DL1_BLOCK_SIZE=64" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=io_addr --args="XLEN=64 -n1"

    # test cache banking
    CONFIGS="-DDNUM_BANKS=1" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=io_addr --args="XLEN=64"
    CONFIGS="-DDNUM_BANKS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=io_addr --args="XLEN=64"
    CONFIGS="-DDNUM_BANKS=2" ./ci/blackbox.sh --driver=simx --cores=1 --app=io_addr --args="XLEN=64"

    # test cache multi-porting
    CONFIGS="-DDNUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=io_addr --args="XLEN=64"
    CONFIGS="-DDNUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=1 --app=demo --debug --args="XLEN=64 -n1"
    CONFIGS="-DL2_NUM_PORTS=2 -DDNUM_PORTS=2" ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=io_addr --args="XLEN=64"
    CONFIGS="-DL2_NUM_PORTS=4 -DDNUM_PORTS=4" ./ci/blackbox.sh --driver=rtlsim --cores=4 --l2cache --app=io_addr --args="XLEN=64"
    CONFIGS="-DL2_NUM_PORTS=4 -DDNUM_PORTS=4" ./ci/blackbox.sh --driver=simx --cores=4 --l2cache --app=io_addr --args="XLEN=64"

    # test 128-bit MEM block
    CONFIGS=-DMEM_BLOCK_SIZE=16 ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo --args="XLEN=64"

    # test single-bank DRAM
    CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_BANKS=1" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo --args="XLEN=64"

    # test 27-bit DRAM address
    CONFIGS="-DPLATFORM_PARAM_LOCAL_MEMORY_ADDR_WIDTH=27" ./ci/blackbox.sh --driver=vlsim --cores=1 --app=demo --args="XLEN=64"

    echo "configuration tests done!"
}

stress0() 
{
    echo "begin stress0 tests..."

    # test verilator reset values
    CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=sgemm --args="XLEN=64"
    CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=sgemm --args="XLEN=64"
    FPU_CORE=FPU_DEFAULT CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood --args="XLEN=64"
    FPU_CORE=FPU_DEFAULT CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=dogfood --args="XLEN=64"
    CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="XLEN=64"
    CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --cores=2 --clusters=2 --l2cache --l3cache --app=io_addr --args="XLEN=64"
    CONFIGS="-DVERILATOR_RESET_VALUE=0" ./ci/blackbox.sh --driver=vlsim --app=printf --args="XLEN=64"
    CONFIGS="-DVERILATOR_RESET_VALUE=1" ./ci/blackbox.sh --driver=vlsim --app=printf --args="XLEN=64"

    echo "stress0 tests done!"
}

stress1() 
{
    echo "begin stress1 tests..."

    ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --clusters=2 --l3cache --app=sgemm --args="XLEN=64 -n256"

    echo "stress1 tests done!"
}

show_usage()
{
    echo "Vortex 64-bit Regression Test"
    echo "Usage: $0 [-coverage] [-all] [-h|--help]"
}

while [ "$1" != "" ]; do
    case $1 in
        # Passed
        -unittest ) unittest
                ;;
        # Under dev
        -coverage ) coverage
                ;;
        # Still failing
        -tex ) tex
                ;;
        # Passed
        -cluster ) cluster
                ;;
        # Passed
        -debug ) debug
                ;;
        # Passed
        -config ) config
                ;;
        # Passed
        -stress0 ) stress0
                ;;
        # Passed
        -stress1 ) stress1
                ;;
        # Passed
        -stress ) stress0
                  stress1
                ;;
        -all ) coverage
                ;;
        -h | --help ) show_usage
                      exit
                ;;
        * )           show_usage
                      exit 1
    esac
    shift
done
