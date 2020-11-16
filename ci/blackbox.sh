#!/bin/sh

# exit when any command fails
set -e

run_1c()
{
    # test single core
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=1" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_2c()
{
    # test 2 cores
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_4c()
{
    # test 4 cores
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=0" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_4c_l2()
{
    # test 4 cores with L2
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_8c_2l2()
{
    # test 8 cores with 2xL2
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=2 -DNUM_CORES=4 -DL2_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_4c_2l2_l3()
{
    # test 4 cores with L2 and L3
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=2 -DNUM_CORES=2 -DL2_ENABLE=1 -DL3_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_8c_4l2_l3()
{
    # test 8 cores with L2 and L3
    make -C driver/opae/vlsim clean
    CONFIGS="-DNUM_CLUSTERS=4 -DNUM_CORES=2 -DL2_ENABLE=1 -DL3_ENABLE=1" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_debug()
{
    # test debug build
    make -C driver/opae/vlsim clean
    DEBUG=1 CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=1" make -C driver/opae/vlsim > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

run_scope()
{
    # test build with scope analyzer
    make -C driver/opae clean
    SCOPE=1 CONFIGS="-DNUM_CLUSTERS=1 -DNUM_CORES=1" make -C driver/opae > /dev/null 2>&1
    make -C benchmarks/opencl/sgemm run-vlsim
}

usage()
{
    echo "usage: blackbox [[-run_1c] [-run_2c] [-run_4c] [-run_4c_l2] [-run_8c_2l2] [-run_4c_2l2_l3] [-run_8c_4l2_l3] [-run_debug] [-run_scope] [-all] [-h|--help]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -run_1c ) run_1c
                ;;
        -run_2c ) run_2c
                ;;
        -run_4c ) run_4c
                ;;
        -run_4c_l2 ) run_4c_l2
                ;;
        -run_8c_2l2 ) run_8c_2l2
                ;;
        -run_4c_2l2_l3 ) run_4c_2l2_l3
                ;;
        -run_8c_4l2_l3 ) run_8c_4l2_l3
                ;;
        -run_debug ) run_debug
                ;;
        -run_scope ) run_scope
                ;;
        -all ) run_1c
               run_2c
               run_4c
               run_4c_l2
               run_8c_2l2
               run_4c_2l2_l3
               run_8c_4l2_l3
               run_debug
               run_scope
               ;;
        -h | --help ) usage
                      exit
                      ;;
        * )           usage
                      exit 1
    esac
    shift
done