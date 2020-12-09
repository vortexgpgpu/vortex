#!/bin/sh

# exit when any command fails
set -e

show_usage()
{
    echo "Vortex BlackBox Test Driver v1.0"
    echo "Usage: [[--clusters=#n] [--cores=#n] [--warps=#n] [--threads=#n] [--l2cache] [--l3cache] [[--driver=rtlsim|vlsim] [--debug] [--scope] [--perf] [--app=vecadd|sgemm|basic|demo|dogfood] [--args=<args>] [--help]]"
}

DRIVER=vlsim
APP=sgemm
CLUSTERS=1
CORES=2
WARPS=4
THREADS=4
L2=0
L3=0
DEBUG=0
SCOPE=0
HAS_ARGS=0

for i in "$@"
do
case $i in
    --driver=*)
        DRIVER=${i#*=}
        shift
        ;;
    --app=*)
        APP=${i#*=}
        shift
        ;;
    --clusters=*)
        CLUSTERS=${i#*=}
        shift
        ;;
    --cores=*)
        CORES=${i#*=}
        shift
        ;;
    --warps=*)
        WARPS=${i#*=}
        shift
        ;;
    --threads=*)
        THREADS=${i#*=}
        shift
        ;;
    --l2cache)
        L2=1
        shift
        ;;
    --l3cache)
        L3=1
        shift
        ;;
    --debug)
        DEBUG=1
        shift
        ;;
    --scope)
        SCOPE=1
        shift
        ;;
    --perf)
        PERF_FLAG=-DPERF_ENABLE
        shift
        ;;
    --args=*)
        ARGS=${i#*=}
        HAS_ARGS=1
        shift
        ;;
    --help)
        show_usage
        exit 0
        ;;
    *)
    show_usage   
    exit -1       
    ;;
esac
done

case $DRIVER in
    rtlsim)
        DRIVER_PATH=driver/rtlsim
        DRIVER_EXTRA=
        ;;
    vlsim)
        DRIVER_PATH=driver/opae
        DRIVER_EXTRA=vlsim
        ;;
    asesim)
        DRIVER_PATH=driver/opae
        DRIVER_EXTRA=asesim
        ;;
    fpga)
        DRIVER_PATH=driver/opae
        DRIVER_EXTRA=fpga
        ;;
    *)
        echo "invalid driver: $DRIVER"
        exit -1
        ;;
esac

case $APP in
    sgemm)
        APP_PATH=benchmarks/opencl/sgemm
        ;;
    vecadd)
        APP_PATH=benchmarks/opencl/vacadd
        ;;
    basic)
        APP_PATH=driver/tests/basic
        ;;
    demo)
        APP_PATH=driver/tests/demo
        ;;
    dogfood)
        APP_PATH=driver/tests/dogfood
        ;;
    *)
        echo "invalid app: $APP"
        exit -1
        ;;
esac

CONFIGS="-DNUM_CLUSTERS=$CLUSTERS -DNUM_CORES=$CORES -DNUM_WARPS=$WARPS -DNUM_THREADS=$THREADS -DL2_ENABLE=$L2 -DL3_ENABLE=$L3 $PERF_FLAG"

echo "CONFIGS=$CONFIGS"

make -C $DRIVER_PATH clean

if [ $DEBUG -eq 1 ]
then    
    if [ $SCOPE -eq 1 ]
    then
        DEBUG=1 SCOPE=1 CONFIGS="$CONFIGS" make -s -C $DRIVER_PATH $DRIVER_EXTRA
    else
        DEBUG=1 CONFIGS="$CONFIGS" make -s -C $DRIVER_PATH $DRIVER_EXTRA
    fi    
    
    if [ $HAS_ARGS -eq 1 ]
    then
        OPTS=$ARGS make -C $APP_PATH run-$DRIVER > run.log 2>&1
    else
        make -C $APP_PATH run-$DRIVER > run.log 2>&1
    fi
else
    if [ $SCOPE -eq 1 ]
    then
        SCOPE=1 CONFIGS="$CONFIGS" make -s -C $DRIVER_PATH $DRIVER_EXTRA
    else
        CONFIGS="$CONFIGS" make -s -C $DRIVER_PATH $DRIVER_EXTRA
    fi
    
    if [ $HAS_ARGS -eq 1 ]
    then
        OPTS=$ARGS make -C $APP_PATH run-$DRIVER
    else
        make -C $APP_PATH run-$DRIVER
    fi
fi