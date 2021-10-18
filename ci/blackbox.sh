#!/bin/sh

show_usage()
{
    echo "Vortex BlackBox Test Driver v1.0"
    echo "Usage: [[--clusters=#n] [--cores=#n] [--warps=#n] [--threads=#n] [--l2cache] [--l3cache] [[--driver=rtlsim|vlsim|simx] [--debug] [--scope] [--perf] [--app=vecadd|sgemm|basic|demo|dogfood] [--args=<args>] [--help]]"
}

SCRIPT_DIR=$(dirname "$0")
VORTEX_HOME=$SCRIPT_DIR/..

DRIVER=vlsim
APP=sgemm
CLUSTERS=1
CORES=1
WARPS=4
THREADS=4
L2=0
L3=0
DEBUG=0
SCOPE=0
HAS_ARGS=0
DEBUG_LEVEL=1

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
        CORES=1        
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
        DRIVER_PATH=$VORTEX_HOME/driver/rtlsim
        ;;
    vlsim)
        DRIVER_PATH=$VORTEX_HOME/driver/vlsim
        ;;
    asesim)
        DRIVER_PATH=$VORTEX_HOME/driver/asesim
        ;;
    fpga)
        DRIVER_PATH=$VORTEX_HOME/driver/fpga
        ;; 
    simx)
        DRIVER_PATH=$VORTEX_HOME/driver/simx
        DEBUG_LEVEL=3
        ;;
    *)
        echo "invalid driver: $DRIVER"
        exit -1
        ;;
esac

if [ -d "$VORTEX_HOME/tests/opencl/$APP" ];
then
    APP_PATH=$VORTEX_HOME/tests/opencl/$APP
elif [ -d "$VORTEX_HOME/tests/regression/$APP" ];
then
    APP_PATH=$VORTEX_HOME/tests/regression/$APP
else
    echo "Application folder found: $APP"
    exit -1
fi

CONFIGS="-DNUM_CLUSTERS=$CLUSTERS -DNUM_CORES=$CORES -DNUM_WARPS=$WARPS -DNUM_THREADS=$THREADS -DL2_ENABLE=$L2 -DL3_ENABLE=$L3 $PERF_FLAG $CONFIGS"

echo "CONFIGS=$CONFIGS"

make -C $DRIVER_PATH clean

status=0

if [ $DEBUG -eq 1 ]
then    
    if [ $SCOPE -eq 1 ]
    then
        DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    else
        DEBUG=$DEBUG_LEVEL CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    fi    
    
    if [ $HAS_ARGS -eq 1 ]
    then
        OPTS=$ARGS make -C $APP_PATH run-$DRIVER > run.log 2>&1
        status=$?
    else
        make -C $APP_PATH run-$DRIVER > run.log 2>&1
        status=$?
    fi
    
    if [ -f "$APP_PATH/trace.vcd" ]
    then 
        mv -f $APP_PATH/trace.vcd .
    fi
else
    if [ $SCOPE -eq 1 ]
    then
        SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    else
        CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    fi
    
    if [ $HAS_ARGS -eq 1 ]
    then
        OPTS=$ARGS make -C $APP_PATH run-$DRIVER
        status=$?
    else
        make -C $APP_PATH run-$DRIVER
        status=$?
    fi
fi

exit $status