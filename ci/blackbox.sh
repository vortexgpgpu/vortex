#!/bin/sh

show_usage()
{
    echo "Vortex BlackBox Test Driver v1.0"
    echo "Usage: [[--clusters=#n] [--cores=#n] [--warps=#n] [--threads=#n] [--l2cache] [--l3cache] [[--driver=#name] [--app=#app] [--args=#args] [--debug=#level] [--scope] [--perf=#class] [--help]]"
}

SCRIPT_DIR=$(dirname "$0")
VORTEX_HOME=$SCRIPT_DIR/..

DRIVER=vlsim
APP=sgemm
CLUSTERS=1
CORES=1
WARPS=4
THREADS=4
L2=
L3=
DEBUG=0
DEBUG_LEVEL=0
SCOPE=0
HAS_ARGS=0
PERF_CLASS=0

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
        L2=-DL2_ENABLE
        shift
        ;;
    --l3cache)
        L3=-DL3_ENABLE
        shift
        ;;
    --debug=*)
        DEBUG_LEVEL=${i#*=}
        DEBUG=1
        shift
        ;;
    --scope)
        SCOPE=1
        CORES=1        
        shift
        ;;
    --perf=*)
        PERF_FLAG=-DPERF_ENABLE
        PERF_CLASS=${i#*=}    
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

CONFIGS="-DNUM_CLUSTERS=$CLUSTERS -DNUM_CORES=$CORES -DNUM_WARPS=$WARPS -DNUM_THREADS=$THREADS $L2 $L3 $PERF_FLAG $CONFIGS"

echo "CONFIGS=$CONFIGS"

BLACKBOX_CACHE=blackbox.$DRIVER.cache

if [ -f "$BLACKBOX_CACHE" ]
then 
    LAST_CONFIGS=`cat $BLACKBOX_CACHE`
fi

if [ "$CONFIGS+$DEBUG+$SCOPE" != "$LAST_CONFIGS" ]; 
then
    make -C $DRIVER_PATH clean
fi

echo "$CONFIGS+$DEBUG+$SCOPE" > $BLACKBOX_CACHE

# export performance monitor class identifier
export PERF_CLASS=$PERF_CLASS

status=0

# ensure config update
make -C hw config

# ensure the stub driver is present
make -C $VORTEX_HOME/driver/stub

if [ $DEBUG -ne 0 ]
then    
    if [ $SCOPE -eq 1 ]
    then
        echo "running: DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH"
        DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    else
        echo "running: DEBUG=$DEBUG_LEVEL CONFIGS="$CONFIGS" make -C $DRIVER_PATH"
        DEBUG=$DEBUG_LEVEL CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    fi    
    
    if [ $HAS_ARGS -eq 1 ]
    then
        echo "running: OPTS=$ARGS make -C $APP_PATH run-$DRIVER > run.log 2>&1"
        OPTS=$ARGS make -C $APP_PATH run-$DRIVER > run.log 2>&1
        status=$?
    else
        echo "running: make -C $APP_PATH run-$DRIVER > run.log 2>&1"
        make -C $APP_PATH run-$DRIVER > run.log 2>&1
        status=$?
    fi
    
    if [ -f "$APP_PATH/trace.vcd" ]
    then 
        mv -f $APP_PATH/trace.vcd .
    fi
else
    echo "driver initialization..."
    if [ $SCOPE -eq 1 ]
    then
        echo "running: SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH"
        SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    else
        echo "running: CONFIGS="$CONFIGS" make -C $DRIVER_PATH"
        CONFIGS="$CONFIGS" make -C $DRIVER_PATH
    fi
    
    echo "running application..."
    if [ $HAS_ARGS -eq 1 ]
    then
        echo "running: OPTS=$ARGS make -C $APP_PATH run-$DRIVER"
        OPTS=$ARGS make -C $APP_PATH run-$DRIVER
        status=$?
    else
        echo "running: make -C $APP_PATH run-$DRIVER"
        make -C $APP_PATH run-$DRIVER
        status=$?
    fi
fi

exit $status