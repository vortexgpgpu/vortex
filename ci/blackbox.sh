#!/bin/sh

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

show_usage()
{
    echo "Vortex BlackBox Test Driver v1.0"
    echo "Usage: $0 [[--clusters=#n] [--cores=#n] [--warps=#n] [--threads=#n] [--l2cache] [--l3cache] [[--driver=#name] [--app=#app] [--args=#args] [--debug=#level] [--scope] [--perf=#class] [--rebuild=#n] [--log=logfile] [--help]]"
}

show_help()
{
    show_usage
    echo "  where"
    echo "--driver: gpu, simx, rtlsim, oape, xrt"
    echo "--app: any subfolder test under regression or opencl"
    echo "--class: 0=disable, 1=pipeline, 2=memsys"
    echo "--rebuild: 0=disable, 1=force, 2=auto, 3=temp"
}

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$SCRIPT_DIR/..

DRIVER=simx
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
REBUILD=2
TEMPBUILD=0
LOGFILE=run.log

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
    --rebuild=*)
        REBUILD=${i#*=}
        shift
        ;;
    --log=*)
        LOGFILE=${i#*=}
        shift
        ;;
    --help)
        show_help
        exit 0
        ;;
    *)
        show_usage
        exit -1
        ;;
esac
done

if [ $REBUILD -eq 3 ];
then
    REBUILD=1
    TEMPBUILD=1
fi

case $DRIVER in
    gpu)
        DRIVER_PATH=
        ;;
    simx)
        DRIVER_PATH=$ROOT_DIR/runtime/simx
        ;;
    rtlsim)
        DRIVER_PATH=$ROOT_DIR/runtime/rtlsim
        ;;
    opae)
        DRIVER_PATH=$ROOT_DIR/runtime/opae
        ;;
    xrt)
        DRIVER_PATH=$ROOT_DIR/runtime/xrt
        ;;
    *)
        echo "invalid driver: $DRIVER"
        exit -1
        ;;
esac

if [ -d "$ROOT_DIR/tests/opencl/$APP" ];
then
    APP_PATH=$ROOT_DIR/tests/opencl/$APP
elif [ -d "$ROOT_DIR/tests/regression/$APP" ];
then
    APP_PATH=$ROOT_DIR/tests/regression/$APP
else
    echo "Application folder not found: $APP"
    exit -1
fi

if [ "$DRIVER" = "gpu" ];
then
    # running application
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

    exit $status
fi

CONFIGS="-DNUM_CLUSTERS=$CLUSTERS -DNUM_CORES=$CORES -DNUM_WARPS=$WARPS -DNUM_THREADS=$THREADS $L2 $L3 $PERF_FLAG $CONFIGS"

echo "CONFIGS=$CONFIGS"

if [ $REBUILD -ne 0 ]
then
    BLACKBOX_CACHE=blackbox.$DRIVER.cache
    if [ -f "$BLACKBOX_CACHE" ]
    then
        LAST_CONFIGS=`cat $BLACKBOX_CACHE`
    fi

    if [ $REBUILD -eq 1 ] || [ "$CONFIGS+$DEBUG+$SCOPE" != "$LAST_CONFIGS" ];
    then
        make -C $DRIVER_PATH clean-driver > /dev/null
        echo "$CONFIGS+$DEBUG+$SCOPE" > $BLACKBOX_CACHE
    fi
fi

# export performance monitor class identifier
export VORTEX_PROFILING=$PERF_CLASS

status=0

# ensure config update
make -C $ROOT_DIR/hw config > /dev/null

# ensure the stub driver is present
make -C $ROOT_DIR/runtime/stub > /dev/null

if [ $DEBUG -ne 0 ]
then
    # running application
    if [ $TEMPBUILD -eq 1 ]
    then
        # setup temp directory
        TEMPDIR=$(mktemp -d)
        mkdir -p "$TEMPDIR/$DRIVER"

        # driver initialization
        if [ $SCOPE -eq 1 ]
        then
            echo "running: DESTDIR=$TEMPDIR/$DRIVER DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            DESTDIR="$TEMPDIR/$DRIVER" DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        else
            echo "running: DESTDIR=$TEMPDIR/$DRIVER DEBUG=$DEBUG_LEVEL CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            DESTDIR="$TEMPDIR/$DRIVER" DEBUG=$DEBUG_LEVEL CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        fi

        # running application
        if [ $HAS_ARGS -eq 1 ]
        then
            echo "running: VORTEX_RT_PATH=$TEMPDIR OPTS=$ARGS make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
            DEBUG=1 VORTEX_RT_PATH=$TEMPDIR OPTS=$ARGS make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1
            status=$?
        else
            echo "running: VORTEX_RT_PATH=$TEMPDIR make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
            DEBUG=1 VORTEX_RT_PATH=$TEMPDIR make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1
            status=$?
        fi

        # cleanup temp directory
        trap "rm -rf $TEMPDIR" EXIT
    else
        # driver initialization
        if [ $SCOPE -eq 1 ]
        then
            echo "running: DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            DEBUG=$DEBUG_LEVEL SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        else
            echo "running: DEBUG=$DEBUG_LEVEL CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            DEBUG=$DEBUG_LEVEL CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        fi

        # running application
        if [ $HAS_ARGS -eq 1 ]
        then
            echo "running: OPTS=$ARGS make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
            DEBUG=1 OPTS=$ARGS make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1
            status=$?
        else
            echo "running: make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
            DEBUG=1 make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1
            status=$?
        fi
    fi

    if [ -f "$APP_PATH/trace.vcd" ]
    then
        mv -f $APP_PATH/trace.vcd .
    fi
else
    if [ $TEMPBUILD -eq 1 ]
    then
        # setup temp directory
        TEMPDIR=$(mktemp -d)
        mkdir -p "$TEMPDIR/$DRIVER"

        # driver initialization
        if [ $SCOPE -eq 1 ]
        then
            echo "running: DESTDIR=$TEMPDIR/$DRIVER SCOPE=1 CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            DESTDIR="$TEMPDIR/$DRIVER" SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        else
            echo "running: DESTDIR=$TEMPDIR/$DRIVER CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            DESTDIR="$TEMPDIR/$DRIVER" CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        fi

        # running application
        if [ $HAS_ARGS -eq 1 ]
        then
            echo "running: VORTEX_RT_PATH=$TEMPDIR OPTS=$ARGS make -C $APP_PATH run-$DRIVER"
            VORTEX_RT_PATH=$TEMPDIR OPTS=$ARGS make -C $APP_PATH run-$DRIVER
            status=$?
        else
            echo "running: VORTEX_RT_PATH=$TEMPDIR make -C $APP_PATH run-$DRIVER"
            VORTEX_RT_PATH=$TEMPDIR make -C $APP_PATH run-$DRIVER
            status=$?
        fi

        # cleanup temp directory
        trap "rm -rf $TEMPDIR" EXIT
    else

        # driver initialization
        if [ $SCOPE -eq 1 ]
        then
            echo "running: SCOPE=1 CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            SCOPE=1 CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        else
            echo "running: CONFIGS=$CONFIGS make -C $DRIVER_PATH"
            CONFIGS="$CONFIGS" make -C $DRIVER_PATH > /dev/null
        fi

        # running application
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
fi

exit $status
