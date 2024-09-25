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

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$SCRIPT_DIR/..

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

add_option() {
    if [ -n "$1" ]; then
        echo "$1 $2"
    else
        echo "$2"
    fi
}

DEFAULTS() {
    DRIVER=simx
    APP=sgemm
    DEBUG=0
    DEBUG_LEVEL=0
    SCOPE=0
    HAS_ARGS=0
    PERF_CLASS=0
    CONFIGS="$CONFIGS"
    REBUILD=2
    TEMPBUILD=0
    LOGFILE=run.log
}

parse_args() {
    DEFAULTS
    for i in "$@"; do
        case $i in
            --driver=*) DRIVER=${i#*=} ;;
            --app=*)    APP=${i#*=} ;;
            --clusters=*) CONFIGS=$(add_option "$CONFIGS" "-DNUM_CLUSTERS=${i#*=}") ;;
            --cores=*)  CONFIGS=$(add_option "$CONFIGS" "-DNUM_CORES=${i#*=}") ;;
            --warps=*)  CONFIGS=$(add_option "$CONFIGS" "-DNUM_WARPS=${i#*=}") ;;
            --threads=*) CONFIGS=$(add_option "$CONFIGS" "-DNUM_THREADS=${i#*=}") ;;
            --l2cache)  CONFIGS=$(add_option "$CONFIGS" "-DL2_ENABLE") ;;
            --l3cache)  CONFIGS=$(add_option "$CONFIGS" "-DL3_ENABLE") ;;
            --perf=*)   CONFIGS=$(add_option "$CONFIGS" "-DPERF_ENABLE"); PERF_CLASS=${i#*=} ;;
            --debug=*)  DEBUG=1; DEBUG_LEVEL=${i#*=} ;;
            --scope)    SCOPE=1; ;;
            --args=*)   HAS_ARGS=1; ARGS=${i#*=} ;;
            --rebuild=*) REBUILD=${i#*=} ;;
            --log=*)    LOGFILE=${i#*=} ;;
            --help)     show_help; exit 0 ;;
            *)          show_usage; exit 1 ;;
        esac
    done

    if [ $REBUILD -eq 3 ];
    then
        REBUILD=1
        TEMPBUILD=1
    fi
}

set_driver_path() {
    case $DRIVER in
        gpu) DRIVER_PATH="" ;;
        simx|rtlsim|opae|xrt) DRIVER_PATH="$ROOT_DIR/runtime/$DRIVER" ;;
        *) echo "Invalid driver: $DRIVER"; exit 1 ;;
    esac
}

set_app_path() {
    if [ -d "$ROOT_DIR/tests/opencl/$APP" ]; then
        APP_PATH="$ROOT_DIR/tests/opencl/$APP"
    elif [ -d "$ROOT_DIR/tests/regression/$APP" ]; then
        APP_PATH="$ROOT_DIR/tests/regression/$APP"
    else
        echo "Application folder not found: $APP"
        exit 1
    fi
}

build_driver() {
    local cmd_opts=""
    [ $DEBUG -ne 0 ] && cmd_opts=$(add_option "$cmd_opts" "DEBUG=$DEBUG_LEVEL")
    [ $SCOPE -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "SCOPE=1")
    [ $TEMPBUILD -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "DESTDIR=\"$TEMPDIR\"")
    [ -n "$CONFIGS" ] && cmd_opts=$(add_option "$cmd_opts" "CONFIGS=\"$CONFIGS\"")

    if [ -n "$cmd_opts" ]; then
        echo "Running: $cmd_opts make -C $DRIVER_PATH > /dev/null"
        eval "$cmd_opts make -C $DRIVER_PATH > /dev/null"
    else
        echo "Running: make -C $DRIVER_PATH > /dev/null"
        make -C $DRIVER_PATH > /dev/null
    fi
}

run_app() {
    local cmd_opts=""
    [ $DEBUG -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "DEBUG=1")
    [ $TEMPBUILD -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "VORTEX_RT_PATH=\"$TEMPDIR\"")
    [ $HAS_ARGS -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "OPTS=\"$ARGS\"")

    if [ $DEBUG -ne 0 ]; then
        if [ -n "$cmd_opts" ]; then
            echo "Running: $cmd_opts make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
            eval "$cmd_opts make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
        else
            echo "Running: make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1"
            make -C $APP_PATH run-$DRIVER > $LOGFILE 2>&1
        fi
    else
        if [ -n "$cmd_opts" ]; then
            echo "Running: $cmd_opts make -C $APP_PATH run-$DRIVER"
            eval "$cmd_opts make -C $APP_PATH run-$DRIVER"
        else
            echo "Running: make -C $APP_PATH run-$DRIVER"
            make -C $APP_PATH run-$DRIVER
        fi
    fi
    status=$?
    return $status
}

main() {
    parse_args "$@"
    set_driver_path
    set_app_path

    # execute on default installed GPU
    if [ "$DRIVER" = "gpu" ]; then
        run_app
        exit $?
    fi

    if [ -n "$CONFIGS" ]; then
        echo "CONFIGS=$CONFIGS"
    fi

    if [ $REBUILD -ne 0 ]; then
        BLACKBOX_CACHE=blackbox.$DRIVER.cache
        LAST_CONFIGS=$(cat "$BLACKBOX_CACHE" 2>/dev/null || echo "")

        if [ $REBUILD -eq 1 ] || [ "$CONFIGS+$DEBUG+$SCOPE" != "$LAST_CONFIGS" ]; then
            make -C $DRIVER_PATH clean-driver > /dev/null
            echo "$CONFIGS+$DEBUG+$SCOPE" > "$BLACKBOX_CACHE"
        fi
    fi

    export VORTEX_PROFILING=$PERF_CLASS

    make -C "$ROOT_DIR/hw" config > /dev/null
    make -C "$ROOT_DIR/runtime/stub" > /dev/null

    if [ $TEMPBUILD -eq 1 ]; then
        # setup temp directory
        TEMPDIR=$(mktemp -d)
        mkdir -p "$TEMPDIR"
        # build stub driver
        echo "running: DESTDIR=$TEMPDIR make -C $ROOT_DIR/runtime/stub"
        DESTDIR="$TEMPDIR" make -C $ROOT_DIR/runtime/stub > /dev/null
        # register tempdir cleanup on exit
        trap "rm -rf $TEMPDIR" EXIT
    fi

    build_driver
    run_app
    status=$?

    if [ $DEBUG -eq 1 ] && [ -f "$APP_PATH/trace.vcd" ]; then
        mv -f $APP_PATH/trace.vcd .
    fi

    if [ $SCOPE -eq 1 ] && [ -f "$APP_PATH/scope.vcd" ]; then
        mv -f $APP_PATH/scope.vcd .
    fi

    exit $status
}

main "$@"