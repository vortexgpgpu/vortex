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
    echo "Usage: $0 [[--clusters=#n] [--cores=#n] [--warps=#n] [--threads=#n] [--l2cache] [--l3cache] [[--driver=#name] [--app=#app] [--args=#args] [--debug=#level] [--scope] [--saif] [--perf=#class] [--vcd_file=#file] [--saif_file=#file] [--log=logfile] [--nohup] [--help]]"
}

show_help()
{
    show_usage
    echo "  where"
    echo "--driver: gpu, simx, rtlsim, oape, xrt"
    echo "--app: any subfolder test under regression or opencl"
    echo "--class: 0=disable, 1=pipeline, 2=memsys"
    echo "--nohup: build and run in temp directory"
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
    SAIF=0
    HAS_ARGS=0
    HAS_NP=0
    PERF_CLASS=0
    CONFIGS="$CONFIGS"
    TEMPBUILD=0
    LOGFILE=run.log
    VCD_FILE=$PWD/trace.vcd
    SAIF_FILE=$PWD/trace.saif
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
            --scope)    SCOPE=1 ;;
            --saif)     SAIF=1 ;;
            --vcd_file=*)  VCD_FILE=${i#*=} ;;
            --saif_file=*) SAIF_FILE=${i#*=} ;;
            --args=*)   HAS_ARGS=1; ARGS=${i#*=} ;;
            --np=*)     HAS_NP=1; NP=${i#*=} ;;
            --log=*)    LOGFILE=${i#*=} ;;
            --nohup)    TEMPBUILD=1 ;;
            --help)     show_help; exit 0 ;;
            --*)        echo "Invalid argument: $i"; show_usage; exit 1 ;;
            *)          show_usage; exit 1 ;;
        esac
    done
}

set_driver_path() {
    case $DRIVER in
        gpu) DRIVER_PATH="" ;;
        simx|rtlsim|opae|xrt) DRIVER_PATH="$ROOT_DIR/sw/runtime/$DRIVER" ;;
        *) echo "Invalid driver: $DRIVER"; exit 1 ;;
    esac
}

set_app_path() {
    if [ -d "$APP" ]; then
        APP_PATH="$APP"
    elif [ -d "$ROOT_DIR/tests/$APP" ]; then
        APP_PATH="$ROOT_DIR/tests/$APP"
    elif [ -d "$ROOT_DIR/tests/regression/$APP" ]; then
        APP_PATH="$ROOT_DIR/tests/regression/$APP"
    elif [ -d "$ROOT_DIR/tests/opencl/$APP" ]; then
        APP_PATH="$ROOT_DIR/tests/opencl/$APP"
    elif [ -d "$ROOT_DIR/tests/hip/$APP" ]; then
        APP_PATH="$ROOT_DIR/tests/hip/$APP"
    else
        echo "Application folder not found: $APP"
        exit 1
    fi
}

build_driver() {
    local cmd_opts=""
    [ $DEBUG -ne 0 ] && cmd_opts=$(add_option "$cmd_opts" "DEBUG=$DEBUG_LEVEL")
    [ $SCOPE -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "SCOPE=1")
    [ $SAIF -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "SAIF=1")
    [ $TEMPBUILD -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "DESTDIR=\"$TEMPDIR\"")
    [ -n "$CONFIGS" ] && cmd_opts=$(add_option "$cmd_opts" "CONFIGS=\"$CONFIGS\"")
    cmd_opts=$(add_option "$cmd_opts" "make -C $DRIVER_PATH > /dev/null")
    echo "Running: $cmd_opts"
    eval "$cmd_opts"
    status=$?
    if [ $status -ne 0 ]; then
        echo "Error building driver: $DRIVER_PATH"
        exit $status
    fi
}

run_app() {
    local cmd_opts=""
    [ $DEBUG -ne 0 ] && cmd_opts=$(add_option "$cmd_opts" "DEBUG=$DEBUG_LEVEL")
    [ $TEMPBUILD -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "VORTEX_RT_LIB=\"$TEMPDIR\"")
    [ $HAS_ARGS -eq 1 ] && cmd_opts=$(add_option "$cmd_opts" "OPTS=\"$ARGS\"")
    [ -n "$CONFIGS" ] && cmd_opts=$(add_option "$cmd_opts" "CONFIGS=\"$CONFIGS\"")
    cmd_opts=$(add_option "$cmd_opts" "make -C \"$APP_PATH\" run-$DRIVER")
    [ $DEBUG -ne 0 ] && cmd_opts=$(add_option "$cmd_opts" "> $LOGFILE 2>&1")
    echo "Running: $cmd_opts"
    eval "$cmd_opts"
    status=$?
    return $status
}

main() {
    parse_args "$@"
    set_driver_path
    set_app_path

    if [ $SAIF -eq 1 ] && [ "$DRIVER" = "simx" ]; then
        echo "Error: SAIF is not supported with the simx driver"
        exit 1
    fi

    # execute on default installed GPU
    if [ "$DRIVER" = "gpu" ]; then
        run_app
        exit $?
    fi

    if [ -n "$CONFIGS" ]; then
        echo "CONFIGS=$CONFIGS"
    fi

    export VORTEX_PROFILING=$PERF_CLASS
    export VCD_FILE=$VCD_FILE
    export SAIF_FILE=$SAIF_FILE

    make -C "$ROOT_DIR/hw" config > /dev/null
    make -C "$ROOT_DIR/sw/runtime/stub" > /dev/null

    if [ $TEMPBUILD -eq 1 ]; then
        # setup temp directory
        TEMPDIR=$(mktemp -d)
        mkdir -p "$TEMPDIR"
        # build stub driver
        echo "Running: DESTDIR=$TEMPDIR make -C $ROOT_DIR/sw/runtime/stub"
        DESTDIR="$TEMPDIR" make -C $ROOT_DIR/sw/runtime/stub > /dev/null
        # stage a per-invocation copy of the app dir so concurrent trials do not
        # race on the shared `config.stamp` / build artifacts. Keep it as a
        # sibling of the original so relative paths (`../../..`, `../common.mk`)
        # still resolve.
        STAGED_APP="${APP_PATH%/}.trial.$$.$(date +%N)"
        cp -r "$APP_PATH" "$STAGED_APP"
        APP_PATH="$STAGED_APP"
        # register tempdir + staged app cleanup on exit
        trap "rm -rf $TEMPDIR $STAGED_APP" EXIT
    fi

    build_driver
    run_app
    status=$?

    exit $status
}

main "$@"