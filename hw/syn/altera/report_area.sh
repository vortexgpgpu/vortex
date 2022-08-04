#!/bin/bash

# Dump Area Report
# first argument is the project name

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(realpath "${SCRIPT_DIR}")"

PROJECT_DIR=$1
PROJECT=$2
MODE=${3-fit}

echo "Running quartus_sh -t $SCRIPT_DIR/report_area.tcl $PROJECT $MODE in $PROJECT_DIR ..."

pushd $PROJECT_DIR
quartus_sh -t $SCRIPT_DIR/report_area.tcl $PROJECT $MODE
popd