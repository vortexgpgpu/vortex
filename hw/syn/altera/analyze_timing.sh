#!/bin/bash

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

# Timing Analysis
# first argument is the project name

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(realpath "${SCRIPT_DIR}")"

PROJECT_DIR=$1
PROJECT=$2
MODE=${3-fit}

echo "Running quartus_sh -t $SCRIPT_DIR/report_area.tcl $PROJECT $MODE in $PROJECT_DIR ..."

pushd $PROJECT_DIR
quartus_sta -t $SCRIPT_DIR/analyze_timing.tcl $PROJECT $MODE
popd