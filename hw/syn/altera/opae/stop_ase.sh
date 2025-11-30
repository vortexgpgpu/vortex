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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

BUILD_DIR=$(realpath $1)

# Export ASE_WORKDIR variable
export ASE_WORKDIR=$BUILD_DIR/synth/work

# stop the simulator (kill process group)
if [ -f "$ASE_WORKDIR/.ase_ready.pid" ]; then
    SIM_PID=$(grep '^pid' "$ASE_WORKDIR/.ase_ready.pid" | cut -d'=' -f2 | tr -d ' ')
    echo "  [DBG]  stopping ASE simulator (pid=$SIM_PID)"
    kill -- -$(ps -o pgid= $SIM_PID | grep -o '[0-9]*')
    wait $SIM_PID 2> /dev/null
else
    echo "ASE PID file does not exist."
fi