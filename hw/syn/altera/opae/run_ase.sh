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

PROGRAM=$(basename "$2")
PROGRAM_DIR=`dirname $2`

POCL_RT_PATH=$TOOLDIR/pocl/runtime
VORTEX_RT_PATH=$SCRIPT_DIR/../../../../runtime

# Export ASE_WORKDIR variable
export ASE_WORKDIR=$BUILD_DIR/synth/work

# cleanup incomplete runs
rm -f $ASE_WORKDIR/.app_lock.pid 
rm -f $ASE_WORKDIR/.ase_ready.pid
rm -f $BUILD_DIR/synth/nohup.out

# Start Simulator in background (capture processs group pid)
pushd $BUILD_DIR/synth 
echo "  [DBG]  starting ASE simnulator (stdout saved to '$BUILD_DIR/synth/nohup.out')"
setsid make sim &> /dev/null &
SIM_PID=$!
popd

# Wait for simulator readiness
# When .ase_ready is created in the $ASE_WORKDIR, ASE is ready for simulation
while [ ! -f $ASE_WORKDIR/.ase_ready.pid ]
do
  sleep 1
done

# run application
pushd $PROGRAM_DIR
shift 2
echo "  [DBG]  running ./$PROGRAM $*"
ASE_LOG=0 LD_LIBRARY_PATH=$POCL_RT_PATH/lib:$VORTEX_RT_PATH/opae:$LD_LIBRARY_PATH ./$PROGRAM $*
popd

# stop the simulator (kill process group)
kill -- -$(ps -o pgid= $SIM_PID | grep -o '[0-9]*')
wait $SIM_PID 2> /dev/null