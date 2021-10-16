#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

BUILD_DIR=$1

PROGRAM=$(basename "$2")
PROGRAM_DIR=`dirname $2`

VORTEX_DRV_PATH=$SCRIPT_DIR/../../../driver

# Export ASE_WORKDIR variable
export ASE_WORKDIR=$SCRIPT_DIR/$BUILD_DIR/work

shift 2

# cleanup incomplete runs
rm -f $ASE_WORKDIR/.app_lock.pid 
rm -f $ASE_WORKDIR/.ase_ready.pid
rm -f $SCRIPT_DIR/$BUILD_DIR/nohup.out

# Start Simulator in background
pushd $SCRIPT_DIR/$BUILD_DIR 
echo "  [DBG]  starting ASE simnulator (stdout saved to '$SCRIPT_DIR/$BUILD_DIR/nohup.out')"
nohup make sim & 
popd

# Wait for simulator readiness
# When .ase_ready is created in the $ASE_WORKDIR, ASE is ready for simulation
while [ ! -f $ASE_WORKDIR/.ase_ready.pid ]
do
  sleep 1
done

# run application
pushd $PROGRAM_DIR
echo "  [DBG]  running ./$PROGRAM $*"
ASE_LOG=0 LD_LIBRARY_PATH=$POCL_RT_PATH/lib:$VORTEX_DRV_PATH/asesim:$LD_LIBRARY_PATH ./$PROGRAM $*
popd