#!/bin/bash

SCRIPT_DIR=$PWD
PROGRAM=$(basename "$1")
PROGRAM_DIR=`dirname $1`

# Export ASE_WORKDIR variable
export ASE_WORKDIR=$SCRIPT_DIR/build_ase/work

shift 1

# cleanup incomplete runs
rm -rf $ASE_WORKDIR/.app_lock.pid $ASE_WORKDIR/.ase_ready.pid

# Start Simulator in background
pushd $SCRIPT_DIR/build_ase 
echo "  [DBG]  starting ASE simnulator"
nohup make sim  & 
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
ASE_LOG=0 LD_LIBRARY_PATH=../../opae/ase:$LD_LIBRARY_PATH ./$PROGRAM $*
popd