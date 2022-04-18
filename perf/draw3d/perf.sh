#!/bin/bash

LOG=./perf/draw3d/draw3d_perf.log
declare -a traces=(vase filmtv skybox coverflow evilskull polybump tekkaman carnival)

# exit when any command fails
set -e

# ensure build
make -s

# draw3d benchmarks
for TRACE in "${traces[@]}"
do
    echo -e "\n**************************************\n" >> $LOG
    echo -e "draw3d $TRACE benchmark\n" >> $LOG
    CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-t$TRACE.cgltrace -w8 -h8" >> $LOG
done