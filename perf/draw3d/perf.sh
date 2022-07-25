#!/bin/bash

LOG=./perf/draw3d/draw3d_perf.log
declare -a traces=(vase filmtv skybox coverflow evilskull polybump tekkaman carnival)

# exit when any command fails
set -e

# ensure build
make -s

# draw3d benchmarks
draw3d(){
    echo > $LOG # clear log
    for TRACE in "${traces[@]}"
    do
        echo -e "\n**************************************\n" >> $LOG
        echo -e "draw3d $TRACE benchmark\n" >> $LOG
        if [ $ALL = true ]
        then
            CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-t$TRACE.cgltrace -w512 -h512" >> $LOG
        else
            CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --app=draw3d --args="-t$TRACE.cgltrace -w512 -h512" | grep 'PERF' >> $LOG
        fi
    done
    echo "draw3d tests done!"
}

usage()
{
    echo "usage: [-a|--all] [-h|--help]"
}

case $1 in
    -a | --all )
        ALL=true
        draw3d
        ;;
    -h | --help )
        usage
        ;;
    -* | --* )
        echo "invalid option"
        usage
        ;;
    * )
        ALL=false
        draw3d
        ;;             
esac