#!/bin/bash

# exit when any command fails
set -e

TEST=raster
WIDTH=256
HEIGHT=256
DRIVER=simx
CORES=1

SCRIPT_DIR=$(dirname "$0")
VORTEX_HOME=${SCRIPT_DIR}/../..
LOG_DIR=${SCRIPT_DIR}

raster()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=(vase filmtv skybox coverflow evilskull polybump tekkaman carnival)

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --app=draw3d --args="-onull -t$mode.cgltrace -w${WIDTH} -h${HEIGHT}" | grep 'Total elapsed time:' >> $LOG_FILE
    done
}

rastertile() 
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=(3 4 5 6 7)

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE -DRASTER_TILE_LOGSIZE=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -k$mode -tcarnival.cgltrace -w${WIDTH} -h${HEIGHT}" >> $LOG_FILE
    done
}

gpusw()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=("" "-z" "-y" "-x")

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --threads=1 --app=draw3d --args="-onull -tcarnival.cgltrace -w${WIDTH} -h${HEIGHT} ${mode}" >> $LOG_FILE
    done
}

rcache()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=("" "-DRCACHE_DISABLE")

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE $mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --threads=1 --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=4 >> $LOG_FILE
    done
}

ocache()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=("" "-DOCACHE_DISABLE")

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE $mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --threads=1 --app=draw3d --args="-onull -tcarnival.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=5 >> $LOG_FILE
    done
}

rslice()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=(1 2 4)

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE -DRASTER_NUM_SLICES=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --threads=1 --app=draw3d --args="-onull -tvase.cgltrace -e -w${WIDTH} -h${HEIGHT}" --perf=4 >> $LOG_FILE
    done
}

oslice()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=(1 2 4)

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do        
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE -DNUM_ROP_UNITS=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --threads=1 --app=draw3d --args="-onull -tcarnival.cgltrace -e -w${WIDTH} -h${HEIGHT}" --perf=5 >> $LOG_FILE
    done
}

show_usage()
{
    echo "Vortex Graphics Perf Test"
    echo "Usage: [--driver=#n] [--cores=#n] [--width=#n] [--height=#n] [--test=raster|rastertile|gpusw|rcache|ocache|rslice|oslice] [--help]"
}

for i in "$@"
do
case $i in
    --driver=*)
        DRIVER=${i#*=}
        shift
        ;;
    --cores=*)
        CORES=${i#*=}
        shift
        ;;
    --width=*)
        WIDTH=${i#*=}
        shift
        ;;
    --height=*)
        HEIGHT=${i#*=}
        shift
        ;;
    --test=*)
        TEST=${i#*=}
        shift
        ;;
    --help) 
        show_usage
        exit 0
        ;;
    *)  
        show_usage
        exit -1
        ;;
esac
done

echo "begin $TEST tests"

case $TEST in
    raster)
        CORES=1
        raster
        CORES=2
        raster
        CORES=4
        raster
        CORES=8
        raster
        CORES=16
        raster
        ;;
    rastertile)
        CORES=1
        rastertile
        CORES=2
        rastertile
        CORES=4
        rastertile
        CORES=8
        rastertile
        CORES=16
        rastertile
        ;;
    gpusw)
        CORES=1
        gpusw
        CORES=2
        gpusw
        CORES=4
        gpusw
        CORES=8
        gpusw
        CORES=16
        gpusw
        ;;
    rcache)
        CORES=1
        rcache
        CORES=2
        rcache
        CORES=4
        rcache
        CORES=8
        rcache
        CORES=16
        rcache
        ;;
    ocache)
        CORES=1
        ocache
        CORES=2
        ocache
        CORES=4
        ocache
        CORES=8
        ocache
        CORES=16
        ocache
        ;;
    rslice)
        CORES=1
        rslice
        CORES=4
        rslice
        CORES=16
        rslice
        ;;
    oslice)
        CORES=4
        oslice
        CORES=8
        oslice
        CORES=16
        oslice
        ;;
    *)
        echo "invalid test: $TEST"
        exit -1
        ;;
esac
    
echo "$TEST tests done!"