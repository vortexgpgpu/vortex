#!/bin/bash

# exit when any command fails
set -e

TEST=perf
WIDTH=256
HEIGHT=256
DRIVER=simx
CORES=1

SCRIPT_DIR=$(dirname "$0")
VORTEX_HOME=${SCRIPT_DIR}/../..
LOG_DIR=${SCRIPT_DIR}

perf()
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

gpusw()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=("" "-x" "-y")

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT} ${mode}" >> $LOG_FILE
    done
}

rtile() 
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=(3 4 5 6 7)

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE -DRASTER_TILE_LOGSIZE=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -k$mode -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" >> $LOG_FILE
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
        CONFIGS="-DEXT_GFX_ENABLE -DRASTER_NUM_SLICES=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=4 >> $LOG_FILE
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
        CONFIGS="-DEXT_GFX_ENABLE -DNUM_ROP_UNITS=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=5 >> $LOG_FILE
    done
}

tslice()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=(1 2 4)

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE -DNUM_TEX_UNITS=$mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=3 >> $LOG_FILE
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
        CONFIGS="-DEXT_GFX_ENABLE $mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=4 >> $LOG_FILE
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
        CONFIGS="-DEXT_GFX_ENABLE $mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=5 >> $LOG_FILE
    done
}

tcache()
{
    SUFFIX=${TEST}_${DRIVER}_${CORES}c_${WIDTH}x${HEIGHT}
    LOG_FILE=${LOG_DIR}/${SUFFIX}.log
        
    declare -a modes=("" "-DTCACHE_DISABLE")

    echo > $LOG_FILE # clear log
    for mode in "${modes[@]}"
    do
        echo -e "\n###############################################################################\n" >> $LOG_FILE
        echo -e "$TEST mode=$mode" >> $LOG_FILE
        CONFIGS="-DEXT_GFX_ENABLE $mode" ${VORTEX_HOME}/ci/blackbox.sh --driver=${DRIVER} --cores=${CORES} --app=draw3d --args="-onull -tvase.cgltrace -w${WIDTH} -h${HEIGHT}" --perf=3 >> $LOG_FILE
    done
}

show_usage()
{
    echo "Vortex Graphics Perf Test"
    echo "Usage: [--driver=#n] [--cores=#n] [--width=#n] [--height=#n] [--test=perf|gpusw|rtile|rcache|ocache|tcache|rslice|oslice|tslice] [--help]"
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

# clear blackbox cache
rm -f blackbox.*.cache

echo "begin $TEST tests"

case $TEST in
    perf)
        CORES=1
        perf
        CORES=2
        perf
        CORES=4
        perf
        CORES=8
        perf
        CORES=16
        perf
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
    rtile)
        CORES=1
        rtile
        CORES=4
        rtile
        CORES=16
        rtile
        ;;
    rcache)
        CORES=1
        rcache
        CORES=4
        rcache
        CORES=16
        rcache
        ;;
    ocache)
        CORES=1
        ocache
        CORES=4
        ocache
        CORES=16
        ocache
        ;;
    tcache)
        CORES=1
        tcache
        CORES=4
        tcache
        CORES=16
        tcache
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
        CORES=1
        oslice
        CORES=4
        oslice
        CORES=16
        oslice
        ;;
    tslice)
        CORES=1
        tslice
        CORES=4
        tslice
        CORES=16
        tslice
        ;;
    *)
        echo "invalid test: $TEST"
        exit -1
        ;;
esac
    
echo "$TEST tests done!"