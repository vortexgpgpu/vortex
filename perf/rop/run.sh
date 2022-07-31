#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

rtlsim()
{
	LOG=./perf/rop/rop_perf.log

	echo "begin rtlsim rop tests"

	# Clear log
	echo > $LOG

	for i in 1 4 16
	do
		echo "NUM_CORES = " $i >> $LOG
		CONFIGS="-DEXT_ROP_ENABLE" ./ci/blackbox.sh --driver=rtlsim --cores=$i --warps=4 --threads=4 --app=rop --args="-w128 -h128 -b -d" --perf=4 | grep 'PERF\|Total elapsed time' >> $LOG
		echo "**************************************" >> $LOG
	done

	echo "rtlsim rop tests done!"
}


simx()
{
	LOG=./perf/rop/rop_perf.log
	
	echo "begin simx rop tests"

	# Clear log
	echo > $LOG

	for i in 1 4 16
	do
		echo "NUM_CORES = " $i >> $LOG
		CONFIGS="-DEXT_GFX_ENABLE" ./ci/blackbox.sh --driver=simx --cores=$i --warps=4 --threads=4 --app=rop --args="-w128 -h128 -b -d" --perf=4 | grep 'PERF' >> $LOG
		echo "**************************************" >> $LOG
	done

	echo "simx rop tests done!"
}


usage()
{
    echo "usage: [-r] [-s] [-h|--help]"
}

case $1 in
    -r ) rtlsim
            ;;
    -s ) simx
            ;;
	-c ) carnival
		;;
    -h | --help ) usage
                    ;;
    * ) simx
        ;;             
esac
shift