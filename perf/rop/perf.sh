#!/bin/sh

# exit when any command fails
set -e

SCRIPT_DIR=$(dirname "$0")
VORTEX_HOME=$SCRIPT_DIR/../../

# ensure build
make -C $VORTEX_HOME -s

echo $SCRIPT_DIR
echo $VORTEX_HOME

rtlsim()
{
	LOG=./perf/rop/perf_rtlsim.log

	echo "begin rtlsim rop tests"

	# Clear log
	echo > $LOG

	for i in {1..16}
	do
		echo "NUM_CORES = " $i >> $LOG
		CONFIGS="-DEXT_ROP_ENABLE" $VORTEX_HOME/ci/blackbox.sh --driver=rtlsim --cores=$i --app=rop --args="-rwhitebox_128.png -w128 -h128" --perf=4 | grep 'PERF' >> $LOG
		echo "**************************************" >> $LOG
	done

	echo "rtlsim rop tests done!"
}


simx()
{
	LOG=./perf/rop/perf_simx.log
	
	echo "begin simx rop tests"

	# Clear log
	echo > $LOG

	for i in {1..16}
	do
		echo "NUM_CORES = " $i >> $LOG
		CONFIGS="-DEXT_GFX_ENABLE" $VORTEX_HOME/ci/blackbox.sh --driver=simx --cores=$i --app=rop --args="-rwhitebox_128.png -w128 -h128" --perf=4 | grep 'PERF' >> $LOG
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
    -h | --help ) usage
                    ;;
    * ) simx
        ;;             
esac
shift