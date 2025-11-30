#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

sgemm()
{
echo "begin cache tests"

CONFIGS="-DICACHE_NUM_WAYS=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n64" --perf=1 | grep 'PERF' > cache_perf.log
echo -e "\n**************************************\n" >> cache_perf.log
CONFIGS="-DDCACHE_NUM_WAYS=2" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n64" --perf=1 | grep 'PERF' >> cache_perf.log
echo -e "\n**************************************\n" >> cache_perf.log
CONFIGS="-DICACHE_NUM_WAYS=4" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n64" --perf=1 | grep 'PERF' >> cache_perf.log
echo -e "\n**************************************\n" >> cache_perf.log
CONFIGS="-DDCACHE_NUM_WAYS=4" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n64" --perf=1 | grep 'PERF' >> cache_perf.log
echo -e "\n**************************************\n" >> cache_perf.log
CONFIGS="-DICACHE_NUM_WAYS=8" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n64" --perf=1 | grep 'PERF' >> cache_perf.log
echo -e "\n**************************************\n" >> cache_perf.log
CONFIGS="-DDCACHE_NUM_WAYS=8" ./ci/blackbox.sh --driver=rtlsim --app=sgemm --args="-n64" --perf=1 | grep 'PERF' >> cache_perf.log

echo "cache tests done!"
}

usage()
{
    echo "usage: [-s] [-h|--help]"
}

case $1 in
    -s ) sgemm
            ;;
    -h | --help ) usage
                    ;;
    * ) sgemm
        ;;
esac
shift