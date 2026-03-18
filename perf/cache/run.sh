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