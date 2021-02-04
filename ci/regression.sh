#!/bin/bash

./ci/test_runtime.sh
./ci/test_riscv_isa.sh  
./ci/test_opencl.sh
./ci/test_driver.sh  
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --perf --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --debug --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=vlsim --cores=1 --scope --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --l2cache --app=demo --args="-n1"
./ci/travis_run.py ./ci/blackbox.sh --driver=rtlsim --cores=2 --clusters=2 --l2cache --l3cache --app=demo --args="-n1"