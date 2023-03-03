# FPGA Startup and Configuration Guide 

OPAE Environment Setup
----------------------

    $ source /opt/inteldevstack/init_env_user.sh
    $ export OPAE_HOME=/opt/opae/1.1.2
    $ export PATH=$OPAE_HOME/bin:$PATH
    $ export C_INCLUDE_PATH=$OPAE_HOME/include:$C_INCLUDE_PATH
    $ export LIBRARY_PATH=$OPAE_HOME/lib:$LIBRARY_PATH
    $ export LD_LIBRARY_PATH=$OPAE_HOME/lib:$LD_LIBRARY_PATH
    $ export RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
    $ export PATH=:/opt/verilator/bin:$PATH
    $ export VERILATOR_ROOT=/opt/verilator

OPAE Build
------------------

The FPGA has to following configuration options:
- DEVICE_FAMILY=arria10 | stratix10
- NUM_CORES=#n

Command line:

    $ cd hw/syn/opae
    $ NUM_CORES=4 make build

A new folder (ex: `build_fpga_4c`) will be created and the build will start and take ~30-480 min to complete.


OPAE Build Configuration
------------------------

The hardware configuration file `/hw/rtl/VX_config.vh` defines all the hardware parameters that can be modified when build the processor.For example, have the following parameters that can be configured:
- `NUM_WARPS`:   Number of warps per cores
- `NUM_THREADS`: Number of threads per warps
- `PERF_ENABLE`: enable the use of all profile counters

You configure the syntesis build from the command line:

    $ CONFIGS="-DPERF_ENABLE -DNUM_THREADS=8" make build

OPAE Build Progress
-------------------

You could check the last 10 lines in the build log for possible errors until build completion.

    $ tail -n 10 ./build_fpga_<num-of-cores>c/build.log

Check if the build is still running by looking for quartus_sh, quartus_syn, or quartus_fit programs.

    $ ps -u <username>

If the build fails and you need to restart it, clean up the build folder using the following command:

    $ make clean

The file `vortex_afu.gbs` should exist when the build is done:

    $ ls -lsa ./build_fpga_<num-of-cores>c/vortex_afu.gbs


Signing the bitstream and Programming the FPGA
----------------------------------------------

    $ cd ./build_fpga_<num-of-cores>c
    $ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
    $ fpgasupdate vortex_afu_unsigned_ssl.gbs

FPGA sample test running OpenCL sgemm kernel
--------------------------------------------

Run the following from the Vortex root directory

    $ ./ci/blackbox.sh --driver=fpga --app=sgemm --args="-n64"

