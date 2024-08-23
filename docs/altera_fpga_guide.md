# FPGA Startup and Configuration Guide

OPAE Environment Setup
----------------------

    $ source /opt/inteldevstack/init_env_user.sh
    $ export OPAE_HOME=/opt/opae/1.1.2
    $ export PATH=$OPAE_HOME/bin:$PATH
    $ export C_INCLUDE_PATH=$OPAE_HOME/include:$C_INCLUDE_PATH
    $ export LIBRARY_PATH=$OPAE_HOME/lib:$LIBRARY_PATH
    $ export LD_LIBRARY_PATH=$OPAE_HOME/lib:$LD_LIBRARY_PATH

OPAE Build
------------------

The FPGA has to following configuration options:
- DEVICE_FAMILY=arria10 | stratix10
- NUM_CORES=#n

Command line:

    $ cd hw/syn/altera/opae
    $ PREFIX=test1 TARGET=fpga NUM_CORES=4 make

A new folder (ex: `test1_xxx_4c`) will be created and the build will start and take ~30-480 min to complete.
Setting TARGET=ase will build the project for simulation using Intel ASE.


OPAE Build Configuration
------------------------

The hardware configuration file `/hw/rtl/VX_config.vh` defines all the hardware parameters that can be modified when build the processor.For example, have the following parameters that can be configured:
- `NUM_WARPS`:   Number of warps per cores
- `NUM_THREADS`: Number of threads per warps
- `PERF_ENABLE`: enable the use of all profile counters

You configure the syntesis build from the command line:

    $ CONFIGS="-DPERF_ENABLE -DNUM_THREADS=8" make

OPAE Build Progress
-------------------

You could check the last 10 lines in the build log for possible errors until build completion.

    $ tail -n 10 <build_dir>/build.log

Check if the build is still running by looking for quartus_sh, quartus_syn, or quartus_fit programs.

    $ ps -u <username>

If the build fails and you need to restart it, clean up the build folder using the following command:

    $ make clean

The bitstream file `vortex_afu.gbs` should exist when the build is done:

    $ ls -lsa <build_dir>/synth/vortex_afu.gbs


Signing the bitstream and Programming the FPGA
----------------------------------------------

    $ cd <build_dir>
    $ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
    $ fpgasupdate vortex_afu_unsigned_ssl.gbs

Sample FPGA Run Test
--------------------

Ensure you have the correct opae runtime for the FPGA target

    $ make -C runtime/opae clean
    $ TARGET=FPGA make -C runtime/opae

Run the following from your Vortex build directory

    $ TARGET=fpga ./ci/blackbox.sh --driver=opae --app=sgemm --args="-n128"

