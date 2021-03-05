# Flubber FPGA Startup and Configuration Guide 

Flubber OPAE setup
------------------

    $ source /opt/inteldevstack/init_env_user.sh
    $ export OPAE_HOME=/opt/opae/1.1.2
    $ export PATH=$OPAE_HOME/bin:$PATH
    $ export C_INCLUDE_PATH=$OPAE_HOME/include:$C_INCLUDE_PATH
    $ export LIBRARY_PATH=$OPAE_HOME/lib:$LIBRARY_PATH
    $ export LD_LIBRARY_PATH=$OPAE_HOME/lib:$LD_LIBRARY_PATH
    $ export RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
    $ export PATH=:/opt/verilator/bin:$PATH
    $ export VERILATOR_ROOT=/opt/verilator


Flubber OPAE build
------------------

The Flubber FPGA has to following configuration options:
- 1 core fpga (fpga-1c)
- 2 cores fpga (fpga-2c)
- 4 cores fpga (fpga-4c)
- 8 cores fpga (fpga-8c)
- 16 cores fpga (fpga-16c)
    $ cd hw/syn/opae
    $ make fpga-`# of cores`c
Example: `make fpga-4c`

A new folder *build_fpga_`# of cores`c* will be created and the build will start and take ~30-45 min to complete.

Flubber Config Build Progress
-----------------------------

You could check the last 10 lines in the build log for possible errors until build completion.
    $ tail -n 10 ./build_fpga_`# of cores`c/build.log
Example: `tail -n 10 ./build_fpga_4c/build.log`

Check if the build is still running by looking for quartus_sh, quartus_syn, or quartus_fit programs.
    $ ps -u `username`

If the build fails and you need to restart it, clean up the build folder using the following command:
    $ make clean-fpga-`# of cores`c
Example: `make clean-fpga-4c`

The file `vortex_afu.gbs` should exist when the build is done:
    $ ls -lsa ./build_fpga_`# of cores`c/vortex_afu.gbs

Signing the bitstream and Programming the FPGA
----------------------------------------------

    $ cd ./build_fpga_`# of cores`c/
    $ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
    $ fpgasupdate vortex_afu_unsigned_ssl.gbs

FPGA sample test running OpenCL sgemm kernel
--------------------------------------------

Run the following from the Vortex root directory
    $ ./ci/blackbox.sh --driver=fpga --app=sgemm --args="-n64"

Build Script Configuration
--------------------------

Inside the ci folder there is a script called `blackbox.sh` which runs different tests on the Vortex processor with different configurations. Run:
    $ ./ci/blackbox.sh --help
To see the different configuration options available.
The most important ones are `--driver`, which runs the Vortex test on either the fpga, rtlsim, vlsim, or simx simulators, and `--perf`, which enables the profiling counters for each core.