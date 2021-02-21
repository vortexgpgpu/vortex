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

A new folder *build_fpga_`# of cores`c* will be created and the build will start and the build will take ~30-45 min to complete.
You could check last 10 lines in build log for possible errors or build completion.
    $ tail -n 10 ./build_fpga_`# of cores`c/build.log
Check if the build is still running by looking for quartus_sh, quartus_syn, or quartus_fit programs.
    $ ps -u `username`
If the build fails and you need to restart it, clean up the build folder using the following command:
    $ make clean-fpga-`# of cores`c
The following file should exist when the build is done:
    $ ls -lsa ./build_fpga_`# of cores`c/vortex_afu.gbs

Signing the bitstream
---------------------
    $ cd ./build_fpga_`# of cores`c/
    $ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs


Programming the FPGA
--------------------
    $ fpgasupdate vortex_afu_unsigned_ssl.gbs

FPGA sample test running OpenCL sgemm kernel
--------------------------------------------
Run the following from the Vortex root directory
    $ ./ci/blackbox.sh --driver=fpga --app=sgemm --args="-n64"