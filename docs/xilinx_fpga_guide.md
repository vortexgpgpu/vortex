# FPGA Startup and Configuration Guide

XRT Environment Setup
----------------------

    $ source /opt/xilinx/Vitis/2023.1/settings64.sh
    $ source /opt/xilinx/xrt/setup.sh


Check Installed FPGA Platforms
------------------------------

    $ platforminfo -l


Build FPGA image
----------------

    $ cd hw/syn/xilinx/xrt
    $ PREFIX=test1 PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1 TARGET=hw NUM_CORES=4 make

Will run the synthesis under new build directory: BUILD_DIR := "\<PREFIX>\_\<PLATFORM>\_\<TARGET>"

The generated bitstream will be located under <BUILD_DIR>/bin/vortex_afu.xclbin

Sample FPGA Run Test
--------------------

Ensure you have the correct opae runtime for the FPGA target

    $ make -C runtime/xrt clean
    $ TARGET=hw make -C runtime/xrt

Run the following from your Vortex build directory

    $ TARGET=hw FPGA_BIN_DIR=<BUILD_DIR>/bin ./ci/blackbox.sh --driver=xrt --app=sgemm --args="-n128"