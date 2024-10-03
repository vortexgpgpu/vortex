# FPGA Startup and Configuration Guide

## Gaining Access to FPGA's with CRNCH
If you are associated with Georgia Tech and need remote access to the FPGA's, you can utilize CRNCH's server.

## What is CRNCH?

**C**enter for **R**esearch into **N**ovel **C**omputing **H**ierarchies

## What does CRNCH Offer?

**The Rogues Gallery (RG)**: new concept focused on developing our understanding of next-generation hardware with a focus on unorthodox and uncommon technologies. **RG** will acquire new and unique hardware (ie, the aforementioned “*rogues*”) from vendors, research labs, and startups and make this hardware available to students, faculty, and industry collaborators within a managed data center environment

## Why are the Rouges Important?

By exposing students and researchers to this set of unique hardware, we hope to foster cross-cutting discussions about hardware designs that will drive future *performance improvements in computing long after the Moore’s Law era of “cheap transistors” ends*.

## How is the Rouges Gallery Funded?

Rogues Gallery testbed is primarily supported by the National Science Foundation (NSF) under NSF Award Number [#2016701](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2016701&HistoricalAwards=false)

## Rouges Gallery Documentation

You can read about RG in more detail on its official documentation [page](https://gt-crnch-rg.readthedocs.io/en/main/index.html#).

You can listen to a talk about RG [here](https://mediaspace.gatech.edu/media/Jeff%20Young%20-%20Rogues%20Gallery%20-%20CRNCH%20Summit%202021/1_lqlgr0jj)

[CRNCH Summit 2023](https://github.com/gt-crnch/crnch-summit-2023/tree/main)

## Request Access for Rouges Gallery

You should use [this form](https://crnch-rg.cc.gatech.edu/request-rogues-gallery-access/) to request access to RG’s reconfigurable computing (vortex fpga) resources. You should receive an email with your ticket item being created. Once it gets processed, you should get an email confirmed your access has been granted. It might take some time to get processed.

## How to Access Rouges Gallery?

CRNCH resources do not require any VPN access for GT members so you can head to the web url for open on-demand: [rg-ood.crnch.gatech.edu](http://rg-ood.crnch.gatech.edu/)

Alternatively, you can `ssh` into rg with: `ssh <your-gt-acctname>@rg-login.crnch.gatech.edu`

(`ssh usubramanya3@rg-login.crnch.gatech.edu`)

Once you’ve logged in, you can use Slurm to request other nodes within the testbed. See more information on Slurm at [this page](https://gt-crnch-rg.readthedocs.io/en/main/general/using-slurm.html).

Note that you can also use VSCode to log into the Rogues Gallery via its terminal functionality. See [this page for more details](https://gt-crnch-rg.readthedocs.io/en/main/general/visual-studio-code.html).

## **What Machines are Available in the Rogues Gallery?**

Complete list of machines can be found [here](https://gt-crnch-rg.readthedocs.io/en/main/general/rg-hardware.html). 

## Which Machine do we Need from RG?

There are three primary nodes you might use. The table below summarizes:

| Name | Device | Description |
| --- | --- | --- |
| flubber1 | u50 | can synthesize vortex |
| flubber4 | u250 | missing HBM |
| flubber5 | u280 | can synthesize vortex |


*Note*: The `USERSCRATCH` folder is synchronized between all RG nodes. That means you can upload your files to `rg-login` and have them available on `flubber[1,4-5`. Changes on one node will be reflected across all nodes.

## How to Access flubber for Synthesis?

Now that you have the files prepared and available on the FPGA node, you can start the synthesis.  To run on hardware we need a rg-xilinx-fpga-hw cluster which includes **flubber[1,4-5]**. First `ssh` into the rouges gallery:

```bash
ssh <username>[@rg-login.crnch.gatech.edu](mailto:usubramanya3@rg-login.crnch.gatech.edu)
```

Then, to access the hardware node you need to `ssh` into flubber:

```bash
ssh flubber1
```

## Synthesis for Xillinx Boards

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

---

The directory `hw/syn/xilinx/xrt` contains the makefile used to synthesize Vortex.

For long-running jobs, invocation of this makefile can be made of the following form:

`[CONFIGS=<vortex macros>] [PREFIX=<prefix directory name>] [NUM_CORES=<#>] TARGET=hw|hw_emu PLATFORM=<platform baseName> nohup make > <log filename> 2>&1 &`

For example:

```bash
CONFIGS="-DL2_ENABLE -DDCACHE_SIZE=8192" PREFIX=build_4c_u280 NUM_CORES=4 TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 nohup make > build_u280_hw_4c.log 2>&1 &
```

The build is complete when the bitstream file `vortex_afu.xclbin` exists in `<prefix directory name><platform baseName>hw|hw_emu/bin`.

## Running a Program on FPGA

The blackbox.sh script in `ci` can be used to run a test with Vortex’s xrt driver using the following command:

`FPGA_BIN_DIR=<path to bitstream directory> TARGET=hw|hw_emu PLATFORM=<platform baseName> ./ci/blackbox.sh --driver=xrt --app=<test name>`

For example:

`FPGA_BIN_DIR=`realpath hw/syn/xilinx/xrt/build_4c_u280_xilinx_u280_gen3x16_xdma_1_202211_1_hw/bin` TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 ./ci/blackbox.sh --driver=xrt --app=demo`

## Synthesis for Intel (Altera) Boards

To set up the environment, source the XRT setup.sh and other Xilinx scripts. For example:

```
source /opt/xilinx/xrt/setup.sh
source /tools/reconfig/xilinx/Vivado/2022.1/settings64.sh
source /tools/reconfig/xilinx/Vitis/2022.1/settings64.sh

```

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

The file `vortex_afu.gbs` should exist when the build is done:

    $ ls -lsa <build_dir>/synth/vortex_afu.gbs


Signing the bitstream and Programming the FPGA
----------------------------------------------

    $ cd <build_dir>
    $ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
    $ fpgasupdate vortex_afu_unsigned_ssl.gbs

FPGA sample test running OpenCL sgemm kernel
--------------------------------------------

Run the following from the Vortex root directory

    $ TARGET=fpga ./ci/blackbox.sh --driver=opae --app=sgemm --args="-n128"

