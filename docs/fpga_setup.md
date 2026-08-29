# FPGA Startup and Configuration Guide

## Gaining Access to FPGA's with CRNCH
If you are associated with Georgia Tech (or related workshops) you can use CRNCH's server to gain remote access to FPGA's. Otherwise, you can skip to the Xilinx or Intel (Altera) synthesis steps below.

## What is CRNCH?

**C**enter for **R**esearch into **N**ovel **C**omputing **H**ierarchies

## What does CRNCH Offer?

**The Rogues Gallery (RG)**: new concept focused on developing our understanding of next-generation hardware with a focus on unorthodox and uncommon technologies. **RG** will acquire new and unique hardware (ie, the aforementioned “*rogues*”) from vendors, research labs, and startups and make this hardware available to students, faculty, and industry collaborators within a managed data center environment

## Why are the Rouges Important?

By exposing students and researchers to this set of unique hardware, we hope to foster cross-cutting discussions about hardware designs that will drive future *performance improvements in computing long after the Moore’s Law era of “cheap transistors” ends*. Specifically, the Rouges Gallery contains FPGA's which can be synthesized into Vortex hardware.

## How is the Rouges Gallery Funded?

Rogues Gallery testbed is primarily supported by the National Science Foundation (NSF) under NSF Award Number [#2016701](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2016701&HistoricalAwards=false)

## Rouges Gallery Documentation

You can read about RG in more detail on its official documentation [page](https://gt-crnch-rg.readthedocs.io/en/main/index.html#).

You can listen to a talk about RG [here](https://mediaspace.gatech.edu/media/Jeff%20Young%20-%20Rogues%20Gallery%20-%20CRNCH%20Summit%202021/1_lqlgr0jj)

[CRNCH Summit 2023](https://github.com/gt-crnch/crnch-summit-2023/tree/main)

## Request Access for Rouges Gallery

You should use [this form](https://crnch-rg.cc.gatech.edu/request-rogues-gallery-access/) to request access to RG’s reconfigurable computing (vortex fpga) resources. You should receive an email with your ticket item being created. Once it gets processed, you should get an email confirmed your access has been granted. It might take some time to get processed.

## How to Access Rouges Gallery?
There are two methods of accessing CRNCH's Rouges Gallery
1) Web-based GUI: [rg-ood.crnch.gatech.edu](http://rg-ood.crnch.gatech.edu/)
2) SSH: `ssh <your-gt-username>@rg-login.crnch.gatech.edu`


## Where should I keep my files?
The CRNCH servers have a folder called `USERSCRATCH` which can be found in your home directory: `echo $HOME`. You should keep all your files in this folder since it is available across all the Rouges Gallery Nodes.

## **What Machines are Available in the Rogues Gallery?**

Complete list of machines can be found [here](https://gt-crnch-rg.readthedocs.io/en/main/general/rg-hardware.html). Furthermore, you can find detailed information about the FPGA hardware [here](https://gt-crnch-rg.readthedocs.io/en/main/reconfig/xilinx/xilinx-getting-started.html).

## Allocate an FPGA Node
Once you’ve connected to the CRNCH login node, you can use the Slurm scheduler to request an interactive job using `salloc`. This [page](https://gt-crnch-rg.readthedocs.io/en/main/general/using-slurm.html) explains why we use Slurm to request resources. Documentation for `salloc` can be found [here](https://gt-crnch-rg.readthedocs.io/en/main/general/using-slurm-examples.html). And here.


To request 16 cores and 64GB of RAM for 6 hours on flubber9, a fpga dev node:
```bash
salloc -p rg-fpga --nodes=1 --ntasks-per-node=16 --mem=64G --nodelist flubber1 --time=06:00:00
```
Synthesis for Xilinx Boards
----------------------
Once you are logged in, you will need to complete some first time configurations. If you are interested in the Intel (Altera) synthesis steps, scroll down below.

### Source Configuration Scripts
```
# From any directory
$ source /opt/xilinx/xrt/setup.sh
$ source /tools/reconfig/xilinx/Vitis/2023.1/settings64.sh
```

If you are using an apptainer,
```
$ export LD_LIBRARY_PATH=/opt/boost-1.66/lib/:/opt/openssl-1.1/lib/:$LD_LIBRARY_PATH
```


### Check Installed FPGA Platforms
`platforminfo -l` which tells us the correct name of the platform installed on the current fpga node. It should be used for the `PLATFORM` variable below. Otherwise, if there is an error then there was an issue with the previous two commands.

### Install Vortex Toolchain
The Xilinx synthesis process requires verilator to generate the bitstream. Eventually, you will need the whole toolchain to run the bitstream on the FPGA. Therefore, the Vortex toolchain and can be installed as follows. If you complete these steps properly, you should only need to complete them once and you can skip to `Activate Vortex Toolchain`
```
# Make a build directory from root and configure scripts for your environment
mkdir build && cd build && ../configure --tooldir=$HOME/tools

# Install the whole prebuilt toolchain
./ci/toolchain_install.sh
```

### Verify Toolchain Install
```
# Tools are invoked by absolute path from the Vortex Makefiles -- no
# shell env sourcing required. Confirm the install with:
$TOOLDIR/verilator/bin/verilator --version
```

### Build the FPGA Bitstream
The root directory contains the path `hw/syn/xilinx/xrt` which has the makefile used to generate the Vortex bitstream.

```
    $ cd hw/syn/xilinx/xrt
    $ PREFIX=test1 PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1 TARGET=hw NUM_CORES=1 make > build_u250_hw_1c.log 2>&1 &
```
Will run the synthesis under new build directory: BUILD_DIR := "\<PREFIX>\_\<PLATFORM>\_\<TARGET>"
The generated bitstream will be located under <BUILD_DIR>/bin/vortex_afu.xclbin

`NUM_CORES=N` here is a Makefile shorthand that selects a pre-defined cluster/core/L2 combination (see `hw/syn/xilinx/xrt/Makefile`); it expands to `-DVX_CFG_NUM_CLUSTERS=… -DVX_CFG_NUM_CORES=…` under the hood.

For long-running jobs, invocation of this makefile can be made of the following form:

`[CONFIGS=<vortex macros>] [PREFIX=<prefix directory name>] [NUM_CORES=<#>] TARGET=hw|hw_emu PLATFORM=<platform baseName> nohup make > <log filename> 2>&1 &`

For example:

```bash
CONFIGS="-DVX_CFG_L2_ENABLE -DVX_CFG_DCACHE_SIZE=8192" PREFIX=build_4c_u280 NUM_CORES=4 TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202310_1 nohup make > build_u250_hw_4c.log 2>&1 &
```

The build is complete when the bitstream file `vortex_afu.xclbin` exists in `<prefix directory name><platform baseName>hw|hw_emu/bin`.

### Enable Host Memory Access (required on hardware)

The Vortex AFU is driven by an on-chip **Command Processor (CP)** that fetches its
command ring and other control buffers directly from **host memory** over the
platform's host-memory bridge (CPU RAM exposed to the card, a.k.a. the "host bank"
or slave-bridge). The XRT runtime therefore allocates these buffers in host memory
at device open.

This bank is **disabled by default** and must be enabled once per card (it persists
until disabled or reboot). If it is not enabled, `vx_device_open` fails with:

```
[XRT] ERROR: Failed to allocate host memory buffer (Operation not permitted),
             make sure host bank is enabled (xrt-smi configure --host-mem)
```

Enable it (requires root). `sudo` resets `PATH`, so source the XRT environment
inside the elevated shell and pass the card's BDF (from `xrt-smi examine`):

```bash
sudo bash -c 'source /opt/xilinx/xrt/setup.sh && \
  xrt-smi configure --host-mem -d <BDF> --size 1G enable'
# e.g. -d 0000:3d:00.1
```

- `--size` must be a power of two and is bounded by available host CMA/huge-page
  memory. **256M** is sufficient for the CP ring and typical test buffers; **1G**
  is a safe default with headroom.
- This applies to the **`hw`** flow only. **`hw_emu`** models host memory in the
  emulator and needs no card configuration.

### Running a Program on Xilinx FPGA

The [blackbox.sh](./simulation.md) script within the build directory can be used to run a test with Vortex’s xrt driver using the following command:

`FPGA_BIN_DIR=<path to bitstream directory> TARGET=hw|hw_emu PLATFORM=<platform baseName> ./ci/blackbox.sh --driver=xrt --app=<test name>`

For example:

```FPGA_BIN_DIR=<realpath> hw/syn/xilinx/xrt/build_4c_u280_xilinx_u280_gen3x16_xdma_1_202211_1_hw/bin TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 ./ci/blackbox.sh --driver=xrt --app=demo```

Run sgemm on a U55C `hw` bitstream (host memory enabled per the section above), `-n` sets the matrix size:

```bash
FPGA_BIN_DIR=$(realpath hw/syn/xilinx/xrt/<build dir>/bin) TARGET=hw \
  ./ci/blackbox.sh --driver=xrt --app=sgemm --args="-n128"
```

The same command with `TARGET=hw_emu` against a `hw_emu` build runs it in emulation
(use a small `-n`, e.g. `-n16`, since emulation is slow).

Synthesis for Intel (Altera) Boards
----------------------

> **⚠️ Deprecated / unmaintained.** The Intel (Altera) OPAE flow is legacy and
> is no longer actively maintained or CI-tested. It targets discontinued Intel
> PAC cards (Arria 10 / Stratix 10) and depends on Intel-supplied platform files
> (e.g. `platform_if.vh` provided by the OPAE PIM at Quartus build time), so the
> steps below may be out of date and the build is not guaranteed to work on
> current toolchains. **The supported FPGA path is the Xilinx Alveo / XRT flow
> above.** The notes here are retained for reference only; please use Xilinx XRT
> for new work.

### OPAE Environment Setup


    $ source /opt/inteldevstack/init_env_user.sh
    $ export OPAE_HOME=/opt/opae/1.1.2
    $ export PATH=$OPAE_HOME/bin:$PATH
    $ export C_INCLUDE_PATH=$OPAE_HOME/include:$C_INCLUDE_PATH
    $ export LIBRARY_PATH=$OPAE_HOME/lib:$LIBRARY_PATH
    $ export LD_LIBRARY_PATH=$OPAE_HOME/lib:$LD_LIBRARY_PATH

### OPAE Build

The FPGA has to following configuration options:
- DEVICE_FAMILY=arria10 | stratix10
- NUM_CORES=#n

Command line:

    $ cd hw/syn/altera/opae
    $ PREFIX=test1 TARGET=fpga NUM_CORES=4 make

A new folder (ex: `test1_xxx_4c`) will be created and the build will start and take ~30-480 min to complete.
Setting TARGET=ase will build the project for simulation using Intel ASE.


### OPAE Build Configuration

Hardware parameters live in `VX_config.toml` and `VX_types.toml` at the repo root (auto-generated headers like `VX_config.vh` are produced under `build/` by `configure`). All build-time overrides use the `VX_CFG_*` namespace. For example:
- `VX_CFG_NUM_WARPS`:   Number of warps per core
- `VX_CFG_NUM_THREADS`: Number of threads per warp
- `PERF_ENABLE`: enable the use of all profile counters

You configure the synthesis build from the command line:

    $ CONFIGS="-DPERF_ENABLE -DVX_CFG_NUM_THREADS=8" make

### OPAE Build Progress

You could check the last 10 lines in the build log for possible errors until build completion.

    $ tail -n 10 <build_dir>/build.log

Check if the build is still running by looking for quartus_sh, quartus_syn, or quartus_fit programs.

    $ ps -u <username>

If the build fails and you need to restart it, clean up the build folder using the following command:

    $ make clean

The file `vortex_afu.gbs` should exist when the build is done:

    $ ls -lsa <build_dir>/synth/vortex_afu.gbs


### Signing the bitstream and Programming the FPGA

    $ cd <build_dir>
    $ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
    $ fpgasupdate vortex_afu_unsigned_ssl.gbs

### Sample FPGA Run Test
Ensure you have the correct opae runtime for the FPGA target

```
$ TARGET=FPGA make -C runtime/opae
```

Run the [blackbox.sh](./simulation.md) from your Vortex build directory

```
$ TARGET=fpga ./ci/blackbox.sh --driver=opae --app=sgemm --args="-n128"
```

### FPGA sample test running OpenCL sgemm kernel

You can use the `blackbox.sh` script to run the following from your Vortex build directory

    $ TARGET=fpga ./ci/blackbox.sh --driver=opae --app=sgemm --args="-n128"

### Testing Vortex using OPAE with Intel ASE Simulation
Building ASE synthesis

```$ TARGET=asesim make -C runtime/opae```

Building ASE runtime

```$ TARGET=asesim make -C runtime/opae```

Running ASE simulation

```$ ASE_LOG=0 ASE_WORKDIR=<build_dir>/synth/work TARGET=asesim ./ci/blackbox.sh --driver=opae --app=sgemm --args="-n16"```
