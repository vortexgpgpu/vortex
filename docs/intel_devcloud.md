# Vortex on Intel’s devcloud Arria 10

This is a step-by-step guide to program Vortex on Arria 10 via Intel's devcloud. (Assumes a Linux like environment)
  
- [Devcloud set-up](#devcloud-set-up)
- [Vortex set-up](#vortex-set-up)
  * [Get vortex](#get-vortex)
  * [Toolchain](#toolchain)
  * [Environment variables](#environment-variables)
  * [Test installation](#test-installation)
- [Programming Arria 10 with Vortex](#programming-arria-10-with-vortex)
  * [OPAE set-up](#opae-set-up)
  * [Set environment variables](#set-environment-variables)
  * [Generate bitstream](#generate-bitstream)
  * [Sign the bitstream](#sign-the-bitstream)
  * [Program the FPGA](#program-the-fpga)
  * [Test example](#test-example)
- [References](#references)

## Devcloud set-up

- Sign up for [Intel devcloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html) 
- If you're on a Linux/macOS type system follow the steps [here](https://devcloud.intel.com/oneapi/documentation/connect-with-ssh-linux-macos/) for the initial setup.
- If your setup is successful, typing `ssh devcloud` on your terminal will connect you to devcloud

## Vortex set-up

### Get vortex

Clone the vortex repo

> Devcloud users cannot use `sudo apt-get` from the main [README](https://github.com/vortexgpgpu/vortex#install-development-tools), but devcloud comes with `build-essential` and `git` preinstalled so this step is ignored. 

```bash
$ git clone --recursive https://github.com/vortexgpgpu/vortex.git 
```

### Toolchain

> Devcloud users dont have /opt access (requires admin privileges) but the vortex script `toolchain_install.sh` uses /opt. We will have to edit this script. 

Type these commands from your devcloud home directory.

```bash
$ mkdir BIN
$ cd BIN
$ pwd
```

Your path will look like `/home/uxxxxxx/BIN`, where `uxxxxxx` is your devcloud user-id. Copy this path for later. 

```bash
$ cd vortex/ci
$ nano toolchain_install.sh
```

You’ll have to change line 8:

```bash
# this is the script 'toolchain_install.sh' you're editing
#!/bin/bash

# exit when any command fails
set -e

REPOSITORY=https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/master

DESTDIR="${DESTDIR:=/home/uxxxxxx/BIN}" # EDIT HERE
```

After this you can run the following commands while inside the vortex directory 

```bash
./ci/toolchain_install.sh -all
make -s
```

### Environment variables

Set the following variables correctly:

```bash
export VERILATOR_ROOT=/home/uxxxxxx/BIN/verilator
export PATH=$VERILATOR_ROOT/bin:$PATH
export POCL_CC_PATH=/home/uxxxxxx/BIN/pocl/compiler
export POCL_RT_PATH=/home/uxxxxxx/BIN/pocl/runtime
export LLVM_PREFIX=/home/uxxxxxx/BIN/llvm-riscv 
export RISCV_TOOLCHAIN_PATH=/home/uxxxxxx/BIN/riscv-gnu-toolchain
```

**Note: You might need to set these variables everytime you access devcloud (`ssh devcloud`) unless you include them in your .bashrc script.**

### Test installation

To quickly test your installation run the following from your vortex folder: 

```bash
./ci/blackbox.sh --driver=simx --cores=2 --app=vecadd
```

## Programming Arria 10 with Vortex

```bash
$ ssh devcloud

$ devcloud_login #select option 1 here - Arria 10 PAC Compilation and Programming - RTL AFU, OpenCL & select either release 1.2 or 1.2.1 

$ tools_setup #select option 5 here - Arria 10 PAC Compilation and Programming - RTL AFU, OpenCL - this sets the right env variables
```
### OPAE set-up

Compressed OPAE folders are available at `/opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/sw`. You need to untar OPAE but you don't have permissions to `/opt`, so copy this folder to `/home/uxxxxxx`.

```bash
$ cd /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/
$ pwd #copy this path
$ cd ~
$ mkdir a10_gx_pac_ias_1_2_1_pv
$ cp –r /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv /home/uxxxxxx/a10_gx_pac_ias_1_2_1_pv/
$ cd /home/uxxxxxx/a10_gx_pac_ias_1_2_1_pv/a10_gx_pac_ias_1_2_1_pv/sw
$ tar xvzf opae-1.1.2-2.tar.gz
```
This will create a folder in the current directory called `opae-1.1.2-2`. You will use this path to set some env variables next. You need to do this only once. 

### Set environment variables
Set the following variables correctly. 

>  $OPAE_PLATFORM_ROOT is already set by the `tools_setup` command, try `echo $OPAE_PLATFORM_ROOT` to verify this. 

```bash
export PATH=$OPAE_PLATFORM_ROOT/bin:$PATH
export C_INCLUDE_PATH=/home/uxxxxxx/a10_gx_pac_ias_1_2_1_pv/a10_gx_pac_ias_1_2_1_pv/sw/opae-1.1.2-2/common/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/uxxxxxx/a10_gx_pac_ias_1_2_1_pv/a10_gx_pac_ias_1_2_1_pv/hw/lib:$LD_LIBRARY_PATH
```

**(Note:You might need to set these variables everytime you access devcloud (`ssh devcloud`) unless you include them in your .bashrc script.)**

### Generate bitstream

To generate a .gbs file: 

```bash
# From /home/uxxxxxx

cd vortex/hw/syn/opae
make fpga-4c # make fpga-<num-of-cores>c, you can check the Makefile for other settings and options
```

This is supposed to create a folder called `build_fpga_4c` and run synthesis for a while. After that `cd build_fpga_4c` **should have a `vortex_afu.gbs` file.**


> Note: If `build_fpga_4c` doesn't have `vortex_afu.gbs`
> ```bash
> cd build_fpga_4c
> $OPAE_PLATFORM_ROOT/bin/run.sh
> ```

### Sign the bitstream 

```bash
$ PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
```
### Program the FPGA

```bash
$ fpgasupdate vortex_afu_unsigned_ssl.gbs
```

### Test example

```bash
# from vortex root folder
$ ./ci/blackbox.sh --driver=fpga --app=sgemm --args="-n64"
```

## References
1. [FPGA demo video](https://github.com/vortexgpgpu/vortex_tutorials/blob/main/Slides/vortex_fpga_demo.mp4)
2. [FPGA Startup and Configuration Guide](https://github.com/vortexgpgpu/vortex/blob/master/docs/fpga_setup.md)

