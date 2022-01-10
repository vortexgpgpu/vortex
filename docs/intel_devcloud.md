# Vortex on Intel’s devcloud Arria 10

- Sign up for Intel devcloud and get the ssh and setup so that ssh devcloud works

- Install vortex repo

```bash
git clone --recursive https://github.com/vortexgpgpu/vortex.git` 
cd vortex
```

- Installing toolchain → On devcloud we dont have /opt access but the vortex script `toolchain_install.sh` uses /opt. First do a pwd in your devcloud home directory. Your path will look like `/home/uxxxxxx.`

```bash
mkdir BIN
cd BIN
pwd
```

Your path will look like `/home/uxxxxxx/BIN.` 

```bash
cd vortex/ci
nano toolchain_install.sh
```

So first we’ll have to change that on line 8 to:

```bash
#!/bin/bash

# exit when any command fails
set -e

REPOSITORY=https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/master

DESTDIR="${DESTDIR:=/home/uxxxxxx/BIN}"
```

After this you can run the following command while inside the vortex directory 

```bash
./ci/toolchain_install.sh -all
make -s
```

- You’ll have to set the following variables correctly

```bash
export VERILATOR_ROOT=/home/uxxxxxx/BIN/verilator
export PATH=$VERILATOR_ROOT/bin:$PATH
export POCL_CC_PATH=/home/uxxxxxx/BIN/pocl/compiler
export POCL_RT_PATH=/home/uxxxxxx/BIN/pocl/runtime
export LLVM_PREFIX=/home/uxxxxxx/BIN/llvm-riscv 
export RISCV_TOOLCHAIN_PATH=/home/uxxxxxx/BIN/riscv-gnu-toolchain
```

- To quickly test your installation run

```bash
./ci/blackbox.sh --driver=simx --cores=2 --app=vecadd
```

- Programming Arria 10 with vortex

```bash
$ ssh devcloud

$ devcloud_login #select option 1 here

$ tools_setup #select option 5 here

$ cd /opt/a10/... find the opae tar file

$ copy it over to home and untar it
```

- Set the following variables correctly

```bash
export PATH=$OPAE_PLATFORM_ROOT/bin:$PATH

export C_INCLUDE_PATH=/home/u109558/a10_gx_pac_ias_1_2_1_pv/a10_gx_pac_ias_1_2_1_pv/sw/opae-1.1.2-2/common/include:$C_INCLUDE_PATH

export LD_LIBRARY_PATH=/home/u109558/a10_gx_pac_ias_1_2_1_pv/a10_gx_pac_ias_1_2_1_pv/hw/lib:$LD_LIBRARY_PATH
```

- We’re now going to generate a .gbs file  (link vortex vid)

```bash
cd vortex/hw/syn/opae
make fpga-4c # make fpga-<num-of-cores>c, you can check the Makefile for other configs
```

This is supposed to create a folder called `build_fpga_4c` and run synthesis for a while. After that `cd build_fpga_4c` should have a `vortex_afu.gbs` file. If this doesnt work then

```bash
cd build_fpga_4c
$OPAE_PLATFORM_ROOT/bin/run.sh
```

```bash
PACSign PR -t UPDATE -H openssl_manager -i vortex_afu.gbs -o vortex_afu_unsigned_ssl.gbs
fpgasupdate vortex_afu_unsigned_ssl.gbs
```

- example

```bash
./ci/blackbox.sh --driver=fpga --app=sgemm --args="-n64"
```
