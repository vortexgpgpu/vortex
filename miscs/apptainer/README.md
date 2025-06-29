# Apptainer Build Process

Use the Slurm scheduler to request an interactive job on Flubber9
```
salloc -p rg-fpga --nodes=1 --ntasks-per-node=64 --mem=16G --nodelist flubber9 --time=01:00:00
```

Go to `apptainer` directory

```
$ pwd
vortex/miscs/apptainer

$  apptainer build --no-https vortex.sif vortex.def

```


To start the apptainer,
```
$ chmod +x run_apptainer.sh
$ ./run_apptainer.sh
```


| Note: Set env variables in `run_apptainer.sh` accordingly.


Inside the Apptainer,
```
# should show devices connected to machine on which you are running this command
Apptainer> lsusb

```


# Vortex Simulation inside Apptainer

Go to bind of vortex repo,
```
Apptainer> cd /home/vortex
Apptainer> mkdir build
Apptainer> cd build
Apptainer> ../configure --xlen=32 --tooldir=$HOME/tools


Skip the below 3 steps, If tools are already present in the $HOME/tools
Apptainer> sed -i 's/\btar /tar --no-same-owner /g' ci/toolchain_install.sh
Apptainer> ./ci/toolchain_install.sh --all
Apptainer> sed -i 's/\btar --no-same-owner /tar /g' ci/toolchain_install.sh


Apptainer> source ./ci/toolchain_env.sh
Apptainer> verilator --version
```


### Running SIMX, RTLSIM and XRTSIM
```
Compile the Vortex codebase
Apptainer> make -s

Run the programs by specifying the appropriate driver as shown below:

SIMX
Apptainer> ./ci/blackbox.sh --cores=2 --app=demo --driver=simx

RTLSIM
Apptainer> ./ci/blackbox.sh --cores=2 --app=demo --driver=rtlsim

XRTSIM
Apptainer> ./ci/blackbox.sh --cores=2 --app=demo --driver=xrt
```


### Vortex Bitstream on FPGA

Common Commands
```
Apptainer> source /opt/xilinx/xrt/setup.sh
Apptainer> source /tools/reconfig/xilinx/Vitis/2023.1/settings64.sh

Apptainer> platforminfo -l

```

##### Building Vortex Bitstream

```
Apptainer> pwd
/home/vortex/build

Apptainer> source ci/toolchain_env.sh

Apptainer> verilator --version
Verilator 5.026 2024-06-15 rev v5.026-43-g065f36ab5

Apptainer> cd hw/syn/xilinx/xrt
Apptainer> PREFIX=test1 PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1 TARGET=hw NUM_CORES=1 make > build_u50_hw_1c.log 2>&1 &
Creates ../test1_xilinx_u50_gen3x16_xdma_5_202210_1_hw/bin/vortex_afu.xclbin
```

##### Running Vortex Bitstream on FPGA
```
Apptainer> cd ../../../../
Apptainer> pwd
/home/vortex/build

Apptainer> make -C runtime/ clean

Apptainer> FPGA_BIN_DIR=hw/syn/xilinx/xrt/test1_xilinx_u50_gen3x16_xdma_5_202210_1_hw/bin  TARGET=hw PLATFORM=xilinx_u50_gen3x16_xdma_5_202210_1  ./ci/blackbox.sh --driver=xrt --app=demo

Verify following line being printed:
info: device name=xilinx_u50_gen3x16_xdma_base_5, memory_capacity=0x200000000 bytes, memory_banks=32.
```