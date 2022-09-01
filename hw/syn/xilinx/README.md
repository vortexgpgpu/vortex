## Xilinx Build and Ecosystem Setup

make all TARGET=hw_emu PLATFORM=xilinx_u280_xdma_201920_3 > build.log 2>&1

vivado -mode batch -source scripts/gen_ips.tcl -tclargs  ../../../ip/xilinx/xilinx_u280_xdma_201920_3