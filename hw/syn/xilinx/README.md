## Xilinx Build and Ecosystem Setup

vivado -mode batch -source scripts/gen_ip.tcl -tclargs ip/xilinx_u280_xdma_201920_3

make all TARGET=hw_emu PLATFORM=xilinx_u280_xdma_201920_3 > build.log 2>&1