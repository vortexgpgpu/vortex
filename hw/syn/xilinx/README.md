## Xilinx Build and Ecosystem Setup

vivado -mode batch -source scripts/gen_ip.tcl -tclargs ip/xilinx_u280_xdma_201920_3

make all TARGET=hw PLATFORM=xilinx_u280_xdma_201920_3 REPORT=1 PROFILE=1 > build_hw.log 2>&1

make all TARGET=hw_emu PLATFORM=xilinx_u280_xdma_201920_3 REPORT=1 PROFILE=1 DEBUG=1 > build_hw_emu.log 2>&1

xsim --gui xilinx_u280_xdma_201920_3-0-vortex_afu.wdb &