## Xilinx Build and Ecosystem Setup

vivado -mode batch -source scripts/gen_ip.tcl -tclargs ip/xilinx_u280_xdma_201920_3

make all TARGET=hw PLATFORM=xilinx_u280_xdma_201920_3 PROFILE=1 DEBUG=1 > build_hw.log 2>&1
make all TARGET=hw_emu PLATFORM=xilinx_u280_xdma_201920_3 PROFILE=1 DEBUG=1 > build_hw_emu.log 2>&1

make all TARGET=hw PLATFORM=xilinx_u50_gen3x16_xdma_201920_3 DEBUG=1 > build_u50_hw.log 2>&1
make all TARGET=hw_emu PLATFORM=xilinx_u50_gen3x16_xdma_201920_3 DEBUG=1 > build_u50_hw_emu.log 2>&1

xsim --gui xilinx_u280_xdma_201920_3-0-vortex_afu.wdb &

HW synthesis log
build_xilinx_u280_xdma_201920_3_hw/_x/logs/link/syn/pfm_dynamic_vortex_afu_1_0_synth_1_runme.log

Running:
TARGET=hw PLATFORM=xilinx_u280_xdma_201920_3 ./ci/blackbox.sh --driver=xrt --app=demo
TARGET=hw PLATFORM=xilinx_u50_gen3x16_xdma_201920_3 ./ci/blackbox.sh --driver=xrt --app=demo

TARGET=hw PLATFORM=xilinx_u50_gen3x16_xdma_201920_3 ./ci/blackbox.sh --driver=xrt --app=demo
TARGET=hw_emu PLATFORM=xilinx_u50_gen3x16_xdma_201920_3 ./ci/blackbox.sh --driver=xrt --app=demo

ILA debugging:
ls /dev/xfpga/xvc_pub*
debug_hw --xvc_pcie /dev/xvc_pub.u0 --hw_server
debug_hw --vivado --host localhost --ltx_file ./build_xilinx_u280_xdma_201920_3_hw/_x/link/vivado/vpl/prj/prj.runs/impl_1/debug_nets.ltx &
debug_hw --vivado --host localhost --ltx_file ./build_xilinx_u50_gen3x16_xdma_201920_3_hw/_x/link/vivado/vpl/prj/prj.runs/impl_1/debug_nets.ltx &
