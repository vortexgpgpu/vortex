`ifndef VORTEX_AXI_WRAPPER_VS
`define VORTEX_AXI_WRAPPER_VS

`define SYNTHESIS
`define NDEBUG
`define EXT_F_DISABLE 
`define VIVADO
`define NOGLOBALS

`define NUM_CORES       1 
`define NUM_THREADS     2 
`define NUM_WARPS       2
`define STARTUP_ADDR    32'h80000 
`define IO_BASE_ADDR    32'hFF000
`define IO_ADDR_SIZE    (32'hFFFFF - `IO_BASE_ADDR + 1)
`define IO_COUT_ADDR    (32'hFFFFF - `MEM_BLOCK_SIZE + 1)

`endif