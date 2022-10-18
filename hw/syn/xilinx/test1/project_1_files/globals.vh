`ifndef GLOBALS_VH
`define GLOBALS_VH

`define SYNTHESIS
`define NDEBUG
`define VIVADO

`define EXT_F_DISABLE
//`define EXT_GFX_ENABLE

`define STARTUP_ADDR    32'h80000
`define IO_BASE_ADDR    32'hFF000
`define IO_ADDR_SIZE    (32'hFFFFF - `IO_BASE_ADDR + 1)
`define IO_COUT_ADDR    (32'hFFFFF - `MEM_BLOCK_SIZE + 1)

`endif