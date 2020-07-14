`ifndef VX_INST_META_IF
`define VX_INST_META_IF

`include "VX_define.vh"

interface VX_inst_meta_if ();

    wire [`NUM_THREADS-1:0]   valid;
    wire [31:0]               curr_PC;
    wire [`NW_BITS-1:0]       warp_num;
    wire [31:0]               instruction;

endinterface

`endif