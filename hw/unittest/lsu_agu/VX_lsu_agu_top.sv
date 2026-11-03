`include "VX_define.vh"

// The two-stage AGU (VX_lsu_agu, then VX_lsu_agu_swz on its address) beside
// the single-cycle oracle, combinationally; the registers between the stages
// carry the address unchanged, so the test compares the functions.
module VX_lsu_agu_top import VX_gpu_pkg::*; (
    input  wire [`VX_CFG_XLEN-1:0] base,
    input  wire [`VX_CFG_XLEN-1:0] stride,
    input  wire [11:0]             offset,
    input  wire [1:0]              pack,
    output wire [`VX_CFG_XLEN-1:0] addr_new,
    output wire [`VX_CFG_XLEN-1:0] lin_new,
    output wire [`VX_CFG_XLEN-1:0] addr_ref,
    output wire [`VX_CFG_XLEN-1:0] lin_ref,
    output wire [`VX_CFG_XLEN-1:0] win_bottom,
    output wire [`VX_CFG_XLEN-1:0] win_top
);
    VX_lsu_agu agu (
        .base (base), .stride (stride), .offset (offset), .pack (pack),
        .lin_addr (lin_new)
    );
    VX_lsu_agu_swz swz (
        .lin_addr (lin_new),
        .addr     (addr_new)
    );
    VX_lsu_agu_ref agu_ref (
        .base (base), .stride (stride), .offset (offset), .pack (pack),
        .addr (addr_ref), .lin_addr (lin_ref)
    );
    assign win_bottom = STACK_WINDOW_BOTTOM;
    assign win_top    = STACK_WINDOW_TOP;
endmodule
