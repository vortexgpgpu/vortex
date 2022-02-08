`include "VX_rop_define.vh"

module VX_rop_unit #(  
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_perf_tex_if.master perf_rop_if,
`endif

    // Memory interface
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_csr_if.slave rop_csr_if,
    VX_rop_req_if.slave rop_req_if
);

    // TODO

endmodule