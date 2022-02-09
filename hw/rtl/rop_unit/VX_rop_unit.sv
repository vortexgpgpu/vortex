`include "VX_rop_define.vh"

module VX_rop_unit #(  
    parameter CORE_ID = 0,
    parameter NUM_SLICES = 1
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_perf_rop_if.master perf_rop_if,
`endif

    // Memory interface
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_csr_if.slave rop_csr_if,
    VX_rop_req_if.slave rop_req_if
);

    rop_csrs_t rop_csrs;

    VX_rop_csr #(
        .CORE_ID (CORE_ID)
    ) rop_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .rop_csr_if (rop_csr_if),
        .rop_req_if (rop_req_if),

        // outputs
        .rop_csrs   (rop_csrs)
    );

    // TODO: remove
    `UNUSED_VAR (rop_csrs)

    // TODO: remove
    `UNUSED_VAR (rop_req_if.valid)
    `UNUSED_VAR (rop_req_if.uuid)
    `UNUSED_VAR (rop_req_if.wid)
    `UNUSED_VAR (rop_req_if.tmask)
    `UNUSED_VAR (rop_req_if.PC)
    `UNUSED_VAR (rop_req_if.rd)
    `UNUSED_VAR (rop_req_if.wb)
    `UNUSED_VAR (rop_req_if.tmask)    
    `UNUSED_VAR (rop_req_if.x)
    `UNUSED_VAR (rop_req_if.y)
    `UNUSED_VAR (rop_req_if.color)
    assign rop_req_if.ready = 0;

    // TODO: remove
    `UNUSED_VAR (rop_csr_if.write_enable);
    `UNUSED_VAR (rop_csr_if.write_addr);
    `UNUSED_VAR (rop_csr_if.write_data);
    `UNUSED_VAR (rop_csr_if.write_uuid);

    // TODO: remove
    assign perf_rop_if.mem_reads = 0;
    assign perf_rop_if.mem_writes = 0;
    assign perf_rop_if.mem_latency = 0;

    // TODO: remove
    assign cache_req_if.valid = 0;
    assign cache_req_if.rw = 0;
    assign cache_req_if.byteen = 0;
    assign cache_req_if.addr = 0;
    assign cache_req_if.data = 0;     
    assign cache_req_if.tag = 0;
    `UNUSED_VAR (cache_req_if.ready)

    // TODO: remove
    `UNUSED_VAR (cache_rsp_if.valid)
    `UNUSED_VAR (cache_rsp_if.tmask)
    `UNUSED_VAR (cache_rsp_if.data)        
    `UNUSED_VAR (cache_rsp_if.tag)
    assign cache_rsp_if.ready = 0;

endmodule