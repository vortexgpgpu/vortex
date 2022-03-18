`include "VX_rop_define.vh"

module VX_rop_unit #(    
    parameter CLUSTER_ID = 0,    
    parameter NUM_SLICES = 1,
    parameter NUM_LANES  = 4
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_rop_perf_if.master rop_perf_if,
`endif

    // Memory interface
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_rop_dcr_if.master rop_dcr_if,
    VX_rop_req_if.slave rop_req_if
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    // TODO: remove
    rop_dcrs_t dcrs = rop_dcr_if.data;
    `UNUSED_VAR (dcrs)

    // TODO: remove
    `UNUSED_VAR (rop_req_if.valid)
    `UNUSED_VAR (rop_req_if.tmask)
    `UNUSED_VAR (rop_req_if.pos_x)
    `UNUSED_VAR (rop_req_if.pos_y)    
    `UNUSED_VAR (rop_req_if.color)
    `UNUSED_VAR (rop_req_if.depth)
    `UNUSED_VAR (rop_req_if.backface)
    assign rop_req_if.ready = 0;

    // TODO: remove
    assign rop_perf_if.mem_reads = 0;
    assign rop_perf_if.mem_writes = 0;
    assign rop_perf_if.mem_latency = 0;

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

    reg blend_valid_in = 1;
    reg blend_ready_out = 1;
    rgba_t [NUM_LANES-1:0] src_color = 0;
    rgba_t [NUM_LANES-1:0] dst_color = 0;    
    wire blend_ready_in;
    wire blend_valid_out;    
    rgba_t [NUM_LANES-1:0] out_color;

    VX_rop_blend #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES)
    ) blend (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (blend_ready_in),
        .valid_in  (blend_valid_in),        
        .ready_out (blend_ready_out),
        .valid_out (blend_valid_out),
        .dcrs      (dcrs),
        .src_color (src_color),
        .dst_color (dst_color),
        .color_out (out_color)
    );

    wire [31:0] buf_addr = rop_dcr_if.zbuf_addr + (rop_req_if.pos_y * rop_dcr_if.zbuf_pitch) + (rop_req_if.pos_x * 4);


    VX_rop_mem #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_LANES  (NUM_LANES)
    ) mem (
        .clk       (clk),
        .reset     (reset),
        .req_valid (rop_req_if.valid),
        .req_rw    (1'b0),
        .req_mask  (rop_req_if.tmask),
        .req_byteen(4'b1111),
        .req_addr  (buf_addr),
        .req_data  (0),
        .req_tag   (),
        .req_ready (),

        .rsp_valid (),
        .rsp_mask  (),
        .rsp_data  (),
        .rsp_tag   (),
        .rsp_ready ()
    )

    `UNUSED_VAR (out_color)
    `UNUSED_VAR (blend_ready_in)
    `UNUSED_VAR (blend_valid_out)

endmodule