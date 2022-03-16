`include "VX_rop_define.vh"

module VX_rop_ds #(
    parameter DEPTH_TEST = 1,
    parameter CLUSTER_ID = 0,
    parameter NUM_LANES  = 4
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_rop_req_if.slave rop_req_if,
    VX_rop_dcr_if.slave rop_dcr_if
);

    localparam MASK = 24'h7FFFFF;

    // Depth Buffer
    wire                    zbuf_req_valid;
    wire                    zbuf_req_rw;
    wire [`NUM_THREADS-1:0] zbuf_req_mask;
    wire [3:0]              zbuf_req_byteen;
    wire [31:0]             zbuf_req_addr;
    wire [31:0]             zbuf_req_data;
    wire [`UUID_BITS-1:0]   zbuf_req_tag;
    wire                    zbuf_req_ready;

    wire                    zbuf_rsp_valid;
    wire [`NUM_THREADS-1:0] zbuf_rsp_mask;
    wire [31:0]             zbuf_rsp_data;
    wire [`UUID_BITS-1:0]   zbuf_rsp_tag;
    wire                    zbuf_rsp_ready;

    wire zbuf_req_fire;

    // Depth Test
    wire [31:0] depth_ref;
    wire [31:0] depth_val;
    wire        passed;

    ///////////////////////////////////////////////////////////////

    // Read depth value from the depth buffer

    assign zbuf_req_valid  = rop_req_if.valid;
    assign zbuf_req_mask   = rop_req_if.tmask;
    assign zbuf_req_byteen = 4'b1111;
    assign zbuf_req_addr   = rop_dcr_if.zbuf_addr + (rop_req_if.y * rop_dcr_if.zbuf_pitch) + (rop_req_if.x * 4);
    assign zbuf_req_tag    = rop_req_if.uuid;

    assign zbuf_req_fire = zbuf_req_valid & zbuf_req_ready;
     
    VX_rop_mem #(
        .NUM_REQS (NUM_LANES),
        .TAGW (`UUID_BITS)
    ) zbuf_streamer (
        .clk            (clk),
        .reset          (reset),

        .req_valid      (zbuf_req_valid),
        .req_rw         (zbuf_req_rw),
        .req_mask       (zbuf_req_mask),
        .req_byteen     (zbuf_req_byteen),
        .req_addr       (zbuf_req_addr),
        .req_data       (zbuf_req_data),
        .req_tag        (zbuf_req_tag),
        .req_ready      (zbuf_req_ready),

        .rsp_valid      (zbuf_rsp_valid),
        .rsp_mask       (zbuf_rsp_mask),
        .rsp_data       (zbuf_rsp_data),
        .rsp_tag        (zbuf_rsp_tag),
        .rsp_ready      (zbuf_rsp_ready)
    );

    ///////////////////////////////////////////////////////////////

    // Compare depth value with a reference value

    assign depth_ref = rop_req_if.depth & MASK;
    assign depth_val = zbuf_rsp_data & MASK;

    VX_rop_compare #(
        .DATAW (32)
    ) do_compare (
        .func   (rop_dcr_if.depth_func),
        .a      (depth_ref),
        .b      (depth_val),
        .result (passed)
    );

    wire rw = zbuf_rsp_valid && passed && (| rop_dcr_if.depth_mask);

    ///////////////////////////////////////////////////////////////

    // Write value into depth buffer

    VX_pipe_register #(
        .DATAW	(1 + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({rw,        depth_val}),
        .data_out ({zbuf_req_rw, zbuf_req_data})
    );

    assign rop_req_if.ready = zbuf_req_fire & zbuf_req_rw;

    ///////////////////////////////////////////////////////////////

endmodule