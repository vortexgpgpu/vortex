`include "VX_rop_define.vh"

module VX_rop_req_arb #(
    parameter NUM_REQS = 1
) (
    input wire clk,
    input wire reset,

    // input requests    
    VX_rop_req_if.slave     req_in[NUM_REQS],

    // output request
    VX_rop_req_if.master    req_out
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    // TODO
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        wire valid = req_in[i].valid;
        wire [`NUM_THREADS-1:0][15:0] pos_x = req_in[i].pos_x;
        wire [`NUM_THREADS-1:0][15:0] pos_y = req_in[i].pos_y;
        wire [`NUM_THREADS-1:0][31:0] color = req_in[i].color;
        wire [`NUM_THREADS-1:0][`ROP_DEPTH_BITS-1:0] depth = req_in[i].depth;
        `UNUSED_VAR (valid)
        `UNUSED_VAR (pos_x)
        `UNUSED_VAR (pos_y)
        `UNUSED_VAR (color)
        `UNUSED_VAR (depth)
        assign req_in[i].ready = 0;
    end

    // TODO
    assign req_out.valid = 0;
    assign req_out.pos_x = 0;
    assign req_out.pos_y = 0;
    assign req_out.color = 0;
    assign req_out.depth = 0;
    `UNUSED_VAR (req_out.ready)

endmodule