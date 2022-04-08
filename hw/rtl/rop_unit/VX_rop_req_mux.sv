`include "VX_rop_define.vh"

import VX_rop_types::*;

module VX_rop_req_mux #(
    parameter NUM_REQS       = 1,
    parameter BUFFERED_REQ   = 0,
    parameter string ARBITER = "R"
) (
    input wire clk,
    input wire reset,

    // input requests    
    VX_rop_req_if.slave     req_in_if[NUM_REQS],

    // output request
    VX_rop_req_if.master    req_out_if
);

    localparam REQ_DATAW = 2 * `NUM_THREADS + 2 * (`NUM_THREADS * `ROP_DIM_BITS) + (`NUM_THREADS * 32) + (`NUM_THREADS * `ROP_DEPTH_BITS);

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0] req_valid_in;
        wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;
        wire [NUM_REQS-1:0] req_ready_in;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign req_valid_in[i] = req_in_if[i].valid;
            assign req_data_in[i] = {req_in_if[i].tmask, req_in_if[i].pos_x, req_in_if[i].pos_y, req_in_if[i].color, req_in_if[i].depth, req_in_if[i].backface};
            assign req_in_if[i].ready = req_ready_in[i];
        end        

        VX_stream_mux #(            
            .NUM_REQS (NUM_REQS),
            .DATAW    (REQ_DATAW),
            .BUFFERED (BUFFERED_REQ),
            .ARBITER  (ARBITER)
        ) req_mux (
            .clk       (clk),
            .reset     (reset),
            `UNUSED_PIN (sel_in),
            .valid_in  (req_valid_in),
            .data_in   (req_data_in),
            .ready_in  (req_ready_in),
            .valid_out (req_out_if.valid),
            .data_out  ({req_out_if.tmask, req_out_if.pos_x, req_out_if.pos_y, req_out_if.color, req_out_if.depth, req_out_if.backface}),
            .ready_out (req_out_if.ready)
        );

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_out_if.valid    = req_in_if[0].valid;

        assign req_out_if.tmask    = req_in_if[0].tmask;
        assign req_out_if.pos_x    = req_in_if[0].pos_x;
        assign req_out_if.pos_y    = req_in_if[0].pos_y;
        assign req_out_if.color    = req_in_if[0].color;
        assign req_out_if.depth    = req_in_if[0].depth;
        assign req_out_if.backface = req_in_if[0].backface;

        assign req_in_if[0].ready  = req_out_if.ready;

    end

endmodule
