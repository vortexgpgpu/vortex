`include "VX_rop_define.vh"

module VX_rop_req_demux #(
    parameter NUM_REQS       = 1,
    parameter NUM_LANES      = 1,
    parameter BUFFERED       = 0,
    parameter string ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input request   
    VX_rop_req_if.slave  req_in_if,

    // output requests
    VX_rop_req_if.master req_out_if [NUM_REQS]
);
    localparam REQ_DATAW = NUM_LANES * (1 + 2 * `ROP_DIM_BITS + $bits(rgba_t) + `ROP_DEPTH_BITS + 1);

    if (NUM_REQS > 1) begin
        wire [NUM_REQS-1:0]                req_valid_out;
        wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_out;
        wire [NUM_REQS-1:0]                req_ready_out;

        VX_stream_demux #(
            .NUM_REQS   (NUM_REQS),
            .DATAW      (REQ_DATAW),
            .BUFFERED   (BUFFERED),
            .ARBITER    (ARBITER)
        ) req_demux (
            .clk        (clk),
            .reset      (reset),
            `UNUSED_PIN (sel_in),
            .valid_in   (req_in_if.valid),
            .data_in    ({req_in_if.tmask, req_in_if.pos_x, req_in_if.pos_y, req_in_if.color, req_in_if.depth, req_in_if.backface}),
            .ready_in   (req_in_if.ready),
            .valid_out  (req_valid_out),
            .data_out   (req_data_out),
            .ready_out  (req_ready_out)
        );
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign req_out_if[i].valid = req_valid_out[i];
            assign {req_out_if[i].tmask, req_out_if[i].pos_x, req_out_if[i].pos_y, req_out_if[i].color, req_out_if[i].depth, req_out_if[i].backface} = req_data_out[i];
            assign req_ready_out[i] = req_out_if[i].ready;
        end

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_out_if[0].valid = req_in_if.valid;
        assign req_out_if[0].tmask = req_in_if.tmask;
        assign req_out_if[0].pos_x = req_in_if.pos_x;
        assign req_out_if[0].pos_y = req_in_if.pos_y;
        assign req_out_if[0].color = req_in_if.color;
        assign req_out_if[0].depth = req_in_if.depth;
        assign req_out_if[0].backface = req_in_if.backface;
        assign req_in_if.ready = req_out_if[0].ready;

    end

endmodule
