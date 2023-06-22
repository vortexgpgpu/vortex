`include "VX_define.vh"

module VX_gbar_arb #(
    parameter NUM_REQS = 1,
    parameter BUFFERED_REQ = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_gbar_bus_if.slave    bus_in_if[NUM_REQS],
    VX_gbar_bus_if.master   bus_out_if
);

    localparam REQ_DATAW = `NB_BITS + `UP(`NC_BITS) + `UP(`NC_BITS);

    // arbitrate request

    wire [NUM_REQS-1:0]                req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;
    wire [NUM_REQS-1:0]                req_ready_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i]  = {bus_in_if[i].req_id, bus_in_if[i].req_size_m1, bus_in_if[i].req_core_id};
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        .NUM_LANES   (1),
        .DATAW       (REQ_DATAW),
        .ARBITER     (ARBITER),
        .BUFFERED    (BUFFERED_REQ)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (req_valid_in),
        .ready_in   (req_ready_in),
        .data_in    (req_data_in),
        .data_out   ({bus_out_if.req_id, bus_out_if.req_size_m1, bus_out_if.req_core_id}),
        .valid_out  (bus_out_if.req_valid),
        .ready_out  (bus_out_if.req_ready)
    );

    // broadcast response

    reg rsp_valid;
    reg [`NB_BITS-1:0] rsp_id;

    always @(posedge clk) begin
        if (reset) begin
            rsp_valid <= 0;
        end else begin
            rsp_valid <= bus_out_if.rsp_valid;
        end
        rsp_id <= bus_out_if.rsp_id;
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign bus_in_if[i].rsp_valid = rsp_valid;
        assign bus_in_if[i].rsp_id = rsp_id;
    end

endmodule
