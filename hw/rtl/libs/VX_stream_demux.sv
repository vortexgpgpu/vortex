`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_demux #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter DATAW          = 1,
    parameter string ARBITER = "",
    parameter LOCK_ENABLE    = 1,
    parameter BUFFERED       = 0,
    parameter NUM_REQS       = (NUM_OUTPUTS + NUM_INPUTS - 1) / NUM_INPUTS,
    localparam LOG_NUM_REQS  = `LOG2UP(NUM_REQS)
) (
    input  wire clk,
    input  wire reset,

    input wire [NUM_LANES-1:0][`UP(LOG_NUM_REQS)-1:0]       sel_in,

    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             valid_in,
    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in,    
    output wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             ready_in,

    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out,
    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out,
    input  wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out
);
    `STATIC_ASSERT ((NUM_OUTPUTS >= NUM_INPUTS), ("invalid parameter"))

    if (NUM_OUTPUTS > NUM_INPUTS)  begin

        wire [NUM_REQS-1:0][NUM_INPUTS-1:0][NUM_LANES-1:0] sel_ready;

        wire [NUM_LANES-1:0][LOG_NUM_REQS-1:0] sel_index;
        wire [NUM_LANES-1:0][NUM_REQS-1:0]     sel_onehot;           

        if (ARBITER != "") begin   
            `UNUSED_VAR (sel_in)        
            wire [NUM_REQS-1:0]     arb_requests;
            wire [LOG_NUM_REQS-1:0] arb_index;
            wire [NUM_REQS-1:0]     arb_onehot;
            wire                    arb_unlock;

            for (genvar i = 0; i < NUM_REQS; ++i) begin
                assign arb_requests[i] = (| sel_ready[i]);
            end

            if ((NUM_INPUTS * NUM_LANES) > 1) begin
                assign arb_unlock = | (valid_in & ready_in);
            end else begin
                assign arb_unlock = valid_in & ready_in; 
            end

            VX_generic_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (LOCK_ENABLE),
                .TYPE        (ARBITER)
            ) arbiter (
                .clk          (clk),
                .reset        (reset),
                .requests     (arb_requests),  
                .unlock       (arb_unlock),
                `UNUSED_PIN   (grant_valid),
                .grant_index  (arb_index),
                .grant_onehot (arb_onehot)
            );

            for (genvar i = 0; i < NUM_LANES; i++) begin
                assign sel_index[i]  = arb_index;
                assign sel_onehot[i] = arb_onehot;
            end
        end else begin
            assign sel_index = sel_in;
            reg [NUM_LANES-1:0][NUM_REQS-1:0] sel_onehot_r;
            always @(*) begin
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    sel_onehot_r[i]            = '0;
                    sel_onehot_r[i][sel_in[i]] = 1;
                end
            end
            assign sel_onehot = sel_onehot_r;
        end

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin
                for (genvar k = 0; k < NUM_REQS; ++k) begin            
                    localparam ii = k * NUM_INPUTS + i;
                    if (ii < NUM_OUTPUTS) begin
                        VX_skid_buffer #(
                            .DATAW    (DATAW),
                            .PASSTHRU (BUFFERED == 0),
                            .OUT_REG  (BUFFERED > 1)
                        ) out_buffer (
                            .clk       (clk),
                            .reset     (reset),
                            .valid_in  (valid_in[i][j] && sel_onehot[j][k]),
                            .data_in   (data_in[i][j]),
                            .ready_in  (sel_ready[k][i][j]),
                            .valid_out (valid_out[ii][j]),
                            .data_out  (data_out[ii][j]),
                            .ready_out (ready_out[ii][j])
                        );
                    end
                end
                assign ready_in[i][j] = sel_ready[sel_index[j]][i][j];
            end
        end

    end else begin

        `UNUSED_VAR (sel_in)

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin
                VX_skid_buffer #(
                    .DATAW    (DATAW),
                    .PASSTHRU (BUFFERED == 0),
                    .OUT_REG  (BUFFERED > 1)
                ) out_buffer (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in[i][j]),
                    .data_in   (data_in[i][j]),
                    .ready_in  (ready_in[i][j]),      
                    .valid_out (valid_out[i][j]),
                    .data_out  (data_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
            end
        end

    end
    
endmodule
`TRACING_ON