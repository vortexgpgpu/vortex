`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_mux #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter DATAW          = 1,
    parameter string ARBITER = "",
    parameter LOCK_ENABLE    = 1,
    parameter BUFFERED       = 0,
    parameter NUM_REQS       = (NUM_INPUTS + NUM_OUTPUTS - 1) / NUM_OUTPUTS,
    localparam LOG_NUM_REQS  = `CLOG2(NUM_REQS)
) (
    input  wire clk,
    input  wire reset,

    input wire  [`UP(LOG_NUM_REQS)-1:0]                     sel_in,

    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             valid_in,
    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in,
    output wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             ready_in,

    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out,
    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out,    
    input  wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out
);
    `STATIC_ASSERT ((NUM_INPUTS >= NUM_OUTPUTS), ("invalid parameter"))

    if (NUM_INPUTS > NUM_OUTPUTS)  begin    

        wire [NUM_REQS-1:0][NUM_OUTPUTS-1:0][NUM_LANES-1:0]             valid_in_r;
        wire [NUM_REQS-1:0][NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in_r;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin
                localparam ii = i * NUM_OUTPUTS + j;
                if (ii < NUM_INPUTS) begin
                    assign valid_in_r[i][j] = valid_in[ii];
                    assign data_in_r[i][j]  = data_in[ii];
                end else begin
                    assign valid_in_r[i][j] = 0;
                    assign data_in_r[i][j]  = 'x;
                end
            end
        end        

        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            sel_valid;
        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] sel_data;
        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            sel_ready;
        
        wire [LOG_NUM_REQS-1:0] sel_index;
        wire [NUM_REQS-1:0]     sel_onehot;

        if (ARBITER != "") begin   
            `UNUSED_VAR (sel_in)        
            wire [NUM_REQS-1:0] arb_requests;
            wire                arb_unlock;

            if (NUM_LANES > 1) begin
                for (genvar i = 0; i < NUM_REQS; ++i) begin
                    assign arb_requests[i] = (| valid_in_r[i]);
                end
            end else begin
                for (genvar i = 0; i < NUM_REQS; ++i) begin
                    assign arb_requests[i] = valid_in_r[i];
                end
            end            

            if ((NUM_OUTPUTS * NUM_LANES) > 1) begin
                assign arb_unlock = | (sel_valid & sel_ready);
            end else begin
                assign arb_unlock = sel_valid & sel_ready;
            end

            VX_generic_arbiter #(
                .NUM_REQS    (NUM_REQS),
                .LOCK_ENABLE (1),
                .TYPE        (ARBITER)
            ) arbiter (
                .clk          (clk),
                .reset        (reset),
                .requests     (arb_requests),  
                .unlock       (arb_unlock),
                `UNUSED_PIN   (grant_valid),
                .grant_index  (sel_index),
                .grant_onehot (sel_onehot)
            );
        end else begin
            assign sel_index = sel_in;
            reg [NUM_REQS-1:0] sel_onehot_r;
            always @(*) begin
                sel_onehot_r = '0;
                sel_onehot_r[sel_in] = 1;
            end
            assign sel_onehot = sel_onehot_r;
        end        

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin
                assign sel_valid[i][j] = valid_in_r[sel_index][i][j];
                assign sel_data[i][j]  = data_in_r[sel_index][i][j];
                for (genvar k = 0; k < NUM_REQS; ++k) begin
                    localparam ii = k * NUM_OUTPUTS + i;
                    if (ii < NUM_INPUTS) begin
                        assign ready_in[ii][j] = sel_ready[i][j] && sel_onehot[k];
                    end
                end
            end
        end

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin
                VX_skid_buffer #(
                    .DATAW    (DATAW),
                    .PASSTHRU (BUFFERED == 0),
                    .OUT_REG  (BUFFERED > 1)
                ) out_buffer (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (sel_valid[i][j]),
                    .data_in   (sel_data[i][j]),
                    .ready_in  (sel_ready[i][j]),
                    .valid_out (valid_out[i][j]),
                    .data_out  (data_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
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