`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_switch #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter DATAW          = 1,
    parameter LOCK_ENABLE    = 1,
    parameter BUFFERED       = 0,
    parameter NUM_REQS       = (NUM_INPUTS > NUM_OUTPUTS) ? ((NUM_INPUTS + NUM_OUTPUTS - 1) / NUM_OUTPUTS) : ((NUM_OUTPUTS + NUM_INPUTS - 1) / NUM_INPUTS),
    parameter SEL_COUNT      = `MIN(NUM_INPUTS, NUM_OUTPUTS),
    localparam LOG_NUM_REQS  = `CLOG2(NUM_REQS)
) (
    input  wire clk,
    input  wire reset,

    input wire  [SEL_COUNT-1:0][`UP(LOG_NUM_REQS)-1:0]      sel_in,

    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             valid_in,
    input  wire [NUM_INPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in,
    output wire [NUM_INPUTS-1:0][NUM_LANES-1:0]             ready_in,

    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out,
    output wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out,    
    input  wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out
);
    localparam NUM_REQS_A = 1 << LOG_NUM_REQS;

    if (NUM_INPUTS > NUM_OUTPUTS) begin

        wire [NUM_REQS_A-1:0][NUM_OUTPUTS-1:0][NUM_LANES-1:0]             valid_in_r;
        wire [NUM_REQS_A-1:0][NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in_r;

        for (genvar i = 0; i < NUM_REQS_A; ++i) begin
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

        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out_r;
        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out_r;
        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out_r;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            assign valid_out_r[i] = valid_in_r[sel_in[i]][i];
            assign data_out_r[i]  = data_in_r[sel_in[i]][i];
        end

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin
                localparam ii = i * NUM_OUTPUTS + j;                
                if (ii < NUM_INPUTS) begin                    
                    assign ready_in[ii] = ready_out_r[j] & {NUM_LANES{(sel_in[j] == LOG_NUM_REQS'(i))}};
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
                    .valid_in  (valid_out_r[i][j]),
                    .data_in   (data_out_r[i][j]),
                    .ready_in  (ready_out_r[i][j]),
                    .valid_out (valid_out[i][j]),
                    .data_out  (data_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
            end
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin
    
        wire [NUM_REQS_A-1:0][NUM_INPUTS-1:0][NUM_LANES-1:0] valid_out_r;
        wire [NUM_REQS_A-1:0][NUM_INPUTS-1:0][NUM_LANES-1:0] ready_out_r;

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS_A; ++j) begin
                assign valid_out_r[j][i] = valid_in[i] & {NUM_LANES{(sel_in[i] == LOG_NUM_REQS'(j))}};
            end
            assign ready_in[i] = ready_out_r[sel_in[i]][i];
        end

        for (genvar i = 0; i < NUM_REQS_A; ++i) begin
            for (genvar j = 0; j < NUM_INPUTS; ++j) begin
                localparam ii = i * NUM_INPUTS + j;
                if (ii < NUM_OUTPUTS) begin
                    for (genvar k = 0; k < NUM_LANES; ++k) begin
                        VX_skid_buffer #(
                            .DATAW    (DATAW),
                            .PASSTHRU (BUFFERED == 0),
                            .OUT_REG  (BUFFERED > 1)
                        ) out_buffer (
                            .clk       (clk),
                            .reset     (reset),
                            .valid_in  (valid_out_r[i][j][k]),
                            .data_in   (data_in[j][k]),
                            .ready_in  (ready_out_r[i][j][k]),
                            .valid_out (valid_out[ii][k]),
                            .data_out  (data_out[ii][k]),
                            .ready_out (ready_out[ii][k])
                        );
                    end
                end else begin
                    assign ready_out_r[i][j] = '0;
                end                
            end
        end
    
    end else begin
    
        `UNUSED_VAR (sel_in)

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
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