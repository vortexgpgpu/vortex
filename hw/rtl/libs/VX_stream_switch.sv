`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_switch #(
    parameter NUM_INPUTS    = 1,
    parameter NUM_OUTPUTS   = 1,
    parameter NUM_LANES     = 1,
    parameter DATAW         = 1,
    parameter BUFFERED      = 0,
    parameter NUM_REQS      = (NUM_INPUTS > NUM_OUTPUTS) ? ((NUM_INPUTS + NUM_OUTPUTS - 1) / NUM_OUTPUTS) : ((NUM_OUTPUTS + NUM_INPUTS - 1) / NUM_INPUTS),
    parameter SEL_COUNT     = `MIN(NUM_INPUTS, NUM_OUTPUTS),
    parameter MAX_FANOUT    = 8,
    parameter LOG_NUM_REQS  = `CLOG2(NUM_REQS)
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
    localparam BUF_SIZE = (BUFFERED == 3) ? 1 : ((BUFFERED != 0) ? 2 : 0);

    if (NUM_INPUTS > NUM_OUTPUTS) begin

        wire [NUM_OUTPUTS-1:0][NUM_REQS-1:0][NUM_LANES-1:0]             valid_in_r;
        wire [NUM_OUTPUTS-1:0][NUM_REQS-1:0][NUM_LANES-1:0][DATAW-1:0]  data_in_r;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                localparam ii = i * NUM_REQS + j;
                if (ii < NUM_INPUTS) begin
                    assign valid_in_r[i][j] = valid_in[ii];
                    assign data_in_r[i][j]  = data_in[ii];
                end else begin
                    assign valid_in_r[i][j] = 0;
                    assign data_in_r[i][j]  = '0;
                end
            end
        end        

        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            valid_out_r;
        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][DATAW-1:0] data_out_r;
        wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]            ready_out_r;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            assign valid_out_r[i] = valid_in_r[i][sel_in[i]];
            assign data_out_r[i]  = data_in_r[i][sel_in[i]];
        end

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                localparam ii = i * NUM_REQS + j;                
                if (ii < NUM_INPUTS) begin                    
                    assign ready_in[ii] = ready_out_r[i] & {NUM_LANES{(sel_in[i] == LOG_NUM_REQS'(j))}};
                end
            end
        end

        `RESET_RELAY_EX (out_buf_reset, reset, 1, ((NUM_OUTPUTS * NUM_LANES) > MAX_FANOUT) ? 0 : -1);

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin
                localparam ii = i * NUM_LANES + j;
                VX_elastic_buffer #(
                    .DATAW    (DATAW),
                    .SIZE     (BUF_SIZE),
                    .OUT_REG  (BUFFERED != 1)
                ) out_buf (
                    .clk       (clk),
                    .reset     (out_buf_reset),
                    .valid_in  (valid_out_r[i][j]),                    
                    .ready_in  (ready_out_r[i][j]),
                    .data_in   (data_out_r[i][j]),
                    .data_out  (data_out[i][j]),
                    .valid_out (valid_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
            end
        end

    end else if (NUM_OUTPUTS > NUM_INPUTS) begin
    
        wire [NUM_INPUTS-1:0][NUM_REQS-1:0][NUM_LANES-1:0] valid_out_r;
        wire [NUM_INPUTS-1:0][NUM_REQS-1:0][NUM_LANES-1:0] ready_out_r;

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                assign valid_out_r[i][j] = valid_in[i] & {NUM_LANES{(sel_in[i] == LOG_NUM_REQS'(j))}};
            end
            assign ready_in[i] = ready_out_r[i][sel_in[i]];
        end

        `RESET_RELAY_EX (out_buf_reset, reset, 1, (NUM_OUTPUTS > MAX_FANOUT) ? 0 : -1);

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin
            for (genvar j = 0; j < NUM_REQS; ++j) begin
                localparam ii = i * NUM_REQS + j;
                if (ii < NUM_OUTPUTS) begin
                    for (genvar k = 0; k < NUM_LANES; ++k) begin
                        VX_elastic_buffer #(
                            .DATAW    (DATAW),
                            .SIZE     (BUF_SIZE),
                            .OUT_REG  (BUFFERED != 1)
                        ) out_buf (
                            .clk       (clk),
                            .reset     (out_buf_reset),
                            .valid_in  (valid_out_r[i][j][k]),
                            .ready_in  (ready_out_r[i][j][k]),
                            .data_in   (data_in[i][k]),                                                     
                            .data_out  (data_out[ii][k]),
                            .valid_out (valid_out[ii][k]),
                            .ready_out (ready_out[ii][k])
                        );
                    end
                end else begin
                    `UNUSED_VAR (valid_out_r[i][j])
                    assign ready_out_r[i][j] = '0;
                end                
            end
        end
    
    end else begin
    
        `UNUSED_VAR (sel_in)

        `RESET_RELAY_EX (out_buf_reset, reset, 1, ((NUM_OUTPUTS * NUM_LANES) > MAX_FANOUT) ? 0 : -1);

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
            for (genvar j = 0; j < NUM_LANES; ++j) begin
                VX_elastic_buffer #(
                    .DATAW    (DATAW),
                    .SIZE     (BUF_SIZE),
                    .OUT_REG  (BUFFERED != 1)
                ) out_buf (
                    .clk       (clk),
                    .reset     (out_buf_reset),
                    .valid_in  (valid_in[i][j]),
                    .ready_in  (ready_in[i][j]),
                    .data_in   (data_in[i][j]),
                    .data_out  (data_out[i][j]),
                    .valid_out (valid_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
            end
        end

    end
    
endmodule
`TRACING_ON
