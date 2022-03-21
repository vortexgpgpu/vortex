`include "VX_platform.vh"

module VX_stream_demux #(
    parameter NUM_REQS = 1,
    parameter LANES    = 1,
    parameter DATAW    = 1,
    parameter BUFFERED = 0,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire clk,
    input  wire reset,

    input wire [LANES-1:0][LOG_NUM_REQS-1:0] sel_in,

    input  wire [LANES-1:0]            valid_in,
    input  wire [LANES-1:0][DATAW-1:0] data_in,    
    output wire [LANES-1:0]            ready_in,

    output wire [NUM_REQS-1:0][LANES-1:0]            valid_out,
    output wire [NUM_REQS-1:0][LANES-1:0][DATAW-1:0] data_out,
    input  wire [NUM_REQS-1:0][LANES-1:0]            ready_out
  );
  
    if (NUM_REQS > 1)  begin

        for (genvar j = 0; j < LANES; ++j) begin

            reg [NUM_REQS-1:0]  valid_in_sel;
            wire [NUM_REQS-1:0] ready_in_sel;

            always @(*) begin
                valid_in_sel            = '0;
                valid_in_sel[sel_in[j]] = valid_in[j];
            end

            assign ready_in[j] = ready_in_sel[sel_in[j]]; 

            for (genvar i = 0; i < NUM_REQS; i++) begin
                VX_skid_buffer #(
                    .DATAW    (DATAW),
                    .PASSTHRU (0 == BUFFERED),
                    .OUT_REG  (2 == BUFFERED)
                ) out_buffer (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (valid_in_sel[i]),        
                    .data_in   (data_in[j]),
                    .ready_in  (ready_in_sel[i]),      
                    .valid_out (valid_out[i][j]),
                    .data_out  (data_out[i][j]),
                    .ready_out (ready_out[i][j])
                );
            end
        end

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (sel_in)
        
        assign valid_out = valid_in;        
        assign data_out  = data_in;
        assign ready_in  = ready_out;

    end
    
endmodule
