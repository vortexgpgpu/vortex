`include "VX_platform.vh"

module VX_stream_demux #(
    parameter NUM_REQS = 1,
    parameter DATAW    = 1,
    parameter BUFFERED = 0,
    localparam LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire clk,
    input  wire reset,

    input wire [LOG_NUM_REQS-1:0] sel,

    input  wire             valid_in,
    input  wire [DATAW-1:0] data_in,    
    output wire             ready_in,

    output wire [NUM_REQS-1:0]            valid_out,
    output wire [NUM_REQS-1:0][DATAW-1:0] data_out,
    input  wire [NUM_REQS-1:0]            ready_out
  );
  
    if (NUM_REQS > 1)  begin

        reg [NUM_REQS-1:0]             valid_out_unqual;
        wire [NUM_REQS-1:0][DATAW-1:0] data_out_unqual;
        wire [NUM_REQS-1:0]            ready_out_unqual;

        always @(*) begin
            valid_out_unqual = '0;
            valid_out_unqual[sel] = valid_in;
        end
        
        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign data_out_unqual[i] = data_in;      
        end
        
        assign ready_in = ready_out_unqual[sel]; 

        for (genvar i = 0; i < NUM_REQS; i++) begin  
            VX_skid_buffer #(
                .DATAW    (DATAW),
                .PASSTHRU (!BUFFERED)
            ) out_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (valid_out_unqual[i]),        
                .data_in   (data_out_unqual[i]),
                .ready_in  (ready_out_unqual[i]),      
                .valid_out (valid_out[i]),
                .data_out  (data_out[i]),
                .ready_out (ready_out[i])
            );                   
        end

    end else begin
    
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (sel)
        
        assign valid_out = valid_in;        
        assign data_out  = data_in;
        assign ready_in  = ready_out;        

    end
    
endmodule