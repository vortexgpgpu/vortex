`include "VX_platform.vh"

`TRACING_OFF
module VX_shift_register_nr #( 
    parameter DATAW  = 1,
    parameter DEPTH  = 1,
    parameter NTAPS  = 1,
    parameter DEPTHW = $clog2(DEPTH),
    parameter [(DEPTHW*NTAPS)-1:0] TAPS = {NTAPS{DEPTHW'(DEPTH-1)}}
) (
    input wire clk,
    input wire enable,
    input wire [DATAW-1:0]          data_in,
    output wire [(NTAPS*DATAW)-1:0] data_out
);
    reg [DEPTH-1:0][DATAW-1:0] entries;

    always @(posedge clk) begin
        if (enable) begin                    
            for (integer i = DEPTH-1; i > 0; --i)
                entries[i] <= entries[i-1];
            entries[0] <= data_in;
        end
    end

    for (genvar i = 0; i < NTAPS; ++i) begin
        assign data_out [i*DATAW+:DATAW] = entries [TAPS[i*DEPTHW+:DEPTHW]];
    end

endmodule

module VX_shift_register_wr #( 
    parameter DATAW  = 1, 
    parameter DEPTH  = 1,
    parameter NTAPS  = 1,
    parameter DEPTHW = $clog2(DEPTH),
    parameter [(DEPTHW*NTAPS)-1:0] TAPS = {NTAPS{DEPTHW'(DEPTH-1)}}
) (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [DATAW-1:0]          data_in,
    output wire [(NTAPS*DATAW)-1:0] data_out
);
    reg [DEPTH-1:0][DATAW-1:0] entries;

    always @(posedge clk) begin
        if (reset) begin
            entries <= '0;
        end else if (enable) begin                    
            for (integer i = DEPTH-1; i > 0; --i)
                entries[i] <= entries[i-1];
            entries[0] <= data_in;
        end
    end

    for (genvar i = 0; i < NTAPS; ++i) begin
        assign data_out [i*DATAW+:DATAW] = entries [TAPS[i*DEPTHW+:DEPTHW]];
    end

endmodule

module VX_shift_register #( 
    parameter DATAW  = 1, 
    parameter RESETW = 0,
    parameter DEPTH  = 1,
    parameter NTAPS  = 1,
    parameter DEPTHW = $clog2(DEPTH),
    parameter [(DEPTHW*NTAPS)-1:0] TAPS = {NTAPS{DEPTHW'(DEPTH-1)}}
) (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire [DATAW-1:0]          data_in,
    output wire [(NTAPS*DATAW)-1:0] data_out
);
    if (RESETW != 0) begin
        if (RESETW == DATAW) begin
    
            VX_shift_register_wr #(
                .DATAW (DATAW),
                .DEPTH (DEPTH),
                .NTAPS (NTAPS),
                .TAPS  (TAPS)
            ) sr (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (data_in),
                .data_out (data_out)
            );
    
        end else begin
    
            VX_shift_register_wr #(
                .DATAW (RESETW),
                .DEPTH (DEPTH),
                .NTAPS (NTAPS),
                .TAPS  (TAPS)
            ) sr_wr (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (data_in[DATAW-1:DATAW-RESETW]),
                .data_out (data_out[DATAW-1:DATAW-RESETW])
            );

            VX_shift_register_nr #(
                .DATAW (DATAW-RESETW),
                .DEPTH (DEPTH),
                .NTAPS (NTAPS),
                .TAPS  (TAPS)
            ) sr_nr (
                .clk      (clk),
                .enable   (enable),
                .data_in  (data_in[DATAW-RESETW-1:0]),
                .data_out (data_out[DATAW-RESETW-1:0])
            );

        end

    end else begin        

        `UNUSED_VAR (reset)
    
        VX_shift_register_nr #(
            .DATAW (DATAW),
            .DEPTH (DEPTH),
            .NTAPS (NTAPS),
            .TAPS  (TAPS)
        ) sr (
            .clk      (clk),
            .enable   (enable),
            .data_in  (data_in),
            .data_out (data_out)
        );

    end    

endmodule
`TRACING_ON
