`include "VX_platform.vh"

`TRACING_OFF
module VX_multiplier #(
    parameter WIDTHA  = 1,
    parameter WIDTHB  = 1,
    parameter WIDTHP  = 1,
    parameter SIGNED  = 0,
    parameter LATENCY = 0
) (
    input wire clk,    
    input wire enable,
    input wire [WIDTHA-1:0]  dataa,
    input wire [WIDTHB-1:0]  datab,
    output wire [WIDTHP-1:0] result
);

`ifdef QUARTUS

    lpm_mult mult (
        .clock  (clk),
        .clken  (enable),
        .dataa  (dataa),
        .datab  (datab),
        .result (result),        
        .aclr   (1'b0),
        .sclr   (1'b0),
        .sum    (1'b0)
    );

    defparam mult.lpm_type   = "LPM_MULT",
             mult.lpm_widtha = WIDTHA,
             mult.lpm_widthb = WIDTHB,
             mult.lpm_widthp = WIDTHP,
             mult.lpm_representation = SIGNED ? "SIGNED" : "UNSIGNED",
             mult.lpm_pipeline = LATENCY,
             mult.lpm_hint   = "DEDICATED_MULTIPLIER_CIRCUITRY=YES,MAXIMIZE_SPEED=9";
`else

    wire [WIDTHP-1:0] result_unqual;

    if (SIGNED) begin
        assign result_unqual = $signed(dataa) * $signed(datab);
    end else begin
        assign result_unqual = dataa * datab;
    end
    
    if (LATENCY == 0) begin
        assign result = result_unqual;
    end else begin        
        reg [WIDTHP-1:0] result_pipe [LATENCY-1:0];
        always @(posedge clk) begin
            if (enable) begin
                result_pipe[0] <= result_unqual;
            end
        end
        for (genvar i = 1; i < LATENCY; i++) begin
            always @(posedge clk) begin
                if (enable) begin
                    result_pipe[i] <= result_pipe[i-1];
                end
            end
        end
        assign result = result_pipe[LATENCY-1]; 
    end

`endif

endmodule
`TRACING_ON
