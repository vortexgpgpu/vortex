`include "VX_define.vh"

module VX_multiplier #(
    parameter WIDTHA = 1,
    parameter WIDTHB = 1,
    parameter WIDTHP = 1,
    parameter SIGNED = 0,
    parameter PIPELINE = 0
) (
    input wire clk,
    input wire reset,

    input wire clk_en,
    input wire [WIDTHA-1:0]  dataa,
    input wire [WIDTHB-1:0]  datab,
    output wire [WIDTHP-1:0] result
);

`ifdef QUARTUS

    lpm_mult quartus_mult (
        .clock  (clk),
        .dataa  (dataa),
        .datab  (datab),
        .result (result),
        .sclr   (reset),
        .aclr   (1'b0),
        .clken  (clk_en),
        .sum    (1'b0)
    );

    defparam quartus_mult.lpm_type = "LPM_MULT",
             quartus_mult.lpm_widtha = WIDTHA,
             quartus_mult.lpm_widthb = WIDTHB,
             quartus_mult.lpm_widthp = WIDTHP,
             quartus_mult.lpm_representation = SIGNED ? "SIGNED" : "UNSIGNED",
             quartus_mult.lpm_pipeline = PIPELINE,
             quartus_mult.lpm_hint = "DEDICATED_MULTIPLIER_CIRCUITRY=YES,MAXIMIZE_SPEED=9";
`else

    wire [WIDTHP-1:0] result_unqual;

    if (SIGNED) begin
        assign result_unqual = $signed(dataa) * $signed(datab);
    end else begin
        assign result_unqual = dataa * datab;
    end
    
    if (PIPELINE == 0) begin
        assign result = result_unqual;
    end else begin
        
        reg [WIDTHP-1:0] result_pipe [0:PIPELINE-1];

        genvar i;
        for (i = 0; i < PIPELINE; i++) begin
            always @(posedge clk) begin
                if (reset) begin
                    result_pipe[i] <= 0;
                end
                else if (clk_en) begin
                    if (i == 0) begin
                        result_pipe[i] <= result_unqual;
                    end else begin
                        result_pipe[i] <= result_pipe[i-1];
                    end                    
                end
            end
        end
        
        assign result = result_pipe[PIPELINE-1]; 
    end

`endif

endmodule