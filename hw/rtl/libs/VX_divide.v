`include "VX_platform.vh"

module VX_divide #(
    parameter WIDTHN = 1,
    parameter WIDTHD = 1,
    parameter WIDTHQ = 1,
    parameter WIDTHR = 1,
    parameter NSIGNED = 0,
    parameter DSIGNED = 0,
    parameter PIPELINE = 0
) (
    input wire clk,
    input wire reset,

    input wire clk_en,
    input wire [WIDTHN-1:0] numer,
    input wire [WIDTHD-1:0] denom,

    output wire [WIDTHQ-1:0] quotient,
    output wire [WIDTHR-1:0] remainder
);

`ifdef QUARTUS

    wire [WIDTHN-1:0] quotient_unqual;
    wire [WIDTHD-1:0] remainder_unqual;

    lpm_divide divide (
        .clock    (clk),        
        .numer    (numer),
        .denom    (denom),
        .quotient (quotient_unqual),
        .remain   (remainder_unqual),
        .clken    (clk_en)
    );

    defparam
		divide.lpm_type = "LPM_DIVIDE",
        divide.lpm_widthn = WIDTHN,        
		divide.lpm_widthd = WIDTHD,		
		divide.lpm_nrepresentation = NSIGNED ? "SIGNED" : "UNSIGNED",
        divide.lpm_drepresentation = DSIGNED ? "SIGNED" : "UNSIGNED",
		divide.lpm_hint = "MAXIMIZE_SPEED=6,LPM_REMAINDERPOSITIVE=FALSE",
		divide.lpm_pipeline = PIPELINE;

    assign quotient  = quotient_unqual [WIDTHQ-1:0];
    assign remainder = remainder_unqual [WIDTHR-1:0];

`else

    reg [WIDTHN-1:0] quotient_unqual;
    reg [WIDTHD-1:0] remainder_unqual;

    always @(*) begin   
    `ifndef SYNTHESIS    
        // this edge case kills verilator in some cases by causing a division
        // overflow exception. INT_MIN / -1 (on x86)
        if (numer == {1'b1, (WIDTHN-1)'(1'b0)}
         && denom == {WIDTHD{1'b1}}) begin
            quotient_unqual  = 0;
            remainder_unqual = 0;
        end else
    `endif
        begin
            if (NSIGNED && DSIGNED) begin
                quotient_unqual  = $signed(numer) / $signed(denom);
                remainder_unqual = $signed(numer) % $signed(denom);
            end 
            else if (NSIGNED && !DSIGNED) begin
                quotient_unqual  = $signed(numer) / denom;
                remainder_unqual = $signed(numer) % denom;
            end 
            else if (!NSIGNED && DSIGNED) begin
                quotient_unqual  = numer / $signed(denom);
                remainder_unqual = numer % $signed(denom);
            end 
            else begin
                quotient_unqual  = numer / denom;
                remainder_unqual = numer % denom;        
            end
        end
    end

    if (PIPELINE == 0) begin
        assign quotient  = quotient_unqual [WIDTHQ-1:0];
        assign remainder = remainder_unqual [WIDTHR-1:0];
    end else begin
        reg [WIDTHN-1:0] quotient_pipe [0:PIPELINE-1];
        reg [WIDTHD-1:0] remainder_pipe [0:PIPELINE-1];

        genvar i;
        for (i = 0; i < PIPELINE; i++) begin
            always @(posedge clk) begin
                if (reset) begin
                    quotient_pipe[i]  <= 0;
                    remainder_pipe[i] <= 0;
                end
                else if (clk_en) begin
                    if (i == 0) begin
                        quotient_pipe[i]  <= quotient_unqual;
                        remainder_pipe[i] <= remainder_unqual;
                    end else begin
                        quotient_pipe[i]  <= quotient_pipe[i-1];
                        remainder_pipe[i] <= remainder_pipe[i-1];
                    end                    
                end
            end
        end

        assign quotient  = quotient_pipe[PIPELINE-1][WIDTHQ-1:0];
        assign remainder = remainder_pipe[PIPELINE-1][WIDTHR-1:0];
    end    

`endif

endmodule