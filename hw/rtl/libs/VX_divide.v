`include "VX_define.vh"

module VX_divide #(
    parameter WIDTHN = 1,
    parameter WIDTHD = 1,
    parameter NSIGNED = 0,
    parameter DSIGNED = 0,
    parameter PIPELINE = 0
) (
    input wire clk,
    input wire reset,

    input [WIDTHN-1:0] numer,
    input [WIDTHD-1:0] denom,

    output reg [WIDTHN-1:0] quotient,
    output reg [WIDTHD-1:0] remainder
);

`ifdef QUARTUS

    lpm_divide quartus_div (
        .clock    (clk),        
        .numer    (numer),
        .denom    (denom),
        .quotient (quotient),
        .remain   (remainder),
        .aclr     (1'b0),
        .clken    (1'b1)
    );

    defparam
		quartus_div.lpm_type = "LPM_DIVIDE",
        quartus_div.lpm_widthn = WIDTHN,        
		quartus_div.lpm_widthd = WIDTHD,		
		quartus_div.lpm_nrepresentation = NSIGNED ? "SIGNED" : "UNSIGNED",
        quartus_div.lpm_drepresentation = DSIGNED ? "SIGNED" : "UNSIGNED",
		quartus_div.lpm_hint = "MAXIMIZE_SPEED=6,LPM_REMAINDERPOSITIVE=FALSE",
		quartus_div.lpm_pipeline = PIPELINE;	

`else

    reg [WIDTHN-1:0] quotient_unqual;
    reg [WIDTHD-1:0] remainder_unqual;

    always @(*) begin   
    `ifndef SYNTHESIS    
        // this edge case kills verilator in some cases by causing a division
        // overflow exception. INT_MIN / -1 (on x86)
        if (numer == {1'b1, (WIDTHN-1)'(0)}
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
        assign quotient  = quotient_unqual;
        assign remainder = remainder_unqual;
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
                else begin
                    if (i == 0) begin
                        quotient_pipe[0]  <= quotient_unqual;
                        remainder_pipe[0] <= remainder_unqual;
                    end else begin
                        quotient_pipe[i]  <= quotient_pipe[i-1];
                        remainder_pipe[i] <= remainder_pipe[i-1];    
                    end                     
                end
            end
        end

        assign quotient  = quotient_pipe[PIPELINE-1];
        assign remainder = remainder_pipe[PIPELINE-1];
    end    

`endif

endmodule