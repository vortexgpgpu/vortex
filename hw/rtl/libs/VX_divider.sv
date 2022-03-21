`include "VX_platform.vh"

`TRACING_OFF
module VX_divider #(
    parameter WIDTHN  = 1,
    parameter WIDTHD  = 1,
    parameter WIDTHQ  = 1,
    parameter WIDTHR  = 1,
    parameter NSIGNED = 0,
    parameter DSIGNED = 0,
    parameter LATENCY = 0
) (
    input wire clk,
    input wire enable, 
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
        .clken    (enable),
        .numer    (numer),
        .denom    (denom),
        .quotient (quotient_unqual),
        .remain   (remainder_unqual)
    );

    defparam
        divide.lpm_type     = "LPM_DIVIDE",
        divide.lpm_widthn   = WIDTHN,        
        divide.lpm_widthd   = WIDTHD,
        divide.lpm_nrepresentation = NSIGNED ? "SIGNED" : "UNSIGNED",
        divide.lpm_drepresentation = DSIGNED ? "SIGNED" : "UNSIGNED",
        divide.lpm_hint     = "MAXIMIZE_SPEED=6,LPM_REMAINDERPOSITIVE=FALSE",
        divide.lpm_pipeline = LATENCY;

    assign quotient  = quotient_unqual [WIDTHQ-1:0];
    assign remainder = remainder_unqual [WIDTHR-1:0];

`else

    reg [WIDTHN-1:0] quotient_unqual;
    reg [WIDTHD-1:0] remainder_unqual;

    always @(*) begin   
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

    if (LATENCY == 0) begin
        assign quotient  = quotient_unqual [WIDTHQ-1:0];
        assign remainder = remainder_unqual [WIDTHR-1:0];
    end else begin
        reg [WIDTHN-1:0] quotient_pipe [LATENCY-1:0];
        reg [WIDTHD-1:0] remainder_pipe [LATENCY-1:0];

        for (genvar i = 0; i < LATENCY; i++) begin
            always @(posedge clk) begin                
                if (enable) begin
                    quotient_pipe[i]  <= (0 == i) ? quotient_unqual  : quotient_pipe[i-1];
                    remainder_pipe[i] <= (0 == i) ? remainder_unqual : remainder_pipe[i-1];
                end
            end
        end

        assign quotient  = quotient_pipe[LATENCY-1][WIDTHQ-1:0];
        assign remainder = remainder_pipe[LATENCY-1][WIDTHR-1:0];
    end    

`endif

endmodule
`TRACING_ON
