`include "VX_platform.vh"

`TRACING_OFF
module VX_serial_div #(
    parameter WIDTHN = 1,
    parameter WIDTHD = 1,
    parameter WIDTHQ = 1,
    parameter WIDTHR = 1,
    parameter TAGW   = 1
) (
    input wire clk,
    input wire reset,

    input wire load,
    input wire [WIDTHN-1:0] numer,
    input wire [WIDTHD-1:0] denom,
    input wire signed_mode,

    output wire [WIDTHQ-1:0] quotient,
    output wire [WIDTHR-1:0] remainder,    
    output wire done
);
    localparam MIN_ND = (WIDTHN < WIDTHD) ? WIDTHN : WIDTHD;
    localparam CNTRW = $clog2(WIDTHN+1);

    reg [WIDTHN + MIN_ND:0] working;
    reg [WIDTHD-1:0] denom_r;

    wire neg_numer = signed_mode && numer[WIDTHN-1];
    wire neg_denom = signed_mode && denom[WIDTHD-1];

    wire [WIDTHN-1:0] numer_qual = neg_numer ? -$signed(numer) : numer;
    wire [WIDTHN-1:0] denom_qual = neg_denom ? -$signed(denom) : denom;
    wire [WIDTHN:0]   sub_result = working[WIDTHN + MIN_ND : WIDTHN] - denom_r;

    reg inv_quot, inv_rem;
    reg [CNTRW-1:0] cntr;
    
    always @(posedge clk) begin
        if (reset) begin
            cntr <= WIDTHN;
        end else begin
            if (load) begin                
                cntr <= WIDTHN;  
            end else if (!done) begin
                cntr <= cntr - CNTRW'(1);
            end
        end

        if (load) begin
            working  <= {{WIDTHD{1'b0}}, numer_qual, 1'b0};
            denom_r  <= denom_qual;
            inv_quot <= (denom != 0) && signed_mode && (numer[31] ^ denom[31]);
            inv_rem  <= signed_mode && numer[31];
        end else if (!done) begin                    
            working  <= sub_result[WIDTHD] ? {working[WIDTHN+MIN_ND-1:0], 1'b0} :
                                             {sub_result[WIDTHD-1:0], working[WIDTHN-1:0], 1'b1};
        end
    end

    wire [WIDTHQ-1:0] q = working[WIDTHQ-1:0];
    wire [WIDTHR-1:0] r = working[WIDTHN+WIDTHR:WIDTHN+1];

    assign quotient  = inv_quot ? -$signed(q) : q;
    assign remainder = inv_rem ? -$signed(r) : r;
    assign done      = ~(| cntr);

endmodule
`TRACING_ON
