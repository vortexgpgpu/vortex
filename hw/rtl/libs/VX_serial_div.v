`include "VX_platform.vh"

module VX_serial_div #(
    parameter WIDTHN = 1,
    parameter WIDTHD = 1,
    parameter WIDTHQ = 1,
    parameter WIDTHR = 1,
    parameter LANES  = 1,
    parameter TAGW   = 1
) (
    input wire clk,
    input wire reset,

    input wire valid_in,
    output wire ready_in,
    input wire [LANES-1:0][WIDTHN-1:0] numer,
    input wire [LANES-1:0][WIDTHD-1:0] denom,
    input wire signed_mode,
    input wire [TAGW-1:0] tag_in,

    output wire [LANES-1:0][WIDTHQ-1:0] quotient,
    output wire [LANES-1:0][WIDTHR-1:0] remainder,    
    input wire ready_out,
    output wire valid_out,
    output wire [TAGW-1:0] tag_out
);
    localparam MIN_ND = (WIDTHN < WIDTHD) ? WIDTHN : WIDTHD;
    localparam CNTRW = $clog2(WIDTHN+1);

    reg [LANES-1:0][WIDTHN + MIN_ND:0] working;
    reg [LANES-1:0][WIDTHD-1:0] denom_r;

    wire [LANES-1:0][WIDTHN-1:0] numer_qual;
    wire [LANES-1:0][WIDTHD-1:0] denom_qual;
    wire [LANES-1:0][WIDTHD:0] sub_result;

    reg [LANES-1:0] inv_quot, inv_rem;

    reg [CNTRW-1:0] cntr;
    reg is_busy;

    reg [TAGW-1:0] tag_r;
    
    wire done = ~(| cntr);

    wire push = valid_in && ready_in;
    wire pop = valid_out && ready_out;

    for (genvar i = 0; i < LANES; ++i) begin
        wire negate_numer = signed_mode && numer[i][WIDTHN-1];
        wire negate_denom = signed_mode && denom[i][WIDTHD-1];
        assign numer_qual[i] = (numer[i] ^ {WIDTHN{negate_numer}}) + WIDTHN'(negate_numer);
        assign denom_qual[i] = (denom[i] ^ {WIDTHD{negate_denom}}) + WIDTHD'(negate_denom);
        assign sub_result[i] = working[i][WIDTHN + MIN_ND : WIDTHN] - denom_r[i];
    end
    
    always @(posedge clk) begin
        if (reset) begin
            cntr    <= 0;
            is_busy <= 0;
        end
        else begin
            if (push) begin                
                for (integer i = 0; i < LANES; ++i) begin
                    working[i]  <= {{WIDTHD{1'b0}}, numer_qual[i], 1'b0};                                        
                    denom_r[i]  <= denom_qual[i];
                    inv_quot[i] <= (denom[i] != 0) && signed_mode && (numer[i][31] ^ denom[i][31]);
                    inv_rem[i]  <= signed_mode && numer[i][31];
                end                              
                tag_r   <= tag_in;
                cntr    <= WIDTHN;
                is_busy <= 1;                
            end
            else begin
                if (!done) begin                    
                    for (integer i = 0; i < LANES; ++i) begin                        
                        working[i] <= sub_result[i][WIDTHD] ? {working[i][WIDTHN+MIN_ND-1:0], 1'b0} :
                                        {sub_result[i][WIDTHD-1:0], working[i][WIDTHN-1:0], 1'b1};
                    end         
                    cntr <= cntr - CNTRW'(1);       
                end
            end
            if (pop) begin
                is_busy <= 0;
            end
        end
    end

    for (genvar i = 0; i < LANES; ++i) begin
        assign quotient[i]  = (working[i][WIDTHQ-1:0] ^ {WIDTHQ{inv_quot[i]}}) + WIDTHQ'(inv_quot[i]);
        assign remainder[i] = (working[i][WIDTHN+WIDTHR:WIDTHN+1] ^ {WIDTHR{inv_rem[i]}}) + WIDTHR'(inv_rem[i]);
    end
    assign ready_in  = !is_busy;    
    assign tag_out   = tag_r;
    assign valid_out = is_busy && done; 

    /*reg [LANES-1:0][WIDTHQ-1:0] quotient_r;
    reg [LANES-1:0][WIDTHR-1:0] remainder_r; 
    reg [TAGW-1:0] tag_out_r;
    reg valid_out_r;

    wire stall_out = !ready_out && valid_out_r;
    assign pop = is_busy && done && !stall_out;

    always @(posedge clk) begin
        if (reset) begin
            valid_out_r <= 0;
        end else if (~stall_out) begin
            for (integer i = 0; i < LANES; ++i) begin
                quotient_r[i]  <= (working[i][WIDTHQ-1:0] ^ {WIDTHQ{inv_quot[i]}}) + WIDTHQ'(inv_quot[i]);
                remainder_r[i] <= ((working[i][WIDTHN+WIDTHR-1:WIDTHN] >> 1) ^ {WIDTHR{inv_rem[i]}}) + WIDTHR'(inv_rem[i]);
            end
            tag_out_r   <= tag_r;
            valid_out_r <= is_busy && done; 
        end 
    end   

    assign ready_in  = !is_busy;    
    assign quotient  = quotient_r;
    assign remainder = remainder_r;
    assign tag_out   = tag_out_r;
    assign valid_out = valid_out_r;*/

endmodule