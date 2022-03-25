`include "VX_platform.vh"

// Fast Paralllel scan using Kogge-Stone style prefix tree with configurable operator
// Adapted from BaseJump STL: http://bjump.org/index.html

`TRACING_OFF
module VX_scan #(
    parameter N       = 1,
    parameter OP      = 0,  // 0: XOR, 1: AND, 2: OR
    parameter REVERSE = 0   // 0: LO->HI, 1: HI->LO
) (
    input  wire [N-1:0] data_in,
    output wire [N-1:0] data_out
);
`IGNORE_WARNINGS_BEGIN

    localparam LOGN = $clog2(N);

    wire [LOGN:0][N-1:0] t;   

    // reverses bits
    if (REVERSE) begin
        assign t[0] = data_in;
    end else begin
        assign t[0] = {<<{data_in}};
    end

    // optimize for the common case of small and-scans
    if ((N == 2) && (OP == 1)) begin
	    assign t[LOGN] = {t[0][1], &t[0][1:0]};
    end else if ((N == 3) && (OP == 1)) begin
	    assign t[LOGN] = {t[0][2], &t[0][2:1], &t[0][2:0]};
    end else if ((N == 4) && (OP == 1)) begin
	    assign t[LOGN] = {t[0][3], &t[0][3:2], &t[0][3:1], &t[0][3:0]};
    end else begin
        // general case
        wire [N-1:0] fill;
	    for (genvar i = 0; i < LOGN; i++) begin
            wire [N-1:0] shifted = N'({fill, t[i]} >> (1<<i));
            if (OP == 0) begin
		        assign fill = {N{1'b0}};
		        assign t[i+1] = t[i] ^ shifted;
            end else if (OP == 1) begin
		        assign fill = {N{1'b1}};
		        assign t[i+1] = t[i] & shifted;
            end else if (OP == 2) begin
		        assign fill = {N{1'b0}};
		        assign t[i+1] = t[i] | shifted;
            end
	    end
    end   

    // reverse bits
    if (REVERSE) begin
        assign data_out = t[LOGN];
    end else begin
        for (genvar i = 0; i < N; i++) begin
            assign data_out[i] = t[LOGN][N-1-i];
        end        
    end

`IGNORE_WARNINGS_END
endmodule
`TRACING_ON
