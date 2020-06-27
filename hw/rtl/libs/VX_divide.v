`include "VX_define.vh"

module VX_divide #(
    parameter WIDTHN = 1,
    parameter WIDTHD = 1,
    parameter REP = "UNSIGNED",
    parameter PIPELINE = 0
) (
    input wire clk,
    input wire reset,
    input wire clken,

    input [WIDTHN-1:0] numer,
    input [WIDTHD-1:0] denom,

    output reg [WIDTHN-1:0] quotient,
    output reg [WIDTHD-1:0] remainder
);

`ifdef QUARTUS

    lpm_divide #(
        .LPM_WIDTHN(WIDTHN),
        .LPM_WIDTHD(WIDTHD),
        .LPM_NREPRESENTATION(REP),
        .LPM_DREPRESENTATION(REP),
        .LPM_PIPELINE(PIPELINE),        
        .DSP_BLOCK_BALANCING("LOGIC ELEMENTS"),
        .MAXIMIZE_SPEED(9)
    ) quartus_divider (
        .clock(clk),
        .aclr(reset),
        .clken(clken),
        .numer(numer),
        .denom(denom),
        .quotient(quotient),
        .remain(remainder)
    );

`else

    wire [WIDTHN-1:0] numer_pipe_end;
    wire [WIDTHD-1:0] denom_pipe_end;

    if (PIPELINE == 0) begin
        assign numer_pipe_end = numer;
        assign denom_pipe_end = denom;
    end else begin
        reg [WIDTHN-1:0] numer_pipe [0:PIPELINE-1];
        reg [WIDTHD-1:0] denom_pipe [0:PIPELINE-1];

        genvar i;
        for (i = 0; i < PIPELINE; i++) begin
            always @(posedge clk) begin
                if (reset) begin
                    numer_pipe[i] <= 0;
                    denom_pipe[i] <= 0;
                end
                else if (clken) begin
                    if (i == 0) begin
                        numer_pipe[0] <= 0;
                        denom_pipe[0] <= 0;
                    end else begin
                        numer_pipe[i] <= numer_pipe[i-1];
                        denom_pipe[i] <= denom_pipe[i-1];    
                    end                     
                end
            end
        end

        assign numer_pipe_end = numer_pipe[PIPELINE-1];
        assign denom_pipe_end = denom_pipe[PIPELINE-1];
    end

    always @(*) begin    
        if (denom_pipe_end == 0) begin
            quotient  = {WIDTHN{1'b1}};
            remainder = numer_pipe_end;
        end
    `ifndef SYNTHESIS    
        // this edge case kills verilator in some cases by causing a division
        // overflow exception. INT_MIN / -1 (on x86)
        else if (numer_pipe_end == {1'b1, (WIDTHN-1)'(0)}
              && denom_pipe_end == {WIDTHD{1'b1}}) begin
            quotient  = 0;
            remainder = 0;
        end
    `endif
        else begin
            if (REP == "SIGNED") begin
                quotient  = $signed(numer_pipe_end) / $signed(denom_pipe_end);
                remainder = $signed(numer_pipe_end) % $signed(denom_pipe_end);
            end else begin
                quotient  = numer_pipe_end / denom_pipe_end;
                remainder = numer_pipe_end % denom_pipe_end;        
            end
        end
    end

`endif

endmodule : VX_divide
