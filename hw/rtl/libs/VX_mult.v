`include "VX_define.vh"

module VX_mult #(
    parameter WIDTHA = 1,
    parameter WIDTHB = 1,
    parameter WIDTHP = 1,
    parameter REP = "UNSIGNED",
    parameter PIPELINE = 0
) (
    input               clk,
    input               reset,
    input               clken,

    input [WIDTHA-1:0]  dataa,
    input [WIDTHB-1:0]  datab,

    output reg [WIDTHP-1:0] result
);

`ifdef QUARTUS

    lpm_mult #(
        .LPM_WIDTHA(WIDTHA),
        .LPM_WIDTHB(WIDTHB),
        .LPM_WIDTHP(WIDTHP),
        .LPM_REPRESENTATION(REP),
        .LPM_PIPELINE(PIPELINE),
        .DSP_BLOCK_BALANCING("LOGIC ELEMENTS"),
        .MAXIMIZE_SPEED(9)
    ) quartus_mult (
        .clock(clk),
        .aclr(reset),
        .clken(clken),
        .dataa(dataa),
        .datab(datab),
        .result(result)
    );

`else
    
    wire [WIDTHA-1:0] dataa_pipe_end;
    wire [WIDTHB-1:0] datab_pipe_end;

    if (PIPELINE == 0) begin
        assign dataa_pipe_end = dataa;
        assign datab_pipe_end = datab;
    end else begin
        reg [WIDTHA-1:0] dataa_pipe [0:PIPELINE-1];
        reg [WIDTHB-1:0] datab_pipe [0:PIPELINE-1];

        genvar i;
        for (i = 0; i < PIPELINE; i++) begin
            always @(posedge clk) begin
                if (reset) begin
                    dataa_pipe[i] <= 0;
                    datab_pipe[i] <= 0;
                end
                else if (clken) begin
                    if (i == 0) begin
                        dataa_pipe[0] <= dataa;
                        datab_pipe[0] <= datab;
                    end else begin
                        dataa_pipe[i] <= dataa_pipe[i-1];
                        datab_pipe[i] <= datab_pipe[i-1];
                    end
                end
            end
        end

        assign dataa_pipe_end = dataa_pipe[PIPELINE-1];
        assign datab_pipe_end = datab_pipe[PIPELINE-1];
    end

    if (REP == "SIGNED") begin
        assign result = $signed(dataa_pipe_end) * $signed(datab_pipe_end);
    end
    else begin
        assign result = dataa_pipe_end * datab_pipe_end;
    end

`endif

endmodule: VX_mult
