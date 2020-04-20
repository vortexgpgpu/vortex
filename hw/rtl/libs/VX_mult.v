`include "VX_define.vh"

module VX_mult #(
    parameter WIDTHA=1,
    parameter WIDTHB=1,
    parameter WIDTHP=1,
    parameter REP="UNSIGNED",
    parameter SPEED="MIXED", // "MIXED" or "HIGHEST"
    parameter PIPELINE=0,
    parameter FORCE_LE="NO"
) (
    input clock, aclr, clken,

    input [WIDTHA-1:0] dataa,
    input [WIDTHB-1:0] datab,

    output reg [WIDTHP-1:0] result
);

// synthesis read_comments_as_HDL on
// localparam IMPL = "quartus";
// synthesis read_comments_as_HDL off

// altera translate_off
    localparam IMPL="fallback";
// altera translate_on

    generate

        if (IMPL == "quartus") begin

            localparam lpm_speed = (SPEED == "HIGHEST") ? 10 : 5;

            if (FORCE_LE == "YES") begin
            `IGNORE_WARNINGS_BEGIN    
                lpm_mult #(            
                    .LPM_WIDTHA(WIDTHA),
                    .LPM_WIDTHB(WIDTHB),
                    .LPM_WIDTHP(WIDTHP),
                    .LPM_REPRESENTATION(REP),
                    .LPM_PIPELINE(PIPELINE),
                    .DSP_BLOCK_BALANCING("LOGIC ELEMENTS"),
                    .MAXIMIZE_SPEED(lpm_speed)
                ) quartus_mult (
                    .clock(clock),
                    .aclr(aclr),
                    .clken(clken),
                    .dataa(dataa),
                    .datab(datab),
                    .result(result)
                );
            `IGNORE_WARNINGS_END
            end
            else begin
                lpm_mult#(
                    .LPM_WIDTHA(WIDTHA),
                    .LPM_WIDTHB(WIDTHB),
                    .LPM_WIDTHP(WIDTHP),
                    .LPM_REPRESENTATION(REP),
                    .LPM_PIPELINE(PIPELINE),
                    .MAXIMIZE_SPEED(lpm_speed)
                ) quartus_mult(
                    .clock(clock),
                    .aclr(aclr),
                    .clken(clken),
                    .dataa(dataa),
                    .datab(datab),
                    .result(result)
                );
            end

        end
        else begin

            wire [WIDTHA-1:0] dataa_pipe_end;
            wire [WIDTHB-1:0] datab_pipe_end;
            if (PIPELINE == 0) begin
                assign dataa_pipe_end = dataa;
                assign datab_pipe_end = datab;
            end else begin
                reg [WIDTHA-1:0] dataa_pipe [0:PIPELINE-1];
                reg [WIDTHB-1:0] datab_pipe [0:PIPELINE-1];

                genvar pipe_stage;
                for (pipe_stage = 0; pipe_stage < PIPELINE-1; pipe_stage = pipe_stage+1) begin : pipe_stages
                    always @(posedge clock or posedge aclr) begin
                        if (aclr) begin
                            dataa_pipe[pipe_stage+1] <= 0;
                            datab_pipe[pipe_stage+1] <= 0;
                        end
                        else if (clken) begin
                            dataa_pipe[pipe_stage+1] <= dataa_pipe[pipe_stage];
                            datab_pipe[pipe_stage+1] <= datab_pipe[pipe_stage];
                        end
                    end
                end

                always @(posedge clock or posedge aclr) begin
                    if (aclr) begin
                        dataa_pipe[0] <= 0;
                        datab_pipe[0] <= 0;
                    end
                    else if (clken) begin
                        dataa_pipe[0] <= dataa;
                        datab_pipe[0] <= datab;
                    end
                end

                assign dataa_pipe_end = dataa_pipe[PIPELINE-1];
                assign datab_pipe_end = datab_pipe[PIPELINE-1];
            end

            /* * * * * * * * * * * * * * * * * * * * * * */
            /*  Do the actual fallback computation here  */
            /* * * * * * * * * * * * * * * * * * * * * * */

            if (REP == "SIGNED") begin
                assign result = $signed($signed(dataa_pipe_end)*$signed(datab_pipe_end));
            end
            else begin
                assign result = dataa_pipe_end*datab_pipe_end;
            end

        end
    endgenerate

endmodule: VX_mult
