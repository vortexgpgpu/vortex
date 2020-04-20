module VX_divide #(
    parameter WIDTHN=1,
    parameter WIDTHD=1,
    parameter NREP="UNSIGNED",
    parameter DREP="UNSIGNED",
    parameter SPEED="MIXED", // "MIXED" or "HIGHEST"
    parameter PIPELINE=0
) (
    input clock, aclr, clken,

    input [WIDTHN-1:0] numer,
    input [WIDTHD-1:0] denom,

    output reg [WIDTHN-1:0] quotient,
    output reg [WIDTHD-1:0] remainder
);

// synthesis read_comments_as_HDL on
// localparam IMPL = "quartus";
// synthesis read_comments_as_HDL off

// altera translate_off
    localparam IMPL="fallback";
// altera translate_on

    generate
        if (NREP != DREP) begin
        `IGNORE_WARNINGS_BEGIN
            different_nrep_drep_not_yet_supported non_existing_module();
        `IGNORE_WARNINGS_END
        end

        if (IMPL == "quartus") begin

            localparam lpm_speed=SPEED == "HIGHEST" ? 9:5;

            lpm_divide #(
                .LPM_WIDTHN(WIDTHN),
                .LPM_WIDTHD(WIDTHD),
                .LPM_NREPRESENTATION(NREP),
                .LPM_DREPRESENTATION(DREP),
                .LPM_PIPELINE(PIPELINE),
                .LPM_REMAINDERPOSITIVE("FALSE"), // emulate verilog % operator
                .MAXIMIZE_SPEED(lpm_speed)
            ) quartus_divider (
                .clock(clock),
                .aclr(aclr),
                .clken(clken),
                .numer(numer),
                .denom(denom),
                .quotient(quotient),
                .remain(remainder)
            );
        end
        else begin

            wire [WIDTHN-1:0] numer_pipe_end;
            wire [WIDTHD-1:0] denom_pipe_end;
            if (PIPELINE == 0) begin
                assign numer_pipe_end = numer;
                assign denom_pipe_end = denom;
            end else begin
                reg [WIDTHN-1:0] numer_pipe [0:PIPELINE-1];
                reg [WIDTHD-1:0] denom_pipe [0:PIPELINE-1];

                genvar pipe_stage;
                for (pipe_stage = 0; pipe_stage < PIPELINE-1; pipe_stage = pipe_stage+1) begin : pipe_stages
                    always @(posedge clock or posedge aclr) begin
                        if (aclr) begin
                            numer_pipe[pipe_stage+1] <= 0;
                            denom_pipe[pipe_stage+1] <= 0;
                        end
                        else if (clken) begin
                            numer_pipe[pipe_stage+1] <= numer_pipe[pipe_stage];
                            denom_pipe[pipe_stage+1] <= denom_pipe[pipe_stage];
                        end
                    end
                end

                always @(posedge clock or posedge aclr) begin
                    if (aclr) begin
                        numer_pipe[0] <= 0;
                        denom_pipe[0] <= 0;
                    end
                    else if (clken) begin
                        numer_pipe[0] <= numer;
                        denom_pipe[0] <= denom;
                    end
                end

                assign numer_pipe_end = numer_pipe[PIPELINE-1];
                assign denom_pipe_end = denom_pipe[PIPELINE-1];
            end

            /* * * * * * * * * * * * * * * * * * * * * * */
            /*  Do the actual fallback computation here  */
            /* * * * * * * * * * * * * * * * * * * * * * */

            if (NREP == "SIGNED") begin

                /*VX_divide_ifnal_signed #(
                    .WIDTHN,
                    .WIDTHD
                )div(
                    .numer(numer_pipe_end),
                    .denom(denom_pipe_end),
                    .quotient,
                    .remainder
                );*/

                always @(*) begin
                    if (denom_pipe_end == 0) begin
                        quotient = 32'hffffffff;
                        remainder = numer_pipe_end;
                    end
                    else if (denom_pipe_end == 32'hffffffff && numer_pipe_end == 32'h80000000) begin
                        // this edge case kills verilator in some cases by causing a division
                        // overflow exception. INT_MIN / -1 (on x86)
                        quotient = 0;
                        remainder = 0;
                    end
                    else begin
                        quotient = $signed($signed(numer_pipe_end)/$signed(denom_pipe_end));
                        remainder = $signed($signed(numer_pipe_end)%$signed(denom_pipe_end));
                    end
                end

            end
            else begin
                assign quotient = (denom_pipe_end == 0) ? 32'hffffffff : numer_pipe_end/denom_pipe_end;
                assign remainder = (denom_pipe_end == 0) ? numer_pipe_end : numer_pipe_end%denom_pipe_end;
            end

        end
    endgenerate

endmodule : VX_divide
