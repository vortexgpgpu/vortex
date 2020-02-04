module VX_divide
    #(
        parameter WIDTHN=1,
        parameter WIDTHD=1,
        parameter NREP="UNSIGNED",
        parameter DREP="UNSIGNED",
        parameter SPEED="MIXED", // "MIXED" or "HIGHEST"
        parameter PIPELINE=0
    )
    (
        input clock, aclr, clken,

        input [WIDTHN-1:0] numer,
        input [WIDTHD-1:0] denom,

        output [WIDTHN-1:0] quotient,
        output [WIDTHD-1:0] remainder
    );

// synthesis read_comments_as_HDL on
// localparam IMPL = "quartus";
// synthesis read_comments_as_HDL off

// altera translate_off
    localparam IMPL="fallback";
// altera translate_on

    generate
        if (NREP != DREP) begin
            different_nrep_drep_not_yet_supported non_existing_module();
        end

        if (IMPL == "quartus") begin

            localparam lpm_speed=SPEED == "HIGHEST" ? 9:5;

            lpm_divide#(
                .LPM_WIDTHN(WIDTHN),
                .LPM_WIDTHD(WIDTHD),
                .LPM_NREPRESENTATION(NREP),
                .LPM_DREPRESENTATION(DREP),
                .LPM_PIPELINE(PIPELINE),
                .LPM_REMAINDERPOSITIVE("FALSE"), // emulate verilog % operator
                .MAXIMIZE_SPEED(lpm_speed)
            ) quartus_divider(
                .clock(clock),
                .aclr(aclr),
                .clken(clken),
                .numer(numer),
                .denom(denom),
                .quotient(quotient),
                .remainder(remainder)
            );

        end
        else if (PIPELINE == 0) begin
            if (NREP == "SIGNED") begin
                assign quotient = $signed($signed(numer)/$signed(denom));
                assign remainder = $signed($signed(numer)%$signed(denom));
            end
            else begin
                assign quotient = numer/denom;
                assign remainder = numer%denom;
            end
        end
        else begin

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

            wire [WIDTHN-1:0] numer_pipe_end;
            assign numer_pipe_end = numer_pipe[PIPELINE-1];
            wire [WIDTHD-1:0] denom_pipe_end;
            assign denom_pipe_end = denom_pipe[PIPELINE-1];

            if (NREP == "SIGNED") begin
                assign quotient = $signed($signed(numer_pipe_end)/$signed(denom_pipe_end));
                assign remainder = $signed($signed(numer_pipe_end)%$signed(denom_pipe_end));
            end
            else begin
                assign quotient = numer_pipe_end/denom_pipe_end;
                assign remainder = numer_pipe_end%denom_pipe_end;
            end

        end
    endgenerate

endmodule: VX_divide