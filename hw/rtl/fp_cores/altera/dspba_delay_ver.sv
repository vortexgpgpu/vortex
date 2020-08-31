// Legal Notice: Copyright 2017 Intel Corporation.  All rights reserved.
// Your use of  Intel  Corporation's design tools,  logic functions and other
// software and tools,  and its AMPP  partner logic functions, and  any output
// files  any of the  foregoing  device programming or simulation files),  and
// any associated  documentation or information are expressly subject  to  the
// terms and conditions  of the Intel FPGA Software License Agreement,
// Intel  MegaCore  Function  License  Agreement, or other applicable license
// agreement,  including,  without limitation,  that your use  is for the sole
// purpose of  programming  logic  devices  manufactured by Intel and sold by
// Intel or its authorized  distributors.  Please  refer  to  the  applicable
// agreement for further details.

module dspba_delay_ver
#(
    parameter width = 8,
    parameter depth = 1,
    parameter reset_high = 1'b1,
    parameter reset_kind = "ASYNC" 
) (
    input clk,
    input aclr,
    input ena,
    input [width-1:0] xin,
    output [width-1:0] xout
);

    wire reset;
    reg [width-1:0] delays [depth-1:0];

    assign reset = aclr ^ reset_high;
    
    generate
        if (depth > 0)
        begin
            genvar i;
            for (i = 0; i < depth; ++i)
            begin : delay_block
                if (reset_kind == "ASYNC") 
                begin : sync_reset
                always @ (posedge clk or negedge reset)
                    begin: a
                        if (!reset) begin
                            delays[i] <= 0;
                        end else begin
                            if (ena) begin
                                if (i > 0) begin
                                    delays[i] <= delays[i - 1];
                                end else begin
                                    delays[i] <= xin;
                                end
                            end
                        end
                    end
                end

                if (reset_kind == "SYNC") 
                begin : async_reset
                always @ (posedge clk)
                    begin: a
                        if (!reset) begin
                            delays[i] <= 0;
                        end else begin
                            if (ena) begin
                                if (i > 0) begin
                                    delays[i] <= delays[i - 1];
                                end else begin
                                    delays[i] <= xin;
                                end
                            end
                        end
                    end
                end

                if (reset_kind == "NONE") 
                begin : no_reset
                always @ (posedge clk)
                    begin: a
                        if (ena) begin
                            if (i > 0) begin
                                delays[i] <= delays[i - 1];
                            end else begin
                                delays[i] <= xin;
                            end
                        end
                    end
                end
            end

            assign xout = delays[depth - 1];
        end else begin
            assign xout = xin;
        end
    endgenerate
    
endmodule