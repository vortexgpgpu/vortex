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

//------------------------------------------------------------------------------

module dspba_sync_reg_ver
#(
    parameter width1 = 8,
    parameter width2 = 8,
    parameter depth = 2,
    parameter pulse_multiplier = 1,
    parameter counter_width = 8,
    parameter init_value = 0,
    parameter reset1_high = 1'b1,
    parameter reset2_high = 1'b1,
    parameter reset_kind = "ASYNC" 
) (
    input clk1,
    input aclr1,
    input [0 : 0] ena,
    input [width1-1 : 0] xin,
    output [width1-1 : 0] xout,
    input clk2,
    input aclr2,
    output [width2-1 : 0] sxout
);
wire [width1-1 : 0] init_value_internal;

wire reset1;
wire reset2;

reg iclk_enable;
reg [width1-1 : 0] iclk_data;
reg [width2-1 : 0] oclk_data;

// For Synthesis this means: preserve this registers and do not merge any other flip-flops with synchronizer flip-flops 
// For TimeQuest this means: identify these flip-flops as synchronizer to enable automatic MTBF analysis
(* altera_attribute = {"-name ADV_NETLIST_OPT_ALLOWED NEVER_ALLOW; -name SYNCHRONIZER_IDENTIFICATION FORCED; -name DONT_MERGE_REGISTER ON; -name PRESERVE_REGISTER ON"} *) reg [depth-1 : 0] sync_regs;

wire oclk_enable;

wire ena_internal;
reg [counter_width-1 : 0] counter;

assign init_value_internal = init_value;

assign reset1 = aclr1 ^ reset1_high;
assign reset2 = aclr2 ^ reset2_high;

generate
    if (pulse_multiplier == 1)
    begin: no_multiplication
        assign ena_internal = ena[0];
    end
endgenerate

generate 
    if (pulse_multiplier > 1)
    begin: multiplu_ena_pulse
        if (reset_kind == "ASYNC")
        begin: async_reset
            always @ (posedge clk1 or negedge reset1)
            begin
                if (reset1 == 1'b0) begin
                    counter <= 0;
                end else begin
                    if (counter > 0) begin
                        if (counter == pulse_multiplier - 1) begin
                            counter <= 0;
                        end else begin
                            counter <= counter + 2'd1;
                        end
                    end else begin
                        if (ena[0] == 1'b1) begin
                            counter <= 1;
                        end
                    end
                end
            end
        end
        if (reset_kind == "SYNC")
        begin: sync_reset
            always @ (posedge clk1)
            begin
                if (reset1 == 1'b0) begin
                    counter <= 0;
                end else begin
                    if (counter > 0) begin
                        if (counter == pulse_multiplier - 1) begin
                            counter <= 0;
                        end else begin
                            counter <= counter + 2'd1;
                        end
                    end else begin
                        if (ena[0] == 1'b1) begin
                            counter <= 1;
                        end
                    end
                end
            end
        end
        if (reset_kind == "NONE")
        begin: no_reset
            always @ (posedge clk1)
            begin
                if (counter > 0) begin
                    if (counter == pulse_multiplier - 1) begin
                        counter <= 0;
                    end else begin
                        counter <= counter + 2'd1;
                    end
                end else begin
                    if (ena[0] == 1'b1) begin
                        counter <= 1;
                    end
                end
            end
        end
        
        assign ena_internal = counter > 0 ? 1'b1 : ena[0];
    end
endgenerate

assign oclk_enable = sync_regs[depth - 1];

generate
    if (reset_kind == "ASYNC")
    begin: iclk_async_reset 
        always @ (posedge clk1 or negedge reset1) 
        begin
           if (reset1 == 1'b0) begin
               iclk_data <= init_value_internal;
               iclk_enable <= 1'b0;
           end else begin
               iclk_enable <= ena_internal;
               if (ena[0] == 1'b1) begin
                   iclk_data <= xin;
               end
           end
        end
    end
    if (reset_kind == "SYNC")
    begin: iclk_sync_reset 
        always @ (posedge clk1) 
        begin
           if (reset1 == 1'b0) begin
               iclk_data <= init_value_internal;
               iclk_enable <= 1'b0;
           end else begin
               iclk_enable <= ena_internal;
               if (ena[0] == 1'b1) begin
                   iclk_data <= xin;
               end
           end
        end
    end
    if (reset_kind == "NONE")
    begin: iclk_no_reset 
        always @ (posedge clk1) 
        begin
           iclk_enable <= ena_internal;
           if (ena[0] == 1'b1) begin
               iclk_data <= xin;
           end
        end
    end
endgenerate

generate
    genvar i;
    for (i = 0; i < depth; ++i)
    begin: sync_regs_block
        if (reset_kind == "ASYNC") 
        begin: sync_reg_async_reset
            always @ (posedge clk2 or negedge reset2) begin
                if (reset2 == 1'b0) begin
                    sync_regs[i] <= 1'b0;
                end else begin
                    if (i > 0) begin
                        sync_regs[i] <= sync_regs[i - 1];
                    end else begin
                        sync_regs[i] <= iclk_enable;
                    end
                end
            end
        end
        if (reset_kind == "SYNC") 
        begin: sync_reg_sync_reset
            always @ (posedge clk2) begin
                if (reset2 == 1'b0) begin
                    sync_regs[i] <= 1'b0;
                end else begin
                    if (i > 0) begin
                        sync_regs[i] <= sync_regs[i - 1];
                    end else begin
                        sync_regs[i] <= iclk_enable;
                    end
                end
            end
        end
        if (reset_kind == "NONE") 
        begin: sync_reg_no_reset
            always @ (posedge clk2) begin
                if (i > 0) begin
                    sync_regs[i] <= sync_regs[i - 1];
                end else begin
                    sync_regs[i] <= iclk_enable;
                end
            end
        end
    end
endgenerate

generate
    if (reset_kind == "ASYNC")
    begin: oclk_async_reset
        always @ (posedge clk2 or negedge reset2)
        begin
            if (reset2 == 1'b0) begin
                oclk_data <= init_value_internal[width2-1 : 0];
            end else begin
                if (oclk_enable == 1'b1) begin
                    oclk_data <= iclk_data[width2-1 : 0];
                end
            end
        end
    end
    if (reset_kind == "SYNC")
    begin: oclk_sync_reset
        always @ (posedge clk2)
        begin
            if (reset2 == 1'b0) begin
                oclk_data <= init_value_internal[width2-1 : 0];
            end else begin
                if (oclk_enable == 1'b1) begin
                    oclk_data <= iclk_data[width2-1 : 0];
                end
            end
        end
    end
    if (reset_kind == "NONE")
    begin: oclk_no_reset
        always @ (posedge clk2)
        begin
            if (oclk_enable == 1'b1) begin
                oclk_data <= iclk_data[width2-1 : 0];
            end
        end
    end
endgenerate

assign xout = iclk_data;
assign sxout = oclk_data;

endmodule

//------------------------------------------------------------------------------

module dspba_pipe
#(
    parameter num_bits   = 8,
    parameter num_stages = 0,
    parameter init_value = 1'bx
) (
    input  clk,
    input  [num_bits-1:0] d,
    output [num_bits-1:0] q
);
    logic [num_bits-1:0] init_stage = { num_bits { init_value } };

    generate
        if (num_stages > 0)
        begin
            reg [num_bits-1:0] stage_array[num_stages-1:0];

            genvar i;
            for (i = 0; i < num_stages; ++i)
            begin : g_pipe
                always @ (posedge clk) begin
                    if (i>0) begin
                        stage_array[i] <= stage_array[i-1];
                    end else begin
                        stage_array[i] <= d;
                    end
                end
            end
            initial begin
                stage_array = '{ num_stages { init_stage } };
            end

            assign q = stage_array[num_stages-1];

        end else begin
            assign q = d;
        end
    endgenerate

endmodule
