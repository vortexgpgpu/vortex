`include "VX_define.vh"

module VX_alu_unit (
    input wire         clk,
    input wire         reset,
    input wire [31:0]  src_a,
    input wire [31:0]  src_b,
    input wire         src_rs2,
    input wire [31:0]  itype_immed,
    input wire [19:0]  upper_immed,
    input wire [4:0]   alu_op,
    input wire [31:0]  curr_PC,
    output reg [31:0]  alu_result,
    output reg         alu_stall
);
    wire[31:0] div_result_unsigned;
    wire[31:0] div_result_signed;

    wire[31:0] rem_result_unsigned;    
    wire[31:0] rem_result_signed;
  
    wire[63:0] mul_result;    

    wire[31:0] alu_in1 = src_a;
    wire[31:0] alu_in2 = (src_rs2 == `RS2_IMMED) ? itype_immed : src_b;

    wire[31:0] upper_immed_s = {upper_immed, {12{1'b0}}};

    reg [7:0] inst_delay;
    reg [7:0] curr_inst_delay;
        
    always @(*) begin
        case (alu_op)
            `ALU_DIV,
            `ALU_DIVU,
            `ALU_REM,
            `ALU_REMU:  inst_delay = `DIV_LATENCY;
            `ALU_MUL,
            `ALU_MULH,
            `ALU_MULHSU,
            `ALU_MULHU: inst_delay = `MUL_LATENCY;
            default:    inst_delay = 0;
        endcase
    end

    wire inst_stalled = (curr_inst_delay != inst_delay);

    always @(posedge clk) begin
        if (reset) begin            
            curr_inst_delay <= 0;
        end else begin
            curr_inst_delay <= inst_stalled ? (curr_inst_delay + 1) : 0; 
        end        
    end

    assign alu_stall = inst_stalled;

    always @(*) begin
        case (alu_op)
            `ALU_ADD:       alu_result = $signed(alu_in1) + $signed(alu_in2);
            `ALU_SUB:       alu_result = $signed(alu_in1) - $signed(alu_in2);
            `ALU_SLLA:      alu_result = alu_in1 << alu_in2[4:0];
            `ALU_SLT:       alu_result = ($signed(alu_in1) < $signed(alu_in2)) ? 32'h1 : 32'h0;
            `ALU_SLTU:      alu_result = alu_in1 < alu_in2 ? 32'h1 : 32'h0;
            `ALU_XOR:       alu_result = alu_in1 ^ alu_in2;
            `ALU_SRL:       alu_result = alu_in1 >> alu_in2[4:0];
            `ALU_SRA:       alu_result = $signed(alu_in1)  >>> alu_in2[4:0];
            `ALU_OR:        alu_result = alu_in1 | alu_in2;
            `ALU_AND:       alu_result = alu_in2 & alu_in1;
            `ALU_SUBU:      alu_result = (alu_in1 >= alu_in2) ? 32'h0 : 32'hffffffff;
            `ALU_LUI:       alu_result = upper_immed_s;
            `ALU_AUIPC:     alu_result = $signed(curr_PC) + $signed(upper_immed_s);
            // TODO: profitable to roll these exceptional cases into inst_delay_tmp to avoid pipeline when possible?
            `ALU_MUL:       alu_result = mul_result[31:0];
            `ALU_MULH:      alu_result = mul_result[63:32];
            `ALU_MULHSU:    alu_result = mul_result[63:32];
            `ALU_MULHU:     alu_result = mul_result[63:32];
            `ALU_DIV:       alu_result = (alu_in2 == 0) ? 32'hffffffff : div_result_signed;
            `ALU_DIVU:      alu_result = (alu_in2 == 0) ? 32'hffffffff : div_result_unsigned;
            `ALU_REM:       alu_result = (alu_in2 == 0) ? alu_in1 : rem_result_signed;
            `ALU_REMU:      alu_result = (alu_in2 == 0) ? alu_in1 : rem_result_unsigned;
            default:        alu_result = 32'h0;
        endcase // alu_op
    end

    VX_divide #(
        .WIDTHN(32),
        .WIDTHD(32),
        .NSIGNED(0),
        .DSIGNED(0),
        .PIPELINE(`DIV_LATENCY)
    ) udiv (
        .clk(clk),
        .reset(reset),
        .numer(alu_in1),
        .denom(alu_in2),
        .quotient(div_result_unsigned),
        .remainder(rem_result_unsigned)
    );

    VX_divide #(
        .WIDTHN(32),
        .WIDTHD(32),
        .NSIGNED(1),
        .DSIGNED(1),
        .PIPELINE(`DIV_LATENCY)
    ) sdiv (
        .clk(clk),
        .reset(reset),
        .numer(alu_in1),
        .denom(alu_in2),
        .quotient(div_result_signed),
        .remainder(rem_result_signed)
    );

    wire [32:0] mul_dataa = {(alu_op == `ALU_MULHU)                          ? 1'b0 : alu_in1[31], alu_in1};
    wire [32:0] mul_datab = {(alu_op == `ALU_MULHU || alu_op == `ALU_MULHSU) ? 1'b0 : alu_in2[31], alu_in2};

    VX_mult #(
        .WIDTHA(33),
        .WIDTHB(33),
        .WIDTHP(66),
        .SIGNED(1),
        .PIPELINE(`MUL_LATENCY)
    ) multiplier (
        .clk(clk),
        .reset(reset),
        .dataa(mul_dataa),
        .datab(mul_datab),
        .result(mul_result)
    );

endmodule