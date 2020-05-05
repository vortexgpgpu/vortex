`include "VX_define.vh"

module VX_alu_unit (
    input wire        clk,
    input wire        reset,
    input wire[31:0]  src_a,
    input wire[31:0]  src_b,
    input wire        src_rs2,
    input wire[31:0]  itype_immed,
    input wire[19:0]  upper_immed,
    input wire[4:0]   alu_op,
    input wire[31:0]  curr_PC,
    output reg[31:0]  alu_result,
    output reg        alu_stall
);

    localparam div_pipeline_len = 20;
    localparam mul_pipeline_len = 8;

    wire[31:0] unsigned_div_result;
    wire[31:0] unsigned_rem_result;
    wire[31:0] signed_div_result;
    wire[31:0] signed_rem_result;

    wire[63:0] mul_data_a, mul_data_b;
    wire[63:0] mul_result;

    wire[31:0] ALU_in1;
    wire[31:0] ALU_in2;

    VX_divide #(
        .WIDTHN(32),
        .WIDTHD(32),
        .SPEED("HIGHEST"),
        .PIPELINE(div_pipeline_len)
    ) unsigned_div (
        .clock(clk),
        .aclr(1'b0),
        .clken(1'b1), // TODO this could be disabled on inactive instructions
        .numer(ALU_in1),
        .denom(ALU_in2),
        .quotient(unsigned_div_result),
        .remainder(unsigned_rem_result)
    );

    VX_divide #(
        .WIDTHN(32),
        .WIDTHD(32),
        .NREP("SIGNED"),
        .DREP("SIGNED"),
        .SPEED("HIGHEST"),
        .PIPELINE(div_pipeline_len)
    ) signed_div (
        .clock(clk),
        .aclr(1'b0),
        .clken(1'b1), // TODO this could be disabled on inactive instructions
        .numer(ALU_in1),
        .denom(ALU_in2),
        .quotient(signed_div_result),
        .remainder(signed_rem_result)
    );

    VX_mult #(
        .WIDTHA(64),
        .WIDTHB(64),
        .WIDTHP(64),
        .SPEED("HIGHEST"),
        .FORCE_LE("YES"),
        .PIPELINE(mul_pipeline_len)
    ) multiplier (
        .clock(clk),
        .aclr(1'b0),
        .clken(1'b1), // TODO this could be disabled on inactive instructions
        .dataa(mul_data_a),
        .datab(mul_data_b),
        .result(mul_result)
    );

    // ALU_MUL, ALU_MULH (signed*signed), ALU_MULHSU (signed*unsigned), ALU_MULHU (unsigned*unsigned)
    wire[63:0] alu_in1_signed = {{32{ALU_in1[31]}}, ALU_in1};
    wire[63:0] alu_in2_signed = {{32{ALU_in2[31]}}, ALU_in2};
    assign mul_data_a = (alu_op == `ALU_MULHU) ? {32'b0, ALU_in1} : alu_in1_signed;
    assign mul_data_b = (alu_op == `ALU_MULHU || alu_op == `ALU_MULHSU) ? {32'b0, ALU_in2} : alu_in2_signed;

    reg [15:0] curr_inst_delay;
    reg [15:0] inst_delay;
    reg inst_was_stalling;

    wire inst_delay_stall = inst_was_stalling ? inst_delay != 0 : curr_inst_delay != 0;
    assign alu_stall = inst_delay_stall;

    always @(*) begin
        case (alu_op)
            `ALU_DIV,
            `ALU_DIVU,
            `ALU_REM,
            `ALU_REMU:  curr_inst_delay = div_pipeline_len;
            `ALU_MUL,
            `ALU_MULH,
            `ALU_MULHSU,
            `ALU_MULHU: curr_inst_delay = mul_pipeline_len;
            default:    curr_inst_delay = 0;
        endcase // alu_op
    end

    always @(posedge clk) begin
        if (reset) begin
            inst_delay <= 0;
            inst_was_stalling <= 0;
        end
        else if (inst_delay_stall) begin
            if (inst_was_stalling) begin
                if (inst_delay > 0)
                    inst_delay <= inst_delay - 1;
            end
            else begin
                inst_was_stalling <= 1;
                inst_delay <= curr_inst_delay - 1;
            end
        end
        else begin
            inst_was_stalling <= 0;
        end
    end

 `ifdef SYN_FUNC
 
    wire which_in2;
    wire[31:0] upper_immed;

    assign which_in2  = src_rs2 == `RS2_IMMED;

    assign ALU_in1 = src_a;
    assign ALU_in2 = which_in2 ? itype_immed : src_b;

    assign upper_immed = {upper_immed, {12{1'b0}}};

    always @(*) begin
        case (alu_op)
            `ALU_ADD:       alu_result = $signed(ALU_in1) + $signed(ALU_in2);
            `ALU_SUB:       alu_result = $signed(ALU_in1) - $signed(ALU_in2);
            `ALU_SLLA:      alu_result = ALU_in1 << ALU_in2[4:0];
            `ALU_SLT:       alu_result = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
            `ALU_SLTU:      alu_result = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
            `ALU_XOR:       alu_result = ALU_in1 ^ ALU_in2;
            `ALU_SRL:       alu_result = ALU_in1 >> ALU_in2[4:0];
            `ALU_SRA:       alu_result = $signed(ALU_in1)  >>> ALU_in2[4:0];
            `ALU_OR:        alu_result = ALU_in1 | ALU_in2;
            `ALU_AND:       alu_result = ALU_in2 & ALU_in1;
            `ALU_SUBU:      alu_result = (ALU_in1 >= ALU_in2) ? 32'h0 : 32'hffffffff;
            `ALU_LUI:       alu_result = upper_immed;
            `ALU_AUIPC:     alu_result = $signed(curr_PC) + $signed(upper_immed);
            // TODO: profitable to roll these exceptional cases into inst_delay to avoid pipeline when possible?
            `ALU_MUL:       alu_result = mul_result[31:0];
            `ALU_MULH:      alu_result = mul_result[63:32];
            `ALU_MULHSU:    alu_result = mul_result[63:32];
            `ALU_MULHU:     alu_result = mul_result[63:32];
            `ALU_DIV:       alu_result = (ALU_in2 == 0) ? 32'hffffffff : signed_div_result;
            `ALU_DIVU:      alu_result = (ALU_in2 == 0) ? 32'hffffffff : unsigned_div_result;
            `ALU_REM:       alu_result = (ALU_in2 == 0) ? ALU_in1 : signed_rem_result;
            `ALU_REMU:      alu_result = (ALU_in2 == 0) ? ALU_in1 : unsigned_rem_result;
            default:        alu_result = 32'h0;
        endcase // alu_op
    end

`else

    wire which_in2;        
    wire[31:0] upper_immed_s;

    assign which_in2  = src_rs2 == `RS2_IMMED;

    assign ALU_in1 = src_a;

    assign ALU_in2 = which_in2 ? itype_immed : src_b;

    assign upper_immed_s = {upper_immed, {12{1'b0}}};

    always @(*) begin
        case (alu_op)
            `ALU_ADD:       alu_result = $signed(ALU_in1) + $signed(ALU_in2);
            `ALU_SUB:       alu_result = $signed(ALU_in1) - $signed(ALU_in2);
            `ALU_SLLA:      alu_result = ALU_in1 << ALU_in2[4:0];
            `ALU_SLT:       alu_result = ($signed(ALU_in1) < $signed(ALU_in2)) ? 32'h1 : 32'h0;
            `ALU_SLTU:      alu_result = ALU_in1 < ALU_in2 ? 32'h1 : 32'h0;
            `ALU_XOR:       alu_result = ALU_in1 ^ ALU_in2;
            `ALU_SRL:       alu_result = ALU_in1 >> ALU_in2[4:0];
            `ALU_SRA:       alu_result = $signed(ALU_in1)  >>> ALU_in2[4:0];
            `ALU_OR:        alu_result = ALU_in1 | ALU_in2;
            `ALU_AND:       alu_result = ALU_in2 & ALU_in1;
            `ALU_SUBU:      alu_result = (ALU_in1 >= ALU_in2) ? 32'h0 : 32'hffffffff;
            `ALU_LUI:       alu_result = upper_immed_s;
            `ALU_AUIPC:     alu_result = $signed(curr_PC) + $signed(upper_immed_s);
            // TODO: profitable to roll these exceptional cases into inst_delay to avoid pipeline when possible?
            `ALU_MUL:       alu_result = mul_result[31:0];
            `ALU_MULH:      alu_result = mul_result[63:32];
            `ALU_MULHSU:    alu_result = mul_result[63:32];
            `ALU_MULHU:     alu_result = mul_result[63:32];
            `ALU_DIV:       alu_result = (ALU_in2 == 0) ? 32'hffffffff : signed_div_result;
            `ALU_DIVU:      alu_result = (ALU_in2 == 0) ? 32'hffffffff : unsigned_div_result;
            `ALU_REM:       alu_result = (ALU_in2 == 0) ? ALU_in1 : signed_rem_result;
            `ALU_REMU:      alu_result = (ALU_in2 == 0) ? ALU_in1 : unsigned_rem_result;
            default:        alu_result = 32'h0;
        endcase // alu_op
    end

`endif

endmodule