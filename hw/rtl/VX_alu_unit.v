`include "VX_define.vh"

module VX_alu_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    VX_alu_req_if   alu_req_if,

    // Outputs
    VX_branch_ctl_if branch_ctl_if,
    VX_exu_to_cmt_if alu_commit_if
);    
    reg [`NUM_THREADS-1:0][31:0] alu_result;            
    wire [`NUM_THREADS-1:0][32:0] sub_result; 
    wire [`NUM_THREADS-1:0][32:0] shift_result;

    wire [`ALU_BITS-1:0]           alu_op = alu_req_if.alu_op;
    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    genvar i;

    for (i = 0; i < `NUM_THREADS; i++) begin    

        wire [32:0] sub_in1  = {(alu_op != `ALU_SLTU) & (alu_op != `ALU_BLTU) & (alu_op != `ALU_BGEU) & alu_in1[i][31], alu_in1[i]};
        wire [32:0] sub_in2  = {(alu_op != `ALU_SLTU) & (alu_op != `ALU_BLTU) & (alu_op != `ALU_BGEU) & alu_in2[i][31], alu_in2[i]};
        assign sub_result[i] = $signed(sub_in1) - $signed(sub_in2);

        wire [32:0] shift_in1  = {(alu_op == `ALU_SRA) & alu_in1[i][31], alu_in1[i]};      
        assign shift_result[i] = $signed(shift_in1) >>> alu_in2[i][4:0];
     
        always @(*) begin
            case (alu_op)                        
                `ALU_SUB:   alu_result[i] = sub_result[i][31:0];
                `ALU_SLL:   alu_result[i] = alu_in1[i] << alu_in2[i][4:0];
                `ALU_SLT,
                `ALU_SLTU:  alu_result[i] = 32'(sub_result[i][32]);            
                `ALU_XOR:   alu_result[i] = alu_in1[i] ^ alu_in2[i];
                `ALU_SRL,
                `ALU_SRA:   alu_result[i] = shift_result[i][31:0];
                `ALU_OR:    alu_result[i] = alu_in1[i] | alu_in2[i];
                `ALU_AND:   alu_result[i] = alu_in1[i] & alu_in2[i];
                default:    alu_result[i] = alu_in1[i] + alu_in2[i]; // ADD, LUI, AUIPC
            endcase
        end       
    end        

    wire [`NT_BITS-1:0] br_result_index;

    VX_priority_encoder #(
        .N(`NUM_THREADS)
    ) choose_alu_result (
        .data_in  (alu_req_if.thread_mask),
        .data_out (br_result_index),
        `UNUSED_PIN (valid_out)
    );

    wire [32:0] br_result = sub_result[br_result_index];
    wire br_sign  = br_result[32];
    wire br_nzero = (| br_result[31:0]);
    wire br_sign_s1;
    wire br_nzero_s1;

    wire [`BR_BITS-1:0] br_op = `IS_BR_OP(alu_req_if.alu_op) ? `BR_OP(alu_req_if.alu_op) : `BR_NO;
    wire [`BR_BITS-1:0] br_op_s1;

    wire [31:0] br_addr = (br_op == `BR_JALR) ? alu_req_if.rs1_data[br_result_index] : alu_req_if.curr_PC;
    wire [31:0] br_dest = $signed(br_addr) + $signed(alu_req_if.offset);

    wire is_jal = (alu_op == `ALU_JAL || alu_op == `ALU_JALR);    
    wire [`NUM_THREADS-1:0][31:0] alu_jal_result = is_jal ? {`NUM_THREADS{alu_req_if.next_PC}} : alu_result;

    wire stall = ~alu_commit_if.ready && alu_commit_if.valid;

    VX_generic_register #(
        .N(1 + `NW_BITS + `ISTAG_BITS + (`NUM_THREADS * 32) + `BR_BITS + 32 + 1 + 1)
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({alu_req_if.valid,    alu_req_if.warp_num,    alu_req_if.issue_tag,    alu_jal_result,     br_op,   br_dest,              br_sign,   br_nzero}),
        .out   ({alu_commit_if.valid, branch_ctl_if.warp_num, alu_commit_if.issue_tag, alu_commit_if.data, br_op_s1, branch_ctl_if.dest, br_sign_s1, br_nzero_s1})
    );       
    
    reg br_taken;
    always @(*) begin
        case (br_op_s1)            
            `BR_NE:  br_taken = br_nzero_s1;
            `BR_EQ:  br_taken = ~br_nzero_s1;
            `BR_LT, 
            `BR_LTU: br_taken = br_sign_s1;
            `BR_GE, 
            `BR_GEU: br_taken = ~br_sign_s1;
            default: br_taken = 1'b1;
        endcase
    end    

    assign branch_ctl_if.valid = alu_commit_if.valid && (br_op_s1 != `BR_NO);
    assign branch_ctl_if.taken = br_taken;

    assign alu_req_if.ready = ~stall;

endmodule