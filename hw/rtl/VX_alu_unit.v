`include "VX_define.vh"

module VX_alu_unit #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,
    
    // Inputs
    VX_alu_req_if       alu_req_if,

    // Outputs
    VX_branch_ctl_if    branch_ctl_if,
    VX_commit_if        alu_commit_if    
);    
    reg [`NUM_THREADS-1:0][31:0] alu_result;    
    reg [`NUM_THREADS-1:0][31:0] add_result;   
    reg [`NUM_THREADS-1:0][32:0] sub_result;
    reg [`NUM_THREADS-1:0][31:0] shr_result;
    reg [`NUM_THREADS-1:0][31:0] msc_result;    

    wire               is_br_op = alu_req_if.is_br_op;
    wire [`ALU_BITS-1:0] alu_op = `ALU_OP(alu_req_if.op_type);
    wire [`BR_BITS-1:0]   br_op = `BR_OP(alu_req_if.op_type);
    wire             alu_signed = `ALU_SIGNED(alu_op);   
    wire [1:0]     alu_op_class = `ALU_OP_CLASS(alu_op); 
    wire                 is_sub = (alu_op == `ALU_SUB);

    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    wire [`NUM_THREADS-1:0][31:0] alu_in1_PC   = alu_req_if.rs1_is_PC ? {`NUM_THREADS{alu_req_if.PC}} : alu_in1;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_imm  = alu_req_if.rs2_is_imm ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_less = (alu_req_if.rs2_is_imm && !is_br_op) ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            add_result[i] = alu_in1_PC[i] + alu_in2_imm[i];  
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [32:0] sub_in1 = {alu_signed & alu_in1[i][31], alu_in1[i]};
        wire [32:0] sub_in2 = {alu_signed & alu_in2_less[i][31], alu_in2_less[i]};
        always @(*) begin
            sub_result[i] = $signed(sub_in1) - $signed(sub_in2);
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    
        wire [32:0] shr_in1 = {alu_signed & alu_in1[i][31], alu_in1[i]};
    `IGNORE_WARNINGS_BEGIN
        wire [32:0] shr_value = $signed(shr_in1) >>> alu_in2_imm[i][4:0]; 
    `IGNORE_WARNINGS_END
        always @(*) begin
            shr_result[i] = shr_value[31:0];
        end
    end        

    for (genvar i = 0; i < `NUM_THREADS; i++) begin 
        always @(*) begin
            case (alu_op)
                `ALU_AND:   msc_result[i] = alu_in1[i] & alu_in2_imm[i];
                `ALU_OR:    msc_result[i] = alu_in1[i] | alu_in2_imm[i];
                `ALU_XOR:   msc_result[i] = alu_in1[i] ^ alu_in2_imm[i];                
                //`ALU_SLL,
                default:    msc_result[i] = alu_in1[i] << alu_in2_imm[i][4:0];
            endcase
        end
    end
            
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            case (alu_op_class)                        
                0: alu_result[i] = add_result[i];
                1: alu_result[i] = {31'b0, sub_result[i][32]};
                2: alu_result[i] = is_sub ? sub_result[i][31:0] : shr_result[i];
                default: alu_result[i] = msc_result[i];
            endcase
        end       
    end
    
    wire is_jal = is_br_op && (br_op == `BR_JAL || br_op == `BR_JALR);
    wire [`NUM_THREADS-1:0][31:0] alu_jal_result = is_jal ? {`NUM_THREADS{alu_req_if.next_PC}} : alu_result; 

    wire [31:0] br_dest    = add_result[alu_req_if.tid]; 
    wire [32:0] cmp_result = sub_result[alu_req_if.tid];   
    
    wire [32:0] cmp_result_r;
    wire is_br_op_r;
`IGNORE_WARNINGS_BEGIN
    wire [`BR_BITS-1:0] br_op_r;
`IGNORE_WARNINGS_END

    // output

    wire stall_out = ~alu_commit_if.ready && alu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1 + `BR_BITS + 32 + 33),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({alu_req_if.valid,    alu_req_if.wid,    alu_req_if.tmask,    alu_req_if.PC,    alu_req_if.rd,    alu_req_if.wb,    alu_jal_result,     is_br_op,   br_op,   br_dest,            cmp_result}),
        .data_out ({alu_commit_if.valid, alu_commit_if.wid, alu_commit_if.tmask, alu_commit_if.PC, alu_commit_if.rd, alu_commit_if.wb, alu_commit_if.data, is_br_op_r, br_op_r, branch_ctl_if.dest, cmp_result_r})
    );

    assign alu_commit_if.eop = 1'b1;
    
    wire is_less  = cmp_result_r[32];
    wire is_equal = ~(| cmp_result_r[31:0]);        

    wire br_neg    = `BR_NEG(br_op_r);    
    wire br_less   = `BR_LESS(br_op_r);
    wire br_static = `BR_STATIC(br_op_r);
    wire br_taken  = ((br_less ? is_less : is_equal) ^ br_neg) | br_static;   

    assign branch_ctl_if.valid = alu_commit_if.valid && alu_commit_if.ready && is_br_op_r;
    assign branch_ctl_if.wid   = alu_commit_if.wid; 
    assign branch_ctl_if.taken = br_taken;

    // can accept new request?
    assign alu_req_if.ready = ~stall_out;

endmodule