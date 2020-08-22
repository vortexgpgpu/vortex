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
    VX_exu_to_cmt_if    alu_commit_if    
);    
    reg [`NUM_THREADS-1:0][31:0] alu_result;
    reg [`NUM_THREADS-1:0][31:0] add_result;   
    reg [`NUM_THREADS-1:0][32:0] sub_result;
    reg [`NUM_THREADS-1:0][31:0] shift_result;
    reg [`NUM_THREADS-1:0][31:0] misc_result;    

    wire valid_r;
    wire [`NW_BITS-1:0] wid_r;    
    wire [`NUM_THREADS-1:0] thread_mask_r;
    wire [31:0] curr_PC_r;
    wire [`NR_BITS-1:0] rd_r;
    wire wb_r;
    wire [`NT_BITS-1:0] tid_r;  
    wire is_sub_r;
    wire [`BR_BITS-1:0] br_op_r;
    wire is_br_op_r, is_br_op_s;
    wire [1:0] alu_op_class_r;
    wire [31:0] next_PC_r; 

    wire               is_br_op = `IS_BR_OP(alu_req_if.op);
    wire [`ALU_BITS-1:0] alu_op = `ALU_OP(alu_req_if.op);
    wire [`BR_BITS-1:0]   br_op = `BR_OP(alu_req_if.op);
    wire             alu_signed = `ALU_SIGNED(alu_op);   
    wire [1:0]     alu_op_class = `ALU_OP_CLASS(alu_op); 
    wire                 is_sub = (alu_op == `ALU_SUB);

    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    wire [`NUM_THREADS-1:0][31:0] alu_in1_PC   = alu_req_if.rs1_is_PC  ? {`NUM_THREADS{alu_req_if.curr_PC}} : alu_in1;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_imm  = alu_req_if.rs2_is_imm ? {`NUM_THREADS{alu_req_if.imm}}     : alu_in2;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_less = (alu_req_if.rs2_is_imm && ~is_br_op) ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(posedge clk) begin
            add_result[i] <= alu_in1_PC[i] + alu_in2_imm[i];  
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [32:0] sub_in1 = {alu_signed & alu_in1[i][31],      alu_in1[i]};
        wire [32:0] sub_in2 = {alu_signed & alu_in2_less[i][31], alu_in2_less[i]};
        always @(posedge clk) begin
            sub_result[i] <= $signed(sub_in1) - $signed(sub_in2);
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    
        wire [32:0] shift_in1 = {alu_signed & alu_in1[i][31], alu_in1[i]};
    `IGNORE_WARNINGS_BEGIN
        wire [32:0] shift_value = $signed(shift_in1) >>> alu_in2_imm[i][4:0]; 
    `IGNORE_WARNINGS_END
        always @(posedge clk) begin
            shift_result[i] <= shift_value[31:0];
        end
    end        

    for (genvar i = 0; i < `NUM_THREADS; i++) begin 
        always @(posedge clk) begin
            case (alu_op)
                `ALU_AND:   misc_result[i] <= alu_in1[i] & alu_in2_imm[i];
                `ALU_OR:    misc_result[i] <= alu_in1[i] | alu_in2_imm[i];
                `ALU_XOR:   misc_result[i] <= alu_in1[i] ^ alu_in2_imm[i];                
                //`ALU_SLL,
                default:    misc_result[i] <= alu_in1[i] << alu_in2_imm[i][4:0];
            endcase
        end
    end
    
    reg [31:0] next_PC = alu_req_if.curr_PC + 4;      

    VX_shift_register #(
        .DATAW(1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + `NT_BITS + 1 + 1 + `BR_BITS + 2 + 32),
        .DEPTH(1)
    ) alu_shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(alu_req_if.ready),
        .in({alu_req_if.valid, alu_req_if.wid, alu_req_if.thread_mask, alu_req_if.curr_PC, alu_req_if.rd, alu_req_if.wb, alu_req_if.tid, is_sub,   is_br_op,   br_op,   alu_op_class,   next_PC}),
        .out({valid_r,         wid_r,          thread_mask_r,          curr_PC_r,          rd_r,          wb_r,          tid_r,          is_sub_r, is_br_op_r, br_op_r, alu_op_class_r, next_PC_r})
    );
            
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            case (alu_op_class_r)                        
                0: alu_result[i] = is_sub_r ? sub_result[i][31:0] : add_result[i];
                1: alu_result[i] = {31'b0, sub_result[i][32]};
                2: alu_result[i] = shift_result[i];
                default: alu_result[i] = misc_result[i];
            endcase
        end       
    end

    // branch handling

    wire br_neg    = `BR_NEG(br_op_r);    
    wire br_less   = `BR_LESS(br_op_r);
    wire br_static = `BR_STATIC(br_op_r);
    wire is_jal    = is_br_op_r && (br_op_r == `BR_JAL || br_op_r == `BR_JALR);

    wire [31:0] br_dest = add_result[tid_r];
    wire [32:0] cmp_result = sub_result[tid_r];
    wire is_less = cmp_result[32];
    wire is_equal = ~(| cmp_result[31:0]);
    wire br_taken = ((br_less ? is_less : is_equal) ^ br_neg) | br_static;        
    
    wire [`NUM_THREADS-1:0][31:0] alu_jal_result = is_jal ? {`NUM_THREADS{next_PC_r}} : alu_result;

    // output

    wire stall_out = ~alu_commit_if.ready && alu_commit_if.valid;

    VX_generic_register #(
        .N(1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1 + 1 + 32)
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_out),
        .flush (0),
        .in    ({valid_r,             wid_r,             thread_mask_r,             curr_PC_r,             rd_r,             wb_r,             alu_jal_result,     is_br_op_r, br_taken,            br_dest}),
        .out   ({alu_commit_if.valid, alu_commit_if.wid, alu_commit_if.thread_mask, alu_commit_if.curr_PC, alu_commit_if.rd, alu_commit_if.wb, alu_commit_if.data, is_br_op_s, branch_ctl_if.taken, branch_ctl_if.dest})
    );

    assign branch_ctl_if.valid = alu_commit_if.valid && alu_commit_if.ready && is_br_op_s;
    assign branch_ctl_if.wid   = alu_commit_if.wid; 

    // can accept new request?
    assign alu_req_if.ready = ~stall_out;

endmodule