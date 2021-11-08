`include "VX_define.vh"

module VX_alu_unit #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
    // Inputs
    VX_alu_req_if.slave     alu_req_if,

    // Outputs
    VX_branch_ctl_if.master branch_ctl_if,
    VX_commit_if.master     alu_commit_if    
);   

    `UNUSED_PARAM (CORE_ID)
    
    reg [`NUM_THREADS-1:0][31:0]  alu_result;    
    wire [`NUM_THREADS-1:0][31:0] add_result;   
    wire [`NUM_THREADS-1:0][32:0] sub_result;
    wire [`NUM_THREADS-1:0][31:0] shr_result;
    reg [`NUM_THREADS-1:0][31:0]  msc_result;

    wire ready_in;    

    `UNUSED_VAR (alu_req_if.op_mod)
    wire                    is_br_op = `INST_ALU_IS_BR(alu_req_if.op_mod);
    wire [`INST_ALU_BITS-1:0] alu_op = `INST_ALU_BITS'(alu_req_if.op_type);
    wire [`INST_BR_BITS-1:0]   br_op = `INST_BR_BITS'(alu_req_if.op_type);
    wire                  alu_signed = `INST_ALU_SIGNED(alu_op);   
    wire [1:0]          alu_op_class = `INST_ALU_OP_CLASS(alu_op); 
    wire                      is_sub = (alu_op == `INST_ALU_SUB);

    wire [`NUM_THREADS-1:0][31:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][31:0] alu_in2 = alu_req_if.rs2_data;

    wire [`NUM_THREADS-1:0][31:0] alu_in1_PC   = alu_req_if.use_PC ? {`NUM_THREADS{alu_req_if.PC}} : alu_in1;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_imm  = alu_req_if.use_imm ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;
    wire [`NUM_THREADS-1:0][31:0] alu_in2_less = (alu_req_if.use_imm && ~is_br_op) ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign add_result[i] = alu_in1_PC[i] + alu_in2_imm[i];
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire [32:0] sub_in1 = {alu_signed & alu_in1[i][31], alu_in1[i]};
        wire [32:0] sub_in2 = {alu_signed & alu_in2_less[i][31], alu_in2_less[i]};
        assign sub_result[i] = sub_in1 - sub_in2;
    end

    for (genvar i = 0; i < `NUM_THREADS; i++) begin    
        wire [32:0] shr_in1 = {alu_signed & alu_in1[i][31], alu_in1[i]};
        assign shr_result[i] = 32'($signed(shr_in1) >>> alu_in2_imm[i][4:0]);
    end        

    for (genvar i = 0; i < `NUM_THREADS; i++) begin 
        always @(*) begin
            case (alu_op)
                `INST_ALU_AND: msc_result[i] = alu_in1[i] & alu_in2_imm[i];
                `INST_ALU_OR:  msc_result[i] = alu_in1[i] | alu_in2_imm[i];
                `INST_ALU_XOR: msc_result[i] = alu_in1[i] ^ alu_in2_imm[i];                
                //`INST_ALU_SLL,
                default:  msc_result[i] = alu_in1[i] << alu_in2_imm[i][4:0];
            endcase
        end
    end
            
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        always @(*) begin
            case (alu_op_class)                        
                2'b00: alu_result[i] = add_result[i];               // ADD, LUI, AUIPC 
                2'b01: alu_result[i] = {31'b0, sub_result[i][32]};  // SLTU, SLT
                2'b10: alu_result[i] = is_sub ? sub_result[i][31:0] // SUB
                                              : shr_result[i];      // SRL, SRA
                // 2'b11,
                default: alu_result[i] = msc_result[i];             // AND, OR, XOR, SLL
            endcase
        end       
    end

    // branch
    
    wire is_jal = is_br_op && (br_op == `INST_BR_JAL || br_op == `INST_BR_JALR);
    wire [`NUM_THREADS-1:0][31:0] alu_jal_result = is_jal ? {`NUM_THREADS{alu_req_if.next_PC}} : alu_result; 

    wire [31:0] br_dest    = add_result[alu_req_if.tid]; 
    wire [32:0] cmp_result = sub_result[alu_req_if.tid];   
    
    wire is_less  = cmp_result[32];
    wire is_equal = ~(| cmp_result[31:0]);        

    // output

    wire                          alu_valid_in;
    wire                          alu_ready_in;
    wire                          alu_valid_out;
    wire                          alu_ready_out;
    wire [`NW_BITS-1:0]           alu_wid;
    wire [`NUM_THREADS-1:0]       alu_tmask;
    wire [31:0]                   alu_PC;
    wire [`NR_BITS-1:0]           alu_rd;   
    wire                          alu_wb; 
    wire [`NUM_THREADS-1:0][31:0] alu_data;

    wire [`INST_BR_BITS-1:0] br_op_r;
    wire [31:0] br_dest_r;
    wire is_less_r;
    wire is_equal_r;
    wire is_br_op_r;

    assign alu_ready_in = alu_ready_out || ~alu_valid_out;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1 + `INST_BR_BITS + 1 + 1 + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (alu_ready_in),
        .data_in  ({alu_valid_in,  alu_req_if.wid, alu_req_if.tmask, alu_req_if.PC, alu_req_if.rd, alu_req_if.wb, alu_jal_result, is_br_op,   br_op,   is_less,   is_equal,   br_dest}),
        .data_out ({alu_valid_out, alu_wid,        alu_tmask,        alu_PC,        alu_rd,        alu_wb,        alu_data,       is_br_op_r, br_op_r, is_less_r, is_equal_r, br_dest_r})
    );

    `UNUSED_VAR (br_op_r)
    wire br_neg    = `INST_BR_NEG(br_op_r);
    wire br_less   = `INST_BR_LESS(br_op_r);
    wire br_static = `INST_BR_STATIC(br_op_r);

    assign branch_ctl_if.valid = alu_valid_out && alu_ready_out && is_br_op_r;
    assign branch_ctl_if.taken = ((br_less ? is_less_r : is_equal_r) ^ br_neg) | br_static;
    assign branch_ctl_if.wid   = alu_wid;
    assign branch_ctl_if.dest  = br_dest_r;

`ifdef EXT_M_ENABLE

    wire                          mul_valid_in;
    wire                          mul_ready_in;
    wire                          mul_valid_out;    
    wire                          mul_ready_out;
    wire [`NW_BITS-1:0]           mul_wid;
    wire [`NUM_THREADS-1:0]       mul_tmask;
    wire [31:0]                   mul_PC;
    wire [`NR_BITS-1:0]           mul_rd;
    wire                          mul_wb;
    wire [`NUM_THREADS-1:0][31:0] mul_data;

    wire [`INST_MUL_BITS-1:0] mul_op = `INST_MUL_BITS'(alu_req_if.op_type);

    VX_muldiv muldiv (
        .clk        (clk),
        .reset      (reset),
        
        // Inputs
        .alu_op     (mul_op),
        .wid_in     (alu_req_if.wid),
        .tmask_in   (alu_req_if.tmask),
        .PC_in      (alu_req_if.PC),
        .rd_in      (alu_req_if.rd),
        .wb_in      (alu_req_if.wb),
        .alu_in1    (alu_req_if.rs1_data), 
        .alu_in2    (alu_req_if.rs2_data),

        // Outputs
        .wid_out    (mul_wid),
        .tmask_out  (mul_tmask),
        .PC_out     (mul_PC),
        .rd_out     (mul_rd),
        .wb_out     (mul_wb),
        .data_out   (mul_data),

        // handshake
        .valid_in   (mul_valid_in),
        .ready_in   (mul_ready_in),
        .valid_out  (mul_valid_out),
        .ready_out  (mul_ready_out)
    );

    wire is_mul_op = `INST_ALU_IS_MUL(alu_req_if.op_mod);

    assign ready_in = is_mul_op ? mul_ready_in : alu_ready_in;

    assign alu_valid_in = alu_req_if.valid && ~is_mul_op;
    assign mul_valid_in = alu_req_if.valid && is_mul_op;

    assign alu_commit_if.valid = alu_valid_out || mul_valid_out;
    assign alu_commit_if.wid   = alu_valid_out ? alu_wid   : mul_wid;
    assign alu_commit_if.tmask = alu_valid_out ? alu_tmask : mul_tmask;
    assign alu_commit_if.PC    = alu_valid_out ? alu_PC    : mul_PC;
    assign alu_commit_if.rd    = alu_valid_out ? alu_rd    : mul_rd;
    assign alu_commit_if.wb    = alu_valid_out ? alu_wb    : mul_wb;
    assign alu_commit_if.data  = alu_valid_out ? alu_data  : mul_data;

    assign alu_ready_out = alu_commit_if.ready;
    assign mul_ready_out = alu_commit_if.ready & ~alu_valid_out; // ALU takes priority

`else 

    assign ready_in = alu_ready_in;

    assign alu_valid_in = alu_req_if.valid;

    assign alu_commit_if.valid = alu_valid_out;
    assign alu_commit_if.wid   = alu_wid;
    assign alu_commit_if.tmask = alu_tmask;
    assign alu_commit_if.PC    = alu_PC; 
    assign alu_commit_if.rd    = alu_rd;    
    assign alu_commit_if.wb    = alu_wb;
    assign alu_commit_if.data  = alu_data;

    assign alu_ready_out = alu_commit_if.ready;

`endif

    assign alu_commit_if.eop = 1'b1;

    // can accept new request?
    assign alu_req_if.ready = ready_in;

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (branch_ctl_if.valid) begin
            dpi_trace("%d: core%0d-branch: wid=%0d, PC=%0h, taken=%b, dest=%0h\n", 
                $time, CORE_ID, branch_ctl_if.wid, alu_commit_if.PC, branch_ctl_if.taken, branch_ctl_if.dest);
        end
    end
`endif

endmodule