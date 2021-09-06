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

    `UNUSED_PARAM (CORE_ID)
    
    reg [`NUM_THREADS-1:0][31:0]  alu_result;    
    wire [`NUM_THREADS-1:0][31:0] add_result;   
    wire [`NUM_THREADS-1:0][32:0] sub_result;
    wire [`NUM_THREADS-1:0][31:0] shr_result;
    reg [`NUM_THREADS-1:0][31:0]  msc_result;

    wire stall_in, stall_out;    

    `UNUSED_VAR (alu_req_if.op_mod)
    wire                    is_br_op = `INST_ALU_IS_BR(alu_req_if.op_mod);
    wire [`INST_ALU_BITS-1:0] alu_op = `INST_ALU_OP(alu_req_if.op_type);
    wire [`INST_BR_BITS-1:0]   br_op = `INST_BR_OP(alu_req_if.op_type);
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

    wire                          result_valid;
    wire [`NW_BITS-1:0]           result_wid;
    wire [`NUM_THREADS-1:0]       result_tmask;
    wire [31:0]                   result_PC;
    wire [`NR_BITS-1:0]           result_rd;   
    wire                          result_wb; 
    wire [`NUM_THREADS-1:0][31:0] result_data;
    wire                          result_is_br;

`ifdef EXT_M_ENABLE

    wire                          mul_ready_in;
    wire                          mul_valid_out;    
    wire                          mul_ready_out;
    wire [`NW_BITS-1:0]           mul_wid;
    wire [`NUM_THREADS-1:0]       mul_tmask;
    wire [31:0]                   mul_PC;
    wire [`NR_BITS-1:0]           mul_rd;
    wire                          mul_wb;
    wire [`NUM_THREADS-1:0][31:0] mul_data;

    wire is_mul_op = `INST_ALU_IS_MUL(alu_req_if.op_mod);
    
    VX_muldiv muldiv (
        .clk        (clk),
        .reset      (reset),
        
        // Inputs
        .alu_op     (`INST_MUL_OP(alu_req_if.op_type)),
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
        .valid_in   (alu_req_if.valid && is_mul_op),
        .ready_in   (mul_ready_in),
        .valid_out  (mul_valid_out),
        .ready_out  (mul_ready_out)
    );

    assign stall_in = (is_mul_op && ~mul_ready_in) 
                   || (~is_mul_op && (mul_valid_out || stall_out));
    
    assign mul_ready_out = ~stall_out;

    assign result_valid = mul_valid_out || (alu_req_if.valid && ~is_mul_op);
    assign result_wid   = mul_valid_out ? mul_wid   : alu_req_if.wid;    
    assign result_tmask = mul_valid_out ? mul_tmask : alu_req_if.tmask;    
    assign result_PC    = mul_valid_out ? mul_PC    : alu_req_if.PC;    
    assign result_rd    = mul_valid_out ? mul_rd    : alu_req_if.rd;    
    assign result_wb    = mul_valid_out ? mul_wb    : alu_req_if.wb;    
    assign result_data  = mul_valid_out ? mul_data  : alu_jal_result;
    assign result_is_br = ~mul_valid_out && is_br_op;

`else 

    assign stall_in = stall_out;

    assign result_valid = alu_req_if.valid;
    assign result_wid   = alu_req_if.wid;  
    assign result_tmask = alu_req_if.tmask;    
    assign result_PC    = alu_req_if.PC; 
    assign result_rd    = alu_req_if.rd;    
    assign result_wb    = alu_req_if.wb;   
    assign result_data  = alu_jal_result;
    assign result_is_br = is_br_op;

`endif

    wire [`INST_BR_BITS-1:0] br_op_r;    
    wire is_less_r;
    wire is_equal_r;
    wire is_br_op_r;

    assign stall_out = ~alu_commit_if.ready && alu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1 + `INST_BR_BITS + 1 + 1 + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({result_valid,        result_wid,        result_tmask,        result_PC,        result_rd,        result_wb,        result_data,        result_is_br, br_op,   is_less,   is_equal,   br_dest}),
        .data_out ({alu_commit_if.valid, alu_commit_if.wid, alu_commit_if.tmask, alu_commit_if.PC, alu_commit_if.rd, alu_commit_if.wb, alu_commit_if.data, is_br_op_r,   br_op_r, is_less_r, is_equal_r, branch_ctl_if.dest})
    );

    assign alu_commit_if.eop = 1'b1;

    `UNUSED_VAR (br_op_r)
    wire br_neg    = `INST_BR_NEG(br_op_r);
    wire br_less   = `INST_BR_LESS(br_op_r);
    wire br_static = `INST_BR_STATIC(br_op_r);

    assign branch_ctl_if.valid = alu_commit_if.valid && alu_commit_if.ready && is_br_op_r;
    assign branch_ctl_if.taken = ((br_less ? is_less_r : is_equal_r) ^ br_neg) | br_static;
    assign branch_ctl_if.wid   = alu_commit_if.wid;

    // can accept new request?
    assign alu_req_if.ready = ~stall_in;

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (branch_ctl_if.valid) begin
            dpi_trace("%d: core%0d-branch: wid=%0d, PC=%0h, taken=%b, dest=%0h\n", 
                $time, CORE_ID, branch_ctl_if.wid, alu_commit_if.PC, branch_ctl_if.taken, branch_ctl_if.dest);
        end
    end
`endif

endmodule