`include "VX_define.vh"
`include "VX_config.vh"

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

    localparam UUID_WIDTH     = `UP(`UUID_BITS);
    localparam NW_WIDTH       = `UP(`NW_BITS);
    localparam RSP_ARB_DATAW  = UUID_WIDTH + NW_WIDTH + `NUM_THREADS + 32 + `NR_BITS + 1 + `NUM_THREADS * `XLEN;
    localparam RSP_ARB_SIZE   = 1 + `EXT_M_ENABLED;
    localparam SHIFT_IMM_BITS = `CLOG2(`XLEN) - 1;


    reg [`NUM_THREADS-1:0][`XLEN-1:0] alu_result;
    reg [`NUM_THREADS-1:0][`XLEN-1:0] add_result;
    reg [`NUM_THREADS-1:0][`XLEN:0] sub_result; // 33 or 65 bits to keep the overflow bit for branch calculations
    reg [`NUM_THREADS-1:0][`XLEN-1:0] shr_result;
    reg [`NUM_THREADS-1:0][`XLEN-1:0] msc_result;

    wire ready_in;    

    `UNUSED_VAR (alu_req_if.op_mod)
    wire                    is_br_op = `INST_ALU_IS_BR(alu_req_if.op_mod);
    wire [`INST_ALU_BITS-1:0] alu_op = `INST_ALU_BITS'(alu_req_if.op_type);
    wire [`INST_BR_BITS-1:0]   br_op = `INST_BR_BITS'(alu_req_if.op_type);
    wire                  alu_signed = `INST_ALU_SIGNED(alu_op);   
    wire [1:0]          alu_op_class = `INST_ALU_OP_CLASS(alu_op); 
    wire                      is_sub = (alu_op == `INST_ALU_SUB);

    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in1 = alu_req_if.rs1_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in2 = alu_req_if.rs2_data;

    wire [`NUM_THREADS-1:0][31:0] trunc_alu_in1;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        // PC operations should only be for 32 bits
        assign trunc_alu_in1[i] = alu_in1[i][31:0];
        // assign trunc_alu_result[i] = alu_result[i][`XLEN-1:0];
    end

    // PC operations should only be for 32 bits
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_A  = alu_req_if.use_PC ? {`NUM_THREADS{`XLEN'(alu_req_if.PC)}} : alu_in1;
    wire [`NUM_THREADS-1:0][31:0] alu_A_trunc = alu_req_if.use_PC ? {`NUM_THREADS{alu_req_if.PC}} : trunc_alu_in1;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_B  = alu_req_if.use_imm ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in2_less = (alu_req_if.use_imm && ~is_br_op) ? {`NUM_THREADS{alu_req_if.imm}} : alu_in2;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        always @(*) begin
            case(alu_op)
                `INST_ALU_ADD, `INST_ALU_AUIPC, `INST_ALU_LUI: add_result[i] = alu_A[i] + alu_B[i];
                `INST_ALU_ADD_W: add_result[i] = `XLEN'($signed(alu_A_trunc[i] + alu_B[i][31:0]));
                default: add_result[i] = alu_A[i] + alu_B[i];
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`XLEN:0] sub_in1 = {alu_signed & alu_in1[i][`XLEN-1], alu_in1[i]};
        wire [`XLEN:0] sub_in2 = {alu_signed & alu_in2_less[i][`XLEN-1], alu_in2_less[i]};

        wire [`XLEN:0] temp_sub_result = sub_in1 - sub_in2;
        always @(*) begin
            case(alu_op)
                `INST_ALU_SUB: sub_result[i] = temp_sub_result;
                `INST_ALU_SUB_W: sub_result[i] = {temp_sub_result[`XLEN], `XLEN'($signed(temp_sub_result[31:0]))};
                default: sub_result[i] = temp_sub_result;
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin    
        wire [`XLEN:0] shr_in1 = {alu_signed & alu_in1[i][`XLEN-1], alu_in1[i]};
        wire [`XLEN-1:0] temp_shr_result = `XLEN'($signed(shr_in1) >>> alu_B[i][SHIFT_IMM_BITS:0]);
        wire [31:0] temp_shr_result_w = 32'($signed(shr_in1) >>> alu_B[i][4:0]);

        always @(*) begin
            case(alu_op)
                `INST_ALU_SRA, `INST_ALU_SRL: shr_result[i] = temp_shr_result;
                `INST_ALU_SRA_W: shr_result[i] = `XLEN'($unsigned(temp_shr_result_w[31:0])); // is this needed or is it already 0 extended?
                `INST_ALU_SRL_W: shr_result[i] = `XLEN'($signed(temp_shr_result_w[31:0]));
                default: shr_result[i] = temp_shr_result;
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [31:0] temp_shift_result = alu_in1[i][31:0] << alu_B[i][4:0]; // only used for SLLW
        always @(*) begin
            case (alu_op)
                `INST_ALU_AND: msc_result[i] = alu_in1[i] & alu_B[i];
                `INST_ALU_OR:  msc_result[i] = alu_in1[i] | alu_B[i];
                `INST_ALU_XOR: msc_result[i] = alu_in1[i] ^ alu_B[i];
                // `INST_ALU_SLL: msc_result[i] = alu_in1[i] << alu_B[i][4:0];
                `INST_ALU_SLL: msc_result[i] = alu_in1[i] << alu_B[i][SHIFT_IMM_BITS:0]; // TODO: CHANGED: adjust this to shift using 6 bits for 64 bit
                `INST_ALU_SLL_W: msc_result[i] = `XLEN'($signed(temp_shift_result[31:0])); // TODO: CHANGED: adjust this to shift using 6 bits for 32 signed bit 
                default:       msc_result[i] = 'x;
            endcase
        end
    end
            
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        always @(*) begin
            case (alu_op_class)                        
                2'b00: alu_result[i] = add_result[i];                               // ADD, LUI, AUIPC, ADDIW, ADDW
                2'b01: alu_result[i] = {{`XLEN-1{1'b0}}, sub_result[i][`XLEN]};     // SLTU, SLT
                2'b10: alu_result[i] = is_sub ? sub_result[i][`XLEN-1:0]            // SUB, SUBW
                                              : shr_result[i];                      // SRL, SRA, SRLW, SRAW, SRLIW, SRAIW, SRLI, SRAI
                 // 2'b11,
                default: alu_result[i] = msc_result[i];                             // AND, OR, XOR, SLL, SLLIW, SLLW, SLLI
            endcase
        end       
    end

    // branch
    
    wire is_jal = is_br_op && (br_op == `INST_BR_JAL || br_op == `INST_BR_JALR);
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_jal_result = is_jal ? {`NUM_THREADS{`XLEN'(alu_req_if.next_PC)}} : alu_result; 

    wire [`XLEN-1:0] br_dest    = add_result[alu_req_if.tid][`XLEN-1:0];
    wire [`XLEN:0] cmp_result = sub_result[alu_req_if.tid][`XLEN:0];
    
    wire is_less  = cmp_result[`XLEN];
    wire is_equal = ~(| cmp_result[`XLEN-1:0]);        

    // output

    wire                          alu_valid_in;
    wire                          alu_ready_in;
    wire                          alu_valid_out;
    wire                          alu_ready_out;
    wire [UUID_WIDTH-1:0]         alu_uuid;
    wire [NW_WIDTH-1:0]           alu_wid;
    wire [`NUM_THREADS-1:0]       alu_tmask;
    wire [31:0]                   alu_PC;
    wire [`NR_BITS-1:0]           alu_rd;   
    wire                          alu_wb; 
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_data;

    wire [`NUM_THREADS-1:0][`XLEN-1:0] full_alu_data;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign full_alu_data[i] =alu_data[i];
    end

    wire [`INST_BR_BITS-1:0] br_op_r;
    wire [`XLEN-1:0] br_dest_r;
    wire is_less_r;
    wire is_equal_r;
    wire is_br_op_r;

    assign alu_ready_in = alu_ready_out || ~alu_valid_out;

    VX_pipe_register #(
        .DATAW  (1 + UUID_WIDTH + NW_WIDTH + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * `XLEN) + 1 + `INST_BR_BITS + 1 + 1 + `XLEN),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (alu_ready_in),
        .data_in  ({alu_valid_in,  alu_req_if.uuid, alu_req_if.wid, alu_req_if.tmask, alu_req_if.PC, alu_req_if.rd, alu_req_if.wb, alu_jal_result, is_br_op,   br_op,   is_less,   is_equal,   br_dest}),
        .data_out ({alu_valid_out, alu_uuid,        alu_wid,        alu_tmask,        alu_PC,        alu_rd,        alu_wb,        alu_data,       is_br_op_r, br_op_r, is_less_r, is_equal_r, br_dest_r})
    );

    `UNUSED_VAR (br_op_r)
    wire br_neg    = `INST_BR_NEG(br_op_r);
    wire br_less   = `INST_BR_LESS(br_op_r);
    wire br_static = `INST_BR_STATIC(br_op_r);

    assign branch_ctl_if.valid = alu_valid_out && alu_ready_out && is_br_op_r;
    assign branch_ctl_if.taken = ((br_less ? is_less_r : is_equal_r) ^ br_neg) | br_static;
    assign branch_ctl_if.wid   = alu_wid;
    assign branch_ctl_if.dest  = br_dest_r[`XLEN-1:0];

`ifdef EXT_M_ENABLE

    wire                          mul_valid_in;
    wire                          mul_ready_in;
    wire                          mul_valid_out;    
    wire                          mul_ready_out;
    wire [UUID_WIDTH-1:0]         mul_uuid;
    wire [NW_WIDTH-1:0]           mul_wid;
    wire [`NUM_THREADS-1:0]       mul_tmask;
    wire [31:0]                   mul_PC;
    wire [`NR_BITS-1:0]           mul_rd;
    wire                          mul_wb;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] mul_data;

    wire [`INST_MUL_BITS-1:0] mul_op = `INST_MUL_BITS'(alu_req_if.op_type);

    `RESET_RELAY (muldiv_reset, reset);

    VX_muldiv muldiv (
        .clk        (clk),
        .reset      (muldiv_reset),
        
        // Inputs
        .valid_in   (mul_valid_in),
        .alu_op     (mul_op),
        .uuid_in    (alu_req_if.uuid),
        .wid_in     (alu_req_if.wid),
        .tmask_in   (alu_req_if.tmask),
        .PC_in      (alu_req_if.PC),
        .rd_in      (alu_req_if.rd),
        .wb_in      (alu_req_if.wb),
        .alu_in1    (alu_req_if.rs1_data), 
        .alu_in2    (alu_req_if.rs2_data),
        .ready_in   (mul_ready_in),

        // Outputs
        .valid_out  (mul_valid_out),
        .wid_out    (mul_wid),
        .uuid_out   (mul_uuid),
        .tmask_out  (mul_tmask),
        .PC_out     (mul_PC),
        .rd_out     (mul_rd),
        .wb_out     (mul_wb),
        .data_out   (mul_data),
        .ready_out  (mul_ready_out)
    );

    wire is_mul_op = `INST_ALU_IS_MUL(alu_req_if.op_mod);

    assign alu_valid_in = alu_req_if.valid && ~is_mul_op;
    assign mul_valid_in = alu_req_if.valid && is_mul_op;
    assign ready_in     = is_mul_op ? mul_ready_in : alu_ready_in;

`else 

    assign alu_valid_in = alu_req_if.valid;
    assign ready_in     = alu_ready_in;

`endif

    // send response

    VX_stream_arb #(
        .NUM_INPUTS (RSP_ARB_SIZE),
        .DATAW      (RSP_ARB_DATAW),        
        .ARBITER    ("R"),
        .BUFFERED   (1)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({
            alu_valid_out
        `ifdef EXT_M_ENABLE
            , mul_valid_out
        `endif
        }),
        .ready_in  ({
            alu_ready_out
        `ifdef EXT_M_ENABLE
            , mul_ready_out
        `endif
        }),
        .data_in   ({
            {alu_uuid, alu_wid, alu_tmask, alu_PC, alu_rd, alu_wb, full_alu_data}
        `ifdef EXT_M_ENABLE
            , {mul_uuid, mul_wid, mul_tmask, mul_PC, mul_rd, mul_wb, mul_data}
        `endif
        }),
        .data_out  ({alu_commit_if.uuid, alu_commit_if.wid, alu_commit_if.tmask, alu_commit_if.PC, alu_commit_if.rd, alu_commit_if.wb, alu_commit_if.data}),
        .valid_out (alu_commit_if.valid),        
        .ready_out (alu_commit_if.ready)
    );

    assign alu_commit_if.eop = 1'b1;

    // can accept new request?
    assign alu_req_if.ready = ready_in;

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (branch_ctl_if.valid) begin
            `TRACE(1, ("%d: core%0d-branch: wid=%0d, PC=0x%0h, taken=%b, dest=0x%0h (#%0d)\n",
                $time, CORE_ID, branch_ctl_if.wid, alu_PC, branch_ctl_if.taken, branch_ctl_if.dest, alu_uuid));
        end
    end
`endif

endmodule
