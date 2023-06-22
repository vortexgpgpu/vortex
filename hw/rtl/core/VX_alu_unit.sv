`include "VX_define.vh"

module VX_alu_unit #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
    // Inputs
    VX_alu_exe_if.slave     alu_exe_if,

    // Outputs
    VX_branch_ctl_if.master branch_ctl_if,
    VX_commit_if.master     alu_commit_if    
);   

    `UNUSED_PARAM (CORE_ID)

    localparam UUID_WIDTH     = `UP(`UUID_BITS);
    localparam NW_WIDTH       = `UP(`NW_BITS);
    localparam RSP_ARB_DATAW  = UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + `NUM_THREADS * `XLEN;
    localparam RSP_ARB_SIZE   = 1 + `EXT_M_ENABLED;
    localparam SHIFT_IMM_BITS = `CLOG2(`XLEN);

    wire [`NUM_THREADS-1:0][`XLEN-1:0] add_result;
    wire [`NUM_THREADS-1:0][`XLEN:0]   sub_result; // +1 bit for branch compare
    wire [`NUM_THREADS-1:0][`XLEN-1:0] shr_result;
    reg  [`NUM_THREADS-1:0][`XLEN-1:0] msc_result;
    
    wire [`NUM_THREADS-1:0][`XLEN-1:0] add_result_w;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] sub_result_w;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] shr_result_w;
    reg  [`NUM_THREADS-1:0][`XLEN-1:0] msc_result_w;

    reg [`NUM_THREADS-1:0][`XLEN-1:0] alu_result;

    wire ready_in;

`ifdef XLEN_64
    wire is_alu_w = `INST_ALU_IS_W(alu_exe_if.op_mod);
`else
    wire is_alu_w = 0;
`endif

    `UNUSED_VAR (alu_exe_if.op_mod)

    wire [`INST_ALU_BITS-1:0] alu_op = `INST_ALU_BITS'(alu_exe_if.op_type);
    wire [`INST_BR_BITS-1:0]   br_op = `INST_BR_BITS'(alu_exe_if.op_type);
     wire                    is_br_op = `INST_ALU_IS_BR(alu_exe_if.op_mod);
    wire                   is_sub_op = `INST_ALU_IS_SUB(alu_op);
    wire                  alu_signed = `INST_ALU_SIGNED(alu_op);   
    wire [1:0]          alu_op_class = `INST_ALU_CLASS(alu_op);
    
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in1 = alu_exe_if.rs1_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in2 = alu_exe_if.rs2_data;

    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in1_PC  = alu_exe_if.use_PC ? {`NUM_THREADS{alu_exe_if.PC}} : alu_in1;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in2_imm = alu_exe_if.use_imm ? {`NUM_THREADS{alu_exe_if.imm}} : alu_in2;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_in2_br  = (alu_exe_if.use_imm && ~is_br_op) ? {`NUM_THREADS{alu_exe_if.imm}} : alu_in2;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign add_result[i] = alu_in1_PC[i] + alu_in2_imm[i];
        assign add_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] + alu_in2_imm[i][31:0]));
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`XLEN:0] sub_in1 = {alu_signed & alu_in1[i][`XLEN-1], alu_in1[i]};
        wire [`XLEN:0] sub_in2 = {alu_signed & alu_in2_br[i][`XLEN-1], alu_in2_br[i]};
        assign sub_result[i] = sub_in1 - sub_in2;
        assign sub_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] - alu_in2_imm[i][31:0]));
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin    
        wire [`XLEN:0] shr_in1 = {alu_signed && alu_in1[i][`XLEN-1], alu_in1[i]};        
        assign shr_result[i] = `XLEN'($signed(shr_in1) >>> alu_in2_imm[i][SHIFT_IMM_BITS-1:0]);
        wire [32:0] shr_in1_w = {alu_signed && alu_in1[i][31], alu_in1[i][31:0]};
        wire [31:0] shr_res_w = 32'($signed(shr_in1_w) >>> alu_in2_imm[i][4:0]);
        assign shr_result_w[i] = `XLEN'($signed(shr_res_w));
    end

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        always @(*) begin
            case (alu_op[1:0])
                2'b00: msc_result[i] = alu_in1[i] & alu_in2_imm[i]; // AND
                2'b01: msc_result[i] = alu_in1[i] | alu_in2_imm[i]; // OR
                2'b10: msc_result[i] = alu_in1[i] ^ alu_in2_imm[i]; // XOR
                2'b11: msc_result[i] = alu_in1[i] << alu_in2_imm[i][SHIFT_IMM_BITS-1:0]; // SLL
            endcase
        end
        assign msc_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] << alu_in2_imm[i][4:0]));
    end
            
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`XLEN-1:0] slt_sub_result = is_sub_op ? sub_result[i][`XLEN-1:0] : `XLEN'(sub_result[i][`XLEN]);
        always @(*) begin
            case ({is_alu_w, alu_op_class})                        
                3'b000: alu_result[i] = add_result[i];      // ADD, LUI, AUIPC
                3'b001: alu_result[i] = slt_sub_result;     // SUB, SLTU, SLT
                3'b010: alu_result[i] = shr_result[i];      // SRL, SRA, SRLI, SRAI
                3'b011: alu_result[i] = msc_result[i];      // AND, OR, XOR, SLL, SLLI
                3'b100: alu_result[i] = add_result_w[i];    // ADDIW, ADDW
                3'b101: alu_result[i] = sub_result_w[i];    // SUBW
                3'b110: alu_result[i] = shr_result_w[i];    // SRLW, SRAW, SRLIW, SRAIW
                3'b111: alu_result[i] = msc_result_w[i];    // SLLW
            endcase
        end       
    end

    // branch

    wire [`XLEN-1:0] br_dest  = add_result[alu_exe_if.tid][`XLEN-1:0];
    wire [`XLEN:0] cmp_result = sub_result[alu_exe_if.tid][`XLEN:0];
    
    wire is_less  = cmp_result[`XLEN];
    wire is_equal = ~(| cmp_result[`XLEN-1:0]);
    wire is_br_static = `INST_BR_STATIC(br_op);
    
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_jal_result;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign alu_jal_result[i] = (is_br_op && is_br_static) ? alu_exe_if.next_PC : alu_result[i]; 
    end   

    // output

    wire                    alu_valid_in;
    wire                    alu_ready_in;
    wire                    alu_valid_out;
    wire                    alu_ready_out;
    wire [UUID_WIDTH-1:0]   alu_uuid;
    wire [NW_WIDTH-1:0]     alu_wid;
    wire [`NUM_THREADS-1:0] alu_tmask;
    wire [`XLEN-1:0]        alu_PC;
    wire [`NR_BITS-1:0]     alu_rd;   
    wire                    alu_wb; 
    wire [`NUM_THREADS-1:0][`XLEN-1:0] alu_data;

    wire [`INST_BR_BITS-1:0] br_op_r;
    wire [`XLEN-1:0] br_dest_r;
    wire is_less_r;
    wire is_equal_r;
    wire is_br_op_r;

    assign alu_ready_in = alu_ready_out || ~alu_valid_out;

    VX_pipe_register #(
        .DATAW  (1 + UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + (`NUM_THREADS * `XLEN) + 1 + `INST_BR_BITS + 1 + 1 + `XLEN),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (alu_ready_in),
        .data_in  ({alu_valid_in,  alu_exe_if.uuid, alu_exe_if.wid, alu_exe_if.tmask, alu_exe_if.PC, alu_exe_if.rd, alu_exe_if.wb, alu_jal_result, is_br_op,   br_op,   is_less,   is_equal,   br_dest}),
        .data_out ({alu_valid_out, alu_uuid,        alu_wid,        alu_tmask,        alu_PC,        alu_rd,        alu_wb,        alu_data,       is_br_op_r, br_op_r, is_less_r, is_equal_r, br_dest_r})
    );

    `UNUSED_VAR (br_op_r)
    wire is_br_neg_r    = `INST_BR_NEG(br_op_r);
    wire is_br_less_r   = `INST_BR_LESS(br_op_r);
    wire is_br_static_r = `INST_BR_STATIC(br_op_r);

    assign branch_ctl_if.valid = alu_valid_out && alu_ready_out && is_br_op_r;
    assign branch_ctl_if.taken = ((is_br_less_r ? is_less_r : is_equal_r) ^ is_br_neg_r) | is_br_static_r;
    assign branch_ctl_if.wid   = alu_wid;
    assign branch_ctl_if.dest  = br_dest_r[`XLEN-1:0];

`ifdef EXT_M_ENABLE

    wire                    muldiv_valid_in;
    wire                    muldiv_ready_in;
    wire                    muldiv_valid_out;    
    wire                    muldiv_ready_out;
    wire [UUID_WIDTH-1:0]   muldiv_uuid;
    wire [NW_WIDTH-1:0]     muldiv_wid;
    wire [`NUM_THREADS-1:0] muldiv_tmask;
    wire [`XLEN-1:0]        muldiv_PC;
    wire [`NR_BITS-1:0]     muldiv_rd;
    wire                    muldiv_wb;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] muldiv_data;

    wire [`INST_M_BITS-1:0] muldiv_op = `INST_M_BITS'(alu_exe_if.op_type);

    `RESET_RELAY (muldiv_reset, reset);

    VX_muldiv muldiv (
        .clk        (clk),
        .reset      (muldiv_reset),
        
        // Inputs
        .valid_in   (muldiv_valid_in),
        .alu_op     (muldiv_op),
        .op_mod     (alu_exe_if.op_mod),
        .uuid_in    (alu_exe_if.uuid),
        .wid_in     (alu_exe_if.wid),
        .tmask_in   (alu_exe_if.tmask),
        .PC_in      (alu_exe_if.PC),
        .rd_in      (alu_exe_if.rd),
        .wb_in      (alu_exe_if.wb),
        .alu_in1    (alu_exe_if.rs1_data), 
        .alu_in2    (alu_exe_if.rs2_data),
        .ready_in   (muldiv_ready_in),

        // Outputs
        .valid_out  (muldiv_valid_out),
        .wid_out    (muldiv_wid),
        .uuid_out   (muldiv_uuid),
        .tmask_out  (muldiv_tmask),
        .PC_out     (muldiv_PC),
        .rd_out     (muldiv_rd),
        .wb_out     (muldiv_wb),
        .data_out   (muldiv_data),
        .ready_out  (muldiv_ready_out)
    );

    wire is_muldiv_op = `INST_ALU_IS_M(alu_exe_if.op_mod);

    assign alu_valid_in = alu_exe_if.valid && ~is_muldiv_op;
    assign muldiv_valid_in = alu_exe_if.valid && is_muldiv_op;
    assign ready_in = is_muldiv_op ? muldiv_ready_in : alu_ready_in;

`else 

    assign alu_valid_in = alu_exe_if.valid;
    assign ready_in = alu_ready_in;

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
            , muldiv_valid_out
        `endif
        }),
        .ready_in  ({
            alu_ready_out
        `ifdef EXT_M_ENABLE
            , muldiv_ready_out
        `endif
        }),
        .data_in   ({
            {alu_uuid, alu_wid, alu_tmask, alu_PC, alu_rd, alu_wb, alu_data}
        `ifdef EXT_M_ENABLE
            , {muldiv_uuid, muldiv_wid, muldiv_tmask, muldiv_PC, muldiv_rd, muldiv_wb, muldiv_data}
        `endif
        }),
        .data_out  ({alu_commit_if.uuid, alu_commit_if.wid, alu_commit_if.tmask, alu_commit_if.PC, alu_commit_if.rd, alu_commit_if.wb, alu_commit_if.data}),
        .valid_out (alu_commit_if.valid),        
        .ready_out (alu_commit_if.ready)
    );

    assign alu_commit_if.eop = 1'b1;

    // can accept new request?
    assign alu_exe_if.ready = ready_in;

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (branch_ctl_if.valid) begin
            `TRACE(1, ("%d: core%0d-branch: wid=%0d, PC=0x%0h, taken=%b, dest=0x%0h (#%0d)\n",
                $time, CORE_ID, branch_ctl_if.wid, alu_PC, branch_ctl_if.taken, branch_ctl_if.dest, alu_uuid));
        end
    end
`endif

endmodule
