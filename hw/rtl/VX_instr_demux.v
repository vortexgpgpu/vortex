`include "VX_define.vh"

module VX_instr_demux (
    input wire      clk,
    input wire      reset,

    // inputs
    VX_decode_if    execute_if,
    VX_gpr_read_if  gpr_read_if,
    VX_csr_to_issue_if csr_to_issue_if,

    // outputs
    VX_alu_req_if   alu_req_if,
    VX_lsu_req_if   lsu_req_if,
    VX_csr_req_if   csr_req_if,
    VX_mul_req_if   mul_req_if,
    VX_fpu_req_if   fpu_req_if,
    VX_gpu_req_if   gpu_req_if    
);
    wire [`NT_BITS-1:0] tid;
    VX_priority_encoder #(
        .N(`NUM_THREADS)
    ) tid_select (
        .data_in  (execute_if.thread_mask),
        .data_out (tid),
        `UNUSED_PIN (valid_out)
    );

    wire [31:0] next_PC = execute_if.curr_PC + 4;

    // ALU unit

    wire alu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_ALU);
    wire alu_req_ready;
    wire is_br_op = `IS_BR_MOD(execute_if.op_mod);

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + 32 + `ALU_BR_BITS + 1 + 32 + 1 + 1 + `NR_BITS + 1 + `NT_BITS)
    ) alu_reg (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (alu_req_ready),
        .valid_in  (alu_req_valid),
        .data_in   ({execute_if.wid, execute_if.thread_mask, execute_if.curr_PC, next_PC,            `ALU_BR_OP(execute_if.op_type), is_br_op,            execute_if.imm, execute_if.rs1_is_PC, execute_if.rs2_is_imm, execute_if.rd, execute_if.wb, tid}),
        .data_out  ({alu_req_if.wid, alu_req_if.thread_mask, alu_req_if.curr_PC, alu_req_if.next_PC, alu_req_if.op_type,             alu_req_if.is_br_op, alu_req_if.imm, alu_req_if.rs1_is_PC, alu_req_if.rs2_is_imm, alu_req_if.rd, alu_req_if.wb, alu_req_if.tid}),
        .ready_out (alu_req_if.ready),
        .valid_out (alu_req_if.valid)
    );

    VX_gpr_bypass #(
        .DATAW (2 * `NUM_THREADS * 32),
        .PASSTHRU (1) // ALU has no back-pressure, bypass not needed
    ) alu_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (alu_req_valid && alu_req_ready),
        .data_in   ({gpr_read_if.rs1_data, gpr_read_if.rs2_data}),
        .data_out  ({alu_req_if.rs1_data,  alu_req_if.rs2_data}),
        .pop       (alu_req_if.valid && alu_req_if.ready)
    );

    // lsu unit

    wire lsu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_LSU);
    wire lsu_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + 1 + `BYTEEN_BITS + 32 + `NR_BITS + 1)
    ) lsu_reg (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (lsu_req_ready),
        .valid_in  (lsu_req_valid),
        .data_in   ({execute_if.wid, execute_if.thread_mask, execute_if.curr_PC, `LSU_RW(execute_if.op_type), `LSU_BE(execute_if.op_type), execute_if.imm,    execute_if.rd, execute_if.wb}),
        .data_out  ({lsu_req_if.wid, lsu_req_if.thread_mask, lsu_req_if.curr_PC, lsu_req_if.rw,             lsu_req_if.byteen,         lsu_req_if.offset, lsu_req_if.rd, lsu_req_if.wb}),
        .ready_out (lsu_req_if.ready),
        .valid_out (lsu_req_if.valid)
    );

    VX_gpr_bypass #(
        .DATAW ((2 * `NUM_THREADS * 32))
    ) lsu_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (lsu_req_valid && lsu_req_ready),
        .data_in   ({gpr_read_if.rs1_data, gpr_read_if.rs2_data}),
        .data_out  ({lsu_req_if.base_addr, lsu_req_if.store_data}),
        .pop       (lsu_req_if.valid && lsu_req_if.ready)
    );

    // csr unit

    wire csr_req_valid = execute_if.valid && (execute_if.ex_type == `EX_CSR);
    wire csr_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `CSR_BITS + `CSR_ADDR_BITS + `NR_BITS + 1 + 1)
    ) csr_reg (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (csr_req_ready),
        .valid_in  (csr_req_valid),
        .data_in   ({execute_if.wid, execute_if.thread_mask, execute_if.curr_PC, `CSR_OP(execute_if.op_type), execute_if.imm[`CSR_ADDR_BITS-1:0], execute_if.rd, execute_if.wb, 1'b0}),
        .data_out  ({csr_req_if.wid, csr_req_if.thread_mask, csr_req_if.curr_PC, csr_req_if.op_type,          csr_req_if.csr_addr,                csr_req_if.rd, csr_req_if.wb, csr_req_if.is_io}),
        .ready_out (csr_req_if.ready),
        .valid_out (csr_req_if.valid)
    );
    
    reg tmp_rs2_is_imm;
    reg [`NR_BITS-1:0] tmp_rs1;

    always @(posedge clk) begin
        tmp_rs2_is_imm <= execute_if.rs2_is_imm;
        tmp_rs1        <= execute_if.rs1;
    end

    wire [31:0] csr_req_mask = tmp_rs2_is_imm ? 32'(tmp_rs1) : gpr_read_if.rs1_data[0];

    VX_gpr_bypass #(
        .DATAW (32)
    ) csr_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (csr_req_valid && csr_req_ready),
        .data_in   (csr_req_mask),
        .data_out  (csr_req_if.csr_mask),
        .pop       (csr_req_if.valid && csr_req_if.ready)
    );

    // mul unit

`ifdef EXT_M_ENABLE
    wire mul_req_valid = execute_if.valid && (execute_if.ex_type == `EX_MUL);
    wire mul_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `MUL_BITS + `NR_BITS + 1)
    ) mul_reg (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (mul_req_ready),
        .valid_in  (mul_req_valid),
        .data_in   ({execute_if.wid, execute_if.thread_mask, execute_if.curr_PC, `MUL_OP(execute_if.op_type), execute_if.rd, execute_if.wb}),
        .data_out  ({mul_req_if.wid, mul_req_if.thread_mask, mul_req_if.curr_PC, mul_req_if.op_type,          mul_req_if.rd, mul_req_if.wb}),
        .ready_out (mul_req_if.ready),
        .valid_out (mul_req_if.valid)
    );   

    VX_gpr_bypass #(
        .DATAW ((2 * `NUM_THREADS * 32))
    ) mul_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (mul_req_valid && mul_req_ready),
        .data_in   ({gpr_read_if.rs1_data, gpr_read_if.rs2_data}),
        .data_out  ({mul_req_if.rs1_data,  mul_req_if.rs2_data}),
        .pop       (mul_req_if.valid && mul_req_if.ready)
    );
`endif

    // fpu unit

`ifdef EXT_F_ENABLE
    wire fpu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_FPU);
    wire fpu_req_ready;

    // resolve dynamic FRM    
    assign csr_to_issue_if.wid = execute_if.wid;
    wire [`FRM_BITS-1:0] fpu_frm = (execute_if.op_mod == `FRM_DYN) ? csr_to_issue_if.frm : execute_if.op_mod;    

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `FPU_BITS + `FRM_BITS + `NR_BITS + 1)
    ) fpu_reg (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (fpu_req_ready),
        .valid_in  (fpu_req_valid),                     
        .data_in   ({execute_if.wid, execute_if.thread_mask, execute_if.curr_PC, `FPU_OP(execute_if.op_type), fpu_frm,        execute_if.rd, execute_if.wb}),
        .data_out  ({fpu_req_if.wid, fpu_req_if.thread_mask, fpu_req_if.curr_PC, fpu_req_if.op_type,          fpu_req_if.frm, fpu_req_if.rd, fpu_req_if.wb}),
        .ready_out (fpu_req_if.ready),
        .valid_out (fpu_req_if.valid)
    );

    VX_gpr_bypass #(
        .DATAW ((3 * `NUM_THREADS * 32))
    ) fpu_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (fpu_req_valid && fpu_req_ready),
        .data_in   ({gpr_read_if.rs1_data, gpr_read_if.rs2_data, gpr_read_if.rs3_data}),
        .data_out  ({fpu_req_if.rs1_data,  fpu_req_if.rs2_data,  fpu_req_if.rs3_data}),
        .pop       (fpu_req_if.valid && fpu_req_if.ready)
    );
`endif

    // gpu unit

    wire gpu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_GPU);
    wire gpu_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + 32 + `GPU_BITS + `NR_BITS + 1)
    ) gpu_reg (
        .clk       (clk),
        .reset     (reset),
        .ready_in  (gpu_req_ready),
        .valid_in  (gpu_req_valid),
        .data_in   ({execute_if.wid, execute_if.thread_mask, execute_if.curr_PC, next_PC,            `GPU_OP(execute_if.op_type), execute_if.rd, execute_if.wb}),
        .data_out  ({gpu_req_if.wid, gpu_req_if.thread_mask, gpu_req_if.curr_PC, gpu_req_if.next_PC, gpu_req_if.op_type,          gpu_req_if.rd, gpu_req_if.wb}),
        .ready_out (gpu_req_if.ready),
        .valid_out (gpu_req_if.valid)
    ); 

    VX_gpr_bypass #(
        .DATAW ((`NUM_THREADS * 32) + 32)
    ) gpu_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (gpu_req_valid && gpu_req_ready),
        .data_in   ({gpr_read_if.rs1_data, gpr_read_if.rs2_data[0]}),
        .data_out  ({gpu_req_if.rs1_data,  gpu_req_if.rs2_data}),
        .pop       (gpu_req_if.valid && gpu_req_if.ready)
    );

    // can take next request?
    assign execute_if.ready = (alu_req_ready && (execute_if.ex_type == `EX_ALU))
                           || (lsu_req_ready && (execute_if.ex_type == `EX_LSU))
                           || (csr_req_ready && (execute_if.ex_type == `EX_CSR))
                       `ifdef EXT_M_ENABLE
                           || (mul_req_ready && (execute_if.ex_type == `EX_MUL))
                       `endif
                       `ifdef EXT_F_ENABLE
                           || (fpu_req_ready && (execute_if.ex_type == `EX_FPU))
                       `endif
                           || (gpu_req_ready && (execute_if.ex_type == `EX_GPU));    
    
endmodule