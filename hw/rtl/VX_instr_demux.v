`include "VX_define.vh"

module VX_instr_demux (
    input wire      clk,
    input wire      reset,

    // inputs
    VX_decode_if    execute_if,
    VX_gpr_rsp_if   gpr_rsp_if,

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
        .data_in  (execute_if.tmask),
        .data_out (tid),
        `UNUSED_PIN (valid_out)
    );

    wire [31:0] next_PC = execute_if.PC + 4;

    // ALU unit

    wire alu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_ALU);
    wire alu_req_ready;

    wire is_br_op = `IS_BR_MOD(execute_if.op_mod);

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + 32 + `ALU_BR_BITS + 1 + 32 + 1 + 1 + `NR_BITS + 1 + `NT_BITS + (2 * `NUM_THREADS * 32)),
        .NOBACKPRESSURE (1) // ALU has no back pressure
    ) alu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (alu_req_valid),
        .ready_in  (alu_req_ready),
        .data_in   ({execute_if.wid, execute_if.tmask, execute_if.PC, next_PC,            `ALU_BR_OP(execute_if.op_type), is_br_op,            execute_if.imm, execute_if.rs1_is_PC, execute_if.rs2_is_imm, execute_if.rd, execute_if.wb, tid,            gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data}),
        .data_out  ({alu_req_if.wid, alu_req_if.tmask, alu_req_if.PC, alu_req_if.next_PC, alu_req_if.op_type,             alu_req_if.is_br_op, alu_req_if.imm, alu_req_if.rs1_is_PC, alu_req_if.rs2_is_imm, alu_req_if.rd, alu_req_if.wb, alu_req_if.tid, alu_req_if.rs1_data, alu_req_if.rs2_data}),
        .valid_out (alu_req_if.valid),
        .ready_out (alu_req_if.ready)
    );

    // lsu unit

    wire lsu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_LSU);
    wire lsu_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + 1 + `BYTEEN_BITS + 32 + `NR_BITS + 1 + (2 * `NUM_THREADS * 32))
    ) lsu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_req_valid),
        .ready_in  (lsu_req_ready),
        .data_in   ({execute_if.wid, execute_if.tmask, execute_if.PC, `LSU_RW(execute_if.op_type), `LSU_BE(execute_if.op_type), execute_if.imm,    execute_if.rd, execute_if.wb, gpr_rsp_if.rs1_data,  gpr_rsp_if.rs2_data}),
        .data_out  ({lsu_req_if.wid, lsu_req_if.tmask, lsu_req_if.PC, lsu_req_if.rw,               lsu_req_if.byteen,           lsu_req_if.offset, lsu_req_if.rd, lsu_req_if.wb, lsu_req_if.base_addr, lsu_req_if.store_data}),
        .valid_out (lsu_req_if.valid),
        .ready_out (lsu_req_if.ready)
    );

    // csr unit

    wire csr_req_valid = execute_if.valid && (execute_if.ex_type == `EX_CSR);
    wire csr_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `CSR_BITS + `CSR_ADDR_BITS + `NR_BITS + 1 + 1 + `NR_BITS + 32)
    ) csr_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_req_valid),
        .ready_in  (csr_req_ready),
        .data_in   ({execute_if.wid, execute_if.tmask, execute_if.PC, `CSR_OP(execute_if.op_type), execute_if.imm[`CSR_ADDR_BITS-1:0], execute_if.rd, execute_if.wb, execute_if.rs2_is_imm, execute_if.rs1, gpr_rsp_if.rs1_data[0]}),
        .data_out  ({csr_req_if.wid, csr_req_if.tmask, csr_req_if.PC, csr_req_if.op_type,          csr_req_if.csr_addr,                csr_req_if.rd, csr_req_if.wb, csr_req_if.rs2_is_imm, csr_req_if.rs1, csr_req_if.rs1_data}),
        .valid_out (csr_req_if.valid),
        .ready_out (csr_req_if.ready)
    );

    // mul unit

`ifdef EXT_M_ENABLE
    wire mul_req_valid = execute_if.valid && (execute_if.ex_type == `EX_MUL);
    wire mul_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `MUL_BITS + `NR_BITS + 1 + (2 * `NUM_THREADS * 32))
    ) mul_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mul_req_valid),
        .ready_in  (mul_req_ready),
        .data_in   ({execute_if.wid, execute_if.tmask, execute_if.PC, `MUL_OP(execute_if.op_type), execute_if.rd, execute_if.wb, gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data}),
        .data_out  ({mul_req_if.wid, mul_req_if.tmask, mul_req_if.PC, mul_req_if.op_type,          mul_req_if.rd, mul_req_if.wb, mul_req_if.rs1_data, mul_req_if.rs2_data}),
        .valid_out (mul_req_if.valid),
        .ready_out (mul_req_if.ready)
    );   
`endif

    // fpu unit

`ifdef EXT_F_ENABLE
    wire fpu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_FPU);
    wire fpu_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `FPU_BITS + `MOD_BITS + `NR_BITS + 1 + (3 * `NUM_THREADS * 32))
    ) fpu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fpu_req_valid),
        .ready_in  (fpu_req_ready),
        .data_in   ({execute_if.wid, execute_if.tmask, execute_if.PC, `FPU_OP(execute_if.op_type), execute_if.op_mod, execute_if.rd, execute_if.wb, gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data, gpr_rsp_if.rs3_data}),
        .data_out  ({fpu_req_if.wid, fpu_req_if.tmask, fpu_req_if.PC, fpu_req_if.op_type,          fpu_req_if.op_mod, fpu_req_if.rd, fpu_req_if.wb, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data}),
        .valid_out (fpu_req_if.valid),
        .ready_out (fpu_req_if.ready)
    );
`else
    `UNUSED_VAR (gpr_rsp_if.rs3_data)
`endif

    // gpu unit

    wire gpu_req_valid = execute_if.valid && (execute_if.ex_type == `EX_GPU);
    wire gpu_req_ready;

    VX_skid_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + 32 + `GPU_BITS + `NR_BITS + 1 + (`NUM_THREADS * 32 + 32))
    ) gpu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (gpu_req_valid),
        .ready_in  (gpu_req_ready),
        .data_in   ({execute_if.wid, execute_if.tmask, execute_if.PC, next_PC,            `GPU_OP(execute_if.op_type), execute_if.rd, execute_if.wb, gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data[0]}),
        .data_out  ({gpu_req_if.wid, gpu_req_if.tmask, gpu_req_if.PC, gpu_req_if.next_PC, gpu_req_if.op_type,          gpu_req_if.rd, gpu_req_if.wb, gpu_req_if.rs1_data, gpu_req_if.rs2_data}),
        .valid_out (gpu_req_if.valid),
        .ready_out (gpu_req_if.ready)
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