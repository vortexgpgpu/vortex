`include "VX_define.vh"

module VX_dispatch (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_ibuffer_if.slave     ibuffer_if,
    VX_gpr_rsp_if.slave     gpr_rsp_if,

    // outputs
    VX_alu_req_if.master    alu_req_if,
    VX_lsu_req_if.master    lsu_req_if,
    VX_csr_req_if.master    csr_req_if,
`ifdef EXT_F_ENABLE
    VX_fpu_req_if.master    fpu_req_if,
`endif
    VX_gpu_req_if.master    gpu_req_if    
);
    wire [`NT_BITS-1:0] tid;
    wire alu_req_ready;
    wire lsu_req_ready;
    wire csr_req_ready;
`ifdef EXT_F_ENABLE
    wire fpu_req_ready;
`endif
    wire gpu_req_ready;

    VX_lzc #(
        .N (`NUM_THREADS)
    ) tid_select (
        .in_i       (ibuffer_if.tmask),
        .cnt_o      (tid),
        `UNUSED_PIN (valid_o)
    );

    wire [31:0] next_PC = ibuffer_if.PC + 4;

    // ALU unit

    wire alu_req_valid = ibuffer_if.valid && (ibuffer_if.ex_type == `EX_ALU);
    wire [`INST_ALU_BITS-1:0] alu_op_type = `INST_ALU_BITS'(ibuffer_if.op_type);
    
    VX_skid_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + 32 + `INST_ALU_BITS + `INST_MOD_BITS + 32 + 1 + 1 + `NR_BITS + 1 + `NT_BITS + (2 * `NUM_THREADS * 32)),
        .OUT_REG (1)
    ) alu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (alu_req_valid),
        .ready_in  (alu_req_ready),
        .data_in   ({ibuffer_if.wid, ibuffer_if.tmask, ibuffer_if.PC, next_PC,            alu_op_type,        ibuffer_if.op_mod, ibuffer_if.imm, ibuffer_if.use_PC, ibuffer_if.use_imm, ibuffer_if.rd, ibuffer_if.wb, tid,            gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data}),
        .data_out  ({alu_req_if.wid, alu_req_if.tmask, alu_req_if.PC, alu_req_if.next_PC, alu_req_if.op_type, alu_req_if.op_mod, alu_req_if.imm, alu_req_if.use_PC, alu_req_if.use_imm, alu_req_if.rd, alu_req_if.wb, alu_req_if.tid, alu_req_if.rs1_data, alu_req_if.rs2_data}),
        .valid_out (alu_req_if.valid),
        .ready_out (alu_req_if.ready)
    );

    // lsu unit

    wire lsu_req_valid = ibuffer_if.valid && (ibuffer_if.ex_type == `EX_LSU);
    wire [`INST_LSU_BITS-1:0] lsu_op_type = `INST_LSU_BITS'(ibuffer_if.op_type);
    wire lsu_is_fence = `INST_LSU_IS_FENCE(ibuffer_if.op_mod);
    wire lsu_is_prefetch = `INST_LSU_IS_PREFETCH(ibuffer_if.op_mod);

    VX_skid_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + `INST_LSU_BITS + 1 + 32 + `NR_BITS + 1 + (2 * `NUM_THREADS * 32) + 1),
        .OUT_REG (1)
    ) lsu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_req_valid),
        .ready_in  (lsu_req_ready),
        .data_in   ({ibuffer_if.wid, ibuffer_if.tmask, ibuffer_if.PC, lsu_op_type,        lsu_is_fence,        ibuffer_if.imm,    ibuffer_if.rd, ibuffer_if.wb, gpr_rsp_if.rs1_data,  gpr_rsp_if.rs2_data, lsu_is_prefetch}),
        .data_out  ({lsu_req_if.wid, lsu_req_if.tmask, lsu_req_if.PC, lsu_req_if.op_type, lsu_req_if.is_fence, lsu_req_if.offset, lsu_req_if.rd, lsu_req_if.wb, lsu_req_if.base_addr, lsu_req_if.store_data, lsu_req_if.is_prefetch}),
        .valid_out (lsu_req_if.valid),
        .ready_out (lsu_req_if.ready)
    );

    // csr unit

    wire csr_req_valid = ibuffer_if.valid && (ibuffer_if.ex_type == `EX_CSR);
    wire [`INST_CSR_BITS-1:0] csr_op_type = `INST_CSR_BITS'(ibuffer_if.op_type);
    wire [`CSR_ADDR_BITS-1:0] csr_addr = ibuffer_if.imm[`CSR_ADDR_BITS-1:0];
    wire [`NRI_BITS-1:0] csr_imm = ibuffer_if.imm[`CSR_ADDR_BITS +: `NRI_BITS];
    wire [31:0] csr_rs1_data = gpr_rsp_if.rs1_data[tid];

    VX_skid_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + `INST_CSR_BITS + `CSR_ADDR_BITS + `NR_BITS + 1 + 1 + `NRI_BITS + 32),
        .OUT_REG (1)
    ) csr_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_req_valid),
        .ready_in  (csr_req_ready),
        .data_in   ({ibuffer_if.wid, ibuffer_if.tmask, ibuffer_if.PC, csr_op_type,        csr_addr,        ibuffer_if.rd, ibuffer_if.wb, ibuffer_if.use_imm, csr_imm,        csr_rs1_data}),
        .data_out  ({csr_req_if.wid, csr_req_if.tmask, csr_req_if.PC, csr_req_if.op_type, csr_req_if.addr, csr_req_if.rd, csr_req_if.wb, csr_req_if.use_imm, csr_req_if.imm, csr_req_if.rs1_data}),
        .valid_out (csr_req_if.valid),
        .ready_out (csr_req_if.ready)
    );

    // fpu unit

`ifdef EXT_F_ENABLE
    wire fpu_req_valid = ibuffer_if.valid && (ibuffer_if.ex_type == `EX_FPU);
    wire [`INST_FPU_BITS-1:0] fpu_op_type = `INST_FPU_BITS'(ibuffer_if.op_type);
        
    VX_skid_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + `INST_FPU_BITS + `INST_MOD_BITS + `NR_BITS + 1 + (3 * `NUM_THREADS * 32)),
        .OUT_REG (1)
    ) fpu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fpu_req_valid),
        .ready_in  (fpu_req_ready),
        .data_in   ({ibuffer_if.wid, ibuffer_if.tmask, ibuffer_if.PC, fpu_op_type,        ibuffer_if.op_mod, ibuffer_if.rd, ibuffer_if.wb, gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data, gpr_rsp_if.rs3_data}),
        .data_out  ({fpu_req_if.wid, fpu_req_if.tmask, fpu_req_if.PC, fpu_req_if.op_type, fpu_req_if.op_mod, fpu_req_if.rd, fpu_req_if.wb, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data}),
        .valid_out (fpu_req_if.valid),
        .ready_out (fpu_req_if.ready)
    );
`else
    `UNUSED_VAR (gpr_rsp_if.rs3_data)
`endif

    // gpu unit

    wire gpu_req_valid = ibuffer_if.valid && (ibuffer_if.ex_type == `EX_GPU);
    wire [`INST_GPU_BITS-1:0] gpu_op_type = `INST_GPU_BITS'(ibuffer_if.op_type);

    VX_skid_buffer #(
        .DATAW   (`NW_BITS + `NUM_THREADS + 32 + 32 + `INST_GPU_BITS + `INST_MOD_BITS + `NR_BITS + 1 + `NT_BITS  + (3 * `NUM_THREADS * 32)),
        .OUT_REG (1)
    ) gpu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (gpu_req_valid),
        .ready_in  (gpu_req_ready),
        .data_in   ({ibuffer_if.wid, ibuffer_if.tmask, ibuffer_if.PC, next_PC,            gpu_op_type,        ibuffer_if.op_mod, ibuffer_if.rd, ibuffer_if.wb, tid,            gpr_rsp_if.rs1_data, gpr_rsp_if.rs2_data, gpr_rsp_if.rs3_data}),
        .data_out  ({gpu_req_if.wid, gpu_req_if.tmask, gpu_req_if.PC, gpu_req_if.next_PC, gpu_req_if.op_type, gpu_req_if.op_mod, gpu_req_if.rd, gpu_req_if.wb, gpu_req_if.tid, gpu_req_if.rs1_data, gpu_req_if.rs2_data, gpu_req_if.rs3_data}),
        .valid_out (gpu_req_if.valid),
        .ready_out (gpu_req_if.ready)
    ); 

    // can take next request?
    reg ready_r;
    always @(*) begin
        case (ibuffer_if.ex_type) 
        `EX_ALU: ready_r = alu_req_ready;
        `EX_LSU: ready_r = lsu_req_ready;
        `EX_CSR: ready_r = csr_req_ready;
    `ifdef EXT_F_ENABLE
        `EX_FPU: ready_r = fpu_req_ready;
    `endif
        `EX_GPU: ready_r = gpu_req_ready;
        default: ready_r = 1'b1; // ignore NOPs
        endcase
    end
    assign ibuffer_if.ready = ready_r;
    
endmodule