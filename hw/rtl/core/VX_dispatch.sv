`include "VX_define.vh"

module VX_dispatch (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_dispatch_if.slave    dispatch_if,

    // outputs
    VX_alu_exe_if.master    alu_exe_if,
    VX_lsu_exe_if.master    lsu_exe_if,
    VX_csr_exe_if.master    csr_exe_if,
`ifdef EXT_F_ENABLE
    VX_fpu_exe_if.master    fpu_exe_if,
`endif
    VX_gpu_exe_if.master    gpu_exe_if    
);
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

    wire [`UP(`NT_BITS)-1:0] tid;
    wire alu_req_ready;
    wire lsu_req_ready;
    wire csr_req_ready;
`ifdef EXT_F_ENABLE
    wire fpu_req_ready;
`endif
    wire gpu_req_ready;

    VX_lzc #(
        .N       (`NUM_THREADS),
        .REVERSE (1)
    ) tid_select (
        .data_in  (dispatch_if.tmask),
        .data_out (tid),
        `UNUSED_PIN (valid_out)
    );

    wire [`XLEN-1:0] next_PC = dispatch_if.PC + 4;

    // ALU unit

    wire alu_req_valid = dispatch_if.valid && (dispatch_if.ex_type == `EX_ALU);
    wire [`INST_ALU_BITS-1:0] alu_op_type = `INST_ALU_BITS'(dispatch_if.op_type);
    
    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `XLEN + `INST_ALU_BITS + `INST_MOD_BITS + `XLEN + 1 + 1 + `NR_BITS + 1 + `UP(`NT_BITS) + (2 * `NUM_THREADS * `XLEN)),
        .OUT_REG (1)
    ) alu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (alu_req_valid),
        .ready_in  (alu_req_ready),
        .data_in   ({dispatch_if.uuid, dispatch_if.wid, dispatch_if.tmask, dispatch_if.PC, next_PC,            alu_op_type,        dispatch_if.op_mod, dispatch_if.imm, dispatch_if.use_PC, dispatch_if.use_imm, dispatch_if.rd, dispatch_if.wb, tid,            dispatch_if.rs1_data, dispatch_if.rs2_data}),
        .data_out  ({alu_exe_if.uuid,  alu_exe_if.wid,  alu_exe_if.tmask,  alu_exe_if.PC,  alu_exe_if.next_PC, alu_exe_if.op_type, alu_exe_if.op_mod,  alu_exe_if.imm,  alu_exe_if.use_PC,  alu_exe_if.use_imm,  alu_exe_if.rd,  alu_exe_if.wb,  alu_exe_if.tid, alu_exe_if.rs1_data, alu_exe_if.rs2_data}),
        .valid_out (alu_exe_if.valid),
        .ready_out (alu_exe_if.ready)
    );

    // lsu unit

    wire lsu_req_valid = dispatch_if.valid && (dispatch_if.ex_type == `EX_LSU);
    wire [`INST_LSU_BITS-1:0] lsu_op_type = `INST_LSU_BITS'(dispatch_if.op_type);

    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `INST_LSU_BITS + `XLEN + `NR_BITS + 1 + `NUM_THREADS*`XLEN + `NUM_THREADS*`XLEN),
        .OUT_REG (1)
    ) lsu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_req_valid),
        .ready_in  (lsu_req_ready),
        .data_in   ({dispatch_if.uuid, dispatch_if.wid, dispatch_if.tmask, dispatch_if.PC, lsu_op_type,        dispatch_if.imm,    dispatch_if.rd, dispatch_if.wb, dispatch_if.rs1_data,  dispatch_if.rs2_data}),
        .data_out  ({lsu_exe_if.uuid,  lsu_exe_if.wid,  lsu_exe_if.tmask,  lsu_exe_if.PC,  lsu_exe_if.op_type, lsu_exe_if.offset,  lsu_exe_if.rd,  lsu_exe_if.wb,  lsu_exe_if.base_addr, lsu_exe_if.store_data}),
        .valid_out (lsu_exe_if.valid),
        .ready_out (lsu_exe_if.ready)
    );

    // csr unit

    wire csr_req_valid = dispatch_if.valid && (dispatch_if.ex_type == `EX_CSR);
    wire [`INST_CSR_BITS-1:0] csr_op_type = `INST_CSR_BITS'(dispatch_if.op_type);
    wire [`VX_CSR_ADDR_BITS-1:0] csr_addr = dispatch_if.imm[`VX_CSR_ADDR_BITS-1:0];
    wire [`NRI_BITS-1:0] csr_imm = dispatch_if.imm[`VX_CSR_ADDR_BITS +: `NRI_BITS];
    wire [`NUM_THREADS-1:0][31:0] csr_data;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign csr_data[i] = dispatch_if.rs1_data[i][31:0];
    end

    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `INST_CSR_BITS + `VX_CSR_ADDR_BITS + `NR_BITS + 1 + 1 + `NRI_BITS + `UP(`NT_BITS) + (`NUM_THREADS * 32)),
        .OUT_REG (1)
    ) csr_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (csr_req_valid),
        .ready_in  (csr_req_ready),
        .data_in   ({dispatch_if.uuid, dispatch_if.wid, dispatch_if.tmask, dispatch_if.PC, csr_op_type,        csr_addr,        dispatch_if.rd, dispatch_if.wb, dispatch_if.use_imm, csr_imm,        tid,            csr_data}),
        .data_out  ({csr_exe_if.uuid,  csr_exe_if.wid,  csr_exe_if.tmask,  csr_exe_if.PC,  csr_exe_if.op_type, csr_exe_if.addr, csr_exe_if.rd,  csr_exe_if.wb,  csr_exe_if.use_imm,  csr_exe_if.imm, csr_exe_if.tid, csr_exe_if.rs1_data}),
        .valid_out (csr_exe_if.valid),
        .ready_out (csr_exe_if.ready)
    );

    // fpu unit

`ifdef EXT_F_ENABLE
    wire fpu_req_valid = dispatch_if.valid && (dispatch_if.ex_type == `EX_FPU);
    wire [`INST_FPU_BITS-1:0] fpu_op_type = `INST_FPU_BITS'(dispatch_if.op_type);
    wire [`INST_FMT_BITS-1:0] fpu_fmt = dispatch_if.imm[`INST_FMT_BITS-1:0];
    wire [`INST_FRM_BITS-1:0] fpu_frm = dispatch_if.op_mod[`INST_FRM_BITS-1:0];
        
    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `INST_FPU_BITS + `INST_FMT_BITS + `INST_FRM_BITS + `NR_BITS + (3 * `NUM_THREADS * `XLEN)),
        .OUT_REG (1)
    ) fpu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fpu_req_valid),
        .ready_in  (fpu_req_ready),
        .data_in   ({dispatch_if.uuid, dispatch_if.wid, dispatch_if.tmask, dispatch_if.PC, fpu_op_type,        fpu_fmt,        fpu_frm,        dispatch_if.rd, dispatch_if.rs1_data, dispatch_if.rs2_data, dispatch_if.rs3_data}),
        .data_out  ({fpu_exe_if.uuid,  fpu_exe_if.wid,  fpu_exe_if.tmask,  fpu_exe_if.PC,  fpu_exe_if.op_type, fpu_exe_if.fmt, fpu_exe_if.frm, fpu_exe_if.rd,  fpu_exe_if.rs1_data,  fpu_exe_if.rs2_data,  fpu_exe_if.rs3_data}),
        .valid_out (fpu_exe_if.valid),
        .ready_out (fpu_exe_if.ready)
    );
`else
    `UNUSED_VAR (dispatch_if.rs3_data)
`endif

    // gpu unit

    wire gpu_req_valid = dispatch_if.valid && (dispatch_if.ex_type == `EX_GPU);
    wire [`INST_GPU_BITS-1:0] gpu_op_type = `INST_GPU_BITS'(dispatch_if.op_type);

    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `XLEN + `INST_GPU_BITS + `INST_MOD_BITS + `NR_BITS + 1 + `UP(`NT_BITS)  + (3 * `NUM_THREADS * `XLEN)),
        .OUT_REG (1)
    ) gpu_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (gpu_req_valid),
        .ready_in  (gpu_req_ready),
        .data_in   ({dispatch_if.uuid, dispatch_if.wid, dispatch_if.tmask, dispatch_if.PC, next_PC,            gpu_op_type,        dispatch_if.op_mod, dispatch_if.rd, dispatch_if.wb, tid,            dispatch_if.rs1_data, dispatch_if.rs2_data, dispatch_if.rs3_data}),
        .data_out  ({gpu_exe_if.uuid,  gpu_exe_if.wid,  gpu_exe_if.tmask,  gpu_exe_if.PC,  gpu_exe_if.next_PC, gpu_exe_if.op_type, gpu_exe_if.op_mod,  gpu_exe_if.rd,  gpu_exe_if.wb,  gpu_exe_if.tid, gpu_exe_if.rs1_data,  gpu_exe_if.rs2_data,  gpu_exe_if.rs3_data}),
        .valid_out (gpu_exe_if.valid),
        .ready_out (gpu_exe_if.ready)
    ); 

    // can take next request?
    reg ready_r;
    always @(*) begin
        case (dispatch_if.ex_type)
        `EX_LSU: ready_r = lsu_req_ready;
        `EX_CSR: ready_r = csr_req_ready;
    `ifdef EXT_F_ENABLE
        `EX_FPU: ready_r = fpu_req_ready;
    `endif
        `EX_GPU: ready_r = gpu_req_ready;
        //`EX_ALU,
        default: ready_r = alu_req_ready;
        endcase
    end
    assign dispatch_if.ready = ready_r;
    
endmodule
