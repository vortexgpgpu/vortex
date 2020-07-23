`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_wb_if            writeback_if,
    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,
    VX_mul_req_if       mul_req_if,    
    VX_fpu_req_if       fpu_req_if,    
    VX_gpu_req_if       gpu_req_if
);
    VX_gpr_data_if gpr_data_if();
    wire schedule_delay;
    wire gpr_delay;

    wire alu_busy = ~alu_req_if.ready; 
    wire lsu_busy = ~lsu_req_if.ready;
    wire csr_busy = ~csr_req_if.ready;
    wire mul_busy = ~mul_req_if.ready;
    wire fpu_busy = ~mul_req_if.ready;
    wire gpu_busy = ~gpu_req_if.ready;

    VX_scheduler #(
        .CORE_ID(CORE_ID)
    ) scheduler (
        .clk            (clk),
        .reset          (reset), 
        .decode_if      (decode_if),
        .writeback_if   (writeback_if), 
        .gpr_busy       (gpr_delay),
        .alu_busy       (alu_busy),
        .lsu_busy       (lsu_busy),
        .csr_busy       (csr_busy),
        .mul_busy       (mul_busy),
        .fpu_busy       (fpu_busy),
        .gpu_busy       (gpu_busy),      
        .schedule_delay (schedule_delay),        
        `UNUSED_PIN     (is_empty)
    );

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk            (clk),      
        .reset          (reset),     
        .decode_if      (decode_if),        
        .writeback_if   (writeback_if),
        .gpr_data_if    (gpr_data_if),
        .schedule_delay (schedule_delay),
        .gpr_delay      (gpr_delay)
    );

    VX_alu_req_if   alu_req_tmp_if();
    VX_lsu_req_if   lsu_req_tmp_if();
    VX_csr_req_if   csr_req_tmp_if();
    VX_mul_req_if   mul_req_tmp_if();
    VX_fpu_req_if   fpu_req_tmp_if();
    VX_gpu_req_if   gpu_req_tmp_if();    

    VX_issue_mux issue_mux (
        .decode_if     (decode_if),
        .gpr_data_if   (gpr_data_if),
        .alu_req_if    (alu_req_tmp_if),
        .lsu_req_if    (lsu_req_tmp_if),        
        .csr_req_if    (csr_req_tmp_if),
        .mul_req_if    (mul_req_tmp_if),
        .fpu_req_if    (fpu_req_tmp_if),
        .gpu_req_if    (gpu_req_tmp_if)
    );

    wire stall_alu = ~alu_req_if.ready || schedule_delay;
    wire stall_lsu = ~lsu_req_if.ready || schedule_delay;
    wire stall_csr = ~csr_req_if.ready || schedule_delay;
    wire stall_mul = ~mul_req_if.ready || schedule_delay;
    wire stall_fpu = ~fpu_req_if.ready || schedule_delay;
    wire stall_gpu = ~gpu_req_if.ready || schedule_delay;

    wire flush_alu = alu_req_if.ready && schedule_delay;    
    wire flush_lsu = lsu_req_if.ready && schedule_delay;
    wire flush_csr = csr_req_if.ready && schedule_delay;
    wire flush_mul = mul_req_if.ready && schedule_delay;
    wire flush_fpu = fpu_req_if.ready && schedule_delay;
    wire flush_gpu = gpu_req_if.ready && schedule_delay;

    VX_generic_register #(
        .N(`NUM_THREADS +`NW_BITS + 32 + `ALU_BITS + 1 + `NR_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + 32 + 32)
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_alu),
        .flush (flush_alu),
        .in    ({alu_req_tmp_if.valid, alu_req_tmp_if.warp_num, alu_req_tmp_if.curr_PC, alu_req_tmp_if.alu_op, alu_req_tmp_if.wb, alu_req_tmp_if.rd, alu_req_tmp_if.rs1_data, alu_req_tmp_if.rs2_data, alu_req_tmp_if.offset, alu_req_tmp_if.next_PC}),
        .out   ({alu_req_if.valid,     alu_req_if.warp_num,     alu_req_if.curr_PC,     alu_req_if.alu_op,     alu_req_if.wb,     alu_req_if.rd,     alu_req_if.rs1_data,     alu_req_if.rs2_data,     alu_req_if.offset,     alu_req_if.next_PC})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + 1 + `BYTEEN_BITS + 1 + `NR_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + 32)
    ) lsu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_lsu),
        .flush (flush_lsu),
        .in    ({lsu_req_tmp_if.valid, lsu_req_tmp_if.warp_num, lsu_req_tmp_if.curr_PC, lsu_req_tmp_if.rw, lsu_req_tmp_if.byteen, lsu_req_tmp_if.wb, lsu_req_tmp_if.rd, lsu_req_tmp_if.base_addr, lsu_req_tmp_if.offset, lsu_req_tmp_if.store_data}),
        .out   ({lsu_req_if.valid,     lsu_req_if.warp_num,     lsu_req_if.curr_PC,     lsu_req_if.rw,     lsu_req_if.byteen,     lsu_req_if.wb,     lsu_req_if.rd,     lsu_req_if.base_addr,     lsu_req_if.offset,     lsu_req_if.store_data})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `CSR_BITS + 1 + `NR_BITS + `CSR_ADDR_SIZE + 32 + 1)
    ) csr_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_csr),
        .flush (flush_csr),
        .in    ({csr_req_tmp_if.valid, csr_req_tmp_if.warp_num, csr_req_tmp_if.curr_PC, csr_req_tmp_if.csr_op, csr_req_tmp_if.wb, csr_req_tmp_if.rd, csr_req_tmp_if.csr_addr, csr_req_tmp_if.csr_mask, csr_req_tmp_if.is_io}),
        .out   ({csr_req_if.valid,     csr_req_if.warp_num,     csr_req_if.curr_PC,     csr_req_if.csr_op,     csr_req_if.wb,     csr_req_if.rd,     csr_req_if.csr_addr,     csr_req_if.csr_mask,     csr_req_if.is_io})
    );

    VX_generic_register #(
        .N(`NUM_THREADS +`NW_BITS + 32 + `MUL_BITS + 1 + `NR_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32))
    ) mul_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_mul),
        .flush (flush_mul),
        .in    ({mul_req_tmp_if.valid, mul_req_tmp_if.warp_num, mul_req_tmp_if.curr_PC, mul_req_tmp_if.mul_op, mul_req_tmp_if.wb, mul_req_tmp_if.rd, mul_req_tmp_if.rs1_data, mul_req_tmp_if.rs2_data}),
        .out   ({mul_req_if.valid,     mul_req_if.warp_num,     mul_req_if.curr_PC,     mul_req_if.mul_op,     mul_req_if.wb,     mul_req_if.rd,     mul_req_if.rs1_data,     mul_req_if.rs2_data})
    );

    VX_generic_register #(
        .N(`NUM_THREADS +`NW_BITS + 32 + `FPU_BITS + 1 + `NR_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + `FRM_BITS)
    ) fpu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_fpu),
        .flush (flush_fpu),
        .in    ({fpu_req_tmp_if.valid, fpu_req_tmp_if.warp_num, fpu_req_tmp_if.curr_PC, fpu_req_tmp_if.fpu_op, fpu_req_tmp_if.wb, fpu_req_tmp_if.rd, fpu_req_tmp_if.rs1_data, fpu_req_tmp_if.rs2_data, fpu_req_tmp_if.rs3_data, fpu_req_tmp_if.frm}),
        .out   ({fpu_req_if.valid,     fpu_req_if.warp_num,     fpu_req_if.curr_PC,     fpu_req_if.fpu_op,     fpu_req_if.wb,     fpu_req_if.rd,     fpu_req_if.rs1_data,     fpu_req_if.rs2_data,     fpu_req_if.rs3_data,     fpu_req_if.frm})
    );

    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + `GPU_BITS + (`NUM_THREADS * 32) + 32 + 32)
    ) gpu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_gpu),
        .flush (flush_gpu),
        .in    ({gpu_req_tmp_if.valid, gpu_req_tmp_if.warp_num, gpu_req_tmp_if.curr_PC, gpu_req_tmp_if.gpu_op, gpu_req_tmp_if.rs1_data, gpu_req_tmp_if.rs2_data, gpu_req_tmp_if.next_PC}),
        .out   ({gpu_req_if.valid,     gpu_req_if.warp_num,     gpu_req_if.curr_PC,     gpu_req_if.gpu_op,     gpu_req_if.rs1_data,     gpu_req_if.rs2_data,     gpu_req_if.next_PC})
    );    

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if ((| alu_req_tmp_if.valid) && ~stall_alu) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=ALU, op=%0d, wb=%d, rd=%0d, rs1=%0h, rs2=%0h, offset=%0h, next_PC=%0h", $time, CORE_ID, alu_req_tmp_if.warp_num, alu_req_tmp_if.curr_PC, alu_req_tmp_if.alu_op, alu_req_tmp_if.wb, alu_req_tmp_if.rd, alu_req_tmp_if.rs1_data, alu_req_tmp_if.rs2_data, alu_req_tmp_if.offset, alu_req_tmp_if.next_PC);   
        end
        if ((| mul_req_tmp_if.valid) && ~stall_mul) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=MUL, op=%0d, wb=%d, rd=%0d, rs1=%0h, rs2=%0h", $time, CORE_ID, mul_req_tmp_if.warp_num, mul_req_tmp_if.curr_PC, mul_req_tmp_if.mul_op, mul_req_tmp_if.wb, mul_req_tmp_if.rd, mul_req_tmp_if.rs1_data, mul_req_tmp_if.rs2_data);   
        end
        if ((| fpu_req_tmp_if.valid) && ~stall_fpu) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=MUL, op=%0d, wb=%d, rd=%0d, rs1=%0h, rs2=%0h", $time, CORE_ID, fpu_req_tmp_if.warp_num, fpu_req_tmp_if.curr_PC, fpu_req_tmp_if.fpu_op, fpu_req_tmp_if.wb, fpu_req_tmp_if.rd, fpu_req_tmp_if.rs1_data, fpu_req_tmp_if.rs2_data);   
        end
        if ((| lsu_req_tmp_if.valid) && ~stall_lsu) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=LSU, rw=%b, wb=%0d, rd=%0d, byteen=%b, baddr=%0h, offset=%0h", $time, CORE_ID, lsu_req_tmp_if.warp_num, lsu_req_tmp_if.curr_PC, lsu_req_tmp_if.rw, lsu_req_tmp_if.rd, lsu_req_tmp_if.wb, lsu_req_tmp_if.byteen, lsu_req_tmp_if.base_addr, lsu_req_tmp_if.offset);   
        end
        if ((| csr_req_tmp_if.valid) && ~stall_csr) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=CSR, op=%0d, wb=%d, rd=%0d, addr=%0h, mask=%0h", $time, CORE_ID, csr_req_tmp_if.warp_num, csr_req_tmp_if.curr_PC, csr_req_tmp_if.csr_op, csr_req_tmp_if.wb, csr_req_tmp_if.rd, csr_req_tmp_if.csr_addr, csr_req_tmp_if.csr_mask);   
        end
        if ((| gpu_req_tmp_if.valid) && ~stall_gpu) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=GPU, op=%0d, rs1=%0h, rs2=%0h", $time, CORE_ID, gpu_req_tmp_if.warp_num, gpu_req_tmp_if.curr_PC, gpu_req_tmp_if.gpu_op, gpu_req_tmp_if.rs1_data, gpu_req_tmp_if.rs2_data);   
        end
    end
`endif

endmodule