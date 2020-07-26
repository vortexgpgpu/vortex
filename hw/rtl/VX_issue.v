`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_wb_if            writeback_if,
    VX_commit_is_if     commit_is_if, 
    
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
    wire [`ISTAG_BITS-1:0] issue_tag, issue_tmp_tag;

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
        .commit_is_if   (commit_is_if), 
        .gpr_busy       (gpr_delay),
        .alu_busy       (alu_busy),
        .lsu_busy       (lsu_busy),
        .csr_busy       (csr_busy),
        .mul_busy       (mul_busy),
        .fpu_busy       (fpu_busy),
        .gpu_busy       (gpu_busy),  
        .issue_tag      (issue_tag),    
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

    VX_decode_if    decode_tmp_if();
    VX_gpr_data_if  gpr_data_tmp_if();

    wire stall = ~alu_req_if.ready || schedule_delay;
    wire flush = alu_req_if.ready && schedule_delay;  

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + `NUM_THREADS + 32 + 32 + `NR_BITS + `NR_BITS + `NR_BITS + 32 + 1 + 1 + 1 + 1 + `EX_BITS + `OP_BITS + 1 + `NR_BITS + 1 + 1 + 1 + `FRM_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + (`NUM_THREADS * 32))
    ) decode_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (flush),
        .in    ({decode_if.valid,     issue_tag,     decode_if.warp_num,     decode_if.thread_mask,     decode_if.curr_PC,     decode_if.next_PC,     decode_if.rd,     decode_if.rs1,     decode_if.rs2,     decode_if.imm,     decode_if.rs1_is_PC,     decode_if.rs2_is_imm,     decode_if.use_rs1,     decode_if.use_rs2,     decode_if.ex_type,     decode_if.instr_op,     decode_if.wb,     decode_if.rs3,     decode_if.use_rs3,     decode_if.rs1_is_fp,     decode_if.rs2_is_fp,     decode_if.frm,    gpr_data_if.rs1_data,      gpr_data_if.rs2_data,     gpr_data_if.rs3_data}),
        .out   ({decode_tmp_if.valid, issue_tmp_tag, decode_tmp_if.warp_num, decode_tmp_if.thread_mask, decode_tmp_if.curr_PC, decode_tmp_if.next_PC, decode_tmp_if.rd, decode_tmp_if.rs1, decode_tmp_if.rs2, decode_tmp_if.imm, decode_tmp_if.rs1_is_PC, decode_tmp_if.rs2_is_imm, decode_tmp_if.use_rs1, decode_tmp_if.use_rs2, decode_tmp_if.ex_type, decode_tmp_if.instr_op, decode_tmp_if.wb, decode_tmp_if.rs3, decode_tmp_if.use_rs3, decode_tmp_if.rs1_is_fp, decode_tmp_if.rs2_is_fp, decode_tmp_if.frm, gpr_data_tmp_if.rs1_data, gpr_data_tmp_if.rs2_data, gpr_data_tmp_if.rs3_data})
    );

    VX_issue_demux issue_demux (
        .decode_if     (decode_tmp_if),
        .gpr_data_if   (gpr_data_tmp_if),
        .issue_tag     (issue_tmp_tag),
        .alu_req_if    (alu_req_if),
        .lsu_req_if    (lsu_req_if),        
        .csr_req_if    (csr_req_if),
        .mul_req_if    (mul_req_if),
        .fpu_req_if    (fpu_req_if),
        .gpu_req_if    (gpu_req_if)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_req_if.valid && ~stall) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=ALU, istag=%0d, tmask=%b, wb=%d, rd=%0d, rs1_data=%0h, rs2_data=%0h, offset=%0h, next_PC=%0h", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, issue_tmp_tag, decode_tmp_if.thread_mask, decode_tmp_if.wb, decode_tmp_if.rd, alu_req_if.rs1_data, alu_req_if.rs2_data, alu_req_if.offset, alu_req_if.next_PC);   
        end
        if (lsu_req_if.valid && ~stall) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=LSU, istag=%0d, tmask=%b, wb=%0d, rd=%0d, byteen=%b, baddr=%0h, offset=%0h", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, issue_tmp_tag, decode_tmp_if.thread_mask, lsu_req_if.rw, decode_tmp_if.rd, decode_tmp_if.wb, lsu_req_if.byteen, lsu_req_if.base_addr, lsu_req_if.offset);   
        end
        if (csr_req_if.valid && ~stall) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=CSR, istag=%0d, tmask=%b, wb=%d, rd=%0d, addr=%0h, mask=%0h", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, issue_tmp_tag, decode_tmp_if.thread_mask, decode_tmp_if.wb, decode_tmp_if.rd, csr_req_if.csr_addr, csr_req_if.csr_mask);   
        end
        if (mul_req_if.valid && ~stall) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=MUL, istag=%0d, tmask=%b, wb=%d, rd=%0d, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, issue_tmp_tag, decode_tmp_if.thread_mask, decode_tmp_if.wb, decode_tmp_if.rd, mul_req_if.rs1_data, mul_req_if.rs2_data);   
        end
        if (fpu_req_if.valid && ~stall) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=FPU, istag=%0d, tmask=%b, wb=%d, rd=%0d, frm=%0h, rs1_data=%0h, rs2_data=%0h, rs3_data=%0h", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, issue_tmp_tag, decode_tmp_if.thread_mask, decode_tmp_if.wb, decode_tmp_if.rd, fpu_req_if.frm, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data);   
        end
        if (gpu_req_if.valid && ~stall) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=GPU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, issue_tmp_tag, decode_tmp_if.thread_mask, gpu_req_if.rs1_data, gpu_req_if.rs2_data);   
        end
    end
`endif

endmodule