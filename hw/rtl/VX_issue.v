`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    VX_decode_if        decode_if,
    VX_wb_if            writeback_if,
    VX_cmt_to_issue_if  cmt_to_issue_if, 
    
    VX_alu_req_if       alu_req_if,
    VX_lsu_req_if       lsu_req_if,    
    VX_csr_req_if       csr_req_if,
    VX_mul_req_if       mul_req_if,    
    VX_fpu_req_if       fpu_req_if,    
    VX_gpu_req_if       gpu_req_if
);

    wire [`ISTAG_BITS-1:0] issue_tag;
    wire schedule_delay;

    VX_gpr_read_if  gpr_read_if();
    assign gpr_read_if.valid     = decode_if.valid && ~schedule_delay;
    assign gpr_read_if.warp_num  = decode_if.warp_num;
    assign gpr_read_if.rs1       = decode_if.rs1;
    assign gpr_read_if.rs2       = decode_if.rs2;
    assign gpr_read_if.rs3       = decode_if.rs3;
    assign gpr_read_if.use_rs3   = decode_if.use_rs3;

    wire ex_busy = (~alu_req_if.ready && (decode_if.ex_type == `EX_ALU))
                || (~lsu_req_if.ready && (decode_if.ex_type == `EX_LSU))
                || (~csr_req_if.ready && (decode_if.ex_type == `EX_CSR))
            `ifdef EXT_M_ENABLE
                || (~mul_req_if.ready && (decode_if.ex_type == `EX_MUL))
            `endif
            `ifdef EXT_F_ENABLE
                || (~fpu_req_if.ready && (decode_if.ex_type == `EX_FPU))
            `endif
                || (~gpu_req_if.ready && (decode_if.ex_type == `EX_GPU));

    VX_scheduler #(
        .CORE_ID(CORE_ID)
    ) scheduler (
        .clk            (clk),
        .reset          (reset), 
        .decode_if      (decode_if),
        .writeback_if   (writeback_if),
        .cmt_to_issue_if(cmt_to_issue_if), 
        .ex_busy        (ex_busy),  
        .issue_tag      (issue_tag),
        .schedule_delay (schedule_delay)
    );

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk            (clk),      
        .reset          (reset),          
        .writeback_if   (writeback_if),
        .gpr_read_if    (gpr_read_if)
    );

    VX_alu_req_if   alu_req_tmp_if();
    VX_lsu_req_if   lsu_req_tmp_if();    
    VX_csr_req_if   csr_req_tmp_if();
    VX_mul_req_if   mul_req_tmp_if();  
    VX_fpu_req_if   fpu_req_tmp_if();  
    VX_gpu_req_if   gpu_req_tmp_if();

    VX_issue_demux issue_demux (
        .decode_if  (decode_if),
        .gpr_read_if(gpr_read_if),
        .issue_tag  (issue_tag),
        .alu_req_if (alu_req_tmp_if),
        .lsu_req_if (lsu_req_tmp_if),        
        .csr_req_if (csr_req_tmp_if),
        .mul_req_if (mul_req_tmp_if),
        .fpu_req_if (fpu_req_tmp_if),
        .gpu_req_if (gpu_req_tmp_if)
    );  

    wire stall = schedule_delay || ~gpr_read_if.ready;
    assign decode_if.ready = ~stall;
    
    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + 32 + `NUM_THREADS + `ALU_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + 32 + 32)
    ) alu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (~alu_req_if.ready),
        .flush (stall && alu_req_if.ready),
        .in    ({alu_req_tmp_if.valid, alu_req_tmp_if.issue_tag, alu_req_tmp_if.warp_num, alu_req_tmp_if.curr_PC, alu_req_tmp_if.thread_mask, alu_req_tmp_if.alu_op, alu_req_tmp_if.rs1_data, alu_req_tmp_if.rs2_data, alu_req_tmp_if.offset, alu_req_tmp_if.next_PC}),
        .out   ({alu_req_if.valid,     alu_req_if.issue_tag,     alu_req_if.warp_num,     alu_req_if.curr_PC,     alu_req_if.thread_mask,     alu_req_if.alu_op,     alu_req_if.rs1_data,     alu_req_if.rs2_data,     alu_req_if.offset,     alu_req_if.next_PC})
    );

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + 32 + `NUM_THREADS + 1 + `BYTEEN_BITS + (`NUM_THREADS * 32) + 32 + (`NUM_THREADS * 32) + `NR_BITS + 1)
    ) lsu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (~lsu_req_if.ready),
        .flush (stall && lsu_req_if.ready),
        .in    ({lsu_req_tmp_if.valid, lsu_req_tmp_if.issue_tag, lsu_req_tmp_if.warp_num, lsu_req_tmp_if.curr_PC, lsu_req_tmp_if.thread_mask, lsu_req_tmp_if.rw, lsu_req_tmp_if.byteen, lsu_req_tmp_if.base_addr, lsu_req_tmp_if.offset, lsu_req_tmp_if.store_data, lsu_req_tmp_if.rd, lsu_req_tmp_if.wb}),
        .out   ({lsu_req_if.valid,     lsu_req_if.issue_tag,     lsu_req_if.warp_num,     lsu_req_if.curr_PC,     lsu_req_if.thread_mask,     lsu_req_if.rw,     lsu_req_if.byteen,     lsu_req_if.base_addr,     lsu_req_if.offset,     lsu_req_if.store_data,     lsu_req_if.rd,     lsu_req_if.wb})
    );

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + 32 + `NUM_THREADS + `CSR_BITS + `CSR_ADDR_BITS + 32 + 1)
    ) csr_reg (
        .clk   (clk),
        .reset (reset),
        .stall (~csr_req_if.ready),
        .flush (stall && csr_req_if.ready),
        .in    ({csr_req_tmp_if.valid, csr_req_tmp_if.issue_tag, csr_req_tmp_if.warp_num, csr_req_tmp_if.curr_PC, csr_req_tmp_if.thread_mask, csr_req_tmp_if.csr_op, csr_req_tmp_if.csr_addr, csr_req_tmp_if.csr_mask, csr_req_tmp_if.is_io}),
        .out   ({csr_req_if.valid,     csr_req_if.issue_tag,     csr_req_if.warp_num,     csr_req_if.curr_PC,     csr_req_if.thread_mask,     csr_req_if.csr_op,     csr_req_if.csr_addr,     csr_req_if.csr_mask,     csr_req_if.is_io})
    );

`ifdef EXT_M_ENABLE
    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + 32 + `NUM_THREADS + `MUL_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32))
    ) mul_reg (
        .clk   (clk),
        .reset (reset),
        .stall (~mul_req_if.ready),
        .flush (stall && mul_req_if.ready),
        .in    ({mul_req_tmp_if.valid, mul_req_tmp_if.issue_tag, mul_req_tmp_if.warp_num, mul_req_tmp_if.curr_PC, mul_req_tmp_if.thread_mask, mul_req_tmp_if.mul_op, mul_req_tmp_if.rs1_data, mul_req_tmp_if.rs2_data}),
        .out   ({mul_req_if.valid,     mul_req_if.issue_tag,     mul_req_if.warp_num,     mul_req_if.curr_PC,     mul_req_if.thread_mask,     mul_req_if.mul_op,     mul_req_if.rs1_data,     mul_req_if.rs2_data})
    );   
`endif

`ifdef EXT_F_ENABLE
    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + 32 + `NUM_THREADS + `FPU_BITS + `FRM_BITS + (`NUM_THREADS * 32) + (`NUM_THREADS * 32) + (`NUM_THREADS * 32))
    ) fpu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (~fpu_req_if.ready),
        .flush (stall && fpu_req_if.ready),
        .in    ({fpu_req_tmp_if.valid, fpu_req_tmp_if.issue_tag, fpu_req_tmp_if.warp_num, fpu_req_tmp_if.curr_PC, fpu_req_tmp_if.thread_mask, fpu_req_tmp_if.fpu_op, fpu_req_tmp_if.frm, fpu_req_tmp_if.rs1_data, fpu_req_tmp_if.rs2_data, fpu_req_tmp_if.rs3_data}),
        .out   ({fpu_req_if.valid,     fpu_req_if.issue_tag,     fpu_req_if.warp_num,     fpu_req_if.curr_PC,     fpu_req_if.thread_mask,     fpu_req_if.fpu_op,     fpu_req_if.frm,     fpu_req_if.rs1_data,     fpu_req_if.rs2_data,     fpu_req_if.rs3_data})
    );
`endif

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + 32 + `NUM_THREADS + `GPU_BITS + (`NUM_THREADS * 32) + 32 + 32)
    ) gpu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (~gpu_req_if.ready),
        .flush (stall && gpu_req_if.ready),
        .in    ({gpu_req_tmp_if.valid, gpu_req_tmp_if.issue_tag, gpu_req_tmp_if.warp_num, gpu_req_tmp_if.curr_PC, gpu_req_tmp_if.thread_mask, gpu_req_tmp_if.gpu_op, gpu_req_tmp_if.rs1_data, gpu_req_tmp_if.rs2_data, gpu_req_tmp_if.next_PC}),
        .out   ({gpu_req_if.valid,     gpu_req_if.issue_tag,     gpu_req_if.warp_num,     gpu_req_if.curr_PC,     gpu_req_if.thread_mask,     gpu_req_if.gpu_op,     gpu_req_if.rs1_data,     gpu_req_if.rs2_data,     gpu_req_if.next_PC})
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_req_if.valid && alu_req_if.ready) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=ALU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h, offset=%0h, next_PC=%0h", $time, CORE_ID, alu_req_if.warp_num, alu_req_if.curr_PC, alu_req_if.issue_tag, alu_req_if.thread_mask, alu_req_if.rs1_data, alu_req_if.rs2_data, alu_req_if.offset, alu_req_if.next_PC);   
        end
        if (lsu_req_if.valid && lsu_req_if.ready) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=LSU, istag=%0d, tmask=%b, rw=%b, byteen=%b, baddr=%0h, offset=%0h, data=%0h", $time, CORE_ID, lsu_req_if.warp_num, lsu_req_if.curr_PC, lsu_req_if.issue_tag, lsu_req_if.thread_mask, lsu_req_if.rw, lsu_req_if.byteen, lsu_req_if.base_addr, lsu_req_if.offset, lsu_req_if.store_data);   
        end
        if (csr_req_if.valid && csr_req_if.ready) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=CSR, istag=%0d, tmask=%b, addr=%0h, mask=%0h", $time, CORE_ID, csr_req_if.warp_num, csr_req_if.curr_PC, csr_req_if.issue_tag, csr_req_if.thread_mask, csr_req_if.csr_addr, csr_req_if.csr_mask);   
        end
        if (mul_req_if.valid && mul_req_if.ready) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=MUL, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, mul_req_if.warp_num, mul_req_if.curr_PC, mul_req_if.issue_tag, mul_req_if.thread_mask, mul_req_if.rs1_data, mul_req_if.rs2_data);   
        end
        if (fpu_req_if.valid && fpu_req_if.ready) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=FPU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h, rs3_data=%0h", $time, CORE_ID, fpu_req_if.warp_num, fpu_req_if.curr_PC, fpu_req_if.issue_tag, fpu_req_if.thread_mask, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data);   
        end
        if (gpu_req_if.valid && gpu_req_if.ready) begin
            $display("%t: Core%0d-issue: warp=%0d, PC=%0h, ex=GPU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, gpu_req_if.warp_num, gpu_req_if.curr_PC, gpu_req_if.issue_tag, gpu_req_if.thread_mask, gpu_req_if.rs1_data, gpu_req_if.rs2_data);   
        end
    end
`endif

endmodule