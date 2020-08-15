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
    VX_bru_req_if       bru_req_if,
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
    assign gpr_read_if.wid  = decode_if.wid;
    assign gpr_read_if.rs1       = decode_if.rs1;
    assign gpr_read_if.rs2       = decode_if.rs2;
    assign gpr_read_if.rs3       = decode_if.rs3;
    assign gpr_read_if.use_rs3   = decode_if.use_rs3;

    wire ex_busy = (~alu_req_if.ready && (decode_if.ex_type == `EX_ALU))
                || (~bru_req_if.ready && (decode_if.ex_type == `EX_BRU))
                || (~lsu_req_if.ready && (decode_if.ex_type == `EX_LSU))
                || (~csr_req_if.ready && (decode_if.ex_type == `EX_CSR))
            `ifdef EXT_M_ENABLE
                || (~mul_req_if.ready && (decode_if.ex_type == `EX_MUL))
            `endif
            `ifdef EXT_F_ENABLE
                || (~fpu_req_if.ready && (decode_if.ex_type == `EX_FPU))
            `endif
                || (~gpu_req_if.ready && (decode_if.ex_type == `EX_GPU));

    VX_scoreboard #(
        .CORE_ID(CORE_ID)
    ) scoreboard (
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

    VX_issue_if  issue_if();

    assign issue_if.rs1_data = gpr_read_if.rs1_data;
    assign issue_if.rs2_data = gpr_read_if.rs2_data;
    assign issue_if.rs3_data = gpr_read_if.rs3_data;    

    wire [`NT_BITS-1:0] tid;
    VX_priority_encoder #(
        .N(`NUM_THREADS)
    ) sel_src (
        .data_in  (decode_if.thread_mask),
        .data_out (tid),
        `UNUSED_PIN (valid_out)
    );

    wire stall = schedule_delay || ~gpr_read_if.ready;    
    wire flush = stall; // clear output on stall

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + `NR_BITS + 32 + 1 + 1 + `EX_BITS + `OP_BITS + 1 + `FRM_BITS + `NT_BITS)
    ) issue_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (flush),
        .in    ({decode_if.valid, issue_tag,          decode_if.wid, decode_if.thread_mask, decode_if.curr_PC, decode_if.rd, decode_if.rs1, decode_if.imm, decode_if.rs1_is_PC, decode_if.rs2_is_imm, decode_if.ex_type, decode_if.ex_op, decode_if.wb, decode_if.frm, tid}),
        .out   ({issue_if.valid,  issue_if.issue_tag, issue_if.wid,  issue_if.thread_mask,  issue_if.curr_PC,  issue_if.rd,  issue_if.rs1,  issue_if.imm,  issue_if.rs1_is_PC,  issue_if.rs2_is_imm,  issue_if.ex_type,  issue_if.ex_op,  issue_if.wb,  issue_if.frm,  issue_if.tid})
    );

    assign decode_if.ready = issue_if.ready;
    assign issue_if.ready = ~stall;
    
    VX_issue_demux issue_demux (
        .issue_if  (issue_if),
        .alu_req_if (alu_req_if),
        .bru_req_if (bru_req_if),
        .lsu_req_if (lsu_req_if),        
        .csr_req_if (csr_req_if),
        .mul_req_if (mul_req_if),
        .fpu_req_if (fpu_req_if),
        .gpu_req_if (gpu_req_if)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_req_if.valid && alu_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=ALU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, alu_req_if.wid, alu_req_if.curr_PC, alu_req_if.issue_tag, alu_req_if.thread_mask, alu_req_if.rs1_data, alu_req_if.rs2_data);   
        end
        if (bru_req_if.valid && bru_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=BRU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h, offset=%0h", $time, CORE_ID, bru_req_if.wid, bru_req_if.curr_PC, bru_req_if.issue_tag, bru_req_if.thread_mask, bru_req_if.rs1_data, bru_req_if.rs2_data, bru_req_if.offset);
        end
        if (lsu_req_if.valid && lsu_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=LSU, istag=%0d, tmask=%b, rw=%b, byteen=%b, baddr=%0h, offset=%0h, data=%0h", $time, CORE_ID, lsu_req_if.wid, lsu_req_if.curr_PC, lsu_req_if.issue_tag, lsu_req_if.thread_mask, lsu_req_if.rw, lsu_req_if.byteen, lsu_req_if.base_addr, lsu_req_if.offset, lsu_req_if.store_data);   
        end
        if (csr_req_if.valid && csr_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=CSR, istag=%0d, tmask=%b, addr=%0h, mask=%0h", $time, CORE_ID, csr_req_if.wid, csr_req_if.curr_PC, csr_req_if.issue_tag, csr_req_if.thread_mask, csr_req_if.csr_addr, csr_req_if.csr_mask);   
        end
        if (mul_req_if.valid && mul_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=MUL, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, mul_req_if.wid, mul_req_if.curr_PC, mul_req_if.issue_tag, mul_req_if.thread_mask, mul_req_if.rs1_data, mul_req_if.rs2_data);   
        end
        if (fpu_req_if.valid && fpu_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=FPU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h, rs3_data=%0h", $time, CORE_ID, fpu_req_if.wid, fpu_req_if.curr_PC, fpu_req_if.issue_tag, fpu_req_if.thread_mask, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data);   
        end
        if (gpu_req_if.valid && gpu_req_if.ready) begin
            $display("%t: Core%0d-issue: wid=%0d, PC=%0h, ex=GPU, istag=%0d, tmask=%b, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, gpu_req_if.wid, gpu_req_if.curr_PC, gpu_req_if.issue_tag, gpu_req_if.thread_mask, gpu_req_if.rs1_data, gpu_req_if.rs2_data);   
        end
    end
`endif

endmodule