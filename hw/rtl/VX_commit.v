`include "VX_define.vh"

module VX_commit #(
    parameter CORE_ID = 0
) (
    input wire          clk,
    input wire          reset,

    // inputs
    VX_exu_to_cmt_if    alu_commit_if,
    VX_exu_to_cmt_if    lsu_commit_if,  
    VX_exu_to_cmt_if    mul_commit_if,    
    VX_exu_to_cmt_if    csr_commit_if,
    VX_fpu_to_cmt_if    fpu_commit_if,
    VX_exu_to_cmt_if    gpu_commit_if,

    // outputs
    VX_cmt_to_issue_if  cmt_to_issue_if,
    VX_wb_if            writeback_if,
    VX_cmt_to_csr_if    cmt_to_csr_if
);
    // update CRSs

    wire [`NUM_EXS-1:0] commited_mask;
    assign commited_mask = {(alu_commit_if.valid && alu_commit_if.ready),
                            (lsu_commit_if.valid && lsu_commit_if.ready),                            
                            (csr_commit_if.valid && csr_commit_if.ready),
                            (mul_commit_if.valid && mul_commit_if.ready),
                            (fpu_commit_if.valid && fpu_commit_if.ready),
                            (gpu_commit_if.valid && gpu_commit_if.ready)};

    wire [`NE_BITS:0] num_commits;

    VX_countones #(
        .N(`NUM_EXS)
    ) valids_counter (
        .valids(commited_mask),
        .count (num_commits)
    );

    assign cmt_to_csr_if.valid       = (| commited_mask);    
    assign cmt_to_csr_if.warp_num    = cmt_to_issue_if.fpu_data.warp_num;
    assign cmt_to_csr_if.num_commits = num_commits;    
  
    assign cmt_to_csr_if.has_fflags  = (fpu_commit_if.valid && fpu_commit_if.ready) && fpu_commit_if.has_fflags;    
    
    integer i;

    fflags_t fflags;
    always @(*) begin
        fflags = 0;        
        for (i = 0; i < `NUM_THREADS; i++) begin
            if (cmt_to_issue_if.fpu_data.thread_mask[i]) begin
                fflags.NX |= fpu_commit_if.fflags[i].NX;
                fflags.UF |= fpu_commit_if.fflags[i].UF;
                fflags.OF |= fpu_commit_if.fflags[i].OF;
                fflags.DZ |= fpu_commit_if.fflags[i].DZ;
                fflags.NV |= fpu_commit_if.fflags[i].NV;
            end
        end
    end
    assign cmt_to_csr_if.fflags = fflags;

    // Notify issue stage

    assign cmt_to_issue_if.alu_valid = alu_commit_if.valid && alu_commit_if.ready;
    assign cmt_to_issue_if.lsu_valid = lsu_commit_if.valid && lsu_commit_if.ready;
    assign cmt_to_issue_if.csr_valid = csr_commit_if.valid && csr_commit_if.ready;
    assign cmt_to_issue_if.mul_valid = mul_commit_if.valid && mul_commit_if.ready;
    assign cmt_to_issue_if.fpu_valid = fpu_commit_if.valid && fpu_commit_if.ready;
    assign cmt_to_issue_if.gpu_valid = gpu_commit_if.valid && gpu_commit_if.ready;

    assign cmt_to_issue_if.alu_tag = alu_commit_if.issue_tag;
    assign cmt_to_issue_if.lsu_tag = lsu_commit_if.issue_tag;
    assign cmt_to_issue_if.csr_tag = csr_commit_if.issue_tag;
    assign cmt_to_issue_if.mul_tag = mul_commit_if.issue_tag;
    assign cmt_to_issue_if.fpu_tag = fpu_commit_if.issue_tag;
    assign cmt_to_issue_if.gpu_tag = gpu_commit_if.issue_tag;

    VX_writeback #(
        .CORE_ID(CORE_ID)
    ) writeback (
        .clk            (clk),
        .reset          (reset),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),        
        .csr_commit_if  (csr_commit_if),
        .mul_commit_if  (mul_commit_if),
        .fpu_commit_if  (fpu_commit_if),    
        .gpu_commit_if  (gpu_commit_if),
        .cmt_to_issue_if(cmt_to_issue_if),  

        .writeback_if   (writeback_if)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_commit_if.valid && alu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=ALU, istag=%0d, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cmt_to_issue_if.alu_data.warp_num, cmt_to_issue_if.alu_data.curr_PC, alu_commit_if.issue_tag, cmt_to_issue_if.alu_data.thread_mask, cmt_to_issue_if.alu_data.wb, cmt_to_issue_if.alu_data.rd, alu_commit_if.data);
        end
        if (lsu_commit_if.valid && lsu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=LSU, istag=%0d, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cmt_to_issue_if.lsu_data.warp_num, cmt_to_issue_if.lsu_data.curr_PC, lsu_commit_if.issue_tag, cmt_to_issue_if.lsu_data.thread_mask, cmt_to_issue_if.lsu_data.wb, cmt_to_issue_if.lsu_data.rd, lsu_commit_if.data);
        end
        if (csr_commit_if.valid && csr_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=CSR, istag=%0d, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cmt_to_issue_if.csr_data.warp_num, cmt_to_issue_if.csr_data.curr_PC, csr_commit_if.issue_tag, cmt_to_issue_if.csr_data.thread_mask, cmt_to_issue_if.csr_data.wb, cmt_to_issue_if.csr_data.rd, csr_commit_if.data);
        end        
        if (mul_commit_if.valid && mul_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=MUL, istag=%0d, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cmt_to_issue_if.mul_data.warp_num, cmt_to_issue_if.mul_data.curr_PC, mul_commit_if.issue_tag, cmt_to_issue_if.mul_data.thread_mask, cmt_to_issue_if.mul_data.wb, cmt_to_issue_if.mul_data.rd, mul_commit_if.data);
        end        
        if (fpu_commit_if.valid && fpu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=FPU, istag=%0d, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cmt_to_issue_if.fpu_data.warp_num, cmt_to_issue_if.fpu_data.curr_PC, fpu_commit_if.issue_tag, cmt_to_issue_if.fpu_data.thread_mask, cmt_to_issue_if.fpu_data.wb, cmt_to_issue_if.fpu_data.rd, fpu_commit_if.data);
        end
        if (gpu_commit_if.valid && gpu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=GPU, istag=%0d, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, cmt_to_issue_if.gpu_data.warp_num, cmt_to_issue_if.gpu_data.curr_PC, gpu_commit_if.issue_tag, cmt_to_issue_if.gpu_data.thread_mask, cmt_to_issue_if.gpu_data.wb, cmt_to_issue_if.gpu_data.rd, gpu_commit_if.data);
        end
    end
`endif

endmodule







