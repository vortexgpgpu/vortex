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
    VX_writeback_if     writeback_if,
    VX_cmt_to_csr_if    cmt_to_csr_if
);
    localparam CMTW = $clog2(`NUM_THREADS+1);

    // CSRs update

    wire alu_commit_fire = alu_commit_if.valid && alu_commit_if.ready;
    wire lsu_commit_fire = lsu_commit_if.valid && lsu_commit_if.ready;
    wire csr_commit_fire = csr_commit_if.valid && csr_commit_if.ready;
    wire mul_commit_fire = mul_commit_if.valid && mul_commit_if.ready;
    wire fpu_commit_fire = fpu_commit_if.valid && fpu_commit_if.ready;
    wire gpu_commit_fire = gpu_commit_if.valid && gpu_commit_if.ready;

    wire commit_fire = alu_commit_fire
                    || lsu_commit_fire
                    || csr_commit_fire
                    || mul_commit_fire
                    || fpu_commit_fire
                    || gpu_commit_fire;

    wire [`NUM_THREADS-1:0] commit_tmask = alu_commit_fire ? alu_commit_if.tmask:
                                           lsu_commit_fire ? lsu_commit_if.tmask:
                                           csr_commit_fire ? csr_commit_if.tmask:
                                           mul_commit_fire ? mul_commit_if.tmask:
                                           fpu_commit_fire ? fpu_commit_if.tmask:
                                                             gpu_commit_if.tmask;

    wire [CMTW-1:0] commit_size;

    VX_countones #(
        .N(`NUM_THREADS)
    ) commit_ctr (
        .valids(commit_tmask),
        .count (commit_size)
    );

    fflags_t fflags;
    always @(*) begin
        fflags = 0;        
        for (integer i = 0; i < `NUM_THREADS; i++) begin
            if (fpu_commit_if.tmask[i]) begin
                fflags.NX |= fpu_commit_if.fflags[i].NX;
                fflags.UF |= fpu_commit_if.fflags[i].UF;
                fflags.OF |= fpu_commit_if.fflags[i].OF;
                fflags.DZ |= fpu_commit_if.fflags[i].DZ;
                fflags.NV |= fpu_commit_if.fflags[i].NV;
            end
        end
    end

    reg csr_update_r;
    reg [`NW_BITS-1:0] wid_r;
    reg [CMTW-1:0] commit_size_r;
    reg has_fflags_r;
    fflags_t fflags_r;       

    always @(posedge clk) begin
        csr_update_r  <= commit_fire;                
        wid_r         <= fpu_commit_if.wid;        
        commit_size_r <= commit_size; 
        has_fflags_r  <= fpu_commit_if.has_fflags;
        fflags_r      <= fflags;        
    end

    assign cmt_to_csr_if.valid       = csr_update_r;            
    assign cmt_to_csr_if.wid         = wid_r;  
    assign cmt_to_csr_if.commit_size = commit_size_r;
    assign cmt_to_csr_if.has_fflags  = has_fflags_r;    
    assign cmt_to_csr_if.fflags      = fflags_r;

    // Writeback

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

        .writeback_if   (writeback_if)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_commit_if.valid && alu_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=ALU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, alu_commit_if.wid, alu_commit_if.PC, alu_commit_if.tmask, alu_commit_if.wb, alu_commit_if.rd, alu_commit_if.data);
        end
        if (lsu_commit_if.valid && lsu_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=LSU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, lsu_commit_if.wid, lsu_commit_if.PC, lsu_commit_if.tmask, lsu_commit_if.wb, lsu_commit_if.rd, lsu_commit_if.data);
        end
        if (csr_commit_if.valid && csr_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=CSR, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, csr_commit_if.wid, csr_commit_if.PC, csr_commit_if.tmask, csr_commit_if.wb, csr_commit_if.rd, csr_commit_if.data);
        end        
        if (mul_commit_if.valid && mul_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=MUL, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, mul_commit_if.wid, mul_commit_if.PC, mul_commit_if.tmask, mul_commit_if.wb, mul_commit_if.rd, mul_commit_if.data);
        end        
        if (fpu_commit_if.valid && fpu_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=FPU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, fpu_commit_if.wid, fpu_commit_if.PC, fpu_commit_if.tmask, fpu_commit_if.wb, fpu_commit_if.rd, fpu_commit_if.data);
        end
        if (gpu_commit_if.valid && gpu_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=GPU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, gpu_commit_if.wid, gpu_commit_if.PC, gpu_commit_if.tmask, gpu_commit_if.wb, gpu_commit_if.rd, gpu_commit_if.data);
        end
    end
`else    
    `UNUSED_VAR (fpu_commit_if.PC)
`endif

endmodule







