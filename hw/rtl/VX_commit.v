`include "VX_define.vh"

module VX_commit #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs
    VX_commit_if    alu_commit_if,
    VX_commit_if    ld_commit_if,
    VX_commit_if    st_commit_if, 
    VX_commit_if    csr_commit_if,
    VX_commit_if    fpu_commit_if,
    VX_commit_if    gpu_commit_if,

    // outputs
    VX_writeback_if     writeback_if,
    VX_cmt_to_csr_if    cmt_to_csr_if
);
    localparam CMTW = $clog2(3*`NUM_THREADS+1);

    // CSRs update

    wire alu_commit_fire = alu_commit_if.valid && alu_commit_if.ready;
    wire ld_commit_fire  = ld_commit_if.valid && ld_commit_if.ready;
    wire st_commit_fire  = st_commit_if.valid && st_commit_if.ready;
    wire csr_commit_fire = csr_commit_if.valid && csr_commit_if.ready;
    wire fpu_commit_fire = fpu_commit_if.valid && fpu_commit_if.ready;
    wire gpu_commit_fire = gpu_commit_if.valid && gpu_commit_if.ready;

    wire commit_fire = alu_commit_fire
                    || ld_commit_fire
                    || st_commit_fire
                    || csr_commit_fire
                    || fpu_commit_fire
                    || gpu_commit_fire;

    wire [`NUM_THREADS-1:0] commit_tmask1, commit_tmask2, commit_tmask3; 

    assign commit_tmask1 = alu_commit_fire ? alu_commit_if.tmask:
                           ld_commit_fire  ? ld_commit_if.tmask:                                           
                           csr_commit_fire ? csr_commit_if.tmask:
                           fpu_commit_fire ? fpu_commit_if.tmask:
                                             0;

    assign commit_tmask2 = st_commit_fire ? st_commit_if.tmask : 0;
    assign commit_tmask3 = gpu_commit_fire ? gpu_commit_if.tmask : 0;

    wire [CMTW-1:0] commit_size;
    assign commit_size = $countones({commit_tmask3, commit_tmask2, commit_tmask1});

    VX_pipe_register #(
        .DATAW  (1 + CMTW),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({commit_fire,         commit_size}),
        .data_out ({cmt_to_csr_if.valid, cmt_to_csr_if.commit_size})
    );

    // Writeback

    VX_writeback #(
        .CORE_ID(CORE_ID)
    ) writeback (
        .clk            (clk),
        .reset          (reset),

        .alu_commit_if  (alu_commit_if),
        .ld_commit_if   (ld_commit_if),        
        .csr_commit_if  (csr_commit_if),
        .fpu_commit_if  (fpu_commit_if),

        .writeback_if   (writeback_if)
    );

    // store and gpu commits don't writeback  
    assign st_commit_if.ready  = 1'b1;
    assign gpu_commit_if.ready = 1'b1;

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_commit_if.valid && alu_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=ALU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, alu_commit_if.wid, alu_commit_if.PC, alu_commit_if.tmask, alu_commit_if.wb, alu_commit_if.rd, alu_commit_if.data);
        end
        if (ld_commit_if.valid && ld_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=LSU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, ld_commit_if.wid, ld_commit_if.PC, ld_commit_if.tmask, ld_commit_if.wb, ld_commit_if.rd, ld_commit_if.data);
        end
        if (st_commit_if.valid && st_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=LSU, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, st_commit_if.wid, st_commit_if.PC, st_commit_if.tmask, st_commit_if.wb, st_commit_if.rd, st_commit_if.data);
        end
        if (csr_commit_if.valid && csr_commit_if.ready) begin
            $display("%t: core%0d-commit: wid=%0d, PC=%0h, ex=CSR, tmask=%b, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, csr_commit_if.wid, csr_commit_if.PC, csr_commit_if.tmask, csr_commit_if.wb, csr_commit_if.rd, csr_commit_if.data);
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







