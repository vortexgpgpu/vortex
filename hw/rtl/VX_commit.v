`include "VX_define.vh"

module VX_commit #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs
    VX_commit_if    alu_commit_if,
    VX_commit_if    lsu_commit_if,  
    VX_commit_if    mul_commit_if,    
    VX_commit_if    csr_commit_if,
    VX_commit_if    gpu_commit_if,

    // outputs
    VX_wb_if        writeback_if,
    VX_perf_cntrs_if perf_cntrs_if
);

    wire [`NUM_EXS-1:0] commited_mask;
    assign commited_mask = {((| alu_commit_if.valid) && alu_commit_if.ready),
                            ((| lsu_commit_if.valid) && lsu_commit_if.ready),
                            ((| mul_commit_if.valid) && mul_commit_if.ready),
                            ((| csr_commit_if.valid) && csr_commit_if.ready),
                            ((| gpu_commit_if.valid) && gpu_commit_if.ready)};

    wire [`NE_BITS:0] num_commits;

     VX_countones #(
        .N(`NUM_EXS)
    ) valids_counter (
        .valids(commited_mask),
        .count (num_commits)
    );

    wire has_committed = (| commited_mask);

    reg [63:0] total_cycles, total_instrs;
    
    always @(posedge clk) begin
       if (reset) begin
            total_cycles <= 0;
            total_instrs <= 0;
        end else begin
            total_cycles <= total_cycles + 1;
            if (has_committed) begin
                total_instrs <= total_instrs + 64'(num_commits);
            end
        end
    end

    assign perf_cntrs_if.total_cycles = total_cycles;
    assign perf_cntrs_if.total_instrs = total_instrs;

    assign gpu_commit_if.ready = 1'b1; // doesn't writeback

    VX_writeback #(
        .CORE_ID(CORE_ID)
    ) writeback (
        .clk            (clk),
        .reset          (reset),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),        
        .csr_commit_if  (csr_commit_if),
        .mul_commit_if  (mul_commit_if),
        
        .writeback_if   (writeback_if)
    );

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if ((| alu_commit_if.valid) && alu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=ALU, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, alu_commit_if.warp_num, alu_commit_if.curr_PC, alu_commit_if.wb, alu_commit_if.rd, alu_commit_if.data);
        end
        if ((| lsu_commit_if.valid) && lsu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=LSU, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, lsu_commit_if.warp_num, lsu_commit_if.curr_PC, lsu_commit_if.wb, lsu_commit_if.rd, lsu_commit_if.data);
        end
        if ((| mul_commit_if.valid) && mul_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=MUL, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, mul_commit_if.warp_num, mul_commit_if.curr_PC, mul_commit_if.wb, mul_commit_if.rd, mul_commit_if.data);
        end
        if ((| csr_commit_if.valid) && csr_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=CSR, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, csr_commit_if.warp_num, csr_commit_if.curr_PC, csr_commit_if.wb, csr_commit_if.rd, csr_commit_if.data);
        end
        if ((| gpu_commit_if.valid) && gpu_commit_if.ready) begin
            $display("%t: Core%0d-commit: warp=%0d, PC=%0h, ex=GPU, wb=%0d, rd=%0d, data=%0h", $time, CORE_ID, gpu_commit_if.warp_num, gpu_commit_if.curr_PC, gpu_commit_if.wb, gpu_commit_if.rd, gpu_commit_if.data);
        end
    end
`endif

endmodule







