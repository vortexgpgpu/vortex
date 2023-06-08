`include "VX_define.vh"

module VX_commit #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_commit_if.slave      alu_commit_if,
    VX_commit_if.slave      ld_commit_if,
    VX_commit_if.slave      st_commit_if, 
    VX_commit_if.slave      csr_commit_if,
`ifdef EXT_F_ENABLE
    VX_commit_if.slave      fpu_commit_if,
`endif
    VX_commit_if.slave      gpu_commit_if,

    // outputs
    VX_writeback_if.master  writeback_if,
    VX_cmt_to_csr_if.master cmt_to_csr_if,
    VX_cmt_to_fetch_if.master cmt_to_fetch_if,

    // simulation helper signals
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value
);
    localparam NUM_RSPS = `NUM_EX_UNITS;
    localparam COMMIT_SIZEW = $clog2(NUM_RSPS * `NUM_THREADS + 1);

    // CSRs update

    wire alu_commit_fire = alu_commit_if.valid && alu_commit_if.ready;
    wire ld_commit_fire  = ld_commit_if.valid && ld_commit_if.ready;
    wire st_commit_fire  = st_commit_if.valid && st_commit_if.ready;
    wire csr_commit_fire = csr_commit_if.valid && csr_commit_if.ready;
`ifdef EXT_F_ENABLE
    wire fpu_commit_fire = fpu_commit_if.valid && fpu_commit_if.ready;
`endif
    wire gpu_commit_fire = gpu_commit_if.valid && gpu_commit_if.ready;

    wire commit_fire = alu_commit_fire
                    || ld_commit_fire
                    || st_commit_fire
                    || csr_commit_fire
                `ifdef EXT_F_ENABLE
                    || fpu_commit_fire
                `endif
                    || gpu_commit_fire;

    wire [NUM_RSPS * `NUM_THREADS-1:0] commit_tmask = {
        {`NUM_THREADS{alu_commit_fire}} & alu_commit_if.tmask,
        {`NUM_THREADS{ld_commit_fire}}  & ld_commit_if.tmask, 
        {`NUM_THREADS{st_commit_fire}}  & st_commit_if.tmask,
        {`NUM_THREADS{csr_commit_fire}} & csr_commit_if.tmask,
    `ifdef EXT_F_ENABLE
        {`NUM_THREADS{fpu_commit_fire}} & fpu_commit_if.tmask,
    `endif
        {`NUM_THREADS{gpu_commit_fire}} & gpu_commit_if.tmask
    };
    
    wire [COMMIT_SIZEW-1:0] commit_size, commit_size_r;
    wire commit_fire_r;

    `POP_COUNT(commit_size, commit_tmask);

    VX_pipe_register #(
        .DATAW  (1 + COMMIT_SIZEW),
        .DEPTH  (`NUM_THREADS > 2),
        .RESETW (1)
    ) commit_size_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({commit_fire,   commit_size}),
        .data_out ({commit_fire_r, commit_size_r})
    );

    reg [`PERF_CTR_BITS-1:0] instret;

    always @(posedge clk) begin
       if (reset) begin
            instret <= '0;
        end else begin
            if (commit_fire_r) begin
                instret <= instret + `PERF_CTR_BITS'(commit_size_r);
            end
        end
    end
    
    assign cmt_to_csr_if.instret = instret;

    // Committed instructions

    wire final_commit_fire = (alu_commit_fire && alu_commit_if.eop)
                          || (ld_commit_fire && ld_commit_if.eop)
                          || (st_commit_fire && st_commit_if.eop)
                          || (csr_commit_fire && csr_commit_if.eop)
                        `ifdef EXT_F_ENABLE
                          || (fpu_commit_fire && fpu_commit_if.eop)
                        `endif
                          || (gpu_commit_fire && gpu_commit_if.eop);

    wire [NUM_RSPS-1:0] final_commit_tmask = {
        (alu_commit_fire && alu_commit_if.eop),
        (ld_commit_fire && ld_commit_if.eop),
        (st_commit_fire && st_commit_if.eop),
        (csr_commit_fire && csr_commit_if.eop),
    `ifdef EXT_F_ENABLE
        (fpu_commit_fire && fpu_commit_if.eop),
    `endif
        (gpu_commit_fire && gpu_commit_if.eop)
    };

    reg [`EX_UNITS_BITS-1:0] final_commit_size, final_commit_size_r;
    reg final_commit_fire_r;

    `POP_COUNT(final_commit_size, final_commit_tmask);

    VX_pipe_register #(
        .DATAW  (1 + `EX_UNITS_BITS),
        .RESETW (1)
    ) final_commit_size_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({final_commit_fire,   final_commit_size}),
        .data_out ({final_commit_fire_r, final_commit_size_r})
    );

    assign cmt_to_fetch_if.valid = final_commit_fire_r;
    assign cmt_to_fetch_if.committed = final_commit_size_r;

    // Writeback

    VX_writeback #(
        .CORE_ID (CORE_ID)
    ) writeback (
        .clk            (clk),
        .reset          (reset),

        .alu_commit_if  (alu_commit_if),
        .ld_commit_if   (ld_commit_if),        
        .csr_commit_if  (csr_commit_if),
    `ifdef EXT_F_ENABLE
        .fpu_commit_if  (fpu_commit_if),
    `endif
        .gpu_commit_if  (gpu_commit_if),
        .writeback_if   (writeback_if),
        
        .sim_wb_value   (sim_wb_value)
    );

    // store and gpu commits don't writeback  
    assign st_commit_if.ready  = 1'b1;

    `UNUSED_VAR (alu_commit_if.uuid)
    
`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (alu_commit_fire) begin
             `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=ALU, tmask=%b, wb=%0d, rd=%0d, data=", $time, CORE_ID, alu_commit_if.wid, alu_commit_if.PC, alu_commit_if.tmask, alu_commit_if.wb, alu_commit_if.rd));
            `TRACE_ARRAY1D(1, alu_commit_if.data, `NUM_THREADS);
             `TRACE(1, (" (#%0d)\n", alu_commit_if.uuid));
        end
        if (ld_commit_fire) begin
             `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=LSU, tmask=%b, wb=%0d, rd=%0d, data=", $time, CORE_ID, ld_commit_if.wid, ld_commit_if.PC, ld_commit_if.tmask, ld_commit_if.wb, ld_commit_if.rd));
            `TRACE_ARRAY1D(1, ld_commit_if.data, `NUM_THREADS);
             `TRACE(1, (" (#%0d)\n", ld_commit_if.uuid));
        end
        if (st_commit_fire) begin
            `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=LSU, tmask=%b, wb=%0d, rd=%0d (#%0d)\n", $time, CORE_ID, st_commit_if.wid, st_commit_if.PC, st_commit_if.tmask, st_commit_if.wb, st_commit_if.rd, st_commit_if.uuid));
        end
        if (csr_commit_fire) begin
             `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=CSR, tmask=%b, wb=%0d, rd=%0d, data=", $time, CORE_ID, csr_commit_if.wid, csr_commit_if.PC, csr_commit_if.tmask, csr_commit_if.wb, csr_commit_if.rd));
            `TRACE_ARRAY1D(1, csr_commit_if.data, `NUM_THREADS);
             `TRACE(1, (" (#%0d)\n", csr_commit_if.uuid));
        end      
    `ifdef EXT_F_ENABLE
        if (fpu_commit_fire) begin
             `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=FPU, tmask=%b, wb=%0d, rd=%0d, data=", $time, CORE_ID, fpu_commit_if.wid, fpu_commit_if.PC, fpu_commit_if.tmask, fpu_commit_if.wb, fpu_commit_if.rd));
            `TRACE_ARRAY1D(1, fpu_commit_if.data, `NUM_THREADS);
             `TRACE(1, (" (#%0d)\n", fpu_commit_if.uuid));
        end
    `endif
        if (gpu_commit_fire) begin
             `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=GPU, tmask=%b, wb=%0d, rd=%0d, data=", $time, CORE_ID, gpu_commit_if.wid, gpu_commit_if.PC, gpu_commit_if.tmask, gpu_commit_if.wb, gpu_commit_if.rd));
            `TRACE_ARRAY1D(1, gpu_commit_if.data, `NUM_THREADS);
             `TRACE(1, (" (#%0d)\n", gpu_commit_if.uuid));
        end
    end
`endif

endmodule
