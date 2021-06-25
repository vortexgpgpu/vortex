`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_issue

    input wire      clk,
    input wire      reset,

`ifdef PERF_ENABLE
    VX_perf_pipeline_if perf_pipeline_if,
`endif

    VX_decode_if    decode_if,
    VX_writeback_if writeback_if,   
    
    VX_alu_req_if   alu_req_if,
    VX_lsu_req_if   lsu_req_if,    
    VX_csr_req_if   csr_req_if,
    VX_fpu_req_if   fpu_req_if,    
    VX_gpu_req_if   gpu_req_if
);
    VX_instr_sched_if instr_sched_if();
    VX_instr_sched_if execute_if();
    VX_gpr_req_if gpr_req_if();
    VX_gpr_rsp_if gpr_rsp_if();

    wire scoreboard_delay;

    VX_instr_sched #(
        .CORE_ID(CORE_ID)
    ) instr_sched (
        .clk            (clk),
        .reset          (reset), 
        .decode_if      (decode_if),
        .instr_sched_if (instr_sched_if) 
    );

    VX_scoreboard #(
        .CORE_ID(CORE_ID)
    ) scoreboard (
        .clk            (clk),
        .reset          (reset), 
        .instr_sched_if (instr_sched_if),
        .writeback_if   (writeback_if),
        .delay          (scoreboard_delay)
    );
        
    assign gpr_req_if.wid = instr_sched_if.wid;
    assign gpr_req_if.rs1 = instr_sched_if.rs1;
    assign gpr_req_if.rs2 = instr_sched_if.rs2;
    assign gpr_req_if.rs3 = instr_sched_if.rs3;

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk          (clk),      
        .reset        (reset),          
        .writeback_if (writeback_if),
        .gpr_req_if   (gpr_req_if),
        .gpr_rsp_if   (gpr_rsp_if)
    );

    assign execute_if.valid     = instr_sched_if.valid && ~scoreboard_delay;
    assign execute_if.wid       = instr_sched_if.wid;
    assign execute_if.tmask     = instr_sched_if.tmask;
    assign execute_if.PC        = instr_sched_if.PC;
    assign execute_if.ex_type   = instr_sched_if.ex_type;    
    assign execute_if.op_type   = instr_sched_if.op_type; 
    assign execute_if.op_mod    = instr_sched_if.op_mod;    
    assign execute_if.wb        = instr_sched_if.wb;
    assign execute_if.rd        = instr_sched_if.rd;
    assign execute_if.rs1       = instr_sched_if.rs1;
    assign execute_if.imm       = instr_sched_if.imm;        
    assign execute_if.use_PC    = instr_sched_if.use_PC;
    assign execute_if.use_imm   = instr_sched_if.use_imm;

    VX_instr_demux instr_demux (
        .clk        (clk),      
        .reset      (reset),
        .execute_if (execute_if),
        .gpr_rsp_if (gpr_rsp_if),
        .alu_req_if (alu_req_if),
        .lsu_req_if (lsu_req_if),        
        .csr_req_if (csr_req_if),
        .fpu_req_if (fpu_req_if),
        .gpu_req_if (gpu_req_if)
    );     

    // issue the instruction
    assign instr_sched_if.ready = !scoreboard_delay && execute_if.ready;     

    `SCOPE_ASSIGN (issue_fire,        instr_sched_if.valid && instr_sched_if.ready);
    `SCOPE_ASSIGN (issue_wid,         instr_sched_if.wid);
    `SCOPE_ASSIGN (issue_tmask,       instr_sched_if.tmask);
    `SCOPE_ASSIGN (issue_pc,          instr_sched_if.PC);
    `SCOPE_ASSIGN (issue_ex_type,     instr_sched_if.ex_type);
    `SCOPE_ASSIGN (issue_op_type,     instr_sched_if.op_type);
    `SCOPE_ASSIGN (issue_op_mod,      instr_sched_if.op_mod);
    `SCOPE_ASSIGN (issue_wb,          instr_sched_if.wb);
    `SCOPE_ASSIGN (issue_rd,          instr_sched_if.rd);
    `SCOPE_ASSIGN (issue_rs1,         instr_sched_if.rs1);
    `SCOPE_ASSIGN (issue_rs2,         instr_sched_if.rs2);
    `SCOPE_ASSIGN (issue_rs3,         instr_sched_if.rs3);
    `SCOPE_ASSIGN (issue_imm,         instr_sched_if.imm);
    `SCOPE_ASSIGN (issue_use_pc,      instr_sched_if.use_PC);
    `SCOPE_ASSIGN (issue_use_imm,     instr_sched_if.use_imm);
    `SCOPE_ASSIGN (scoreboard_delay,  scoreboard_delay); 
    `SCOPE_ASSIGN (execute_delay,     ~execute_if.ready);    
    `SCOPE_ASSIGN (gpr_rsp_a,         gpr_rsp_if.rs1_data);
    `SCOPE_ASSIGN (gpr_rsp_b,         gpr_rsp_if.rs2_data);
    `SCOPE_ASSIGN (gpr_rsp_c,         gpr_rsp_if.rs3_data);
    `SCOPE_ASSIGN (writeback_valid,   writeback_if.valid);    
    `SCOPE_ASSIGN (writeback_tmask,   writeback_if.tmask);
    `SCOPE_ASSIGN (writeback_wid,     writeback_if.wid);
    `SCOPE_ASSIGN (writeback_pc,      writeback_if.PC);  
    `SCOPE_ASSIGN (writeback_rd,      writeback_if.rd);
    `SCOPE_ASSIGN (writeback_data,    writeback_if.data);
    `SCOPE_ASSIGN (writeback_eop,     writeback_if.eop);

`ifdef PERF_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_ibf_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_scb_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_alu_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_lsu_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_csr_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_gpu_stalls;
`ifdef EXT_F_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_fpu_stalls;
`endif

    always @(posedge clk) begin
        if (reset) begin
            perf_ibf_stalls <= 0;
            perf_scb_stalls <= 0;
            perf_alu_stalls <= 0;
            perf_lsu_stalls <= 0;
            perf_csr_stalls <= 0;
            perf_gpu_stalls <= 0;
        `ifdef EXT_F_ENABLE
            perf_fpu_stalls <= 0;
        `endif
        end else begin
            if (decode_if.valid & !decode_if.ready) begin
                perf_ibf_stalls <= perf_ibf_stalls  + `PERF_CTR_BITS'd1;
            end
            if (instr_sched_if.valid & scoreboard_delay) begin 
                perf_scb_stalls <= perf_scb_stalls  + `PERF_CTR_BITS'd1;
            end
            if (alu_req_if.valid & !alu_req_if.ready) begin
                perf_alu_stalls <= perf_alu_stalls + `PERF_CTR_BITS'd1;
            end
            if (lsu_req_if.valid & !lsu_req_if.ready) begin
                perf_lsu_stalls <= perf_lsu_stalls + `PERF_CTR_BITS'd1;
            end
            if (csr_req_if.valid & !csr_req_if.ready) begin
                perf_csr_stalls <= perf_csr_stalls + `PERF_CTR_BITS'd1;
            end
            if (gpu_req_if.valid & !gpu_req_if.ready) begin
                perf_gpu_stalls <= perf_gpu_stalls + `PERF_CTR_BITS'd1;
            end
        `ifdef EXT_F_ENABLE
            if (fpu_req_if.valid & !fpu_req_if.ready) begin
                perf_fpu_stalls <= perf_fpu_stalls + `PERF_CTR_BITS'd1;
            end
        `endif
        end
    end
    
    assign perf_pipeline_if.ibf_stalls = perf_ibf_stalls;
    assign perf_pipeline_if.scb_stalls = perf_scb_stalls; 
    assign perf_pipeline_if.alu_stalls = perf_alu_stalls;
    assign perf_pipeline_if.lsu_stalls = perf_lsu_stalls;
    assign perf_pipeline_if.csr_stalls = perf_csr_stalls;
    assign perf_pipeline_if.gpu_stalls = perf_gpu_stalls;
`ifdef EXT_F_ENABLE
    assign perf_pipeline_if.fpu_stalls = perf_fpu_stalls;
`endif
`endif

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (alu_req_if.valid && alu_req_if.ready) begin
            $write("%t: core%0d-issue: wid=%0d, PC=%0h, ex=ALU, tmask=%b, rd=%0d, rs1_data=", 
                $time, CORE_ID, alu_req_if.wid, alu_req_if.PC, alu_req_if.tmask, alu_req_if.rd);
            `PRINT_ARRAY1D(alu_req_if.rs1_data, `NUM_THREADS);
            $write(", rs2_data=");
            `PRINT_ARRAY1D(alu_req_if.rs2_data, `NUM_THREADS);
            $write("\n");
        end
        if (lsu_req_if.valid && lsu_req_if.ready) begin
            $write("%t: core%0d-issue: wid=%0d, PC=%0h, ex=LSU, tmask=%b, rd=%0d, offset=%0h, addr=", 
                $time, CORE_ID, lsu_req_if.wid, lsu_req_if.PC, lsu_req_if.tmask, lsu_req_if.rd, lsu_req_if.offset); 
            `PRINT_ARRAY1D(lsu_req_if.base_addr, `NUM_THREADS);
            $write(", data=");
            `PRINT_ARRAY1D(lsu_req_if.store_data, `NUM_THREADS);
            $write("\n");
        end
        if (csr_req_if.valid && csr_req_if.ready) begin
            $write("%t: core%0d-issue: wid=%0d, PC=%0h, ex=CSR, tmask=%b, rd=%0d, addr=%0h, rs1_data=", 
                $time, CORE_ID, csr_req_if.wid, csr_req_if.PC, csr_req_if.tmask, csr_req_if.rd, csr_req_if.addr);   
            `PRINT_ARRAY1D(csr_req_if.rs1_data, `NUM_THREADS);
            $write("\n");
        end
        if (fpu_req_if.valid && fpu_req_if.ready) begin
            $write("%t: core%0d-issue: wid=%0d, PC=%0h, ex=FPU, tmask=%b, rd=%0d, rs1_data=", 
                $time, CORE_ID, fpu_req_if.wid, fpu_req_if.PC, fpu_req_if.tmask, fpu_req_if.rd);   
            `PRINT_ARRAY1D(fpu_req_if.rs1_data, `NUM_THREADS);
            $write(", rs2_data=");
            `PRINT_ARRAY1D(fpu_req_if.rs2_data, `NUM_THREADS);
            $write(", rs3_data=");
            `PRINT_ARRAY1D(fpu_req_if.rs3_data, `NUM_THREADS);
            $write("\n");
        end
        if (gpu_req_if.valid && gpu_req_if.ready) begin
            $write("%t: core%0d-issue: wid=%0d, PC=%0h, ex=GPU, tmask=%b, rd=%0d, rs1_data=", 
                $time, CORE_ID, gpu_req_if.wid, gpu_req_if.PC, gpu_req_if.tmask, gpu_req_if.rd);   
            `PRINT_ARRAY1D(gpu_req_if.rs1_data, `NUM_THREADS);
            $write(", rs2_data=");
            `PRINT_ARRAY1D(gpu_req_if.rs2_data, `NUM_THREADS);
            $write("\n");   
        end
    end
`endif

endmodule