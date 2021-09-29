`include "VX_define.vh"

module VX_issue #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_issue

    input wire      clk,
    input wire      reset,

`ifdef PERF_ENABLE
    VX_perf_pipeline_if.master perf_pipeline_if,
`endif

    VX_decode_if.slave      decode_if,
    VX_writeback_if.slave   writeback_if,   
    
    VX_alu_req_if.master    alu_req_if,
    VX_lsu_req_if.master    lsu_req_if,    
    VX_csr_req_if.master    csr_req_if,
`ifdef EXT_F_ENABLE
    VX_fpu_req_if.master    fpu_req_if,    
`endif
    VX_gpu_req_if.master    gpu_req_if
);
    VX_ibuffer_if ibuffer_if();
    VX_gpr_rsp_if gpr_rsp_if();

    VX_gpr_req_if gpr_req_if();
    assign gpr_req_if.wid = ibuffer_if.wid;
    assign gpr_req_if.rs1 = ibuffer_if.rs1;
    assign gpr_req_if.rs2 = ibuffer_if.rs2;
    assign gpr_req_if.rs3 = ibuffer_if.rs3;

    VX_writeback_if sboard_wb_if();
    assign sboard_wb_if.valid  = writeback_if.valid;
    assign sboard_wb_if.wid    = writeback_if.wid;
    assign sboard_wb_if.PC     = writeback_if.PC;
    assign sboard_wb_if.rd     = writeback_if.rd;
    assign sboard_wb_if.eop    = writeback_if.eop;
    assign sboard_wb_if.ready  = writeback_if.ready;
        
    VX_ibuffer_if sboard_ib_if();
    assign sboard_ib_if.valid  = ibuffer_if.valid && idmux_ib_if.ready;
    assign sboard_ib_if.wid    = ibuffer_if.wid;
    assign sboard_ib_if.PC     = ibuffer_if.PC;   
    assign sboard_ib_if.wb     = ibuffer_if.wb;      
    assign sboard_ib_if.rd     = ibuffer_if.rd;
    assign sboard_ib_if.rd_n   = ibuffer_if.rd_n;        
    assign sboard_ib_if.rs1_n  = ibuffer_if.rs1_n;        
    assign sboard_ib_if.rs2_n  = ibuffer_if.rs2_n;        
    assign sboard_ib_if.rs3_n  = ibuffer_if.rs3_n;        
    assign sboard_ib_if.wid_n  = ibuffer_if.wid_n;

    VX_ibuffer_if idmux_ib_if();
    assign idmux_ib_if.valid   = ibuffer_if.valid && sboard_ib_if.ready;
    assign idmux_ib_if.wid     = ibuffer_if.wid;
    assign idmux_ib_if.tmask   = ibuffer_if.tmask;
    assign idmux_ib_if.PC      = ibuffer_if.PC;
    assign idmux_ib_if.ex_type = ibuffer_if.ex_type;    
    assign idmux_ib_if.op_type = ibuffer_if.op_type; 
    assign idmux_ib_if.op_mod  = ibuffer_if.op_mod;    
    assign idmux_ib_if.wb      = ibuffer_if.wb;
    assign idmux_ib_if.rd      = ibuffer_if.rd;
    assign idmux_ib_if.rs1     = ibuffer_if.rs1;
    assign idmux_ib_if.imm     = ibuffer_if.imm;        
    assign idmux_ib_if.use_PC  = ibuffer_if.use_PC;
    assign idmux_ib_if.use_imm = ibuffer_if.use_imm;

    // issue the instruction
    assign ibuffer_if.ready = sboard_ib_if.ready && idmux_ib_if.ready;

    `RESET_RELAY (ibuf_reset);
    `RESET_RELAY (gpr_reset);
    `RESET_RELAY (demux_reset);

    VX_ibuffer #(
        .CORE_ID(CORE_ID)
    ) ibuffer (
        .clk        (clk),
        .reset      (ibuf_reset), 
        .decode_if  (decode_if),
        .ibuffer_if (ibuffer_if) 
    );

    VX_scoreboard #(
        .CORE_ID(CORE_ID)
    ) scoreboard (
        .clk        (clk),
        .reset      (reset), 
        .ibuffer_if (sboard_ib_if),
        .writeback_if(sboard_wb_if)
    );

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk          (clk),      
        .reset        (gpr_reset),          
        .writeback_if (writeback_if),
        .gpr_req_if   (gpr_req_if),
        .gpr_rsp_if   (gpr_rsp_if)
    );

    VX_instr_demux instr_demux (
        .clk        (clk),      
        .reset      (demux_reset),
        .ibuffer_if (idmux_ib_if),
        .gpr_rsp_if (gpr_rsp_if),
        .alu_req_if (alu_req_if),
        .lsu_req_if (lsu_req_if),        
        .csr_req_if (csr_req_if),
    `ifdef EXT_F_ENABLE
        .fpu_req_if (fpu_req_if),
    `endif
        .gpu_req_if (gpu_req_if)
    );

    `SCOPE_ASSIGN (issue_fire,        ibuffer_if.valid && ibuffer_if.ready);
    `SCOPE_ASSIGN (issue_wid,         ibuffer_if.wid);
    `SCOPE_ASSIGN (issue_tmask,       ibuffer_if.tmask);
    `SCOPE_ASSIGN (issue_pc,          ibuffer_if.PC);
    `SCOPE_ASSIGN (issue_ex_type,     ibuffer_if.ex_type);
    `SCOPE_ASSIGN (issue_op_type,     ibuffer_if.op_type);
    `SCOPE_ASSIGN (issue_op_mod,      ibuffer_if.op_mod);
    `SCOPE_ASSIGN (issue_wb,          ibuffer_if.wb);
    `SCOPE_ASSIGN (issue_rd,          ibuffer_if.rd);
    `SCOPE_ASSIGN (issue_rs1,         ibuffer_if.rs1);
    `SCOPE_ASSIGN (issue_rs2,         ibuffer_if.rs2);
    `SCOPE_ASSIGN (issue_rs3,         ibuffer_if.rs3);
    `SCOPE_ASSIGN (issue_imm,         ibuffer_if.imm);
    `SCOPE_ASSIGN (issue_use_pc,      ibuffer_if.use_PC);
    `SCOPE_ASSIGN (issue_use_imm,     ibuffer_if.use_imm);
    `SCOPE_ASSIGN (scoreboard_delay,  scoreboard_delay); 
    `SCOPE_ASSIGN (execute_delay,     ~idmux_ib_if.ready);    
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
            if (ibuffer_if.valid & scoreboard_delay) begin 
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
            dpi_trace("%d: core%0d-issue: wid=%0d, PC=%0h, ex=ALU, tmask=%b, rd=%0d, rs1_data=", 
                $time, CORE_ID, alu_req_if.wid, alu_req_if.PC, alu_req_if.tmask, alu_req_if.rd);
            `TRACE_ARRAY1D(alu_req_if.rs1_data, `NUM_THREADS);
            dpi_trace(", rs2_data=");
            `TRACE_ARRAY1D(alu_req_if.rs2_data, `NUM_THREADS);
            dpi_trace("\n");
        end
        if (lsu_req_if.valid && lsu_req_if.ready) begin
            dpi_trace("%d: core%0d-issue: wid=%0d, PC=%0h, ex=LSU, tmask=%b, rd=%0d, offset=%0h, addr=", 
                $time, CORE_ID, lsu_req_if.wid, lsu_req_if.PC, lsu_req_if.tmask, lsu_req_if.rd, lsu_req_if.offset); 
            `TRACE_ARRAY1D(lsu_req_if.base_addr, `NUM_THREADS);
            dpi_trace(", data=");
            `TRACE_ARRAY1D(lsu_req_if.store_data, `NUM_THREADS);
            dpi_trace("\n");
        end
        if (csr_req_if.valid && csr_req_if.ready) begin
            dpi_trace("%d: core%0d-issue: wid=%0d, PC=%0h, ex=CSR, tmask=%b, rd=%0d, addr=%0h, rs1_data=", 
                $time, CORE_ID, csr_req_if.wid, csr_req_if.PC, csr_req_if.tmask, csr_req_if.rd, csr_req_if.addr);   
            `TRACE_ARRAY1D(csr_req_if.rs1_data, `NUM_THREADS);
            dpi_trace("\n");
        end
    `ifdef EXT_F_ENABLE
        if (fpu_req_if.valid && fpu_req_if.ready) begin
            dpi_trace("%d: core%0d-issue: wid=%0d, PC=%0h, ex=FPU, tmask=%b, rd=%0d, rs1_data=", 
                $time, CORE_ID, fpu_req_if.wid, fpu_req_if.PC, fpu_req_if.tmask, fpu_req_if.rd);   
            `TRACE_ARRAY1D(fpu_req_if.rs1_data, `NUM_THREADS);
            dpi_trace(", rs2_data=");
            `TRACE_ARRAY1D(fpu_req_if.rs2_data, `NUM_THREADS);
            dpi_trace(", rs3_data=");
            `TRACE_ARRAY1D(fpu_req_if.rs3_data, `NUM_THREADS);
            dpi_trace("\n");
        end
    `endif
        if (gpu_req_if.valid && gpu_req_if.ready) begin
            dpi_trace("%d: core%0d-issue: wid=%0d, PC=%0h, ex=GPU, tmask=%b, rd=%0d, rs1_data=", 
                $time, CORE_ID, gpu_req_if.wid, gpu_req_if.PC, gpu_req_if.tmask, gpu_req_if.rd);   
            `TRACE_ARRAY1D(gpu_req_if.rs1_data, `NUM_THREADS);
            dpi_trace(", rs2_data=");
            `TRACE_ARRAY1D(gpu_req_if.rs2_data, `NUM_THREADS);
            dpi_trace("\n");   
        end
    end
`endif

endmodule