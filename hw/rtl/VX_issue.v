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
    VX_decode_if  ibuf_deq_if();
    VX_decode_if  execute_if();
    VX_gpr_req_if gpr_req_if();
    VX_gpr_rsp_if gpr_rsp_if();

    wire scoreboard_delay;

    VX_ibuffer #(
        .CORE_ID(CORE_ID)
    ) ibuffer (
        .clk          (clk),
        .reset        (reset), 
        .freeze       (1'b0),
        .ibuf_enq_if  (decode_if),
        .ibuf_deq_if  (ibuf_deq_if) 
    );

    VX_scoreboard #(
        .CORE_ID(CORE_ID)
    ) scoreboard (
        .clk          (clk),
        .reset        (reset), 
        .ibuf_deq_if  (ibuf_deq_if),
        .writeback_if (writeback_if),
        .delay        (scoreboard_delay)
    );
        
    assign gpr_req_if.wid = ibuf_deq_if.wid;
    assign gpr_req_if.rs1 = ibuf_deq_if.rs1;
    assign gpr_req_if.rs2 = ibuf_deq_if.rs2;
    assign gpr_req_if.rs3 = ibuf_deq_if.rs3;

    VX_gpr_stage #(
        .CORE_ID(CORE_ID)
    ) gpr_stage (
        .clk          (clk),      
        .reset        (reset),          
        .writeback_if (writeback_if),
        .gpr_req_if   (gpr_req_if),
        .gpr_rsp_if   (gpr_rsp_if)
    );

    assign execute_if.valid     = ibuf_deq_if.valid && ~scoreboard_delay;
    assign execute_if.wid       = ibuf_deq_if.wid;
    assign execute_if.tmask     = ibuf_deq_if.tmask;
    assign execute_if.PC        = ibuf_deq_if.PC;
    assign execute_if.ex_type   = ibuf_deq_if.ex_type;    
    assign execute_if.op_type   = ibuf_deq_if.op_type; 
    assign execute_if.op_mod    = ibuf_deq_if.op_mod;    
    assign execute_if.wb        = ibuf_deq_if.wb;
    assign execute_if.rd        = ibuf_deq_if.rd;
    assign execute_if.rs1       = ibuf_deq_if.rs1;
    assign execute_if.imm       = ibuf_deq_if.imm;        
    assign execute_if.use_PC    = ibuf_deq_if.use_PC;
    assign execute_if.use_imm   = ibuf_deq_if.use_imm;

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
    assign ibuf_deq_if.ready = !scoreboard_delay && execute_if.ready;     

    `SCOPE_ASSIGN (issue_fire,        ibuf_deq_if.valid && ibuf_deq_if.ready);
    `SCOPE_ASSIGN (issue_wid,         ibuf_deq_if.wid);
    `SCOPE_ASSIGN (issue_tmask,       ibuf_deq_if.tmask);
    `SCOPE_ASSIGN (issue_pc,          ibuf_deq_if.PC);
    `SCOPE_ASSIGN (issue_ex_type,     ibuf_deq_if.ex_type);
    `SCOPE_ASSIGN (issue_op_type,     ibuf_deq_if.op_type);
    `SCOPE_ASSIGN (issue_op_mod,      ibuf_deq_if.op_mod);
    `SCOPE_ASSIGN (issue_wb,          ibuf_deq_if.wb);
    `SCOPE_ASSIGN (issue_rd,          ibuf_deq_if.rd);
    `SCOPE_ASSIGN (issue_rs1,         ibuf_deq_if.rs1);
    `SCOPE_ASSIGN (issue_rs2,         ibuf_deq_if.rs2);
    `SCOPE_ASSIGN (issue_rs3,         ibuf_deq_if.rs3);
    `SCOPE_ASSIGN (issue_imm,         ibuf_deq_if.imm);
    `SCOPE_ASSIGN (issue_use_pc,      ibuf_deq_if.use_PC);
    `SCOPE_ASSIGN (issue_use_imm,     ibuf_deq_if.use_imm);
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
    reg [43:0] perf_ibf_stalls;
    reg [43:0] perf_scb_stalls;
    reg [43:0] perf_alu_stalls;
    reg [43:0] perf_lsu_stalls;
    reg [43:0] perf_csr_stalls;
    reg [43:0] perf_gpu_stalls;
`ifdef EXT_F_ENABLE
    reg [43:0] perf_fpu_stalls;
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
                perf_ibf_stalls <= perf_ibf_stalls  + 44'd1;
            end
            if (ibuf_deq_if.valid & scoreboard_delay) begin 
                perf_scb_stalls <= perf_scb_stalls  + 44'd1;
            end
            if (alu_req_if.valid & !alu_req_if.ready) begin
                perf_alu_stalls <= perf_alu_stalls + 44'd1;
            end
            if (lsu_req_if.valid & !lsu_req_if.ready) begin
                perf_lsu_stalls <= perf_lsu_stalls + 44'd1;
            end
            if (csr_req_if.valid & !csr_req_if.ready) begin
                perf_csr_stalls <= perf_csr_stalls + 44'd1;
            end
            if (gpu_req_if.valid & !gpu_req_if.ready) begin
                perf_gpu_stalls <= perf_gpu_stalls + 44'd1;
            end
        `ifdef EXT_F_ENABLE
            if (fpu_req_if.valid & !fpu_req_if.ready) begin
                perf_fpu_stalls <= perf_fpu_stalls + 44'd1;
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
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=ALU, tmask=%b, rd=%0d, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, alu_req_if.wid, alu_req_if.PC, alu_req_if.tmask, alu_req_if.rd, alu_req_if.rs1_data, alu_req_if.rs2_data);   
        end
        if (lsu_req_if.valid && lsu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=LSU, tmask=%b, rd=%0d, baddr=%0h, offset=%0h, data=%0h", $time, CORE_ID, lsu_req_if.wid, lsu_req_if.PC, lsu_req_if.tmask, lsu_req_if.rd, lsu_req_if.base_addr, lsu_req_if.offset, lsu_req_if.store_data);   
        end
        if (csr_req_if.valid && csr_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=CSR, tmask=%b, rd=%0d, addr=%0h, rs1_data=%0h", $time, CORE_ID, csr_req_if.wid, csr_req_if.PC, csr_req_if.tmask, csr_req_if.rd, csr_req_if.addr, csr_req_if.rs1_data);   
        end
        if (fpu_req_if.valid && fpu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=FPU, tmask=%b, rd=%0d, rs1_data=%0h, rs2_data=%0h, rs3_data=%0h", $time, CORE_ID, fpu_req_if.wid, fpu_req_if.PC, fpu_req_if.tmask, fpu_req_if.rd, fpu_req_if.rs1_data, fpu_req_if.rs2_data, fpu_req_if.rs3_data);   
        end
        if (gpu_req_if.valid && gpu_req_if.ready) begin
            $display("%t: core%0d-issue: wid=%0d, PC=%0h, ex=GPU, tmask=%b, rd=%0d, rs1_data=%0h, rs2_data=%0h", $time, CORE_ID, gpu_req_if.wid, gpu_req_if.PC, gpu_req_if.tmask, gpu_req_if.rd, gpu_req_if.rs1_data, gpu_req_if.rs2_data);   
        end
    end
`endif

endmodule