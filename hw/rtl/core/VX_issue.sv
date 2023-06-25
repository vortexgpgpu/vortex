`include "VX_define.vh"
`ifndef NDEBUG
`include "VX_trace.vh"
`endif

module VX_issue #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_pipeline_perf_if.issue perf_issue_if,
`endif

    VX_decode_if.slave      decode_if,
    VX_writeback_if.slave   writeback_if,   
    
    VX_alu_exe_if.master    alu_exe_if,
    VX_lsu_exe_if.master    lsu_exe_if,    
    VX_csr_exe_if.master    csr_exe_if,
`ifdef EXT_F_ENABLE
    VX_fpu_exe_if.master    fpu_exe_if,    
`endif
    VX_gpu_exe_if.master    gpu_exe_if
);
    VX_ibuffer_if       ibuffer_if();    
    VX_gpr_stage_if     gpr_stage_if();
    VX_writeback_if     sboard_wb_if();
    VX_scoreboard_if    scoreboard_if();
    VX_dispatch_if      dispatch_if();

    wire [3:0] in_use_regs;

    // GPR request interface
    assign gpr_stage_if.wid     = ibuffer_if.wid;
    assign gpr_stage_if.rs1     = ibuffer_if.rs1;
    assign gpr_stage_if.rs2     = ibuffer_if.rs2;
    assign gpr_stage_if.rs3     = ibuffer_if.rs3;

    // scoreboard writeback interface
    assign sboard_wb_if.valid   = writeback_if.valid;
    assign sboard_wb_if.uuid    = writeback_if.uuid;
    assign sboard_wb_if.wid     = writeback_if.wid;
    assign sboard_wb_if.tmask   = writeback_if.tmask;
    assign sboard_wb_if.PC      = writeback_if.PC;
    assign sboard_wb_if.rd      = writeback_if.rd;
    assign sboard_wb_if.data    = writeback_if.data;
    assign sboard_wb_if.eop     = writeback_if.eop;
    `UNUSED_VAR (sboard_wb_if.ready)
        
    // scoreboard interface
    assign scoreboard_if.valid  = ibuffer_if.valid && dispatch_if.ready;
    assign scoreboard_if.uuid   = ibuffer_if.uuid;
    assign scoreboard_if.wid    = ibuffer_if.wid;
    assign scoreboard_if.tmask  = ibuffer_if.tmask;
    assign scoreboard_if.PC     = ibuffer_if.PC;   
    assign scoreboard_if.wb     = ibuffer_if.wb;
    assign scoreboard_if.rd     = ibuffer_if.rd;
    assign scoreboard_if.rd_n   = ibuffer_if.rd_n;        
    assign scoreboard_if.rs1_n  = ibuffer_if.rs1_n;        
    assign scoreboard_if.rs2_n  = ibuffer_if.rs2_n;        
    assign scoreboard_if.rs3_n  = ibuffer_if.rs3_n;        
    assign scoreboard_if.wid_n  = ibuffer_if.wid_n;
    
    // dispatch interface
    assign dispatch_if.valid    = ibuffer_if.valid && scoreboard_if.ready;
    assign dispatch_if.uuid     = ibuffer_if.uuid;
    assign dispatch_if.wid      = ibuffer_if.wid;
    assign dispatch_if.tmask    = ibuffer_if.tmask;
    assign dispatch_if.PC       = ibuffer_if.PC;
    assign dispatch_if.ex_type  = ibuffer_if.ex_type;    
    assign dispatch_if.op_type  = ibuffer_if.op_type; 
    assign dispatch_if.op_mod   = ibuffer_if.op_mod;    
    assign dispatch_if.wb       = ibuffer_if.wb;
    assign dispatch_if.use_PC   = ibuffer_if.use_PC;
    assign dispatch_if.use_imm  = ibuffer_if.use_imm;
    assign dispatch_if.imm      = ibuffer_if.imm;
    assign dispatch_if.rd       = ibuffer_if.rd;
    assign dispatch_if.rs1_data = gpr_stage_if.rs1_data;
    assign dispatch_if.rs2_data = gpr_stage_if.rs2_data;
    assign dispatch_if.rs3_data = gpr_stage_if.rs3_data;

    // issue the instruction
    assign ibuffer_if.ready = scoreboard_if.ready && dispatch_if.ready;

    `RESET_RELAY (ibuf_reset, reset);
    `RESET_RELAY (scoreboard_reset, reset);
    `RESET_RELAY (gpr_reset, reset);
    `RESET_RELAY (dispatch_reset, reset);

    VX_ibuffer #(
        .CORE_ID (CORE_ID)
    ) ibuffer (
        .clk        (clk),
        .reset      (ibuf_reset), 
        .decode_if  (decode_if),
        .ibuffer_if (ibuffer_if) 
    );

    VX_scoreboard #(
        .CORE_ID (CORE_ID)
    ) scoreboard (
        .clk           (clk),
        .reset         (scoreboard_reset),         
        .writeback_if  (writeback_if),
        .scoreboard_if (scoreboard_if),
        .in_use_regs   (in_use_regs)
    );

    VX_gpr_stage #(
        .CORE_ID (CORE_ID)
    ) gpr_stage (
        .clk          (clk),      
        .reset        (gpr_reset),          
        .writeback_if (writeback_if),
        .gpr_stage_if (gpr_stage_if)
    );

    VX_dispatch dispatch (
        .clk        (clk),      
        .reset      (dispatch_reset),
        .dispatch_if(dispatch_if),
        .alu_exe_if (alu_exe_if),
        .lsu_exe_if (lsu_exe_if),        
        .csr_exe_if (csr_exe_if),
    `ifdef EXT_F_ENABLE
        .fpu_exe_if (fpu_exe_if),
    `endif
        .gpu_exe_if (gpu_exe_if)
    );

    wire ibuffer_if_fire = ibuffer_if.valid && ibuffer_if.ready;
    
    reg [31:0] timeout_ctr;
    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr <= '0;
        end else begin        
            if (ibuffer_if.valid && ~ibuffer_if.ready) begin
            `ifdef DBG_TRACE_CORE_PIPELINE
                `TRACE(3, ("%d: *** core%0d-stall: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, wb=%0d, cycles=%0d, inuse=%b%b%b%b, dispatch=%b (#%0d)\n",
                    $time, CORE_ID, ibuffer_if.wid, ibuffer_if.PC, ibuffer_if.tmask, ibuffer_if.rd, ibuffer_if.wb, timeout_ctr,
                    in_use_regs[0], in_use_regs[1], in_use_regs[2], in_use_regs[3], ~dispatch_if.ready, ibuffer_if.uuid));
            `endif
                timeout_ctr <= timeout_ctr + 1;
            end else if (ibuffer_if_fire) begin
                timeout_ctr <= '0;
            end
        end
    end
    `RUNTIME_ASSERT(timeout_ctr < `STALL_TIMEOUT,
                    ("%t: *** core%0d-issue-timeout: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, wb=%0d, inuse=%b%b%b%b, dispatch=%b (#%0d)",
                        $time, CORE_ID, ibuffer_if.wid, ibuffer_if.PC, ibuffer_if.tmask, ibuffer_if.rd, ibuffer_if.wb, 
                        in_use_regs[0], in_use_regs[1], in_use_regs[2], in_use_regs[3], ~dispatch_if.ready, ibuffer_if.uuid));    

`ifdef DBG_SCOPE_ISSUE
    if (CORE_ID == 0) begin
    `ifdef SCOPE
        localparam UUID_WIDTH = `UP(`UUID_BITS);
        wire scoreboard_if_not_ready = ~scoreboard_if.ready; 
        wire dispatch_if_not_ready = ~dispatch_if.ready;
        wire writeback_if_valid = writeback_if.valid;
        VX_scope_tap #(
            .SCOPE_ID (2),
            .TRIGGERW (5),
            .PROBEW   (UUID_WIDTH + `NUM_THREADS + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS +
                1 + (`NR_BITS * 4) + `XLEN + 1 + 1 + (`NUM_THREADS * 3 * `XLEN) +
                UUID_WIDTH + `NUM_THREADS + `NR_BITS + (`NUM_THREADS*`XLEN) + 1)
        ) scope_tap (
            .clk(clk),
            .reset(scope_reset),
            .start(1'b0),
            .stop(1'b0),
            .triggers({
                reset, 
                ibuffer_if_fire, 
                scoreboard_if_not_ready, 
                dispatch_if_not_ready, 
                writeback_if_valid
            }),
            .probes({
                ibuffer_if.uuid,
                ibuffer_if.tmask,
                ibuffer_if.ex_type,
                ibuffer_if.op_type,
                ibuffer_if.op_mod,
                ibuffer_if.wb,
                ibuffer_if.rd,
                ibuffer_if.rs1,
                ibuffer_if.rs2,
                ibuffer_if.rs3,
                ibuffer_if.imm,
                ibuffer_if.use_PC,
                ibuffer_if.use_imm,
                dispatch_if.rs1_data,
                dispatch_if.rs2_data,
                dispatch_if.rs3_data,
                writeback_if.uuid,
                writeback_if.tmask,
                writeback_if.rd,
                writeback_if.data,
                writeback_if.eop
            }),
            .bus_in(scope_bus_in),
            .bus_out(scope_bus_out)
        );
    `endif        
    `ifdef CHIPSCOPE
        ila_issue ila_issue_inst (
            .clk    (clk),
            .probe0 ({ibuffer_if.uuid, ibuffer.rs3, ibuffer.rs2, ibuffer.rs1, ibuffer_if.PC, ibuffer_if.tmask, ibuffer_if.wid, ibuffer_if.ex_type, ibuffer_if.op_type, ibuffer_if.ready, ibuffer_if.valid, in_use_regs, scoreboard_if.ready, dispatch_if.ready, ibuffer_if.ready, ibuffer_if.valid}),
            .probe1 ({writeback_if.uuid, writeback_if.data[0], writeback_if.PC, writeback_if.tmask, writeback_if.wid, writeback_if.eop, writeback_if.valid})
        );
    `endif
    end
`else
    `SCOPE_IO_UNUSED()
`endif

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
            perf_ibf_stalls <= '0;
            perf_scb_stalls <= '0;
            perf_alu_stalls <= '0;
            perf_lsu_stalls <= '0;
            perf_csr_stalls <= '0;
            perf_gpu_stalls <= '0;
        `ifdef EXT_F_ENABLE
            perf_fpu_stalls <= '0;
        `endif
        end else begin
            if (decode_if.valid && ~decode_if.ready) begin
                perf_ibf_stalls <= perf_ibf_stalls  + `PERF_CTR_BITS'(1);
            end
            if (scoreboard_if.valid && ~scoreboard_if.ready) begin 
                perf_scb_stalls <= perf_scb_stalls  + `PERF_CTR_BITS'(1);
            end
            if (dispatch_if.valid && ~dispatch_if.ready) begin
                case (dispatch_if.ex_type)
                `EX_ALU: perf_alu_stalls <= perf_alu_stalls + `PERF_CTR_BITS'(1);
            `ifdef EXT_F_ENABLE
                `EX_FPU: perf_fpu_stalls <= perf_fpu_stalls + `PERF_CTR_BITS'(1);
            `endif
                `EX_LSU: perf_lsu_stalls <= perf_lsu_stalls + `PERF_CTR_BITS'(1);
                `EX_CSR: perf_csr_stalls <= perf_csr_stalls + `PERF_CTR_BITS'(1);
                `EX_GPU: perf_gpu_stalls <= perf_gpu_stalls + `PERF_CTR_BITS'(1);
                default:;
                endcase
            end
        end
    end
    
    assign perf_issue_if.ibf_stalls = perf_ibf_stalls;
    assign perf_issue_if.scb_stalls = perf_scb_stalls; 
    assign perf_issue_if.alu_stalls = perf_alu_stalls;
    assign perf_issue_if.lsu_stalls = perf_lsu_stalls;
    assign perf_issue_if.csr_stalls = perf_csr_stalls;
    assign perf_issue_if.gpu_stalls = perf_gpu_stalls;
`ifdef EXT_F_ENABLE
    assign perf_issue_if.fpu_stalls = perf_fpu_stalls;
`endif
`endif

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (dispatch_if.valid && dispatch_if.ready) begin
            `TRACE(1, ("%d: core%0d-issue: wid=%0d, PC=0x%0h, ex=", $time, CORE_ID, dispatch_if.wid, dispatch_if.PC));
            trace_ex_type(1, dispatch_if.ex_type);
            `TRACE(1, (", op="));
            trace_ex_op(1, dispatch_if.ex_type, dispatch_if.op_type, dispatch_if.op_mod, dispatch_if.imm);
            `TRACE(1, (", mod=%0d, tmask=%b, wb=%b, rd=%0d, rs1_data=",  dispatch_if.op_mod, dispatch_if.tmask, dispatch_if.wb, dispatch_if.rd));
            `TRACE_ARRAY1D(1, dispatch_if.rs1_data, `NUM_THREADS);
            `TRACE(1, (", rs2_data="));
            `TRACE_ARRAY1D(1, dispatch_if.rs2_data, `NUM_THREADS);
            `TRACE(1, (", rs3_data="));
            `TRACE_ARRAY1D(1, dispatch_if.rs3_data, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", dispatch_if.uuid));
        end
    end
`endif

endmodule
