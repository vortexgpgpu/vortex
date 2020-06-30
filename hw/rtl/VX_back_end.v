`include "VX_define.vh"

module VX_back_end #(
    parameter CORE_ID = 0
) (
    `SCOPE_SIGNALS_LSU_IO
    `SCOPE_SIGNALS_BE_IO

    input wire clk, 
    input wire reset, 

    // IO CSR
    VX_csr_req_if          io_csr_req,
    VX_wb_if               io_csr_rsp,

    input wire             schedule_delay,

    VX_cache_core_req_if   dcache_req_if,
    VX_cache_core_rsp_if   dcache_rsp_if,
    VX_jal_rsp_if          jal_rsp_if,
    VX_branch_rsp_if       branch_rsp_if,

    VX_backend_req_if      bckE_req_if,
    VX_wb_if               writeback_if,

    VX_warp_ctl_if         warp_ctl_if,

    output wire            mem_delay,
    output wire            exec_delay,
    output wire            gpr_stage_delay,

    output wire            ebreak
);

    wire                no_slot_mem;
    wire                no_slot_exec;


    // LSU input + output
    VX_lsu_req_if       lsu_req_if();
    VX_wb_if            mem_wb_if();

    // Exec unit input + output
    VX_exec_unit_req_if exec_unit_req_if();
    VX_wb_if            inst_exec_wb_if();

    // GPU unit input
    VX_gpu_inst_req_if  gpu_inst_req_if();

    // CSR unit inputs
    VX_csr_req_if       csr_req_if();
    VX_wb_if            csr_wb_if();
    wire                no_slot_csr;
    wire                stall_gpr_csr;

    VX_gpr_stage gpr_stage (
        .clk                (clk),
        .reset              (reset),
        .schedule_delay     (schedule_delay),
        .writeback_if       (writeback_if),
        .bckE_req_if        (bckE_req_if),
        // New
        .exec_unit_req_if   (exec_unit_req_if),
        .lsu_req_if         (lsu_req_if),
        .gpu_inst_req_if    (gpu_inst_req_if),
        .csr_req_if         (csr_req_if),
        .stall_gpr_csr      (stall_gpr_csr),
        // End new
        .memory_delay       (mem_delay),
        .exec_delay         (exec_delay),
        .gpr_stage_delay    (gpr_stage_delay)
    );

    assign ebreak = exec_unit_req_if.is_etype && (| exec_unit_req_if.valid);

    VX_lsu_unit #(
        .CORE_ID(CORE_ID)
    ) lsu_unit (
        `SCOPE_SIGNALS_LSU_BIND

        .clk            (clk),
        .reset          (reset),
        .lsu_req_if     (lsu_req_if),
        .mem_wb_if_p1   (mem_wb_if),
        .dcache_req_if  (dcache_req_if),
        .dcache_rsp_if  (dcache_rsp_if),
        .delay          (mem_delay),
        .no_slot_mem    (no_slot_mem)
    );

    VX_exec_unit exec_unit (
        .clk            (clk),
        .reset          (reset),
        .exec_unit_req_if(exec_unit_req_if),
        .inst_exec_wb_if(inst_exec_wb_if),
        .jal_rsp_if     (jal_rsp_if),
        .branch_rsp_if  (branch_rsp_if),
        .delay          (exec_delay),
        .no_slot_exec   (no_slot_exec)
    );

    VX_gpu_inst gpu_inst (
        .gpu_inst_req_if(gpu_inst_req_if),
        .warp_ctl_if    (warp_ctl_if)
    );

    VX_csr_req_if issued_csr_req();

    VX_wb_if       csr_pipe_rsp();

    VX_csr_arbiter csr_arbiter (
        .clk           (clk),
        .reset         (reset),
        .csr_pipe_stall(stall_gpr_csr),
        .core_csr_req  (csr_req_if),
        .io_csr_req    (io_csr_req),
        .issued_csr_req(issued_csr_req),

        .csr_pipe_rsp  (csr_pipe_rsp),
        .csr_wb_if     (csr_wb_if),
        .csr_io_rsp    (io_csr_rsp)
    
    );

    VX_csr_pipe #(
        .CORE_ID(CORE_ID)
    ) csr_pipe (
        .clk            (clk),
        .reset          (reset),
        .no_slot_csr    (no_slot_csr),
        .csr_req_if     (issued_csr_req),
        .writeback_if   (writeback_if),
        .csr_wb_if      (csr_pipe_rsp),
        .stall_gpr_csr  (stall_gpr_csr)
    );

    VX_writeback writeback (
        .clk            (clk),
        .reset          (reset),
        .mem_wb_if      (mem_wb_if),
        .inst_exec_wb_if(inst_exec_wb_if),
        .csr_wb_if      (csr_wb_if),

        .writeback_if   (writeback_if),
        .no_slot_mem    (no_slot_mem),
        .no_slot_exec   (no_slot_exec),
        .no_slot_csr    (no_slot_csr)
    );   

    `SCOPE_ASSIGN(scope_decode_valid,       bckE_req_if.valid);
    `SCOPE_ASSIGN(scope_decode_warp_num,    bckE_req_if.warp_num);
    `SCOPE_ASSIGN(scope_decode_curr_PC,     bckE_req_if.curr_PC);    
    `SCOPE_ASSIGN(scope_decode_is_jal,      bckE_req_if.is_jal);
    `SCOPE_ASSIGN(scope_decode_rs1,         bckE_req_if.rs1);
    `SCOPE_ASSIGN(scope_decode_rs2,         bckE_req_if.rs2);

    `SCOPE_ASSIGN(scope_execute_valid,      exec_unit_req_if.valid);    
    `SCOPE_ASSIGN(scope_execute_warp_num,   exec_unit_req_if.warp_num);
    `SCOPE_ASSIGN(scope_execute_curr_PC,    exec_unit_req_if.curr_PC);    
    `SCOPE_ASSIGN(scope_execute_rd,         exec_unit_req_if.rd);
    `SCOPE_ASSIGN(scope_execute_a,          exec_unit_req_if.a_reg_data);
    `SCOPE_ASSIGN(scope_execute_b,          exec_unit_req_if.b_reg_data);   
        
    `SCOPE_ASSIGN(scope_writeback_valid,    writeback_if.valid);    
    `SCOPE_ASSIGN(scope_writeback_warp_num, writeback_if.warp_num);
    `SCOPE_ASSIGN(scope_writeback_curr_PC,  writeback_if.curr_PC);  
    `SCOPE_ASSIGN(scope_writeback_wb,       writeback_if.wb);      
    `SCOPE_ASSIGN(scope_writeback_rd,       writeback_if.rd);
    `SCOPE_ASSIGN(scope_writeback_data,     writeback_if.data);

endmodule
