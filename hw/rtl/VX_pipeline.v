`include "VX_define.vh"

module VX_pipeline #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_SIGNALS_ICACHE_IO
    `SCOPE_SIGNALS_DCACHE_IO
    `SCOPE_SIGNALS_CORE_IO
    `SCOPE_SIGNALS_BE_IO
    
    // Clock
    input wire                              clk,
    input wire                              reset,

    // Dcache core request
    output wire [`NUM_THREADS-1:0]          dcache_req_valid,
    output wire [`NUM_THREADS-1:0]          dcache_req_rw,
    output wire [`NUM_THREADS-1:0][3:0]     dcache_req_byteen,
    output wire [`NUM_THREADS-1:0][29:0]    dcache_req_addr,
    output wire [`NUM_THREADS-1:0][31:0]    dcache_req_data,
    output wire [`DCORE_TAG_WIDTH-1:0]      dcache_req_tag,    
    input wire                              dcache_req_ready,

    // Dcache core reponse    
    input wire [`NUM_THREADS-1:0]           dcache_rsp_valid,
    input wire [`NUM_THREADS-1:0][31:0]     dcache_rsp_data,
    input wire [`DCORE_TAG_WIDTH-1:0]       dcache_rsp_tag,    
    output wire                             dcache_rsp_ready,      

    // Icache core request
    output wire                             icache_req_valid,
    output wire                             icache_req_rw,
    output wire [3:0]                       icache_req_byteen,
    output wire [29:0]                      icache_req_addr,
    output wire [31:0]                      icache_req_data,
    output wire [`ICORE_TAG_WIDTH-1:0]      icache_req_tag,    
    input wire                              icache_req_ready,

    // Icache core response    
    input wire                              icache_rsp_valid,
    input wire [31:0]                       icache_rsp_data,
    input wire [`ICORE_TAG_WIDTH-1:0]       icache_rsp_tag,    
    output wire                             icache_rsp_ready,      

    // Status
    output wire                             busy, 
    output wire                             ebreak
);

`DEBUG_BEGIN
    wire scheduler_empty;
`DEBUG_END

    wire memory_delay;
    wire exec_delay;
    wire gpr_stage_delay;
    wire schedule_delay;

    // Dcache
    VX_cache_core_req_if #(
        .NUM_REQUESTS(`NUM_THREADS), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`DCORE_TAG_ID_BITS)
    ) core_dcache_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`NUM_THREADS), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`DCORE_TAG_ID_BITS)
    ) core_dcache_rsp_if();

    // Icache 
    VX_cache_core_req_if #(
        .NUM_REQUESTS(1), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`ICORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`ICORE_TAG_ID_BITS)
    )  core_icache_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(1), 
        .WORD_SIZE(4), 
        .CORE_TAG_WIDTH(`ICORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`ICORE_TAG_ID_BITS)
    )  core_icache_rsp_if();

    // Front-end to Back-end
    VX_frE_to_bckE_req_if    bckE_req_if();

    // Back-end to Front-end
    VX_wb_if                 writeback_if(); 
    VX_branch_rsp_if         branch_rsp_if();
    VX_jal_rsp_if            jal_rsp_if();   

    // Warp controls
    VX_warp_ctl_if           warp_ctl_if();

    VX_front_end #(
        .CORE_ID(CORE_ID)
    ) front_end (
        `SCOPE_SIGNALS_ICACHE_ATTACH

        .clk            (clk),
        .reset          (reset),
        .warp_ctl_if    (warp_ctl_if),
        .bckE_req_if    (bckE_req_if),
        .schedule_delay (schedule_delay),
        .icache_rsp_if  (core_icache_rsp_if),
        .icache_req_if  (core_icache_req_if),
        .jal_rsp_if     (jal_rsp_if),
        .branch_rsp_if  (branch_rsp_if),
        .busy           (busy)
    );

    VX_scheduler scheduler (
        .clk            (clk),
        .reset          (reset),
        .memory_delay   (memory_delay),
        .exec_delay     (exec_delay),
        .gpr_stage_delay(gpr_stage_delay),
        .bckE_req_if    (bckE_req_if),
        .writeback_if   (writeback_if),
        .schedule_delay (schedule_delay),
        .is_empty       (scheduler_empty)
    );

    VX_back_end #(
        .CORE_ID(CORE_ID)
    ) back_end (
        `SCOPE_SIGNALS_DCACHE_ATTACH
        `SCOPE_SIGNALS_BE_ATTACH

        .clk             (clk),
        .reset           (reset),
        .schedule_delay  (schedule_delay),
        .warp_ctl_if     (warp_ctl_if),
        .bckE_req_if     (bckE_req_if),
        .jal_rsp_if      (jal_rsp_if),
        .branch_rsp_if   (branch_rsp_if),    
        .dcache_req_if   (core_dcache_req_if),
        .dcache_rsp_if   (core_dcache_rsp_if),
        .writeback_if    (writeback_if),
        .mem_delay       (memory_delay),
        .exec_delay      (exec_delay),
        .gpr_stage_delay (gpr_stage_delay),        
        .ebreak          (ebreak)
    );

    assign dcache_req_valid  = core_dcache_req_if.core_req_valid;
    assign dcache_req_rw     = core_dcache_req_if.core_req_rw;
    assign dcache_req_byteen = core_dcache_req_if.core_req_byteen;
    assign dcache_req_addr   = core_dcache_req_if.core_req_addr;
    assign dcache_req_data   = core_dcache_req_if.core_req_data;
    assign dcache_req_tag    = core_dcache_req_if.core_req_tag;
    assign core_dcache_req_if.core_req_ready = dcache_req_ready;

    assign core_dcache_rsp_if.core_rsp_valid = dcache_rsp_valid;
    assign core_dcache_rsp_if.core_rsp_data  = dcache_rsp_data;
    assign core_dcache_rsp_if.core_rsp_tag   = dcache_rsp_tag;
    assign dcache_rsp_ready = core_dcache_rsp_if.core_rsp_ready;

    assign icache_req_valid  = core_icache_req_if.core_req_valid;
    assign icache_req_rw     = core_icache_req_if.core_req_rw;
    assign icache_req_byteen = core_icache_req_if.core_req_byteen;
    assign icache_req_addr   = core_icache_req_if.core_req_addr;
    assign icache_req_data   = core_icache_req_if.core_req_data;
    assign icache_req_tag    = core_icache_req_if.core_req_tag;
    assign core_icache_req_if.core_req_ready = icache_req_ready;

    assign core_icache_rsp_if.core_rsp_valid = icache_rsp_valid;
    assign core_icache_rsp_if.core_rsp_data  = icache_rsp_data;
    assign core_icache_rsp_if.core_rsp_tag   = icache_rsp_tag;
    assign icache_rsp_ready = core_icache_rsp_if.core_rsp_ready;

    `SCOPE_ASSIGN(scope_schedule_delay, schedule_delay);    
    `SCOPE_ASSIGN(scope_memory_delay, memory_delay);
    `SCOPE_ASSIGN(scope_exec_delay, exec_delay);
    `SCOPE_ASSIGN(scope_gpr_stage_delay, gpr_stage_delay);

`ifdef DBG_PRINT_WB
    always_ff @(posedge clk) begin
        if ((| writeback_if.valid) && (writeback_if.wb != 0)) begin
            $display("%t: Writeback: wid=%0d, rd=%0d, data=%0h", $time, writeback_if.warp_num, writeback_if.rd, writeback_if.data);
        end
    end
`endif

endmodule // Vortex