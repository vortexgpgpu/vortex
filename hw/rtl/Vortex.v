`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex #( 
    parameter CORE_ID = 0
) (        
    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // DRAM Dcache request
    output wire                         D_dram_req_read,
    output wire                         D_dram_req_write,    
    output wire [`DDRAM_ADDR_WIDTH-1:0] D_dram_req_addr,
    output wire [`DDRAM_LINE_WIDTH-1:0] D_dram_req_data,
    output wire [`DDRAM_TAG_WIDTH-1:0]  D_dram_req_tag,
    input  wire                         D_dram_req_ready,

    // DRAM Dcache reponse    
    input  wire                         D_dram_rsp_valid,
    input  wire [`DDRAM_LINE_WIDTH-1:0] D_dram_rsp_data,
    input  wire [`DDRAM_TAG_WIDTH-1:0]  D_dram_rsp_tag,
    output wire                         D_dram_rsp_ready,

    // DRAM Icache request
    output wire                         I_dram_req_read,
    output wire                         I_dram_req_write,    
    output wire [`IDRAM_ADDR_WIDTH-1:0] I_dram_req_addr,
    output wire [`IDRAM_LINE_WIDTH-1:0] I_dram_req_data,
    output wire [`IDRAM_TAG_WIDTH-1:0]  I_dram_req_tag,
    input  wire                         I_dram_req_ready,

    // DRAM Icache response    
    input  wire                         I_dram_rsp_valid,
    input  wire [`IDRAM_LINE_WIDTH-1:0] I_dram_rsp_data,
    input  wire [`IDRAM_TAG_WIDTH-1:0]  I_dram_rsp_tag,
    output wire                         I_dram_rsp_ready,

    // Snoop request
    input  wire                         snp_req_valid,
    input  wire [`DDRAM_ADDR_WIDTH-1:0] snp_req_addr,
    input  wire [`DSNP_TAG_WIDTH-1:0]   snp_req_tag,
    output wire                         snp_req_ready,

    output wire                         snp_rsp_valid,
    output wire [`DSNP_TAG_WIDTH-1:0]   snp_rsp_tag,
    input  wire                         snp_rsp_ready,

    // I/O request
    output wire                         io_req_read,
    output wire                         io_req_write,    
    output wire[31:0]                   io_req_addr,
    output wire[31:0]                   io_req_data,
    output wire[`BYTE_EN_BITS-1:0]      io_req_byteen,
    output wire[`CORE_REQ_TAG_WIDTH-1:0] io_req_tag,  
    input wire                          io_req_ready,

    // I/O response
    input wire                          io_rsp_valid,
    input wire[31:0]                    io_rsp_data,
    input wire[`CORE_REQ_TAG_WIDTH-1:0] io_rsp_tag,
    output wire                         io_rsp_ready,

    // Status
    output wire                         busy, 
    output wire                         ebreak
);
`DEBUG_BEGIN
    wire scheduler_empty;
`DEBUG_END

    wire memory_delay;
    wire exec_delay;
    wire gpr_stage_delay;
    wire schedule_delay;

    // Dcache Interfaces
    VX_cache_core_req_if #(
        .NUM_REQUESTS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`CORE_TAG_ID_BITS)
    ) dcache_core_req_if(), io_core_req_if(), dcache_io_core_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`CORE_TAG_ID_BITS)
    ) dcache_core_rsp_if(), io_core_rsp_if(), dcache_io_core_rsp_if();

    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
    ) dcache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
    ) dcache_dram_rsp_if();

    assign D_dram_req_write = dcache_dram_req_if.dram_req_write;
    assign D_dram_req_read  = dcache_dram_req_if.dram_req_read;
    assign D_dram_req_addr  = dcache_dram_req_if.dram_req_addr;
    assign D_dram_req_data  = dcache_dram_req_if.dram_req_data;
    assign D_dram_req_tag   = dcache_dram_req_if.dram_req_tag;
    assign dcache_dram_req_if.dram_req_ready = D_dram_req_ready;

    assign dcache_dram_rsp_if.dram_rsp_valid = D_dram_rsp_valid;
    assign dcache_dram_rsp_if.dram_rsp_data  = D_dram_rsp_data;
    assign dcache_dram_rsp_if.dram_rsp_tag   = D_dram_rsp_tag;
    assign D_dram_rsp_ready = dcache_dram_rsp_if.dram_rsp_ready;

    assign io_req_read   = (io_core_req_if.core_req_read[0] != `BYTE_EN_NO);
    assign io_req_write  = (io_core_req_if.core_req_write[0] != `BYTE_EN_NO);
    assign io_req_addr   = io_core_req_if.core_req_addr[0];
    assign io_req_data   = io_core_req_if.core_req_data[0];
    assign io_req_byteen = io_req_read ? io_core_req_if.core_req_read[0] : io_core_req_if.core_req_write[0];
    assign io_req_tag    = io_core_req_if.core_req_tag[0];
    assign io_core_req_if.core_req_ready = io_req_ready;

    assign io_core_rsp_if.core_rsp_valid[0] = io_rsp_valid;
    assign io_core_rsp_if.core_rsp_data[0]  = io_rsp_data;
    assign io_core_rsp_if.core_rsp_tag      = io_rsp_tag;    
    assign io_rsp_ready  = io_core_rsp_if.core_rsp_ready;
    
    // Icache interfaces
    VX_cache_core_req_if #(
        .NUM_REQUESTS(`INUM_REQUESTS), 
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`CORE_TAG_ID_BITS)
    )  icache_core_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`INUM_REQUESTS), 
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`CORE_REQ_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`CORE_TAG_ID_BITS)
    )  icache_core_rsp_if();
    
    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH(`IDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH(`IDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH(`IDRAM_TAG_WIDTH)
    ) icache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH(`IDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH(`IDRAM_TAG_WIDTH)
    ) icache_dram_rsp_if();

    assign I_dram_req_write = icache_dram_req_if.dram_req_write;
    assign I_dram_req_read  = icache_dram_req_if.dram_req_read;
    assign I_dram_req_addr  = icache_dram_req_if.dram_req_addr;
    assign I_dram_req_data  = icache_dram_req_if.dram_req_data;
    assign I_dram_req_tag   = icache_dram_req_if.dram_req_tag;
    assign icache_dram_req_if.dram_req_ready = I_dram_req_ready;

    assign icache_dram_rsp_if.dram_rsp_valid = I_dram_rsp_valid;    
    assign icache_dram_rsp_if.dram_rsp_data  = I_dram_rsp_data;
    assign icache_dram_rsp_if.dram_rsp_tag   = I_dram_rsp_tag;
    assign I_dram_rsp_ready = icache_dram_rsp_if.dram_rsp_ready;    

    ///////////////////////////////////////////////////////////////////////////////

    // Front-end to Back-end
    VX_frE_to_bckE_req_if    bckE_req_if();     // New instruction request to EXE/MEM

    // Back-end to Front-end
    VX_wb_if                 writeback_if();    // Writeback to GPRs
    VX_branch_rsp_if         branch_rsp_if();   // Branch Resolution to Fetch
    VX_jal_rsp_if            jal_rsp_if();      // Jump resolution to Fetch

    // Warp controls
    VX_warp_ctl_if           warp_ctl_if();

    // Cache snooping
    VX_cache_snp_req_if #(
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .SNP_TAG_WIDTH(`DSNP_TAG_WIDTH)
    ) dcache_snp_req_if();

    VX_cache_snp_rsp_if #(
        .SNP_TAG_WIDTH(`DSNP_TAG_WIDTH)
    ) dcache_snp_rsp_if();

    assign dcache_snp_req_if.snp_req_valid = snp_req_valid;
    assign dcache_snp_req_if.snp_req_addr  = snp_req_addr;
    assign dcache_snp_req_if.snp_req_tag   = snp_req_tag;
    assign snp_req_ready                   = dcache_snp_req_if.snp_req_ready;

    assign snp_rsp_valid = dcache_snp_rsp_if.snp_rsp_valid;
    assign snp_rsp_tag   = dcache_snp_rsp_if.snp_rsp_tag;
    assign dcache_snp_rsp_if.snp_rsp_ready = snp_rsp_ready;

    VX_front_end #(
        .CORE_ID(CORE_ID)
    ) front_end (
        .clk            (clk),
        .reset          (reset),
        .warp_ctl_if    (warp_ctl_if),
        .bckE_req_if    (bckE_req_if),
        .schedule_delay (schedule_delay),
        .icache_rsp_if  (icache_core_rsp_if),
        .icache_req_if  (icache_core_req_if),
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
        .clk             (clk),
        .reset           (reset),
        .schedule_delay  (schedule_delay),
        .warp_ctl_if     (warp_ctl_if),
        .bckE_req_if     (bckE_req_if),
        .jal_rsp_if      (jal_rsp_if),
        .branch_rsp_if   (branch_rsp_if),    
        .dcache_req_if   (dcache_io_core_req_if),
        .dcache_rsp_if   (dcache_io_core_rsp_if),
        .writeback_if    (writeback_if),
        .mem_delay       (memory_delay),
        .exec_delay      (exec_delay),
        .gpr_stage_delay (gpr_stage_delay),        
        .ebreak          (ebreak)
    );

    VX_dmem_ctrl dmem_ctrl (
        .clk                (clk),
        .reset              (reset),

        // Core <-> Dcache
        .dcache_core_req_if (dcache_core_req_if),
        .dcache_core_rsp_if (dcache_core_rsp_if),

        // Dram <-> Dcache
        .dcache_dram_req_if (dcache_dram_req_if),
        .dcache_dram_rsp_if (dcache_dram_rsp_if),
        .dcache_snp_req_if  (dcache_snp_req_if),
        .dcache_snp_rsp_if  (dcache_snp_rsp_if),

        // Core <-> Icache
        .icache_core_req_if (icache_core_req_if),
        .icache_core_rsp_if (icache_core_rsp_if),

        // Dram <-> Icache
        .icache_dram_req_if (icache_dram_req_if),
        .icache_dram_rsp_if (icache_dram_rsp_if)
    );

    // use "case equality" to handle uninitialized address value
    wire io_select = ((dcache_io_core_req_if.core_req_addr[0] >= `IO_BUS_BASE_ADDR) === 1'b1);

    VX_dcache_io_arb dcache_io_arb (
        .io_select          (io_select),
        .core_req_if        (dcache_io_core_req_if),
        .dcache_core_req_if (dcache_core_req_if),
        .io_core_req_if     (io_core_req_if),  
        .dcache_core_rsp_if (dcache_core_rsp_if),
        .io_core_rsp_if     (io_core_rsp_if),    
        .core_rsp_if        (dcache_io_core_rsp_if)
    );

endmodule // Vortex





