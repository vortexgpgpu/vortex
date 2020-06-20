`include "VX_define.vh"

module Vortex #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_SIGNALS_ISTAGE_IO
    `SCOPE_SIGNALS_LSU_IO
    `SCOPE_SIGNALS_CORE_IO
    `SCOPE_SIGNALS_ICACHE_IO
    `SCOPE_SIGNALS_PIPELINE_IO
    `SCOPE_SIGNALS_BE_IO
    
    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DRAM Dcache request
    output wire                             D_dram_req_valid,
    output wire                             D_dram_req_rw,    
    output wire [`DDRAM_BYTEEN_WIDTH-1:0]   D_dram_req_byteen,
    output wire [`DDRAM_ADDR_WIDTH-1:0]     D_dram_req_addr,
    output wire [`DDRAM_LINE_WIDTH-1:0]     D_dram_req_data,
    output wire [`DDRAM_TAG_WIDTH-1:0]      D_dram_req_tag,
    input  wire                             D_dram_req_ready,

    // DRAM Dcache reponse    
    input  wire                             D_dram_rsp_valid,
    input  wire [`DDRAM_LINE_WIDTH-1:0]     D_dram_rsp_data,
    input  wire [`DDRAM_TAG_WIDTH-1:0]      D_dram_rsp_tag,
    output wire                             D_dram_rsp_ready,

    // DRAM Icache request
    output wire                             I_dram_req_valid,
    output wire                             I_dram_req_rw,    
    output wire [`IDRAM_BYTEEN_WIDTH-1:0]   I_dram_req_byteen,
    output wire [`IDRAM_ADDR_WIDTH-1:0]     I_dram_req_addr,
    output wire [`IDRAM_LINE_WIDTH-1:0]     I_dram_req_data,
    output wire [`IDRAM_TAG_WIDTH-1:0]      I_dram_req_tag,
    input  wire                             I_dram_req_ready,

    // DRAM Icache response    
    input  wire                             I_dram_rsp_valid,
    input  wire [`IDRAM_LINE_WIDTH-1:0]     I_dram_rsp_data,
    input  wire [`IDRAM_TAG_WIDTH-1:0]      I_dram_rsp_tag,
    output wire                             I_dram_rsp_ready,

    // Snoop request
    input  wire                             snp_req_valid,
    input  wire [`DDRAM_ADDR_WIDTH-1:0]     snp_req_addr,
    input wire                              snp_req_invalidate,
    input  wire [`DSNP_TAG_WIDTH-1:0]       snp_req_tag,
    output wire                             snp_req_ready,

    output wire                             snp_rsp_valid,
    output wire [`DSNP_TAG_WIDTH-1:0]       snp_rsp_tag,
    input  wire                             snp_rsp_ready,

    // I/O request
    output wire                             io_req_valid,
    output wire                             io_req_rw,    
    output wire [3:0]                       io_req_byteen,  
    output wire [29:0]                      io_req_addr,
    output wire [31:0]                      io_req_data,    
    output wire [`DCORE_TAG_WIDTH-1:0]      io_req_tag,  
    input wire                              io_req_ready,

    // I/O response
    input wire                              io_rsp_valid,
    input wire [31:0]                       io_rsp_data,
    input wire [`DCORE_TAG_WIDTH-1:0]       io_rsp_tag,
    output wire                             io_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
    // Dcache Interfaces   

    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
    ) dcache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
    ) dcache_dram_rsp_if();

    assign D_dram_req_valid = dcache_dram_req_if.dram_req_valid;
    assign D_dram_req_rw    = dcache_dram_req_if.dram_req_rw;
    assign D_dram_req_byteen= dcache_dram_req_if.dram_req_byteen;
    assign D_dram_req_addr  = dcache_dram_req_if.dram_req_addr;
    assign D_dram_req_data  = dcache_dram_req_if.dram_req_data;
    assign D_dram_req_tag   = dcache_dram_req_if.dram_req_tag;
    assign dcache_dram_req_if.dram_req_ready = D_dram_req_ready;

    assign dcache_dram_rsp_if.dram_rsp_valid = D_dram_rsp_valid;
    assign dcache_dram_rsp_if.dram_rsp_data  = D_dram_rsp_data;
    assign dcache_dram_rsp_if.dram_rsp_tag   = D_dram_rsp_tag;
    assign D_dram_rsp_ready = dcache_dram_rsp_if.dram_rsp_ready;

    VX_cache_core_req_if #(
        .NUM_REQUESTS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`DCORE_TAG_ID_BITS)
    ) core_dcache_req_if(),arb_dcache_req_if(), arb_io_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`DNUM_REQUESTS), 
        .WORD_SIZE(`DWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`DCORE_TAG_ID_BITS)
    ) core_dcache_rsp_if(), arb_dcache_rsp_if(), arb_io_rsp_if();

    assign io_req_valid  = arb_io_req_if.core_req_valid[0];
    assign io_req_rw     = arb_io_req_if.core_req_rw[0];
    assign io_req_byteen = arb_io_req_if.core_req_byteen[0];
    assign io_req_addr   = arb_io_req_if.core_req_addr[0];
    assign io_req_data   = arb_io_req_if.core_req_data[0];
    assign io_req_tag    = arb_io_req_if.core_req_tag[0];
    assign arb_io_req_if.core_req_ready = io_req_ready;

    assign arb_io_rsp_if.core_rsp_valid   = {{(`NUM_THREADS-1){1'b0}}, io_rsp_valid};
    assign arb_io_rsp_if.core_rsp_data[0] = io_rsp_data;
    assign arb_io_rsp_if.core_rsp_tag     = io_rsp_tag;    
    assign io_rsp_ready = arb_io_rsp_if.core_rsp_ready;
    
    // Icache interfaces
    
    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH(`IDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH(`IDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH(`IDRAM_TAG_WIDTH)
    ) icache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH(`IDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH(`IDRAM_TAG_WIDTH)
    ) icache_dram_rsp_if();

    assign I_dram_req_valid = icache_dram_req_if.dram_req_valid;
    assign I_dram_req_rw    = icache_dram_req_if.dram_req_rw;
    assign I_dram_req_byteen= icache_dram_req_if.dram_req_byteen;
    assign I_dram_req_addr  = icache_dram_req_if.dram_req_addr;
    assign I_dram_req_data  = icache_dram_req_if.dram_req_data;
    assign I_dram_req_tag   = icache_dram_req_if.dram_req_tag;
    assign icache_dram_req_if.dram_req_ready = I_dram_req_ready;

    assign icache_dram_rsp_if.dram_rsp_valid = I_dram_rsp_valid;    
    assign icache_dram_rsp_if.dram_rsp_data  = I_dram_rsp_data;
    assign icache_dram_rsp_if.dram_rsp_tag   = I_dram_rsp_tag;
    assign I_dram_rsp_ready = icache_dram_rsp_if.dram_rsp_ready;  
    
    VX_cache_core_req_if #(
        .NUM_REQUESTS(`INUM_REQUESTS), 
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`DCORE_TAG_ID_BITS)
    ) core_icache_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`INUM_REQUESTS), 
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`DCORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`DCORE_TAG_ID_BITS)
    ) core_icache_rsp_if();

    // Vortex pipeline
    VX_pipeline #(
        .CORE_ID(CORE_ID)
    ) pipeline (
        `SCOPE_SIGNALS_ISTAGE_BIND
        `SCOPE_SIGNALS_LSU_BIND
        `SCOPE_SIGNALS_PIPELINE_BIND
        `SCOPE_SIGNALS_BE_BIND

        .clk(clk),
        .reset(reset),

        // Dcache core request
        .dcache_req_valid   (core_dcache_req_if.core_req_valid),
        .dcache_req_rw      (core_dcache_req_if.core_req_rw),
        .dcache_req_byteen  (core_dcache_req_if.core_req_byteen),
        .dcache_req_addr    (core_dcache_req_if.core_req_addr),
        .dcache_req_data    (core_dcache_req_if.core_req_data),
        .dcache_req_tag     (core_dcache_req_if.core_req_tag),
        .dcache_req_ready   (core_dcache_req_if.core_req_ready),

        // Dcache core reponse    
        .dcache_rsp_valid   (core_dcache_rsp_if.core_rsp_valid),
        .dcache_rsp_data    (core_dcache_rsp_if.core_rsp_data),
        .dcache_rsp_tag     (core_dcache_rsp_if.core_rsp_tag),
        .dcache_rsp_ready   (core_dcache_rsp_if.core_rsp_ready),

        // Dcache core request
        .icache_req_valid   (core_icache_req_if.core_req_valid),
        .icache_req_rw      (core_icache_req_if.core_req_rw),
        .icache_req_byteen  (core_icache_req_if.core_req_byteen),
        .icache_req_addr    (core_icache_req_if.core_req_addr),
        .icache_req_data    (core_icache_req_if.core_req_data),
        .icache_req_tag     (core_icache_req_if.core_req_tag),
        .icache_req_ready   (core_icache_req_if.core_req_ready),

        // Dcache core reponse    
        .icache_rsp_valid   (core_icache_rsp_if.core_rsp_valid),
        .icache_rsp_data    (core_icache_rsp_if.core_rsp_data),
        .icache_rsp_tag     (core_icache_rsp_if.core_rsp_tag),
        .icache_rsp_ready   (core_icache_rsp_if.core_rsp_ready),     

        // Status
        .busy(busy), 
        .ebreak(ebreak)
    );  

    // Cache snooping
    VX_cache_snp_req_if #(
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .SNP_TAG_WIDTH(`DSNP_TAG_WIDTH)
    ) dcache_snp_req_if();

    VX_cache_snp_rsp_if #(
        .SNP_TAG_WIDTH(`DSNP_TAG_WIDTH)
    ) dcache_snp_rsp_if();

    assign dcache_snp_req_if.snp_req_valid      = snp_req_valid;
    assign dcache_snp_req_if.snp_req_addr       = snp_req_addr;
    assign dcache_snp_req_if.snp_req_invalidate = snp_req_invalidate;
    assign dcache_snp_req_if.snp_req_tag        = snp_req_tag;
    assign snp_req_ready                        = dcache_snp_req_if.snp_req_ready;

    assign snp_rsp_valid = dcache_snp_rsp_if.snp_rsp_valid;
    assign snp_rsp_tag   = dcache_snp_rsp_if.snp_rsp_tag;
    assign dcache_snp_rsp_if.snp_rsp_ready = snp_rsp_ready;

    VX_mem_unit #(
        .CORE_ID(CORE_ID)
    ) mem_unit (
        `SCOPE_SIGNALS_ICACHE_BIND

        .clk                (clk),
        .reset              (reset),

        // Core <-> Dcache
        .core_dcache_req_if (arb_dcache_req_if),
        .core_dcache_rsp_if (arb_dcache_rsp_if),

        // Dram <-> Dcache
        .dcache_dram_req_if (dcache_dram_req_if),
        .dcache_dram_rsp_if (dcache_dram_rsp_if),
        .dcache_snp_req_if  (dcache_snp_req_if),
        .dcache_snp_rsp_if  (dcache_snp_rsp_if),

        // Core <-> Icache
        .core_icache_req_if (core_icache_req_if),
        .core_icache_rsp_if (core_icache_rsp_if),

        // Dram <-> Icache
        .icache_dram_req_if (icache_dram_req_if),
        .icache_dram_rsp_if (icache_dram_rsp_if)
    );

    // select io address
    wire is_io_addr = ({core_dcache_req_if.core_req_addr[0], 2'b0} >= `IO_BUS_BASE_ADDR);
    wire io_select = (| core_dcache_req_if.core_req_valid) ? is_io_addr : 0;

    VX_dcache_arb dcache_io_arb (
        .req_select       (io_select),
        .in_core_req_if   (core_dcache_req_if),
        .out0_core_req_if (arb_dcache_req_if),
        .out1_core_req_if (arb_io_req_if),  
        .in0_core_rsp_if  (arb_dcache_rsp_if),
        .in1_core_rsp_if  (arb_io_rsp_if),    
        .out_core_rsp_if  (core_dcache_rsp_if)
    );
    
endmodule // Vortex





