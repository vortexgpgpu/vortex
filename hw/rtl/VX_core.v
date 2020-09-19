`include "VX_define.vh"

module VX_core #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_SIGNALS_ISTAGE_IO
    `SCOPE_SIGNALS_LSU_IO
    `SCOPE_SIGNALS_CACHE_IO
    `SCOPE_SIGNALS_ISSUE_IO
    `SCOPE_SIGNALS_EXECUTE_IO
    
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
    output wire [`NUM_THREADS-1:0]          io_req_valid,
    output wire                             io_req_rw,    
    output wire [`NUM_THREADS-1:0][3:0]     io_req_byteen,  
    output wire [`NUM_THREADS-1:0][29:0]    io_req_addr,
    output wire [`NUM_THREADS-1:0][31:0]    io_req_data,    
    output wire [`DCORE_TAG_WIDTH-1:0]      io_req_tag,  
    input wire                              io_req_ready,

    // I/O response
    input wire                              io_rsp_valid,
    input wire [31:0]                       io_rsp_data,
    input wire [`DCORE_TAG_WIDTH-1:0]       io_rsp_tag,
    output wire                             io_rsp_ready,

    // CSR I/O request
    input  wire                             csr_io_req_valid,
    input  wire [11:0]                      csr_io_req_addr,
    input  wire                             csr_io_req_rw,
    input  wire [31:0]                      csr_io_req_data,
    output wire                             csr_io_req_ready,

    // CSR I/O response
    output wire                             csr_io_rsp_valid,
    output wire [31:0]                      csr_io_rsp_data,
    input wire                              csr_io_rsp_ready,

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
    VX_cache_dram_req_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
    ) dcache_dram_req_if();

    VX_cache_dram_rsp_if #(
        .DRAM_LINE_WIDTH(`DDRAM_LINE_WIDTH),
        .DRAM_TAG_WIDTH(`DDRAM_TAG_WIDTH)
    ) dcache_dram_rsp_if();

    assign D_dram_req_valid = dcache_dram_req_if.valid;
    assign D_dram_req_rw    = dcache_dram_req_if.rw;
    assign D_dram_req_byteen= dcache_dram_req_if.byteen;
    assign D_dram_req_addr  = dcache_dram_req_if.addr;
    assign D_dram_req_data  = dcache_dram_req_if.data;
    assign D_dram_req_tag   = dcache_dram_req_if.tag;
    assign dcache_dram_req_if.ready = D_dram_req_ready;

    assign dcache_dram_rsp_if.valid = D_dram_rsp_valid;
    assign dcache_dram_rsp_if.data  = D_dram_rsp_data;
    assign dcache_dram_rsp_if.tag   = D_dram_rsp_tag;
    assign D_dram_rsp_ready = dcache_dram_rsp_if.ready;

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

    assign io_req_valid  = arb_io_req_if.valid;
    assign io_req_rw     = arb_io_req_if.rw;
    assign io_req_byteen = arb_io_req_if.byteen;
    assign io_req_addr   = arb_io_req_if.addr;
    assign io_req_data   = arb_io_req_if.data;
    assign io_req_tag    = arb_io_req_if.tag;
    assign arb_io_req_if.ready = io_req_ready;

    assign arb_io_rsp_if.valid   = {{(`NUM_THREADS-1){1'b0}}, io_rsp_valid};
    assign arb_io_rsp_if.data[0] = io_rsp_data;
    assign arb_io_rsp_if.tag     = io_rsp_tag;    
    assign io_rsp_ready = arb_io_rsp_if.ready;
    
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

    assign I_dram_req_valid = icache_dram_req_if.valid;
    assign I_dram_req_rw    = icache_dram_req_if.rw;
    assign I_dram_req_byteen= icache_dram_req_if.byteen;
    assign I_dram_req_addr  = icache_dram_req_if.addr;
    assign I_dram_req_data  = icache_dram_req_if.data;
    assign I_dram_req_tag   = icache_dram_req_if.tag;
    assign icache_dram_req_if.ready = I_dram_req_ready;

    assign icache_dram_rsp_if.valid = I_dram_rsp_valid;    
    assign icache_dram_rsp_if.data  = I_dram_rsp_data;
    assign icache_dram_rsp_if.tag   = I_dram_rsp_tag;
    assign I_dram_rsp_ready = icache_dram_rsp_if.ready;  
    
    VX_cache_core_req_if #(
        .NUM_REQUESTS(`INUM_REQUESTS), 
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`ICORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`ICORE_TAG_ID_BITS)
    ) core_icache_req_if();

    VX_cache_core_rsp_if #(
        .NUM_REQUESTS(`INUM_REQUESTS), 
        .WORD_SIZE(`IWORD_SIZE), 
        .CORE_TAG_WIDTH(`ICORE_TAG_WIDTH),
        .CORE_TAG_ID_BITS(`ICORE_TAG_ID_BITS)
    ) core_icache_rsp_if();

    VX_pipeline #(
        .CORE_ID(CORE_ID)
    ) pipeline (
        `SCOPE_SIGNALS_ISTAGE_BIND
        `SCOPE_SIGNALS_LSU_BIND
        `SCOPE_SIGNALS_ISSUE_BIND
        `SCOPE_SIGNALS_EXECUTE_BIND

        .clk(clk),
        .reset(reset),

        // Dcache core request
        .dcache_req_valid   (core_dcache_req_if.valid),
        .dcache_req_rw      (core_dcache_req_if.rw),
        .dcache_req_byteen  (core_dcache_req_if.byteen),
        .dcache_req_addr    (core_dcache_req_if.addr),
        .dcache_req_data    (core_dcache_req_if.data),
        .dcache_req_tag     (core_dcache_req_if.tag),
        .dcache_req_ready   (core_dcache_req_if.ready),

        // Dcache core reponse    
        .dcache_rsp_valid   (core_dcache_rsp_if.valid),
        .dcache_rsp_data    (core_dcache_rsp_if.data),
        .dcache_rsp_tag     (core_dcache_rsp_if.tag),
        .dcache_rsp_ready   (core_dcache_rsp_if.ready),

        // Dcache core request
        .icache_req_valid   (core_icache_req_if.valid),
        .icache_req_rw      (core_icache_req_if.rw),
        .icache_req_byteen  (core_icache_req_if.byteen),
        .icache_req_addr    (core_icache_req_if.addr),
        .icache_req_data    (core_icache_req_if.data),
        .icache_req_tag     (core_icache_req_if.tag),
        .icache_req_ready   (core_icache_req_if.ready),

        // Dcache core reponse    
        .icache_rsp_valid   (core_icache_rsp_if.valid),
        .icache_rsp_data    (core_icache_rsp_if.data),
        .icache_rsp_tag     (core_icache_rsp_if.tag),
        .icache_rsp_ready   (core_icache_rsp_if.ready),     

        // CSR I/O request
        .csr_io_req_valid   (csr_io_req_valid),
        .csr_io_req_rw      (csr_io_req_rw),
        .csr_io_req_addr    (csr_io_req_addr),
        .csr_io_req_data    (csr_io_req_data),
        .csr_io_req_ready   (csr_io_req_ready),

        // CSR I/O response
        .csr_io_rsp_valid   (csr_io_rsp_valid),            
        .csr_io_rsp_data    (csr_io_rsp_data),
        .csr_io_rsp_ready   (csr_io_rsp_ready),

        // Status
        .busy(busy), 
        .ebreak(ebreak)
    );  

    // Cache snooping interfaces
    
    VX_cache_snp_req_if #(
        .DRAM_ADDR_WIDTH(`DDRAM_ADDR_WIDTH),
        .SNP_TAG_WIDTH(`DSNP_TAG_WIDTH)
    ) dcache_snp_req_if();

    VX_cache_snp_rsp_if #(
        .SNP_TAG_WIDTH(`DSNP_TAG_WIDTH)
    ) dcache_snp_rsp_if();

    assign dcache_snp_req_if.valid      = snp_req_valid;
    assign dcache_snp_req_if.addr       = snp_req_addr;
    assign dcache_snp_req_if.invalidate = snp_req_invalidate;
    assign dcache_snp_req_if.tag        = snp_req_tag;
    assign snp_req_ready                = dcache_snp_req_if.ready;

    assign snp_rsp_valid = dcache_snp_rsp_if.valid;
    assign snp_rsp_tag   = dcache_snp_rsp_if.tag;
    assign dcache_snp_rsp_if.ready = snp_rsp_ready;

    VX_mem_unit #(
        .CORE_ID(CORE_ID)
    ) mem_unit (
        `SCOPE_SIGNALS_CACHE_BIND

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

    // select io bus
    wire is_io_addr = ({core_dcache_req_if.addr[0], 2'b0} >= `IO_BUS_BASE_ADDR);
    wire io_req_select = (| core_dcache_req_if.valid) ? is_io_addr : 0;
    wire io_rsp_select = (| arb_io_rsp_if.valid);

    VX_dcache_arb dcache_io_arb (        
        .core_req_in_if   (core_dcache_req_if),
        .core_req_out0_if (arb_dcache_req_if),
        .core_req_out1_if (arb_io_req_if),  
        .core_rsp_in0_if  (arb_dcache_rsp_if),
        .core_rsp_in1_if  (arb_io_rsp_if),    
        .core_rsp_out_if  (core_dcache_rsp_if),
        .select_req       (io_req_select),
        .select_rsp       (io_rsp_select)
    );
    
endmodule





