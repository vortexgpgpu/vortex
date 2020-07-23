`include "VX_define.vh"

module VX_pipeline #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_SIGNALS_ISTAGE_IO
    `SCOPE_SIGNALS_LSU_IO
    `SCOPE_SIGNALS_PIPELINE_IO
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
    
    // CSR I/O Request
    input  wire                             csr_io_req_valid,
    input  wire[11:0]                       csr_io_req_addr,
    input  wire                             csr_io_req_rw,
    input  wire[31:0]                       csr_io_req_data,
    output wire                             csr_io_req_ready,

    // CSR I/O Response
    output wire                             csr_io_rsp_valid,
    output wire[31:0]                       csr_io_rsp_data,
    input wire                              csr_io_rsp_ready,      

    // Status
    output wire                             busy, 
    output wire                             ebreak
);
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

    // CSR I/O
    VX_csr_io_req_if csr_io_req_if();
    assign csr_io_req_if.valid = csr_io_req_valid;
    assign csr_io_req_if.rw    = csr_io_req_rw;
    assign csr_io_req_if.addr  = csr_io_req_addr;
    assign csr_io_req_if.data  = csr_io_req_data;
    assign csr_io_req_ready    = csr_io_req_if.ready;

    VX_csr_io_rsp_if csr_io_rsp_if();
    assign csr_io_rsp_valid    = csr_io_rsp_if.valid; 
    assign csr_io_rsp_data     = csr_io_rsp_if.data; 
    assign csr_io_rsp_if.ready = csr_io_rsp_ready; 

    VX_perf_cntrs_if    perf_cntrs_if();
    VX_decode_if        decode_if();
    VX_branch_ctl_if    branch_ctl_if();
    VX_warp_ctl_if      warp_ctl_if();
    VX_ifetch_rsp_if    ifetch_rsp_if();
    VX_alu_req_if       alu_req_if();
    VX_lsu_req_if       lsu_req_if();
    VX_csr_req_if       csr_req_if();
    VX_mul_req_if       mul_req_if();  
    VX_fpu_req_if       fpu_req_if(); 
    VX_gpu_req_if       gpu_req_if();
    VX_wb_if            writeback_if();     
    VX_wstall_if        wstall_if();
    VX_join_if          join_if();
    VX_commit_if        alu_commit_if();
    VX_commit_if        lsu_commit_if();        
    VX_commit_if        csr_commit_if(); 
    VX_commit_if        mul_commit_if();     
    VX_commit_if        fpu_commit_if();     
    VX_commit_if        gpu_commit_if();     

    VX_fetch #(
        .CORE_ID(CORE_ID)
    ) fetch (
        .clk            (clk),
        .reset          (reset),
        .icache_req_if  (core_icache_req_if),
        .icache_rsp_if  (core_icache_rsp_if), 
        .wstall_if      (wstall_if),
        .join_if        (join_if),        
        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),
        .ifetch_rsp_if  (ifetch_rsp_if),
        .busy           (busy)
    );

    VX_decode #(
        .CORE_ID(CORE_ID)
    ) decode (
        .clk            (clk),
        .reset          (reset),        
        .ifetch_rsp_if  (ifetch_rsp_if),
        .decode_if      (decode_if),
        .wstall_if      (wstall_if),
        .join_if        (join_if)
    );

    VX_issue #(
        .CORE_ID(CORE_ID)
    ) issue (
        .clk            (clk),
        .reset          (reset),        

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),

        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
        .mul_req_if     (mul_req_if),
        .fpu_req_if     (fpu_req_if),
        .gpu_req_if     (gpu_req_if)
    );

    VX_execute #(
        .CORE_ID(CORE_ID)
    ) execute (
        `SCOPE_SIGNALS_LSU_BIND
        .clk            (clk),
        .reset          (reset),    
        
        .dcache_req_if  (core_dcache_req_if),
        .dcache_rsp_if  (core_dcache_rsp_if),
        
        .csr_io_req_if  (csr_io_req_if),
        .csr_io_rsp_if  (csr_io_rsp_if),

        .perf_cntrs_if  (perf_cntrs_if),                 
        
        .alu_req_if     (alu_req_if),
        .lsu_req_if     (lsu_req_if),        
        .csr_req_if     (csr_req_if),
        .mul_req_if     (mul_req_if),
        .fpu_req_if     (fpu_req_if),
        .gpu_req_if     (gpu_req_if),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),        
        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),        
        .csr_commit_if  (csr_commit_if),
        .mul_commit_if  (mul_commit_if),
        .fpu_commit_if  (fpu_commit_if),
        .gpu_commit_if  (gpu_commit_if),        
        
        .ebreak         (ebreak)
    );    

    VX_commit #(
        .CORE_ID(CORE_ID)
    ) commit (
        .clk            (clk),
        .reset          (reset),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),        
        .csr_commit_if  (csr_commit_if),
        .mul_commit_if  (mul_commit_if),
        .fpu_commit_if  (fpu_commit_if),
        .gpu_commit_if  (gpu_commit_if),
        
        .writeback_if   (writeback_if),
        .perf_cntrs_if  (perf_cntrs_if)
    );   

    assign dcache_req_valid  = core_dcache_req_if.valid;
    assign dcache_req_rw     = core_dcache_req_if.rw;
    assign dcache_req_byteen = core_dcache_req_if.byteen;
    assign dcache_req_addr   = core_dcache_req_if.addr;
    assign dcache_req_data   = core_dcache_req_if.data;
    assign dcache_req_tag    = core_dcache_req_if.tag;
    assign core_dcache_req_if.ready = dcache_req_ready;

    assign core_dcache_rsp_if.valid = dcache_rsp_valid;
    assign core_dcache_rsp_if.data  = dcache_rsp_data;
    assign core_dcache_rsp_if.tag   = dcache_rsp_tag;
    assign dcache_rsp_ready = core_dcache_rsp_if.ready;

    assign icache_req_valid  = core_icache_req_if.valid;
    assign icache_req_rw     = core_icache_req_if.rw;
    assign icache_req_byteen = core_icache_req_if.byteen;
    assign icache_req_addr   = core_icache_req_if.addr;
    assign icache_req_data   = core_icache_req_if.data;
    assign icache_req_tag    = core_icache_req_if.tag;
    assign core_icache_req_if.ready = icache_req_ready;

    assign core_icache_rsp_if.valid = icache_rsp_valid;
    assign core_icache_rsp_if.data  = icache_rsp_data;
    assign core_icache_rsp_if.tag   = icache_rsp_tag;
    assign icache_rsp_ready = core_icache_rsp_if.ready;

    `SCOPE_ASSIGN(scope_busy, busy); 
    `SCOPE_ASSIGN(scope_schedule_delay, schedule_delay);    
    `SCOPE_ASSIGN(scope_mem_delay, mem_delay);
    `SCOPE_ASSIGN(scope_exec_delay, exec_delay);
    `SCOPE_ASSIGN(scope_gpr_stage_delay, gpr_delay);

endmodule
