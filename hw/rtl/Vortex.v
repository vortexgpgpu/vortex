`include "VX_define.vh"
`include "VX_cache_config.vh"

module Vortex #( 
    parameter CORE_ID = 0
) (        
    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // IO
    output wire                         io_valid,
    output wire [31:0]                  io_data,

    // DRAM Dcache Req
    output wire                         dram_req_read,
    output wire                         dram_req_write,    
    output wire [31:0]                  dram_req_addr,
    output wire [`DBANK_LINE_SIZE-1:0]  dram_req_data,
    input  wire                         dram_req_ready,

    // DRAM Dcache Rsp    
    input  wire                         dram_rsp_valid,
    input  wire [31:0]                  dram_rsp_addr,
    input  wire [`DBANK_LINE_SIZE-1:0]  dram_rsp_data,
    output wire                         dram_rsp_ready,

    // DRAM Icache Req
    output wire                         I_dram_req_read,
    output wire                         I_dram_req_write,    
    output wire [31:0]                  I_dram_req_addr,
    output wire [`IBANK_LINE_SIZE-1:0]  I_dram_req_data,
    input  wire                         I_dram_req_ready,

    // DRAM Icache Rsp    
    input  wire                         I_dram_rsp_valid,
    input  wire [31:0]                  I_dram_rsp_addr,
    input  wire [`IBANK_LINE_SIZE-1:0]  I_dram_rsp_data,
    output wire                         I_dram_rsp_ready,

    // LLC Snooping
    input  wire                         llc_snp_req_valid,
    input  wire [31:0]                  llc_snp_req_addr,
    output wire                         llc_snp_req_ready,

    // CSR request
    //input  wire                        csr_read_valid;
    //input  wire                        csr_write_valid;    
    //input  wire [`CSR_WIDTH-1:0        csr_index;
    //input  wire                        csr_data_in;
    //output wire [15:0]                 csr_data_out;        

    output wire                         ebreak
);
`DEBUG_BEGIN
    wire scheduler_empty;
`DEBUG_END

    wire memory_delay;
    wire exec_delay;
    wire gpr_stage_delay;
    wire schedule_delay;

    // Dcache Interface
    VX_cache_core_rsp_if #(.NUM_REQUESTS(`DNUM_REQUESTS))  dcache_core_rsp_if();
    VX_cache_core_req_if #(.NUM_REQUESTS(`DNUM_REQUESTS))  dcache_core_req_if();
    VX_cache_core_req_if #(.NUM_REQUESTS(`DNUM_REQUESTS))  dcache_core_req_qual_if();

    VX_cache_dram_req_if #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) cache_dram_req_if();
    VX_cache_dram_rsp_if #(.BANK_LINE_WORDS(`DBANK_LINE_WORDS)) cache_dram_rsp_if();

    assign cache_dram_rsp_if.dram_rsp_valid = dram_rsp_valid;
    assign cache_dram_rsp_if.dram_rsp_addr  = dram_rsp_addr;

    assign dram_req_write  = cache_dram_req_if.dram_req_write;
    assign dram_req_read   = cache_dram_req_if.dram_req_read;
    assign dram_req_addr   = cache_dram_req_if.dram_req_addr;
    assign dram_rsp_ready  = cache_dram_req_if.dram_rsp_ready;

    assign cache_dram_req_if.dram_req_ready = dram_req_ready;

    genvar i;
    generate
        for (i = 0; i < `DBANK_LINE_WORDS; i=i+1) begin
            assign cache_dram_rsp_if.dram_rsp_data[i] = dram_rsp_data[i * 32 +: 32];
            assign dram_req_data[i * 32 +: 32]        = cache_dram_req_if.dram_req_data[i];
        end
    endgenerate

    wire temp_io_valid = (!memory_delay) 
                      && (|dcache_core_req_if.core_req_valid) 
                      && (dcache_core_req_if.core_req_write[0] != `NO_MEM_WRITE) 
                      && (dcache_core_req_if.core_req_addr[0] == `IO_BUS_ADDR);

    wire [31:0] temp_io_data = dcache_core_req_if.core_req_data[0];
    assign io_valid          = temp_io_valid;
    assign io_data           = temp_io_data;

    assign dcache_core_req_qual_if.core_req_valid     = dcache_core_req_if.core_req_valid & {`NUM_THREADS{~io_valid}};        
    assign dcache_core_req_qual_if.core_req_read      = dcache_core_req_if.core_req_read;
    assign dcache_core_req_qual_if.core_req_write     = dcache_core_req_if.core_req_write;
    assign dcache_core_req_qual_if.core_req_addr      = dcache_core_req_if.core_req_addr;
    assign dcache_core_req_qual_if.core_req_data      = dcache_core_req_if.core_req_data;    
    
    assign dcache_core_req_if.core_req_ready = dcache_core_req_qual_if.core_req_ready;

    assign dcache_core_req_qual_if.core_req_rd        = dcache_core_req_if.core_req_rd;
    assign dcache_core_req_qual_if.core_req_wb        = dcache_core_req_if.core_req_wb;
    assign dcache_core_req_qual_if.core_req_warp_num  = dcache_core_req_if.core_req_warp_num;
    assign dcache_core_req_qual_if.core_req_pc        = dcache_core_req_if.core_req_pc;    
    
    VX_cache_core_rsp_if #(.NUM_REQUESTS(`INUM_REQUESTS))  icache_core_rsp_if();
    VX_cache_core_req_if #(.NUM_REQUESTS(`INUM_REQUESTS))  icache_core_req_if();

    VX_cache_dram_req_if #(.BANK_LINE_WORDS(`IBANK_LINE_WORDS)) icache_dram_req_if();
    VX_cache_dram_rsp_if #(.BANK_LINE_WORDS(`IBANK_LINE_WORDS)) icache_dram_rsp_if();

    assign icache_dram_rsp_if.dram_rsp_valid      = I_dram_rsp_valid;
    assign icache_dram_rsp_if.dram_rsp_addr = I_dram_rsp_addr;

    assign I_dram_req_write  = icache_dram_req_if.dram_req_write;
    assign I_dram_req_read   = icache_dram_req_if.dram_req_read;
    assign I_dram_req_addr   = icache_dram_req_if.dram_req_addr;
    assign I_dram_rsp_ready  = icache_dram_req_if.dram_rsp_ready;

    assign icache_dram_req_if.dram_req_ready = I_dram_req_ready;

    genvar j;
    generate
        for (j = 0; j < `IBANK_LINE_WORDS; j = j + 1) begin
            assign icache_dram_rsp_if.dram_rsp_data[j] = I_dram_rsp_data[j * 32 +: 32];
            assign I_dram_req_data[j * 32 +: 32]           = icache_dram_req_if.dram_req_data[j];
        end
    endgenerate

///////////////////////////////////////////////////////////////////////////////

// Front-end to Back-end
VX_frE_to_bckE_req_if    bckE_req_if();     // New instruction request to EXE/MEM

// Back-end to Front-end
VX_wb_if                 writeback_if();     // Writeback to GPRs
VX_branch_rsp_if         branch_rsp_if();   // Branch Resolution to Fetch
VX_jal_rsp_if            jal_rsp_if();      // Jump resolution to Fetch

// Warp controls
VX_warp_ctl_if           warp_ctl_if();

// Cache snooping
VX_cache_snp_req_if   icache_snp_req_if();
VX_cache_snp_req_if   dcache_snp_req_if();
assign dcache_snp_req_if.snp_req_valid  = llc_snp_req_valid;
assign dcache_snp_req_if.snp_req_addr   = llc_snp_req_addr;
assign llc_snp_req_ready                = dcache_snp_req_if.snp_req_ready;

VX_front_end front_end (
    .clk                (clk),
    .reset              (reset),
    .warp_ctl_if        (warp_ctl_if),
    .bckE_req_if        (bckE_req_if),
    .schedule_delay     (schedule_delay),
    .icache_rsp_if      (icache_core_rsp_if),
    .icache_req_if      (icache_core_req_if),
    .jal_rsp_if         (jal_rsp_if),
    .branch_rsp_if      (branch_rsp_if),
    .fetch_ebreak       (ebreak)
);

VX_scheduler scheduler (
    .clk                (clk),
    .reset              (reset),
    .memory_delay       (memory_delay),
    .exec_delay         (exec_delay),
    .gpr_stage_delay    (gpr_stage_delay),
    .bckE_req_if        (bckE_req_if),
    .writeback_if       (writeback_if),
    .schedule_delay     (schedule_delay),
    .is_empty           (scheduler_empty)
);

VX_back_end #(
    .CORE_ID(CORE_ID)
) back_end (
    .clk                 (clk),
    .reset               (reset),
    .schedule_delay      (schedule_delay),
    .warp_ctl_if         (warp_ctl_if),
    .bckE_req_if         (bckE_req_if),
    .jal_rsp_if          (jal_rsp_if),
    .branch_rsp_if       (branch_rsp_if),
    .dcache_rsp_if       (dcache_core_rsp_if),
    .dcache_req_if       (dcache_core_req_if),
    .writeback_if        (writeback_if),
    .mem_delay           (memory_delay),
    .exec_delay          (exec_delay),
    .gpr_stage_delay     (gpr_stage_delay)
);

VX_dmem_ctrl dmem_ctrl (
    .clk                    (clk),
    .reset                  (reset),

    // Dram <-> Dcache
    .dcache_dram_req_if     (cache_dram_req_if),
    .dcache_dram_rsp_if     (cache_dram_rsp_if),
    .dcache_snp_req_if      (dcache_snp_req_if),

    // Dram <-> Icache
    .icache_dram_req_if     (icache_dram_req_if),
    .icache_dram_rsp_if     (icache_dram_rsp_if),
    .icache_snp_req_if      (icache_snp_req_if),

    // Core <-> Icache
    .icache_core_req_if     (icache_core_req_if),
    .icache_core_rsp_if     (icache_core_rsp_if),

    // Core <-> Dcache
    .dcache_core_req_if     (dcache_core_req_qual_if),
    .dcache_core_rsp_if     (dcache_core_rsp_if)
);

endmodule // Vortex





