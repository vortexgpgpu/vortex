`ifndef VX_SCOPE
`define VX_SCOPE

`ifdef SCOPE

`define SCOPE_SIGNALS_DATA_LIST \
        scope_dram_req_addr, \
        scope_dram_req_rw, \
        scope_dram_req_byteen, \
        scope_dram_req_data, \
        scope_dram_req_tag, \
        scope_dram_rsp_data, \
        scope_dram_rsp_tag, \
        scope_snp_req_addr, \
        scope_snp_req_invalidate, \
        scope_snp_req_tag, \
        scope_snp_rsp_tag, \
        scope_icache_req_warp_num, \
        scope_icache_req_addr, \
        scope_icache_req_tag, \
        scope_icache_rsp_data, \
        scope_icache_rsp_tag, \
        scope_dcache_req_warp_num, \
        scope_dcache_req_curr_PC, \
        scope_dcache_req_addr, \
        scope_dcache_req_rw, \
        scope_dcache_req_byteen, \
        scope_dcache_req_data, \
        scope_dcache_req_tag, \
        scope_dcache_rsp_data, \
        scope_dcache_rsp_tag, \
        scope_decode_warp_num, \
        scope_decode_curr_PC, \
        scope_decode_is_jal, \
        scope_decode_rs1, \
        scope_decode_rs2, \
        scope_execute_warp_num, \
        scope_execute_curr_PC, \
        scope_execute_rd, \
        scope_execute_a, \
        scope_execute_b, \
        scope_writeback_warp_num, \
        scope_writeback_curr_PC, \
        scope_writeback_wb, \
        scope_writeback_rd, \
        scope_writeback_data, \
        scope_bank_addr_st0, \
        scope_bank_addr_st1, \
        scope_bank_addr_st2, \
        scope_bank_is_mrvq_st1, \
        scope_bank_miss_st1, \
        scope_bank_dirty_st1, \
        scope_bank_tag_valid_st1, \
        scope_bank_tag_match_st1, \
        scope_bank_force_miss_st1,
         
    
    `define SCOPE_SIGNALS_UPD_LIST \
        scope_dram_req_valid, \
        scope_dram_req_ready, \
        scope_dram_rsp_valid, \
        scope_dram_rsp_ready, \
        scope_snp_req_valid, \
        scope_snp_req_ready, \
        scope_snp_rsp_valid, \
        scope_snp_rsp_ready, \
        scope_icache_req_valid, \
        scope_icache_req_ready, \
        scope_icache_rsp_valid, \
        scope_icache_rsp_ready, \
        scope_dcache_req_valid, \
        scope_dcache_req_ready, \
        scope_dcache_rsp_valid, \
        scope_dcache_rsp_ready, \
        scope_decode_valid, \
        scope_execute_valid, \
        scope_writeback_valid, \
        scope_schedule_delay, \
        scope_memory_delay, \
        scope_exec_delay, \
        scope_gpr_stage_delay, \
        scope_busy, \
        scope_bank_valid_st0, \
        scope_bank_valid_st1, \
        scope_bank_valid_st2, \
        scope_bank_stall_pipe

    `define SCOPE_SIGNALS_DECL \
        wire scope_dram_req_valid; \
        wire [31:0] scope_dram_req_addr; \
        wire scope_dram_req_rw; \
        wire [15:0] scope_dram_req_byteen; \
        wire [127:0] scope_dram_req_data; \
        wire [`VX_DRAM_TAG_WIDTH-1:0] scope_dram_req_tag; \
        wire scope_dram_req_ready; \
        wire scope_dram_rsp_valid; \
        wire [127:0] scope_dram_rsp_data; \
        wire [`VX_DRAM_TAG_WIDTH-1:0] scope_dram_rsp_tag; \
        wire scope_dram_rsp_ready; \
        wire scope_snp_req_valid; \
        wire [31:0] scope_snp_req_addr; \
        wire scope_snp_req_invalidate; \
        wire [`VX_SNP_TAG_WIDTH-1:0] scope_snp_req_tag; \
        wire scope_snp_req_ready; \
        wire scope_snp_rsp_valid; \
        wire [`VX_SNP_TAG_WIDTH-1:0] scope_snp_rsp_tag; \
        wire scope_icache_req_valid; \
        wire [`NW_BITS-1:0] scope_icache_req_warp_num; \
        wire [31:0] scope_icache_req_addr; \
        wire [`ICORE_TAG_WIDTH-1:0] scope_icache_req_tag; \
        wire scope_icache_req_ready; \
        wire scope_icache_rsp_valid; \
        wire [31:0] scope_icache_rsp_data; \
        wire [`ICORE_TAG_WIDTH-1:0] scope_icache_rsp_tag; \
        wire scope_icache_rsp_ready; \
        wire [`NUM_THREADS-1:0] scope_dcache_req_valid; \
        wire [`NW_BITS-1:0] scope_dcache_req_warp_num; \
        wire [31:0] scope_dcache_req_curr_PC; \
        wire [63:0] scope_dcache_req_addr; \
        wire scope_dcache_req_rw; \
        wire [7:0] scope_dcache_req_byteen; \
        wire [63:0] scope_dcache_req_data; \
        wire [`DCORE_TAG_WIDTH-1:0] scope_dcache_req_tag; \
        wire scope_dcache_req_ready; \
        wire [`NUM_THREADS-1:0] scope_dcache_rsp_valid; \
        wire [63:0] scope_dcache_rsp_data; \
        wire [`DCORE_TAG_WIDTH-1:0] scope_dcache_rsp_tag; \
        wire scope_dcache_rsp_ready; \
        wire scope_busy; \
        wire scope_snp_rsp_ready; \
        wire scope_schedule_delay; \
        wire scope_memory_delay; \
        wire scope_exec_delay; \
        wire scope_gpr_stage_delay; \
        wire [`NUM_THREADS-1:0]  scope_decode_valid; \
        wire [`NW_BITS-1:0]  scope_decode_warp_num; \
        wire [31:0] scope_decode_curr_PC; \
        wire        scope_decode_is_jal; \
        wire [4:0]  scope_decode_rs1; \
        wire [4:0]  scope_decode_rs2; \
        wire [`NUM_THREADS-1:0]  scope_execute_valid; \
        wire [`NW_BITS-1:0]  scope_execute_warp_num; \
        wire [31:0] scope_execute_curr_PC; \
        wire [4:0]  scope_execute_rd; \
        wire [63:0] scope_execute_a; \
        wire [63:0] scope_execute_b; \
        wire [`NUM_THREADS-1:0]  scope_writeback_valid; \
        wire [`NW_BITS-1:0]  scope_writeback_warp_num; \
        wire [31:0] scope_writeback_curr_PC; \
        wire [1:0]  scope_writeback_wb; \
        wire [4:0]  scope_writeback_rd; \
        wire [63:0] scope_writeback_data; \
        wire scope_bank_valid_st0; \
        wire scope_bank_valid_st1; \
        wire scope_bank_valid_st2; \
        wire [31:0] scope_bank_addr_st0; \
        wire [31:0] scope_bank_addr_st1; \
        wire [31:0] scope_bank_addr_st2; \
        wire scope_bank_is_mrvq_st1; \
        wire scope_bank_miss_st1; \
        wire scope_bank_dirty_st1; \
        wire scope_bank_tag_valid_st1; \
        wire scope_bank_tag_match_st1; \
        wire scope_bank_force_miss_st1; \
        wire scope_bank_stall_pipe;        

    `define SCOPE_SIGNALS_ISTAGE_IO \
        output wire scope_icache_req_valid, \
        output wire [`NW_BITS-1:0] scope_icache_req_warp_num, \
        output wire [31:0] scope_icache_req_addr, \
        output wire [`ICORE_TAG_WIDTH-1:0] scope_icache_req_tag, \
        output wire scope_icache_req_ready, \
        output wire scope_icache_rsp_valid, \
        output wire [31:0] scope_icache_rsp_data, \
        output wire [`ICORE_TAG_WIDTH-1:0] scope_icache_rsp_tag, \
        output wire scope_icache_rsp_ready,

    `define SCOPE_SIGNALS_LSU_IO \
        output wire [`NUM_THREADS-1:0] scope_dcache_req_valid, \
        output wire [`NW_BITS-1:0] scope_dcache_req_warp_num, \
        output wire [31:0] scope_dcache_req_curr_PC, \
        output wire [63:0] scope_dcache_req_addr, \
        output wire scope_dcache_req_rw, \
        output wire [7:0] scope_dcache_req_byteen, \
        output wire [63:0] scope_dcache_req_data, \
        output wire [`DCORE_TAG_WIDTH-1:0] scope_dcache_req_tag, \
        output wire scope_dcache_req_ready, \
        output wire [`NUM_THREADS-1:0] scope_dcache_rsp_valid, \
        output wire [63:0] scope_dcache_rsp_data, \
        output wire [`DCORE_TAG_WIDTH-1:0] scope_dcache_rsp_tag, \
        output wire scope_dcache_rsp_ready,
        
    `define SCOPE_SIGNALS_CORE_IO \

    `define SCOPE_SIGNALS_CACHE_IO \
        output wire scope_bank_valid_st0, \
        output wire scope_bank_valid_st1, \
        output wire scope_bank_valid_st2, \
        output wire [31:0] scope_bank_addr_st0, \
        output wire [31:0] scope_bank_addr_st1, \
        output wire [31:0] scope_bank_addr_st2, \
        output wire scope_bank_is_mrvq_st1, \
        output wire scope_bank_miss_st1, \
        output wire scope_bank_dirty_st1, \
        output wire scope_bank_tag_valid_st1, \
        output wire scope_bank_tag_match_st1, \
        output wire scope_bank_force_miss_st1, \
        output wire scope_bank_stall_pipe,

    `define SCOPE_SIGNALS_PIPELINE_IO \
        output wire scope_busy, \
        output wire scope_schedule_delay, \
        output wire scope_memory_delay, \
        output wire scope_exec_delay, \
        output wire scope_gpr_stage_delay,

    `define SCOPE_SIGNALS_BE_IO \
        output wire [`NUM_THREADS-1:0] scope_decode_valid, \
        output wire [`NW_BITS-1:0] scope_decode_warp_num, \
        output wire [31:0] scope_decode_curr_PC, \
        output wire        scope_decode_is_jal, \
        output wire [4:0]  scope_decode_rs1, \
        output wire [4:0]  scope_decode_rs2, \
        output wire [`NUM_THREADS-1:0]  scope_execute_valid, \
        output wire [`NW_BITS-1:0]  scope_execute_warp_num, \
        output wire [31:0] scope_execute_curr_PC, \
        output wire [4:0]  scope_execute_rd, \
        output wire [63:0] scope_execute_a, \
        output wire [63:0] scope_execute_b, \
        output wire [`NUM_THREADS-1:0]  scope_writeback_valid, \
        output wire [`NW_BITS-1:0]  scope_writeback_warp_num, \
        output wire [31:0] scope_writeback_curr_PC, \
        output wire [1:0]  scope_writeback_wb, \
        output wire [4:0]  scope_writeback_rd, \
        output wire [63:0] scope_writeback_data,

    `define SCOPE_SIGNALS_ISTAGE_BIND \
        .scope_icache_req_valid (scope_icache_req_valid), \
        .scope_icache_req_warp_num (scope_icache_req_warp_num), \
        .scope_icache_req_addr  (scope_icache_req_addr), \
        .scope_icache_req_tag   (scope_icache_req_tag), \
        .scope_icache_req_ready (scope_icache_req_ready), \
        .scope_icache_rsp_valid (scope_icache_rsp_valid), \
        .scope_icache_rsp_data  (scope_icache_rsp_data), \
        .scope_icache_rsp_tag   (scope_icache_rsp_tag), \
        .scope_icache_rsp_ready (scope_icache_rsp_ready),

    `define SCOPE_SIGNALS_LSU_BIND \
        .scope_dcache_req_valid (scope_dcache_req_valid), \
        .scope_dcache_req_warp_num (scope_dcache_req_warp_num), \
        .scope_dcache_req_curr_PC (scope_dcache_req_curr_PC), \
        .scope_dcache_req_addr  (scope_dcache_req_addr), \
        .scope_dcache_req_rw    (scope_dcache_req_rw), \
        .scope_dcache_req_byteen(scope_dcache_req_byteen), \
        .scope_dcache_req_data  (scope_dcache_req_data), \
        .scope_dcache_req_tag   (scope_dcache_req_tag), \
        .scope_dcache_req_ready (scope_dcache_req_ready), \
        .scope_dcache_rsp_valid (scope_dcache_rsp_valid), \
        .scope_dcache_rsp_data  (scope_dcache_rsp_data), \
        .scope_dcache_rsp_tag   (scope_dcache_rsp_tag), \
        .scope_dcache_rsp_ready (scope_dcache_rsp_ready),

    `define SCOPE_SIGNALS_CORE_BIND \

    `define SCOPE_SIGNALS_CACHE_BIND \
        .scope_bank_valid_st0   (scope_bank_valid_st0), \
        .scope_bank_valid_st1   (scope_bank_valid_st1), \
        .scope_bank_valid_st2   (scope_bank_valid_st2), \
        .scope_bank_addr_st0    (scope_bank_addr_st0), \
        .scope_bank_addr_st1    (scope_bank_addr_st1), \
        .scope_bank_addr_st2    (scope_bank_addr_st2), \
        .scope_bank_is_mrvq_st1 (scope_bank_is_mrvq_st1), \
        .scope_bank_miss_st1    (scope_bank_miss_st1), \
        .scope_bank_dirty_st1   (scope_bank_dirty_st1), \
        .scope_bank_tag_valid_st1 (scope_bank_tag_valid_st1), \
        .scope_bank_tag_match_st1 (scope_bank_tag_match_st1), \
        .scope_bank_force_miss_st1 (scope_bank_force_miss_st1), \
        .scope_bank_stall_pipe  (scope_bank_stall_pipe),

    `define SCOPE_SIGNALS_CACHE_UNBIND \
        /* verilator lint_off PINCONNECTEMPTY */ \
        .scope_bank_valid_st0   (), \
        .scope_bank_valid_st1   (), \
        .scope_bank_valid_st2   (), \
        .scope_bank_addr_st0    (), \
        .scope_bank_addr_st1    (), \
        .scope_bank_addr_st2    (), \
        .scope_bank_is_mrvq_st1 (), \
        .scope_bank_miss_st1    (), \
        .scope_bank_dirty_st1   (), \
        .scope_bank_tag_valid_st1 (), \
        .scope_bank_tag_match_st1 (), \
        .scope_bank_force_miss_st1 (), \
        .scope_bank_stall_pipe  (), \
        /* verilator lint_on PINCONNECTEMPTY */

    `define SCOPE_SIGNALS_CACHE_BANK_SELECT \
        /* verilator lint_off UNUSED */ \
        wire [NUM_BANKS-1:0] scope_per_bank_valid_st0; \
        wire [NUM_BANKS-1:0] scope_per_bank_valid_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_valid_st2; \
        wire [NUM_BANKS-1:0][31:0] scope_per_bank_addr_st0; \
        wire [NUM_BANKS-1:0][31:0] scope_per_bank_addr_st1; \
        wire [NUM_BANKS-1:0][31:0] scope_per_bank_addr_st2; \
        wire [NUM_BANKS-1:0] scope_per_bank_is_mrvq_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_miss_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_dirty_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_tag_valid_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_tag_match_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_force_miss_st1; \
        wire [NUM_BANKS-1:0] scope_per_bank_stall_pipe; \
        /* verilator lint_on UNUSED */ \
        assign scope_bank_valid_st0 = scope_per_bank_valid_st0[0]; \
        assign scope_bank_valid_st1 = scope_per_bank_valid_st1[0]; \
        assign scope_bank_valid_st2 = scope_per_bank_valid_st2[0]; \
        assign scope_bank_addr_st0 = scope_per_bank_addr_st0[0]; \
        assign scope_bank_addr_st1 = scope_per_bank_addr_st1[0]; \
        assign scope_bank_addr_st2 = scope_per_bank_addr_st2[0]; \
        assign scope_bank_is_mrvq_st1 = scope_per_bank_is_mrvq_st1[0]; \
        assign scope_bank_miss_st1 = scope_per_bank_miss_st1[0]; \
        assign scope_bank_dirty_st1 = scope_per_bank_dirty_st1[0]; \
        assign scope_bank_tag_valid_st1 = scope_per_bank_tag_valid_st1[0]; \
        assign scope_bank_tag_match_st1 = scope_per_bank_tag_match_st1[0]; \
        assign scope_bank_force_miss_st1 = scope_per_bank_force_miss_st1[0]; \
        assign scope_bank_stall_pipe = scope_per_bank_stall_pipe[0];

    `define SCOPE_SIGNALS_CACHE_BANK_BIND \
        .scope_bank_valid_st0 (scope_per_bank_valid_st0[i]), \
        .scope_bank_valid_st1 (scope_per_bank_valid_st1[i]), \
        .scope_bank_valid_st2 (scope_per_bank_valid_st2[i]), \
        .scope_bank_addr_st0 (scope_per_bank_addr_st0[i]), \
        .scope_bank_addr_st1 (scope_per_bank_addr_st1[i]), \
        .scope_bank_addr_st2 (scope_per_bank_addr_st2[i]), \
        .scope_bank_is_mrvq_st1 (scope_per_bank_is_mrvq_st1[i]), \
        .scope_bank_miss_st1 (scope_per_bank_miss_st1[i]), \
        .scope_bank_dirty_st1 (scope_per_bank_dirty_st1[i]), \
        .scope_bank_tag_valid_st1 (scope_per_bank_tag_valid_st1[i]), \
        .scope_bank_tag_match_st1 (scope_per_bank_tag_match_st1[i]), \
        .scope_bank_force_miss_st1 (scope_per_bank_force_miss_st1[i]), \
        .scope_bank_stall_pipe (scope_per_bank_stall_pipe[i]),
 
    `define SCOPE_SIGNALS_PIPELINE_BIND \
        .scope_busy             (scope_busy), \
        .scope_schedule_delay   (scope_schedule_delay), \
        .scope_memory_delay     (scope_memory_delay), \
        .scope_exec_delay       (scope_exec_delay), \
        .scope_gpr_stage_delay  (scope_gpr_stage_delay),

    `define SCOPE_SIGNALS_BE_BIND \
        .scope_decode_valid     (scope_decode_valid), \
        .scope_decode_warp_num  (scope_decode_warp_num), \
        .scope_decode_curr_PC   (scope_decode_curr_PC), \
        .scope_decode_is_jal    (scope_decode_is_jal), \
        .scope_decode_rs1       (scope_decode_rs1), \
        .scope_decode_rs2       (scope_decode_rs2), \
        .scope_execute_valid    (scope_execute_valid), \
        .scope_execute_warp_num (scope_execute_warp_num), \
        .scope_execute_curr_PC  (scope_execute_curr_PC), \
        .scope_execute_rd       (scope_execute_rd), \
        .scope_execute_a        (scope_execute_a), \
        .scope_execute_b        (scope_execute_b), \
        .scope_writeback_valid  (scope_writeback_valid), \
        .scope_writeback_warp_num (scope_writeback_warp_num), \
        .scope_writeback_curr_PC(scope_writeback_curr_PC), \
        .scope_writeback_wb     (scope_writeback_wb), \
        .scope_writeback_rd     (scope_writeback_rd), \
        .scope_writeback_data   (scope_writeback_data),

    `define SCOPE_ASSIGN(d,s) assign d = s
`else
    `define SCOPE_SIGNALS_ISTAGE_IO
    `define SCOPE_SIGNALS_LSU_IO
    `define SCOPE_SIGNALS_CORE_IO
    `define SCOPE_SIGNALS_CACHE_IO
    `define SCOPE_SIGNALS_PIPELINE_IO
    `define SCOPE_SIGNALS_BE_IO

    `define SCOPE_SIGNALS_ISTAGE_BIND
    `define SCOPE_SIGNALS_LSU_BIND
    `define SCOPE_SIGNALS_CORE_BIND
    `define SCOPE_SIGNALS_CACHE_BIND
    `define SCOPE_SIGNALS_PIPELINE_BIND
    `define SCOPE_SIGNALS_BE_BIND
    
    `define SCOPE_SIGNALS_CACHE_UNBIND
    `define SCOPE_SIGNALS_CACHE_BANK_SELECT
    `define SCOPE_SIGNALS_CACHE_BANK_BIND
                
    `define SCOPE_ASSIGN(d,s)
`endif

// VX_SCOPE
`endif