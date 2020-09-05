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
        scope_icache_req_wid, \
        scope_icache_req_addr, \
        scope_icache_req_tag, \
        scope_icache_rsp_data, \
        scope_icache_rsp_tag, \
        scope_dcache_req_wid, \
        scope_dcache_req_pc, \
        scope_dcache_req_addr, \
        scope_dcache_req_rw, \
        scope_dcache_req_byteen, \
        scope_dcache_req_data, \
        scope_dcache_req_tag, \
        scope_dcache_rsp_data, \
        scope_dcache_rsp_tag, \
        scope_issue_wid, \
        scope_issue_tmask, \
        scope_issue_pc, \
        scope_issue_ex_type, \
        scope_issue_op_type, \
        scope_issue_op_mod, \
        scope_issue_wb, \
        scope_issue_rd, \
        scope_issue_rs1, \
        scope_issue_rs2, \
        scope_issue_rs3, \
        scope_issue_imm, \
        scope_issue_rs1_is_pc, \
        scope_issue_rs2_is_imm, \
        scope_gpr_rsp_wid, \
        scope_gpr_rsp_pc, \
        scope_gpr_rsp_a, \
        scope_gpr_rsp_b, \
        scope_gpr_rsp_c, \
        scope_writeback_wid, \
        scope_writeback_pc, \
        scope_writeback_rd, \
        scope_writeback_data, \
        scope_bank_addr_st0, \
        scope_bank_addr_st1, \
        scope_bank_addr_st2, \
        scope_bank_is_mrvq_st1, \
        scope_bank_miss_st1, \
        scope_bank_dirty_st1, \
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
        scope_bank_valid_st0, \
        scope_bank_valid_st1, \
        scope_bank_valid_st2, \
        scope_bank_stall_pipe, \
        scope_issue_valid, \
        scope_issue_ready, \
        scope_gpr_rsp_valid, \
        scope_writeback_valid, \
        scope_scoreboard_delay, \
        scope_gpr_delay, \
        scope_execute_delay, \
        scope_busy

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
        wire [`NW_BITS-1:0] scope_icache_req_wid; \
        wire [31:0] scope_icache_req_addr; \
        wire [`ICORE_TAG_ID_BITS-1:0] scope_icache_req_tag; \
        wire scope_icache_req_ready; \
        wire scope_icache_rsp_valid; \
        wire [31:0] scope_icache_rsp_data; \
        wire [`ICORE_TAG_ID_BITS-1:0] scope_icache_rsp_tag; \
        wire scope_icache_rsp_ready; \
        wire [`NUM_THREADS-1:0] scope_dcache_req_valid; \
        wire [`NW_BITS-1:0] scope_dcache_req_wid; \
        wire [31:0] scope_dcache_req_pc; \
        wire [`NUM_THREADS-1:0][31:0] scope_dcache_req_addr; \
        wire scope_dcache_req_rw; \
        wire [`NUM_THREADS-1:0][3:0] scope_dcache_req_byteen; \
        wire [`NUM_THREADS-1:0][31:0] scope_dcache_req_data; \
        wire [`DCORE_TAG_ID_BITS-1:0] scope_dcache_req_tag; \
        wire scope_dcache_req_ready; \
        wire [`NUM_THREADS-1:0] scope_dcache_rsp_valid; \
        wire [`NUM_THREADS-1:0][31:0] scope_dcache_rsp_data; \
        wire [`DCORE_TAG_ID_BITS-1:0] scope_dcache_rsp_tag; \
        wire scope_dcache_rsp_ready; \
        wire scope_snp_rsp_ready; \
        wire [`NW_BITS-1:0] scope_issue_wid; \
        wire [`NUM_THREADS-1:0] scope_issue_tmask; \
        wire [31:0] scope_issue_pc; \
        wire [`EX_BITS-1:0] scope_issue_ex_type; \
        wire [`OP_BITS-1:0] scope_issue_op_type; \
        wire [`MOD_BITS-1:0] scope_issue_op_mod; \
        wire scope_issue_wb; \
        wire [`NR_BITS-1:0] scope_issue_rd; \
        wire [`NR_BITS-1:0] scope_issue_rs1; \
        wire [`NR_BITS-1:0] scope_issue_rs2; \
        wire [`NR_BITS-1:0] scope_issue_rs3; \
        wire [31:0] scope_issue_imm; \
        wire scope_issue_rs1_is_pc; \
        wire scope_issue_rs2_is_imm; \
        wire scope_gpr_rsp_valid; \
        wire [`NW_BITS-1:0] scope_gpr_rsp_wid; \
        wire [31:0] scope_gpr_rsp_pc; \
        wire [`NUM_THREADS-1:0][31:0] scope_gpr_rsp_a; \
        wire [`NUM_THREADS-1:0][31:0] scope_gpr_rsp_b; \
        wire [`NUM_THREADS-1:0][31:0] scope_gpr_rsp_c; \
        wire scope_writeback_valid; \
        wire [`NW_BITS-1:0] scope_writeback_wid; \
        wire [31:0] scope_writeback_pc; \
        wire [`NR_BITS-1:0] scope_writeback_rd; \
        wire [`NUM_THREADS-1:0][31:0] scope_writeback_data; \
        wire scope_bank_valid_st0; \
        wire scope_bank_valid_st1; \
        wire scope_bank_valid_st2; \
        wire [31:0] scope_bank_addr_st0; \
        wire [31:0] scope_bank_addr_st1; \
        wire [31:0] scope_bank_addr_st2; \
        wire scope_bank_is_mrvq_st1; \
        wire scope_bank_miss_st1; \
        wire scope_bank_dirty_st1; \
        wire scope_bank_force_miss_st1; \
        wire scope_bank_stall_pipe; \
        wire scope_issue_valid; \
        wire scope_issue_ready; \
        wire scope_scoreboard_delay; \
        wire scope_gpr_delay; \
        wire scope_execute_delay; \
        wire scope_busy;

    `define SCOPE_SIGNALS_ISTAGE_IO \
        output wire scope_icache_req_valid, \
        output wire [`NW_BITS-1:0] scope_icache_req_wid, \
        output wire [31:0] scope_icache_req_addr, \
        output wire [`ICORE_TAG_ID_BITS-1:0] scope_icache_req_tag, \
        output wire scope_icache_req_ready, \
        output wire scope_icache_rsp_valid, \
        output wire [31:0] scope_icache_rsp_data, \
        output wire [`ICORE_TAG_ID_BITS-1:0] scope_icache_rsp_tag, \
        output wire scope_icache_rsp_ready,

    `define SCOPE_SIGNALS_LSU_IO \
        output wire [`NUM_THREADS-1:0] scope_dcache_req_valid, \
        output wire [`NW_BITS-1:0] scope_dcache_req_wid, \
        output wire [31:0] scope_dcache_req_pc, \
        output wire [`NUM_THREADS-1:0][31:0] scope_dcache_req_addr, \
        output wire scope_dcache_req_rw, \
        output wire [`NUM_THREADS-1:0][3:0] scope_dcache_req_byteen, \
        output wire [`NUM_THREADS-1:0][31:0] scope_dcache_req_data, \
        output wire [`DCORE_TAG_ID_BITS-1:0] scope_dcache_req_tag, \
        output wire scope_dcache_req_ready, \
        output wire [`NUM_THREADS-1:0] scope_dcache_rsp_valid, \
        output wire [`NUM_THREADS-1:0][31:0] scope_dcache_rsp_data, \
        output wire [`DCORE_TAG_ID_BITS-1:0] scope_dcache_rsp_tag, \
        output wire scope_dcache_rsp_ready,

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
        output wire scope_bank_force_miss_st1, \
        output wire scope_bank_stall_pipe,

    `define SCOPE_SIGNALS_ISSUE_IO \
        output wire scope_issue_valid, \
        output wire [`NW_BITS-1:0] scope_issue_wid, \
        output wire [`NUM_THREADS-1:0] scope_issue_tmask, \
        output wire [31:0] scope_issue_pc, \
        output wire [`EX_BITS-1:0] scope_issue_ex_type, \
        output wire [`OP_BITS-1:0] scope_issue_op_type, \
        output wire [`MOD_BITS-1:0] scope_issue_op_mod, \
        output wire scope_issue_wb, \
        output wire [`NR_BITS-1:0] scope_issue_rd, \
        output wire [`NR_BITS-1:0] scope_issue_rs1, \
        output wire [`NR_BITS-1:0] scope_issue_rs2, \
        output wire [`NR_BITS-1:0] scope_issue_rs3, \
        output wire [31:0] scope_issue_imm, \
        output wire scope_issue_rs1_is_pc, \
        output wire scope_issue_rs2_is_imm, \
        output wire scope_writeback_valid, \
        output wire scope_gpr_rsp_valid, \
        output wire [`NW_BITS-1:0] scope_gpr_rsp_wid, \
        output wire [31:0] scope_gpr_rsp_pc, \
        output wire [`NUM_THREADS-1:0][31:0] scope_gpr_rsp_a, \
        output wire [`NUM_THREADS-1:0][31:0] scope_gpr_rsp_b, \
        output wire [`NUM_THREADS-1:0][31:0] scope_gpr_rsp_c, \
        output wire [`NW_BITS-1:0] scope_writeback_wid, \
        output wire [31:0] scope_writeback_pc, \
        output wire [`NR_BITS-1:0] scope_writeback_rd, \
        output wire [`NUM_THREADS-1:0][31:0] scope_writeback_data, \
        output wire scope_issue_ready, \
        output wire scope_scoreboard_delay, \
        output wire scope_gpr_delay, \
        output wire scope_execute_delay,

    `define SCOPE_SIGNALS_EXECUTE_IO
        
    `define SCOPE_SIGNALS_ISTAGE_BIND \
        .scope_icache_req_valid (scope_icache_req_valid), \
        .scope_icache_req_wid   (scope_icache_req_wid), \
        .scope_icache_req_addr  (scope_icache_req_addr), \
        .scope_icache_req_tag   (scope_icache_req_tag), \
        .scope_icache_req_ready (scope_icache_req_ready), \
        .scope_icache_rsp_valid (scope_icache_rsp_valid), \
        .scope_icache_rsp_data  (scope_icache_rsp_data), \
        .scope_icache_rsp_tag   (scope_icache_rsp_tag), \
        .scope_icache_rsp_ready (scope_icache_rsp_ready),

    `define SCOPE_SIGNALS_LSU_BIND \
        .scope_dcache_req_valid (scope_dcache_req_valid), \
        .scope_dcache_req_wid   (scope_dcache_req_wid), \
        .scope_dcache_req_pc    (scope_dcache_req_pc), \
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
        .scope_bank_force_miss_st1(scope_bank_force_miss_st1), \
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
        assign scope_bank_force_miss_st1 = scope_per_bank_force_miss_st1[0]; \
        assign scope_bank_stall_pipe = scope_per_bank_stall_pipe[0];

    `define SCOPE_SIGNALS_CACHE_BANK_BIND \
        .scope_bank_valid_st0   (scope_per_bank_valid_st0[i]), \
        .scope_bank_valid_st1   (scope_per_bank_valid_st1[i]), \
        .scope_bank_valid_st2   (scope_per_bank_valid_st2[i]), \
        .scope_bank_addr_st0    (scope_per_bank_addr_st0[i]), \
        .scope_bank_addr_st1    (scope_per_bank_addr_st1[i]), \
        .scope_bank_addr_st2    (scope_per_bank_addr_st2[i]), \
        .scope_bank_is_mrvq_st1 (scope_per_bank_is_mrvq_st1[i]), \
        .scope_bank_miss_st1    (scope_per_bank_miss_st1[i]), \
        .scope_bank_dirty_st1   (scope_per_bank_dirty_st1[i]), \
        .scope_bank_force_miss_st1 (scope_per_bank_force_miss_st1[i]), \
        .scope_bank_stall_pipe  (scope_per_bank_stall_pipe[i]),
 
    `define SCOPE_SIGNALS_ISSUE_BIND \
        .scope_issue_valid      (scope_issue_valid), \
        .scope_issue_wid        (scope_issue_wid), \
        .scope_issue_tmask      (scope_issue_tmask), \
        .scope_issue_pc         (scope_issue_pc), \
        .scope_issue_ex_type    (scope_issue_ex_type), \
        .scope_issue_op_type    (scope_issue_op_type), \
        .scope_issue_op_mod     (scope_issue_op_mod), \
        .scope_issue_wb         (scope_issue_wb), \
        .scope_issue_rd         (scope_issue_rd), \
        .scope_issue_rs1        (scope_issue_rs1), \
        .scope_issue_rs2        (scope_issue_rs2), \
        .scope_issue_rs3        (scope_issue_rs3), \
        .scope_issue_imm        (scope_issue_imm), \
        .scope_issue_rs1_is_pc  (scope_issue_rs1_is_pc), \
        .scope_issue_rs2_is_imm (scope_issue_rs2_is_imm), \
        .scope_writeback_valid  (scope_writeback_valid), \
        .scope_writeback_wid    (scope_writeback_wid), \
        .scope_writeback_pc     (scope_writeback_pc), \
        .scope_writeback_rd     (scope_writeback_rd), \
        .scope_writeback_data   (scope_writeback_data), \
        .scope_issue_ready      (scope_issue_ready), \
        .scope_gpr_rsp_valid    (scope_gpr_rsp_valid), \
        .scope_gpr_rsp_wid      (scope_gpr_rsp_wid), \
        .scope_gpr_rsp_pc       (scope_gpr_rsp_pc), \
        .scope_gpr_rsp_a        (scope_gpr_rsp_a), \
        .scope_gpr_rsp_b        (scope_gpr_rsp_b), \
        .scope_gpr_rsp_c        (scope_gpr_rsp_c), \
        .scope_scoreboard_delay (scope_scoreboard_delay), \
        .scope_gpr_delay        (scope_gpr_delay), \
        .scope_execute_delay    (scope_execute_delay), \

    `define SCOPE_SIGNALS_EXECUTE_BIND

    `define SCOPE_ASSIGN(d,s) assign d = s
`else
    `define SCOPE_SIGNALS_ISTAGE_IO
    `define SCOPE_SIGNALS_LSU_IO
    `define SCOPE_SIGNALS_CACHE_IO
    `define SCOPE_SIGNALS_ISSUE_IO
    `define SCOPE_SIGNALS_EXECUTE_IO

    `define SCOPE_SIGNALS_ISTAGE_BIND
    `define SCOPE_SIGNALS_LSU_BIND
    `define SCOPE_SIGNALS_CACHE_BIND
    `define SCOPE_SIGNALS_ISSUE_BIND
    `define SCOPE_SIGNALS_EXECUTE_BIND
    
    `define SCOPE_SIGNALS_CACHE_UNBIND
    `define SCOPE_SIGNALS_CACHE_BANK_SELECT
    `define SCOPE_SIGNALS_CACHE_BANK_BIND
                
    `define SCOPE_ASSIGN(d,s)
`endif

// VX_SCOPE
`endif