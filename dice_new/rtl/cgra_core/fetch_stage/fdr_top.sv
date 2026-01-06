`include "VX_define.vh"

module fdr_top
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int TAG_WIDTH      = 48,
    parameter int BITSTREAM_SIZE = 2056
) (
    input logic clk_i,
    input logic rst_i,

    // =========================================================================
    // Memory Bus Interfaces
    // =========================================================================
    VX_mem_bus_if.master metacache_mem_if,
    VX_mem_bus_if.master bitstream_cache_mem_if,

    // =========================================================================
    // Scheduler / FDR Interfaces
    // =========================================================================
    cta_sched_if.slave schedule_if,
    fdr_if.master      fdr_if,

    // =========================================================================
    // SIMT Stack Interface
    // =========================================================================
    input logic [DICE_ADDR_WIDTH-1:0] simt_stack_pc_i,
    input logic                       simt_update_ready_i,

    // SIMT stack update outputs (arbitrated from resolver/monitor)
    output logic                                         simt_update_valid_o,
    output logic                                         simt_update_with_divergence_o,
    output logic [                  DICE_ADDR_WIDTH-1:0] simt_update_next_pc_o,
    output logic [                  DICE_ADDR_WIDTH-1:0] simt_branch_not_taken_pc_o,
    output logic [                  DICE_ADDR_WIDTH-1:0] simt_branch_reconv_pc_o,
    output logic [    DICE_NUM_MAX_THREADS_PER_CORE-1:0] simt_predicate_values_o,
    output logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0] simt_update_hw_cta_id_o,

    // =========================================================================
    // Predicate Register File Interface
    // =========================================================================
    output logic                                                             prf_req_o,
    output logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)+$clog2(DICE_PR_NUM)-1:0] prf_raddr_o,
    input  logic [                        DICE_NUM_MAX_THREADS_PER_CORE-1:0] prf_rdata_i,

    // =========================================================================
    // CTA Status Table Interface
    // =========================================================================
    input dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status_i,

    output logic                            clear_prefetch_valid_o,
    output logic [DICE_HW_CTA_ID_WIDTH-1:0] clear_prefetch_hw_cta_id_o,
    output logic                            predict_miss_flush_o,

    // Branch prediction interface
    output branch_predict_interface_t predict_interface_o,
    output logic                      predict_we_o,

    // Clear divergence for monitor
    output logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0] clear_divergence_cta_id_o,
    output logic                                         clear_divergence_valid_o,

    // =========================================================================
    // CGRA Configuration Memories
    // =========================================================================
    output logic [VX_gpu_pkg::VX_MEM_DATA_WIDTH-1:0] cm0_data_o,
    output logic [((BITSTREAM_SIZE + VX_gpu_pkg::VX_MEM_DATA_WIDTH - 1) / VX_gpu_pkg::VX_MEM_DATA_WIDTH) - 1:0] cm0_chunk_en_o,
    output logic [VX_gpu_pkg::VX_MEM_DATA_WIDTH-1:0] cm1_data_o,
    output logic [((BITSTREAM_SIZE + VX_gpu_pkg::VX_MEM_DATA_WIDTH - 1) / VX_gpu_pkg::VX_MEM_DATA_WIDTH) - 1:0] cm1_chunk_en_o
);

  // ===========================================================================
  // Local Parameters
  // ===========================================================================
  localparam int PrfAddrWidth = $clog2(DICE_NUM_MAX_CTA_PER_CORE) + $clog2(DICE_PR_NUM);
  localparam int CtaIdWidth = $clog2(DICE_NUM_MAX_CTA_PER_CORE);

  // ===========================================================================
  // Internal Signals - Meta Fetch / Decoder
  // ===========================================================================
  pgraph_meta_t                                             meta_internal;
  logic                                                     meta_valid_internal;
  logic                                                     fire_eblock_internal;
  logic                                                     schedule_ready_internal;

  // ===========================================================================
  // Internal Signals - Bitstream
  // ===========================================================================
  logic                 [              DICE_ADDR_WIDTH-1:0] bitstream_addr;
  logic                 [       BITSTREAM_LENGTH_WIDTH-1:0] bitstream_length;
  logic                                                     bitstream_addr_valid_internal;
  logic                                                     done_streaming_internal;

  // ===========================================================================
  // Internal Signals - Branch Handler
  // ===========================================================================
  thread_mask_t                                             branch_mask_internal;
  branch_meta_t                                             branch_meta_internal;
  logic                                                     branch_mask_valid;
  logic                                                     branch_req_valid_internal;
  logic                                                     is_barrier_internal;

  // Pending branch table (shared between resolver and monitor)
  pending_branch_info_t [    DICE_NUM_MAX_CTA_PER_CORE-1:0] pending_branch_table;

  // Foreground (branch_resolver) signals
  logic                                                     fg_prf_req;
  logic                 [                 PrfAddrWidth-1:0] fg_prf_raddr;
  logic                                                     fg_update_valid;
  logic                                                     fg_update_with_divergence;
  logic                 [              DICE_ADDR_WIDTH-1:0] fg_update_next_pc;
  logic                 [              DICE_ADDR_WIDTH-1:0] fg_branch_not_taken_pc;
  logic                 [              DICE_ADDR_WIDTH-1:0] fg_branch_reconv_pc;
  logic                 [DICE_NUM_MAX_THREADS_PER_CORE-1:0] fg_predicate_values;
  logic                 [                   CtaIdWidth-1:0] fg_update_hw_cta_id;
  logic                                                     fg_update_ready;

  // Background (divergence_monitor) signals
  logic                                                     bg_prf_req;
  logic                 [                 PrfAddrWidth-1:0] bg_prf_raddr;
  logic                                                     bg_update_valid;
  logic                                                     bg_update_with_divergence;
  logic                 [              DICE_ADDR_WIDTH-1:0] bg_update_next_pc;
  logic                 [              DICE_ADDR_WIDTH-1:0] bg_branch_not_taken_pc;
  logic                 [              DICE_ADDR_WIDTH-1:0] bg_branch_reconv_pc;
  logic                 [DICE_NUM_MAX_THREADS_PER_CORE-1:0] bg_predicate_values;
  logic                 [                   CtaIdWidth-1:0] bg_update_hw_cta_id;
  logic                                                     bg_update_ready;
  logic                                                     bg_grant;

  // ===========================================================================
  // Internal Signals - Valid Checker
  // ===========================================================================
  logic                                                     clear_prefetch_internal;
  logic                                                     predict_miss_internal;

  // CTA status lookup for current CTA
  logic                 [         DICE_HW_CTA_ID_WIDTH-1:0] current_hw_cta_id;
  assign current_hw_cta_id                 = schedule_if.data.schedule_hw_cta_id;

  // ===========================================================================
  // Scheduler Ready Handshake
  // ===========================================================================
  assign schedule_if.ready                 = schedule_ready_internal;

  // ===========================================================================
  // Pass-through Assignments (schedule_if → fdr_if)
  // ===========================================================================
  assign fdr_if.data.schedule_hw_cta_id    = schedule_if.data.schedule_hw_cta_id;
  assign fdr_if.data.schedule_eblock_id    = schedule_if.data.schedule_eblock_id;
  assign fdr_if.data.schedule_cta_id       = schedule_if.data.schedule_cta_id;
  assign fdr_if.data.schedule_kernel_id    = schedule_if.data.schedule_kernel_id;
  assign fdr_if.data.schedule_grid_size    = schedule_if.data.schedule_grid_size;
  assign fdr_if.data.schedule_cta_size     = schedule_if.data.schedule_cta_size;
  assign fdr_if.data.schedule_hw_cta_size  = schedule_if.data.schedule_hw_cta_size;
  assign fdr_if.data.schedule_smem_per_cta = schedule_if.data.schedule_smem_per_cta;
  assign fdr_if.data.real_active_mask      = branch_mask_internal;

  // ===========================================================================
  // Output Assignments
  // ===========================================================================
  assign clear_prefetch_valid_o            = clear_prefetch_internal;
  assign clear_prefetch_hw_cta_id_o        = current_hw_cta_id;
  assign predict_miss_flush_o              = predict_miss_internal;

  // ===========================================================================
  // PRF/SIMT Arbitration: Foreground has priority -- NEEDS TO BE THOROUGHLY TESTED
  // ===========================================================================
  assign prf_req_o                         = fg_prf_req | bg_prf_req;
  assign prf_raddr_o                       = fg_prf_req ? fg_prf_raddr : bg_prf_raddr;
  assign bg_grant                          = !fg_update_valid;

  always_comb begin
    if (fg_update_valid) begin
      simt_update_valid_o           = 1'b1;
      simt_update_with_divergence_o = fg_update_with_divergence;
      simt_update_next_pc_o         = fg_update_next_pc;
      simt_branch_not_taken_pc_o    = fg_branch_not_taken_pc;
      simt_branch_reconv_pc_o       = fg_branch_reconv_pc;
      simt_predicate_values_o       = fg_predicate_values;
      simt_update_hw_cta_id_o       = fg_update_hw_cta_id;
    end else if (bg_update_valid) begin
      simt_update_valid_o           = 1'b1;
      simt_update_with_divergence_o = bg_update_with_divergence;
      simt_update_next_pc_o         = bg_update_next_pc;
      simt_branch_not_taken_pc_o    = bg_branch_not_taken_pc;
      simt_branch_reconv_pc_o       = bg_branch_reconv_pc;
      simt_predicate_values_o       = bg_predicate_values;
      simt_update_hw_cta_id_o       = bg_update_hw_cta_id;
    end else begin
      simt_update_valid_o           = 1'b0;
      simt_update_with_divergence_o = 1'b0;
      simt_update_next_pc_o         = '0;
      simt_branch_not_taken_pc_o    = '0;
      simt_branch_reconv_pc_o       = '0;
      simt_predicate_values_o       = '0;
      simt_update_hw_cta_id_o       = '0;
    end
  end

  assign fg_update_ready = simt_update_ready_i;
  assign bg_update_ready = simt_update_ready_i && !fg_update_valid;

  // ===========================================================================
  // Meta Fetch
  // ===========================================================================
  meta_fetch #(
      .TAG_WIDTH(TAG_WIDTH)
  ) u_meta_fetch (
      .clk_i               (clk_i),
      .rst_i               (rst_i),
      .schedule_valid_i    (schedule_if.valid),
      .fdr_next_pc_i       (schedule_if.data.schedule_next_pc),
      .schedule_eblock_id_i(schedule_if.data.schedule_eblock_id),
      .schedule_ready_o    (schedule_ready_internal),
      .meta_fetch_bus_if   (metacache_mem_if),
      .outgoing_meta_o     (meta_internal),
      .meta_valid_o        (meta_valid_internal),
      .fire_eblock_i       (fire_eblock_internal)
  );

  // ===========================================================================
  // Decoder
  // ===========================================================================
  decode u_decode (
      .metadata_i               (meta_internal),
      .meta_in_valid_i          (meta_valid_internal),
      .real_active_thread_mask_i(branch_mask_internal),
      .bitstream_addr_o         (bitstream_addr),
      .bitstream_addr_valid_o   (bitstream_addr_valid_internal),
      .bitstream_length_o       (bitstream_length),
      .branch_metadata_o        (branch_meta_internal),
      .branch_req_valid_o       (branch_req_valid_internal),
      .is_barrier_o             (is_barrier_internal),
      .meta_o                   (fdr_if.data.metadata)
  );

  // ===========================================================================
  // Bitstream Fetch/Load
  // ===========================================================================
  bitstream_fetch_load #(
      .TAG_WIDTH     (TAG_WIDTH),
      .BITSTREAM_SIZE(BITSTREAM_SIZE)
  ) u_bitstream_fetch_load (
      .clk_i           (clk_i),
      .rst_i           (rst_i),
      .meta_valid_i    (bitstream_addr_valid_internal),
      .bitstream_addr_i(bitstream_addr),
      .cm0_data_o      (cm0_data_o),
      .cm0_chunk_en_o  (cm0_chunk_en_o),
      .cm1_data_o      (cm1_data_o),
      .cm1_chunk_en_o  (cm1_chunk_en_o),
      .done_streaming_o(done_streaming_internal),
      .cache_bus_if    (bitstream_cache_mem_if),
      .cm_num_o        (fdr_if.data.loaded_buffer)
  );

  // ===========================================================================
  // Valid Checker
  // ===========================================================================
  valid_check u_valid_check (
      .barrier_indicator_i(is_barrier_internal),
      .mask_valid_i       (branch_mask_valid),
      .eblock_pc_i        (schedule_if.data.schedule_next_pc),
      .prefetch_block_i   (schedule_if.data.schedule_prefetch_block),
      .hw_cta_id_i        (current_hw_cta_id),
      .simt_stack_pc_i    (simt_stack_pc_i),
      .bitstream_loaded_i (done_streaming_internal),
      .unresolved_div_i   (cta_status_i[current_hw_cta_id].unresolved_control_divergence),
      .barrier_complete_i (cta_status_i[current_hw_cta_id].is_barrier),
      .prefetch_cleared_i (cta_status_i[current_hw_cta_id].prefetch_cleared),
      .fdr_valid_o        (fdr_if.valid),
      .ex_ready_i         (fdr_if.ready),
      .fire_eblock_o      (fire_eblock_internal),
      .clear_prefetch_o   (clear_prefetch_internal),
      .predict_miss_o     (predict_miss_internal)
  );

  // ===========================================================================
  // Branch Resolver (Foreground) -- NEEDS TO BE THOROUGHLY TESTED
  // ===========================================================================
  branch_resolver u_branch_resolver (
      .clk_i                    (clk_i),
      .rst_i                    (rst_i),
      .branch_metadata_i        (branch_meta_internal),
      .branch_req_valid_i       (branch_req_valid_internal),
      .current_pc_i             (schedule_if.data.schedule_next_pc),
      .ret_i                    (1'b0),
      .hw_cta_id_i              (current_hw_cta_id),
      .init_thread_mask_i       (schedule_if.data.schedule_active_mask),
      .cta_status_i             (cta_status_i[current_hw_cta_id]),
      .prf_req_o                (fg_prf_req),
      .prf_raddr_o              (fg_prf_raddr),
      .prf_rdata_i              (prf_rdata_i),
      .update_valid_o           (fg_update_valid),
      .update_with_divergence_o (fg_update_with_divergence),
      .update_next_pc_o         (fg_update_next_pc),
      .branch_not_taken_pc_o    (fg_branch_not_taken_pc),
      .branch_reconvergence_pc_o(fg_branch_reconv_pc),
      .predicate_regs_value_o   (fg_predicate_values),
      .update_hw_cta_id_o       (fg_update_hw_cta_id),
      .update_ready_i           (fg_update_ready),
      .real_active_thread_mask_o(branch_mask_internal),
      .mask_valid_o             (branch_mask_valid),
      .predict_interface_o      (predict_interface_o),
      .predict_we_o             (predict_we_o),
      .pending_branch_table_o   (pending_branch_table)
  );

  // ===========================================================================
  // Divergence Monitor (Background) -- NEEDS TO BE THOROUGHLY TESTED
  // ===========================================================================
  divergence_monitor u_divergence_monitor (
      .clk_i                    (clk_i),
      .rst_i                    (rst_i),
      .cta_status_i             (cta_status_i),
      .pending_branch_table_i   (pending_branch_table),
      .prf_req_o                (bg_prf_req),
      .prf_raddr_o              (bg_prf_raddr),
      .prf_rdata_i              (prf_rdata_i),
      .update_valid_o           (bg_update_valid),
      .update_with_divergence_o (bg_update_with_divergence),
      .update_next_pc_o         (bg_update_next_pc),
      .branch_not_taken_pc_o    (bg_branch_not_taken_pc),
      .branch_reconvergence_pc_o(bg_branch_reconv_pc),
      .predicate_regs_value_o   (bg_predicate_values),
      .update_hw_cta_id_o       (bg_update_hw_cta_id),
      .update_ready_i           (bg_update_ready),
      .grant_i                  (bg_grant),
      .clear_cta_id_o           (clear_divergence_cta_id_o),
      .clear_divergence_valid_o (clear_divergence_valid_o)
  );

endmodule
