
module dice_core
  import dice_pkg::*;
  import dice_frontend_pkg::*;
(
    input logic clk,
    input logic reset,

    // Host/Dispatcher Interface - CTA Allocation
    input  logic         cta_add_valid_i,
    output logic         cta_add_ready_o,
    input  dice_cta_desc_t new_cta_desc_i,

    output logic         cta_complete_valid_o,
    input  logic         cta_complete_ready_i,
    output dice_cta_id_t cta_done_id_o,

    // Memory Bus Interfaces
    VX_mem_bus_if.master metacache_mem_if,
    VX_mem_bus_if.master bitstream_cache_mem_if,

);

  // Internal Interfaces
  cta_sched_if schedule_if ();
  branch_handler_if bh_if ();
  dice_bh_simt_if simt_stack_update_if ();

  // Internal Signals - CTA Schedule Stage Outputs
  logic [MaxNumCta-1:0] stack_top_valid;
  logic [MaxNumCta-1:0][PcWidth-1:0] stack_top_next_pc;
  logic [MaxNumCta-1:0][PcWidth-1:0] stack_top_reconvergence_pc;
  logic [MaxNumCta-1:0][ThreadWidth-1:0] stack_top_active_mask;
  logic [MaxNumCta-1:0] stack_empty;
  logic [MaxNumCta-1:0] stack_full;

  // Internal Signals - FDR to CTA Schedule Connections
  logic simt_update_valid;
  logic simt_update_with_divergence;
  logic [DICE_ADDR_WIDTH-1:0] simt_update_next_pc;
  logic [DICE_ADDR_WIDTH-1:0] simt_branch_not_taken_pc;
  logic [DICE_ADDR_WIDTH-1:0] simt_branch_reconv_pc;
  logic [DICE_NUM_MAX_THREADS_PER_CORE-1:0] simt_predicate_values;
  logic [CtaIdWidth-1:0] simt_update_hw_cta_id;
  logic [1:0] simt_update_hw_cta_size;

  // Internal Signals - Predicate Register File (internal to core)
  logic prf_req;
  logic [PrfAddrWidth-1:0] prf_raddr;
  logic [DICE_NUM_MAX_THREADS_PER_CORE-1:0] prf_rdata;

  // Internal Signals - CGRA Configuration Memory (internal to core)
  logic [VX_gpu_pkg::VX_MEM_DATA_WIDTH-1:0] cm0_data;
  logic [((BitstreamSize + VX_gpu_pkg::VX_MEM_DATA_WIDTH - 1) / VX_gpu_pkg::VX_MEM_DATA_WIDTH) - 1:0] cm0_chunk_en;
  logic [VX_gpu_pkg::VX_MEM_DATA_WIDTH-1:0] cm1_data;
  logic [((BitstreamSize + VX_gpu_pkg::VX_MEM_DATA_WIDTH - 1) / VX_gpu_pkg::VX_MEM_DATA_WIDTH) - 1:0] cm1_chunk_en;

  // Internal Signals - Branch Prediction (FDR → CTA Status Table)
  branch_predict_interface_t predict_interface;
  logic predict_we;

  // Internal Signals - Divergence Clearing
  logic [CtaIdWidth-1:0] clear_divergence_cta_id;
  logic clear_divergence_valid;

  // Internal Signals - Prefetch Clearing
  logic clear_prefetch_valid;
  logic [DICE_HW_CTA_ID_WIDTH-1:0] clear_prefetch_hw_cta_id;
  logic predict_miss_flush;

  // Stub Signals (downstream stages not implemented)
  logic eblock_commit_valid_stub;
  logic [EblockIdWidth-1:0] eblock_commit_id_stub;
  block_retire_status_t brt_info_stub;
  logic brt_info_write_enable_stub;

  assign eblock_commit_valid_stub = 1'b0;
  assign eblock_commit_id_stub = '0;
  assign brt_info_stub = '0;
  assign brt_info_write_enable_stub = 1'b0;

  // Stub PRF read data (PRF module will be instantiated later)
  assign prf_rdata = '0;

  // SIMT Stack Update Interface Wiring
  assign simt_stack_update_if.update_valid = simt_update_valid;
  assign simt_stack_update_if.update_stack_data.update_with_divergence = simt_update_with_divergence;
  assign simt_stack_update_if.update_stack_data.update_next_pc = simt_update_next_pc;
  assign simt_stack_update_if.update_stack_data.predicate_regs_value = {{(SIMT_STACK_COUNT*SIMT_STACK_THREAD_WIDTH - DICE_NUM_MAX_THREADS_PER_CORE){1'b0}}, simt_predicate_values};
  assign simt_stack_update_if.update_stack_data.branch_not_taken_pc = simt_branch_not_taken_pc;
  assign simt_stack_update_if.update_stack_data.branch_reconvergence_pc = simt_branch_reconv_pc;

  // Branch Handler Interface Wiring
  assign bh_if.bh_data = predict_interface;
  assign bh_if.branch_predict_info_write_enable = predict_we;

  // SIMT Stack Ready for FDR
  logic simt_update_ready;
  assign simt_update_ready = simt_stack_update_if.update_ready;

  // Current CTA Status (from CTA Status Table → FDR)
  dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status;
  assign cta_status = bh_if.cta_status_data;

  // SIMT Stack PC for FDR (from scheduled CTA's stack top)
  logic [DICE_ADDR_WIDTH-1:0] simt_stack_pc;
  assign simt_stack_pc = stack_top_next_pc[schedule_if.data.schedule_hw_cta_id];


  // CTA Schedule Stage
  cta_schedule_stage #(
      .STACK_DEPTH (StackDepth)
  ) u_cta_schedule_stage (
      .clk_i                   (clk),
      .rst_i                   (reset),

      // Host/Dispatcher interface
      .cta_add_valid_i         (cta_add_valid_i),
      .cta_add_ready_o         (cta_add_ready_o),
      .new_cta_all_desc_i      (new_cta_desc_i),

      // CTA completion
      .cta_complete_valid_o    (cta_complete_valid_o),
      .cta_complete_ready_i    (cta_complete_ready_i),
      .cta_done_id_o           (cta_done_id_o),

      // Scheduler output (to FDR)
      .schedule_if             (schedule_if),

      // E-block commit (stub)
      .eblock_commit_valid_i   (eblock_commit_valid_stub),
      .eblock_commit_id_i      (eblock_commit_id_stub),

      // Branch handler interface
      .status_table_bh_if      (bh_if),

      // Block retire table (stub)
      .brt_info_i              (brt_info_stub),
      .brt_info_write_enable_i (brt_info_write_enable_stub),

      // SIMT stack update interface
      .simt_stack_update       (simt_stack_update_if),
      .simt_update_hw_cta_id_i (simt_update_hw_cta_id),
      .simt_update_hw_cta_size_i(simt_update_hw_cta_size),

      // SIMT stack status outputs
      .stack_top_valid_o       (stack_top_valid),
      .stack_top_next_pc_o     (stack_top_next_pc),
      .stack_top_reconvergence_pc_o(stack_top_reconvergence_pc),
      .stack_top_active_mask_o (stack_top_active_mask),
      .stack_empty_o           (stack_empty),
      .stack_full_o            (stack_full)
  );

  // ===========================================================================
  // Fetch-Decode-Resolve Stage (FDR)
  // ===========================================================================
  fdr_top #(
      .TAG_WIDTH     (TagWidth),
      .BITSTREAM_SIZE(BitstreamSize)
  ) u_fdr_top (
      .clk_i                    (clk),
      .rst_i                    (reset),

      // Memory System
      .metacache_mem_if         (metacache_mem_if),
      .bitstream_cache_mem_if   (bitstream_cache_mem_if),

      // From CTA Schedule Stage
      .schedule_if              (schedule_if),

      // To Execute Stage
      .fdr_if                   (fdr_out_if),


      // SIMT stack interface (CTA Schedule Stage)
      .simt_stack_pc_i          (simt_stack_pc),
      .simt_update_ready_i      (simt_update_ready),
      .simt_update_valid_o      (simt_update_valid),
      .simt_update_with_divergence_o(simt_update_with_divergence),
      .simt_update_next_pc_o    (simt_update_next_pc),
      .simt_branch_not_taken_pc_o(simt_branch_not_taken_pc),
      .simt_branch_reconv_pc_o  (simt_branch_reconv_pc),
      .simt_predicate_values_o  (simt_predicate_values),
      .simt_update_hw_cta_id_o  (simt_update_hw_cta_id),

      // Predicate register file (Execute Stage) -> Interface may need to be changed
      .prf_req_o                (prf_req),
      .prf_raddr_o              (prf_raddr),
      .prf_rdata_i              (prf_rdata),

      // CTA status
      .cta_status_i             (cta_status),

      // Prefetch/prediction control
      .clear_prefetch_valid_o   (clear_prefetch_valid),
      .clear_prefetch_hw_cta_id_o(clear_prefetch_hw_cta_id),
      .predict_miss_flush_o     (predict_miss_flush),

      // Branch prediction interface
      .predict_interface_o      (predict_interface),
      .predict_we_o             (predict_we),

      // Divergence clearing
      .clear_divergence_cta_id_o(clear_divergence_cta_id),
      .clear_divergence_valid_o (clear_divergence_valid),

      // CGRA configuration memory (internal)
      .cm0_data_o               (cm0_data),
      .cm0_chunk_en_o           (cm0_chunk_en),
      .cm1_data_o               (cm1_data),
      .cm1_chunk_en_o           (cm1_chunk_en)
  );


  // HW CTA Size Derivation (for SIMT stack update)
  assign simt_update_hw_cta_size = schedule_if.data.schedule_hw_cta_size[1:0];

endmodule
