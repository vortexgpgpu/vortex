`include "dice_define.vh"

module cta_schedule_stage
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int STACK_DEPTH = 32
) (

    input logic clk_i,
    input logic rst_i,

    // Host/Dispatcher interface for new CTA allocation
    input  logic                     cta_add_valid_i,
    output logic                     cta_add_ready_o,
    input  dice_cta_desc_t new_cta_all_desc_i,

    // CTA completion output (to dispatcher)
    output logic                   cta_complete_valid_o,
    input  logic                   cta_complete_ready_i,
    output dice_cta_id_t cta_done_id_o,

    // Scheduler output interface (to FDR stage)
    cta_sched_if.master schedule_if,

    // E-block commit interface (from execution/retire)
    input logic                     eblock_commit_valid_i,
    input logic [EBLOCK_ID_WIDTH-1:0] eblock_commit_id_i,

    // Branch handler / predictor interface (from FDR/execution)
    branch_handler_if.slave status_table_bh_if,

    // Block Retire Table interface
    input block_retire_status_t brt_info_i,
    input logic                           brt_info_write_enable_i,


    // UPDATE INTERFACE (SIMT STACK CONTROLLER AND BRANCH HANDLER)
    dice_bh_simt_if.slave                  simt_stack_update,
    input logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0] simt_update_hw_cta_id_i,
    input logic                 [           1:0] simt_update_hw_cta_size_i,


    // SIMT STACK STATUS - MAY CHANGE TO BE INCLUDED IN BH AND VC IFs
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_top_valid_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_ADDR_WIDTH-1:0] stack_top_next_pc_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_ADDR_WIDTH-1:0] stack_top_reconvergence_pc_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_NUM_MAX_THREADS_PER_CORE/DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_top_active_mask_o,
    // Stack status - individual stack status
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_empty_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_full_o

    //cta status table stuff

);

  // -------------------------------------------------------------------------
  // Local Parameters (derived from packages)
  // -------------------------------------------------------------------------
  localparam int ThreadWidth = DICE_NUM_MAX_THREADS_PER_CORE / DICE_NUM_MAX_CTA_PER_CORE;

  // -------------------------------------------------------------------------
  // Local wires
  // -------------------------------------------------------------------------
  logic active_table_add_ready;
  logic active_table_add_valid;
  dice_cta_desc_t active_table_cta_desc;
  logic [DICE_TID_WIDTH:0] active_table_cta_size;
  logic active_table_pop_valid;
  logic [DICE_HW_CTA_ID_WIDTH-1:0] active_table_pop_hw_id;
  logic active_table_pop_ready;
  logic active_table_out_valid;
  logic active_table_out_ready;
  dice_cta_id_t active_table_out_cta_id;
  logic [DICE_HW_CTA_ID_WIDTH-1:0] active_table_next_empty_idx;
  active_cta_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] active_cta_entries;

  dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status_real;

  assign status_table_bh_if.cta_status_data = cta_status_real;

  // Adapter for cta_scheduler which uses cta_status_t
  cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0]
      scheduler_status_adapter;

  always_comb begin
    for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
      scheduler_status_adapter[i].hw_cta_id   = (DICE_CTA_ID_WIDTH + 1)'(i);
      scheduler_status_adapter[i].is_prefetch = cta_status_real[i].is_prefetch;
      scheduler_status_adapter[i].predict_pc  = cta_status_real[i].predict_pc;
    end
  end

  // SIMT stack update wiring
  logic simt_stack_update_ready;
  assign simt_stack_update.update_ready = simt_stack_update_ready;

  // SIMT stack initialization wiring (cta_controller → simt_stack_controller)
  logic                                           simt_init_valid;
  logic [                $clog2(MAX_NUM_CTA)-1:0] simt_init_hw_cta_id;
  logic [                                    1:0] simt_init_hw_cta_size;
  logic [                           PC_WIDTH-1:0] simt_init_pc;
  logic [                           PC_WIDTH-1:0] simt_init_reconvergence_pc;
  logic                                           simt_init_ready;


  // Create validity bitmap from active_cta_entries
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] active_cta_validty_bitmap;
  always_comb begin
    for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
      active_cta_validty_bitmap[i] = active_cta_entries[i].cta_valid;
    end
  end

  logic clear_entry_valid;
  logic [DICE_HW_CTA_ID_WIDTH-1:0] clear_entry_hw_id;

  // -------------------------------------------------------------------------
  // CTA Controller
  // -------------------------------------------------------------------------
  cta_controller cta_controller_inst (
      .clk_i                  (clk_i),
      .rst_i                  (rst_i),
      .in_cta_valid_i         (cta_add_valid_i),
      .in_cta_ready_o         (cta_add_ready_o),
      .in_cta_desc_i          (new_cta_all_desc_i),
      .comp_cta_valid_o       (cta_complete_valid_o),
      .comp_cta_ready_i       (cta_complete_ready_i),
      .comp_cta_id_o          (cta_done_id_o),
      .add_valid_o            (active_table_add_valid),
      .add_ready_i            (active_table_add_ready),
      .add_cta_info_o         (active_table_cta_desc),
      .add_cta_size_o         (active_table_cta_size),
      .next_empty_cta_index_i (active_table_next_empty_idx),
      .pop_valid_o            (active_table_pop_valid),
      .pop_hw_cta_id_o        (active_table_pop_hw_id),
      .pop_ready_i            (active_table_pop_ready),
      .active_cta_status_i    (active_cta_validty_bitmap),
      .pop_out_valid_i        (active_table_out_valid),
      .pop_out_cta_id_i       (active_table_out_cta_id),
      .init_valid_o           (simt_init_valid),
      .init_hw_cta_id_o       (simt_init_hw_cta_id),
      .init_hw_cta_size_o     (simt_init_hw_cta_size),
      .init_pc_o              (simt_init_pc),
      .init_reconvergence_pc_o(simt_init_reconvergence_pc),
      .init_ready_i           (simt_init_ready),
      .cta_status_table_i     (cta_status_real),
      .clear_entry_valid_o    (clear_entry_valid),
      .clear_entry_hw_id_o    (clear_entry_hw_id)
  );

  // -------------------------------------------------------------------------
  // Active CTA Table
  // -------------------------------------------------------------------------
  active_cta_table active_cta_table_inst (
      .clk_i                 (clk_i),
      .rst_i                 (rst_i),
      .add_ready_o           (active_table_add_ready),
      .add_valid_i           (active_table_add_valid),
      .add_cta_info_i        (active_table_cta_desc),
      .add_cta_size_i        (active_table_cta_size[DICE_TID_WIDTH-1:0]),
      .pop_valid_i           (active_table_pop_valid),
      .pop_hw_cta_id_i       (active_table_pop_hw_id),
      .pop_ready_o           (active_table_pop_ready),
      .out_valid_o           (active_table_out_valid),
      .out_ready_i           (active_table_out_ready),
      .out_cta_id_o          (active_table_out_cta_id),
      .out_cta_size_o        (),
      .out_kernel_id_o       (),
      .active_cta_entries_o  (active_cta_entries),
      .full_o                (),
      .next_empty_cta_index_o(active_table_next_empty_idx)
  );


  // -------------------------------------------------------------------------
  // CTA Scheduler
  // -------------------------------------------------------------------------
  cta_scheduler cta_scheduler_inst (
      .clk_i                  (clk_i),
      .rst_i                  (rst_i),
      .enable_i               (1'b1),
      .active_cta_entries_i   (active_cta_entries),
      .cta_status_entries_i   (scheduler_status_adapter),
      .cta_next_pc_i          (stack_top_next_pc_o),
      .stack_top_active_mask_i(stack_top_active_mask_o),
      .eblock_commit_valid_i  (eblock_commit_valid_i),
      .eblock_commit_id_i     ((DICE_EBLOCK_ID_WIDTH)'(eblock_commit_id_i)),
      .scheduled_eblock       (schedule_if)
  );



  // -------------------------------------------------------------------------
  // CTA Status Table
  // -------------------------------------------------------------------------
  cta_status_table cta_status_table_inst (
      .clk_i                   (clk_i),
      .rst_i                   (rst_i),
      .branch_predict_info_i   (status_table_bh_if.bh_data),
      .branch_predict_info_we_i(status_table_bh_if.branch_predict_info_write_enable),
      .brt_info_i              (brt_info_i),
      .brt_info_we_i           (brt_info_write_enable_i),
      .clear_entry_valid_i     (clear_entry_valid),
      .clear_entry_hw_id_i     (clear_entry_hw_id),
      .cta_status_o            (cta_status_real)
  );


  // -------------------------------------------------------------------------
  // SIMT Stack Controller
  // -------------------------------------------------------------------------
  simt_stack_controller #(
      .STACK_DEPTH (STACK_DEPTH)
  ) simt_stack_controller_inst (
      .clk_i(clk_i),
      .rst_i(rst_i),
      .hw_cta_id_i(simt_update_hw_cta_id_i),
      .hw_cta_size_i(simt_update_hw_cta_size_i),
      .update_valid_i(simt_stack_update.update_valid),
      .update_with_divergence_i(simt_stack_update.update_stack_data.update_with_divergence),
      .update_next_pc_i(simt_stack_update.update_stack_data.update_next_pc),
      .predicate_regs_value_i     (simt_stack_update.update_stack_data.predicate_regs_value[
                                      DICE_NUM_MAX_CTA_PER_CORE*ThreadWidth-1:0]),
      .branch_not_taken_pc_i(simt_stack_update.update_stack_data.branch_not_taken_pc),
      .branch_reconvergence_pc_i(simt_stack_update.update_stack_data.branch_reconvergence_pc),
      .update_ready_o(simt_stack_update_ready),
      .init_valid_i(simt_init_valid),
      .init_hw_cta_id_i(simt_init_hw_cta_id),
      .init_hw_cta_size_i(simt_init_hw_cta_size),
      .init_pc_i(simt_init_pc),
      .init_reconvergence_pc_i(simt_init_reconvergence_pc),
      .init_ready_o(simt_init_ready),
      .stack_top_valid_o(stack_top_valid_o),
      .stack_top_next_pc_o(stack_top_next_pc_o),
      .stack_top_reconvergence_pc_o(stack_top_reconvergence_pc_o),
      .stack_top_active_mask_o(stack_top_active_mask_o),
      .stack_empty_o(stack_empty_o),
      .stack_full_o(stack_full_o)
  );


endmodule
