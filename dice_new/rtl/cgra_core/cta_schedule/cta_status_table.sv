
module cta_status_table
  import dice_pkg::*;
  import dice_frontend_pkg::*;
(
    input logic clk_i,
    input logic rst_i,

    // From branch handler / branch predictor
    input branch_predict_interface_t branch_predict_info_i,
    input logic                    branch_predict_info_we_i,

    // From Block Retire Table (BRT)
    input block_retire_status_t brt_info_i,
    input logic               brt_info_we_i,

    // From cta controller
    input logic clear_entry_valid_i,
    input logic [DICE_HW_CTA_ID_WIDTH-1:0] clear_entry_hw_id_i,

    // Exposed status for each CTA
    output dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status_o
);

  dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status_q;
  dice_cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_status_d;

  logic [DICE_HW_CTA_ID_WIDTH-1:0] bp_cta_id;

  always_comb begin
    bp_cta_id = '0;
    for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
      cta_status_d[i] = cta_status_q[i];
    end

    if (brt_info_we_i) begin
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        cta_status_d[i].has_pending_eblock = brt_info_i.hw_cta_pending[i];
      end
    end

    if (branch_predict_info_we_i) begin
      bp_cta_id = branch_predict_info_i.hw_cta_id;

      cta_status_d[bp_cta_id].prefetch_cleared =
          branch_predict_info_i.unresolved_control_divergence;
      cta_status_d[bp_cta_id].is_return        = branch_predict_info_i.is_return;
      cta_status_d[bp_cta_id].predict_pc       = branch_predict_info_i.predict_pc;
      cta_status_d[bp_cta_id].is_barrier       = branch_predict_info_i.is_barrier;
    end

    if (clear_entry_valid_i) begin
      cta_status_d[clear_entry_hw_id_i].has_pending_eblock = 1'b0;
      cta_status_d[clear_entry_hw_id_i].unresolved_control_divergence = 1'b0;
      cta_status_d[clear_entry_hw_id_i].is_return = 1'b0;
      cta_status_d[clear_entry_hw_id_i].predict_pc = '0;
      cta_status_d[clear_entry_hw_id_i].is_barrier = 1'b0;
      cta_status_d[clear_entry_hw_id_i].prefetch_cleared = 1'b0;
      cta_status_d[clear_entry_hw_id_i].is_prefetch = 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        cta_status_q[i] <= '0;
      end
    end else begin
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        cta_status_q[i] <= cta_status_d[i];
      end
    end
  end

  assign cta_status_o = cta_status_q;

  `ifndef SYNTHESIS
  // Status is cleared in the next cycle after clear_entry_valid_i
  always_ff @(posedge clk_i) begin
    if (clear_entry_valid_i) begin
      assert (cta_status_d[clear_entry_hw_id_i].has_pending_eblock == 1'b0)
      else $error("CleanStatusAfterClear: Status not cleared (check immediate update)");
    end
  end
  `endif

endmodule
