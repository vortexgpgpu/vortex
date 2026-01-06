// =============================================================================
// Module: divergence_monitor.sv
// =============================================================================
// Background branch resolution path. Continuously scans CTA status table for
// CTAs that have:
//   - unresolved_control_divergence = 1
//   - has_pending_eblock = 0
//
// When both conditions are met, reads the pending branch table and predicate
// register file to resolve the deferred branch via SIMT stack controller.
// =============================================================================

`include "dice_define.vh"

module divergence_monitor
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int PcWidth     = DICE_ADDR_WIDTH,
    parameter int ThreadWidth = DICE_NUM_MAX_THREADS_PER_CORE,
    parameter int NumCta      = DICE_NUM_MAX_CTA_PER_CORE,
    parameter int NumPredRegs = DICE_PR_NUM
) (
    input logic clk_i,
    input logic rst_i,

    // =========================================================================
    // CTA Status (read all CTAs)
    // =========================================================================
    input dice_cta_status_t [NumCta-1:0] cta_status_i,

    // =========================================================================
    // Pending Branch Table (from branch_resolver)
    // =========================================================================
    input pending_branch_info_t [NumCta-1:0] pending_branch_table_i,

    // =========================================================================
    // Predicate RF Interface
    // =========================================================================
    output logic                                          prf_req_o,
    output logic [$clog2(NumCta)+$clog2(NumPredRegs)-1:0] prf_raddr_o,
    input  logic [                       ThreadWidth-1:0] prf_rdata_i,

    // =========================================================================
    // SIMT Stack Controller Interface
    // =========================================================================
    output logic                      update_valid_o,
    output logic                      update_with_divergence_o,
    output logic [       PcWidth-1:0] update_next_pc_o,
    output logic [       PcWidth-1:0] branch_not_taken_pc_o,
    output logic [       PcWidth-1:0] branch_reconvergence_pc_o,
    output logic [   ThreadWidth-1:0] predicate_regs_value_o,
    output logic [$clog2(NumCta)-1:0] update_hw_cta_id_o,
    input  logic                      update_ready_i,

    // =========================================================================
    // Grant Signal (from arbiter - active when foreground is not using stack)
    // =========================================================================
    input logic grant_i,

    // =========================================================================
    // To CTA Status Table (clear divergence)
    // =========================================================================
    output logic [$clog2(NumCta)-1:0] clear_cta_id_o,
    output logic                      clear_divergence_valid_o
);

  // ===========================================================================
  // FSM States
  // ===========================================================================
  typedef enum logic [2:0] {
    StateScan,
    StateReadPrf,
    StateWaitStack,
    StateClearStatus
  } state_e;

  state_e current_state_q, next_state;

  // ===========================================================================
  // Internal Signals
  // ===========================================================================
  // Round-robin scanner pointer
  logic                 [$clog2(NumCta)-1:0] monitor_ptr_q;

  // Work detection
  logic                                      found_work;
  logic                 [$clog2(NumCta)-1:0] work_cta_id;

  // Registered CTA ID for multi-cycle operation
  logic                 [$clog2(NumCta)-1:0] target_cta_id_q;
  pending_branch_info_t                      target_branch_info_q;

  // ===========================================================================
  // Work Detection (combinational scan)
  // ===========================================================================
  always_comb begin
    found_work  = 1'b0;
    work_cta_id = monitor_ptr_q;

    // Check the CTA at current pointer
    if (cta_status_i[monitor_ptr_q].unresolved_control_divergence &&
        !cta_status_i[monitor_ptr_q].has_pending_eblock) begin
      found_work  = 1'b1;
      work_cta_id = monitor_ptr_q;
    end
  end

  // ===========================================================================
  // FSM Next State Logic
  // ===========================================================================
  always_comb begin
    next_state = current_state_q;

    case (current_state_q)
      StateScan: begin
        if (found_work && grant_i) begin
          next_state = StateReadPrf;
        end
        // Otherwise keep scanning (ptr advances each cycle)
      end

      StateReadPrf: begin
        // One cycle for PRF read latency
        next_state = StateWaitStack;
      end

      StateWaitStack: begin
        if (update_ready_i) begin
          next_state = StateClearStatus;
        end
      end

      StateClearStatus: begin
        // Clear divergence flag, return to scanning
        next_state = StateScan;
      end

      default: begin
        next_state = StateScan;
      end
    endcase
  end

  // ===========================================================================
  // FSM Sequential Logic
  // ===========================================================================
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      current_state_q      <= StateScan;
      monitor_ptr_q        <= '0;
      target_cta_id_q      <= '0;
      target_branch_info_q <= '0;
    end else begin
      current_state_q <= next_state;

      // Advance monitor pointer in scan state
      if (current_state_q == StateScan) begin
        if (found_work && grant_i) begin
          // Found work, capture target CTA
          target_cta_id_q      <= work_cta_id;
          target_branch_info_q <= pending_branch_table_i[work_cta_id];
        end else begin
          // No work at current pointer, advance
          if (monitor_ptr_q == NumCta[$clog2(NumCta)-1:0] - 1) begin
            monitor_ptr_q <= '0;
          end else begin
            monitor_ptr_q <= monitor_ptr_q + 1'b1;
          end
        end
      end
    end
  end

  // ===========================================================================
  // Predicate RF Interface
  // ===========================================================================
  always_comb begin
    prf_req_o   = 1'b0;
    prf_raddr_o = '0;

    if (current_state_q == StateReadPrf || current_state_q == StateWaitStack) begin
      prf_req_o   = 1'b1;
      prf_raddr_o = {target_cta_id_q, target_branch_info_q.pred_reg};
    end
  end

  // ===========================================================================
  // SIMT Stack Controller Interface
  // ===========================================================================
  always_comb begin
    update_valid_o            = 1'b0;
    update_with_divergence_o  = 1'b0;
    update_next_pc_o          = '0;
    branch_not_taken_pc_o     = '0;
    branch_reconvergence_pc_o = '0;
    predicate_regs_value_o    = '0;
    update_hw_cta_id_o        = '0;

    if (current_state_q == StateWaitStack) begin
      update_valid_o            = 1'b1;
      update_with_divergence_o  = 1'b1;  // Always divergent (that's why it was deferred)
      update_next_pc_o          = target_branch_info_q.taken_pc;
      branch_not_taken_pc_o     = target_branch_info_q.not_taken_pc;
      branch_reconvergence_pc_o = target_branch_info_q.reconv_pc;
      update_hw_cta_id_o        = target_cta_id_q;

      // Apply predicate polarity
      predicate_regs_value_o    = target_branch_info_q.neg_pred ? ~prf_rdata_i : prf_rdata_i;
    end
  end

  // ===========================================================================
  // Clear Divergence Flag
  // ===========================================================================
  always_comb begin
    clear_cta_id_o           = '0;
    clear_divergence_valid_o = 1'b0;

    if (current_state_q == StateClearStatus) begin
      clear_cta_id_o           = target_cta_id_q;
      clear_divergence_valid_o = 1'b1;
    end
  end

endmodule
