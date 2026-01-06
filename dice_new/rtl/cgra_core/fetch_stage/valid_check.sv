/**
 * Valid Checker Module
 *
 * Determines when an e-block can be passed from FDR stage to DE stage.
 * Valid is asserted when ALL conditions are met:
 *   1) Bitstream is loaded
 *   2) E-block's prefetch is cleared OR not a prefetch block
 *   3) Decode/Branch Handler is done (mask_valid from decoder)
 *   4) Barrier condition met
 *   5) PC match (only for prefetch blocks after divergence resolved)
 */
module valid_check
  import dice_pkg::*;
  import dice_frontend_pkg::*;
(
    // -------------------------------------------------------------------------
    // From Decoder
    // -------------------------------------------------------------------------
    input logic barrier_indicator_i,  // This p-graph requires barrier (all prev blocks must finish)
    input logic mask_valid_i,         // Decode done (mask valid)

    //from CS, FDR buffer
    input logic [DICE_ADDR_WIDTH-1:0]    eblock_pc_i,
    input logic                                    prefetch_block_i,
    input logic [DICE_HW_CTA_ID_WIDTH-1:0] hw_cta_id_i,

    // -------------------------------------------------------------------------
    // From SIMT Stack
    // -------------------------------------------------------------------------
    input logic [DICE_ADDR_WIDTH-1:0] simt_stack_pc_i,  // next_pc for branch predict

    // -------------------------------------------------------------------------
    // From Bitstream Loader
    // -------------------------------------------------------------------------
    input logic bitstream_loaded_i,

    // -------------------------------------------------------------------------
    // From CTA Status Table
    // -------------------------------------------------------------------------
    input logic unresolved_div_i,     // Unresolved control divergence for this CTA
    input logic barrier_complete_i,   // Barrier condition met (prev blocks retired)
    input logic prefetch_cleared_i,   // Prefetch has been resolved for this CTA

    // -------------------------------------------------------------------------
    // To FDR-DE Stage Buffer
    // -------------------------------------------------------------------------
    output logic fdr_valid_o,
    input  logic ex_ready_i,

    // -------------------------------------------------------------------------
    // Feedback/Control Signals
    // -------------------------------------------------------------------------
    output logic fire_eblock_o,       // Valid handshake complete (valid && ready)
    output logic clear_prefetch_o,    // Signal to clear prefetch in status table (predict hit)
    output logic predict_miss_o       // Signal predict miss (flush FDR stage)
);

  // ===========================================================================
  // Condition Signals
  // ===========================================================================
  logic pc_match;
  logic pc_match_required;
  logic pc_check_pass;
  logic prefetch_ok;
  logic bitstream_ok;
  logic mask_ok;
  logic barrier_ok;
  logic no_divergence;
  logic can_issue;

  // ===========================================================================
  // Condition 1: Bitstream Loaded
  // ===========================================================================
  assign bitstream_ok = bitstream_loaded_i;

  // ===========================================================================
  // Condition 2: Prefetch OK (not prefetch OR prefetch cleared)
  // ===========================================================================
  assign prefetch_ok = !prefetch_block_i || prefetch_cleared_i;

  // ===========================================================================
  // Condition 3: Decode Done (mask_valid from decoder/branch handler)
  // ===========================================================================
  assign mask_ok = mask_valid_i;

  // ===========================================================================
  // Condition 4: Barrier OK (no barrier required OR barrier complete)
  // ===========================================================================
  assign barrier_ok = !barrier_indicator_i || barrier_complete_i;

  // ===========================================================================
  // Condition 5: No Unresolved Divergence
  // ===========================================================================
  assign no_divergence = !unresolved_div_i;

  // ===========================================================================
  // PC Match Check (only for prefetch blocks after divergence resolved)
  // After unresolved divergence is cleared, check if e-block PC matches
  // the SIMT stack next_pc. This confirms the branch prediction was correct.
  // ===========================================================================
  assign pc_match = (eblock_pc_i == simt_stack_pc_i);

  // PC match is only required for prefetch blocks after divergence is resolved
  // but before prefetch is cleared
  assign pc_match_required = prefetch_block_i && !unresolved_div_i && !prefetch_cleared_i;

  // Pass if PC match not required, or if it matches
  assign pc_check_pass = !pc_match_required || pc_match;

  // ===========================================================================
  // Final Valid Condition
  // ===========================================================================
  assign can_issue = bitstream_ok     &&
                     prefetch_ok      &&
                     mask_ok          &&
                     barrier_ok       &&
                     no_divergence    &&
                     pc_check_pass;

  // ===========================================================================
  // Outputs
  // ===========================================================================

  // Valid to DE Stage
  assign fdr_valid_o = can_issue;

  // Fire when valid AND DE stage ready (handshake complete)
  assign fire_eblock_o = can_issue && ex_ready_i;

  // Clear prefetch on branch predict hit:
  // When PC match was required and we matched, signal to clear prefetch state
  assign clear_prefetch_o = pc_match_required && pc_match && can_issue;

  // Predict miss: prefetch block, divergence resolved, but PC mismatch
  // This should trigger a flush of the FDR stage
  assign predict_miss_o = pc_match_required && !pc_match;

endmodule
