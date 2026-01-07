/**
 * SIMT Stack Status Interface
 * Carries per-CTA SIMT stack status from CTA Schedule Stage to FDR.
 */
interface simt_stack_status_if
  import dice_pkg::*;
  import dice_frontend_pkg::*;
();

  // Full status array for all CTAs
  simt_stack_status_entry_t [DICE_NUM_MAX_CTA_PER_CORE-1:0] status;

  // CTA Schedule Stage produces status
  modport master (
    output status
  );

  // FDR consumes status
  modport slave (
    input status
  );

endinterface
