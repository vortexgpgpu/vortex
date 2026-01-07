/**
 * Branch Control Interface
 * Carries divergence and prefetch clearing signals from FDR to CTA Schedule Stage.
 */
interface branch_control_if
  import dice_pkg::*;
  import dice_frontend_pkg::*;
();

  branch_control_t ctrl;

  // FDR produces control signals
  modport master (
    output ctrl
  );

  // CTA Schedule Stage / other consumers
  modport slave (
    input ctrl
  );

endinterface
