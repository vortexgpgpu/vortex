`ifndef DICE_PKG_VH
`define DICE_PKG_VH

`include "dice_define.vh"

package dice_pkg;

  // =========================================================
  // Derived parameters (computed from configuration)
  // =========================================================
  parameter int DICE_ADDR_WIDTH      = `DICE_ADDR_WIDTH;
  parameter int DICE_KERNEL_ID_WIDTH = $clog2(`DICE_MAX_KERNEL_ID);
  parameter int DICE_CTA_ID_WIDTH    = $clog2(`DICE_MAX_GRID_SIZE);
  parameter int DICE_TID_WIDTH       = $clog2(`DICE_NUM_MAX_THREADS_PER_CORE);
  parameter int DICE_HW_CTA_ID_WIDTH = $clog2(`DICE_NUM_MAX_CTA_PER_CORE);
  parameter int DICE_EBLOCK_ID_WIDTH = `DICE_NUM_RETIRE_TABLE_ENTRIES + 4;
  parameter int DICE_CLUSTER_ID_WIDTH= $clog2(`DICE_NUM_CGRA_CLUSTERS);
  parameter int DICE_CORE_ID_WIDTH   = $clog2(`DICE_NUM_CGRA_CORES);
  parameter int DICE_SMEM_SIZE_WIDTH = $clog2(`DICE_SMEM_SIZE_PER_CORE);

  // =========================================================
  // Type definitions
  // =========================================================
  typedef struct packed {
    logic [DICE_CTA_ID_WIDTH:0]       x; //one more bit than needed to represent max value
    logic [DICE_CTA_ID_WIDTH:0]       y; //one more bit than needed to represent max value
    logic [DICE_CTA_ID_WIDTH:0]       z; //one more bit than needed to represent max value
  } dice_grid_size_t; // Grid size descriptor

  typedef struct packed {
    logic [DICE_TID_WIDTH:0]       x; //one more bit than needed to represent max value
    logic [DICE_TID_WIDTH:0]       y; //one more bit than needed to represent max value
    logic [DICE_TID_WIDTH:0]       z; //one more bit than needed to represent max value
  } dice_cta_size_t; // CTA size descriptor

  typedef struct packed {
    logic [DICE_CTA_ID_WIDTH-1:0]       x;
    logic [DICE_CTA_ID_WIDTH-1:0]       y;
    logic [DICE_CTA_ID_WIDTH-1:0]       z; 
  } dice_cta_id_t; // CTA ID descriptor

  typedef struct packed {
    logic [DICE_TID_WIDTH-1:0]       x;
    logic [DICE_TID_WIDTH-1:0]       y;
    logic [DICE_TID_WIDTH-1:0]       z; 
  } dice_tid_t; // Thread ID descriptor

  typedef struct packed {
    // IDs and geometry
    logic [DICE_KERNEL_ID_WIDTH-1:0] kernel_id;
    dice_grid_size_t                grid_size;
    dice_cta_size_t                 cta_size;
    // Resources for backend to determine shared memory address for each CTA
    logic [DICE_SMEM_SIZE_WIDTH-1:0] smem_per_cta;

    // Initial
    logic [DICE_ADDR_WIDTH-1:0] start_pc;
    logic [DICE_ADDR_WIDTH-1:0] arg_ptr; //might not need
  } dice_kernel_desc_t; // Kernel descriptor for top driver to receive kernel launch info

  typedef struct packed {
    dice_kernel_desc_t           kernel_desc;
    dice_cta_id_t                cta_id;
  } dice_cta_desc_t; // CTA descriptor passed to CGRA core front end

  typedef struct packed {
    logic unresolved_control_divergence;
    logic [DICE_ADDR_WIDTH-1:0] predict_pc;
    logic still_pending_in_BRT;
    logic return_pending;
  } dice_cta_status_t; // CTA status descriptor

  typedef struct packed {
    logic [DICE_HW_CTA_ID_WIDTH-1:0] hw_cta_id;
    logic unresolved_control_divergence;
    logic [DICE_ADDR_WIDTH-1:0] predict_pc;
    logic is_return;
  } branch_predict_interface_t; // Branch prediction interface descriptor

  typedef struct packed {
    logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] hw_cta_pending;
  } block_retire_status_t; // Block retire status descriptor

endpackage

`endif // DICE_PKG_VH
