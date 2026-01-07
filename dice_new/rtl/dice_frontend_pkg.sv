`include "dice_define.vh"

package dice_frontend_pkg;
  // Import dice_pkg in package body (allowed per lowRISC style guide for same-IP)
  import dice_pkg::*;


  localparam int BITSTREAM_LENGTH_WIDTH = 8;
  localparam int MAX_EBLOCK = `DICE_NUM_MAX_CTA_PER_CORE + `DICE_NUM_RETIRE_TABLE_ENTRIES;
  localparam int EBLOCK_ID_WIDTH = $clog2(MAX_EBLOCK);
  localparam int SIMT_STACK_COUNT = `DICE_NUM_MAX_CTA_PER_CORE;
  localparam int SIMT_STACK_THREAD_WIDTH = `DICE_NUM_MAX_THREADS_PER_CORE;
  localparam int SIMT_STACK_DEPTH = 32;

  // Predicate register configuration (moved from defines for module use)
  localparam int DICE_PR_NUM = `DICE_PR_NUM;

  localparam int REG_NUM = `DICE_GPR_NUM + `DICE_PR_NUM + `DICE_CR_NUM;


  // =========================================================
  // Type definitions
  // =========================================================


  /**
  * Branch Metadata Structure
  * Defines control logic for p-graph branching, including predicate dependencies,
  * jump targets, and hardware reconvergence points.
  */
  typedef struct packed {
    logic branch_ena;  // Branch enable: active if branch is associated with current p-graph
    logic branch_uni;  // Universal branch: if set, ignore branch_pred_reg
    logic [$clog2(DICE_PR_NUM)-1:0] branch_pred_reg;  // Predicate register dependency index
    logic branch_neg_pred;  // Polarity: 1 = jump if pred is 0; 0 = jump if pred is 1

    // Jump Target Calculation:
    // Actual PC = Current_PC + (branch_jump_target_offset * Metadata_Length)
    logic [$clog2(`DICE_MAX_PGRAPHS)-1:0] branch_jump_target_offset;

    // Reconvergence Calculation:
    // Reconvergence PC = Current_PC + (branch_reconv_offset * Metadata_Length)
    logic [$clog2(`DICE_MAX_PGRAPHS)-1:0] branch_reconv_offset;
  } branch_meta_t;

  //metadata
  typedef struct packed {
    logic [DICE_ADDR_WIDTH-1:0]                                   bitstream_addr;
    logic [BITSTREAM_LENGTH_WIDTH-1:0]                            bitstream_length;
    logic [1:0]                                                   unrolling_factor;
    logic [7:0]                                                   lat;
    logic [REG_NUM-1:0]                                           in_regs_bitmap;
    logic [REG_NUM-1:0]                                           out_regs_bitmap;
    logic [$clog2(`DICE_CGRA_MEM_PORTS-1):0][$clog2(REG_NUM)-1:0] ld_dest_regs;
    logic [$clog2(`DICE_CGRA_MEM_PORTS-1):0]                      num_stores;
    branch_meta_t                                                 branch_meta;
    logic                                                         barrier;
    logic                                                         parameter_load;
  } pgraph_meta_t;

  //thread mask -> typedef may not be needed
  typedef logic [`DICE_NUM_MAX_THREADS_PER_CORE-1:0] thread_mask_t;

  typedef struct packed {
    logic [BITSTREAM_LENGTH_WIDTH-1:0] bitstream_length;
    logic [REG_NUM-1:0] in_regs_bitmap;
    logic [REG_NUM-1:0] out_regs_bitmap;
    logic [$clog2(`DICE_CGRA_MEM_PORTS-1):0][$clog2(REG_NUM)-1:0] ld_dest_regs;
    logic [$clog2(`DICE_CGRA_MEM_PORTS-1):0] num_stores;
    logic [1:0] unrolling_factor;
    logic [7:0] lat;
    logic parameter_load;
  } fdr_meta_t;

  //stage borders
  typedef struct packed {
    logic [DICE_HW_CTA_ID_WIDTH-1:0]             schedule_hw_cta_id;
    logic [DICE_ADDR_WIDTH-1:0]                  schedule_next_pc;
    logic [EBLOCK_ID_WIDTH-1:0]                  schedule_eblock_id;
    thread_mask_t                                schedule_active_mask;
    logic                                        schedule_prefetch_block;
    dice_cta_id_t                                schedule_cta_id;
    dice_grid_size_t                             schedule_grid_size;
    dice_cta_size_t                              schedule_cta_size;
    logic [DICE_KERNEL_ID_WIDTH-1:0]             schedule_kernel_id;
    logic [DICE_SMEM_SIZE_WIDTH-1:0]             schedule_smem_per_cta;
    logic [dice_pkg::DICE_HW_CTA_SIZE_WIDTH-1:0] schedule_hw_cta_size;
  } schedule_eblock_t;


  typedef struct packed {
    // IDs
    logic [DICE_HW_CTA_ID_WIDTH-1:0] schedule_hw_cta_id;
    logic [EBLOCK_ID_WIDTH-1:0]      schedule_eblock_id;
    dice_cta_id_t                    schedule_cta_id;
    logic [DICE_KERNEL_ID_WIDTH-1:0] schedule_kernel_id;

    // Geometry & resources
    dice_grid_size_t                             schedule_grid_size;
    dice_cta_size_t                              schedule_cta_size;
    logic [dice_pkg::DICE_HW_CTA_SIZE_WIDTH-1:0] schedule_hw_cta_size; //CHANGE THIS TO HW SIZE
    logic [1:0] hw_cta_size;
    logic [DICE_SMEM_SIZE_WIDTH-1:0]             schedule_smem_per_cta;

    // Execution state
    logic [DICE_NUM_MAX_THREADS_PER_CORE-1:0] real_active_mask;

    // Metadata
    fdr_meta_t metadata;
    logic      loaded_buffer;
  } fdr_t;



  // CTA table entry structure
  typedef struct packed {
    logic [DICE_CTA_ID_WIDTH:0] hw_cta_id;  //may not need if it is indexed by this
    logic cta_valid;
    dice_cta_id_t cta_id;
    dice_grid_size_t grid_size;
    dice_cta_size_t cta_size;
    logic [DICE_KERNEL_ID_WIDTH-1:0] kernel_id;
    logic [DICE_SMEM_SIZE_WIDTH-1:0] smem_per_cta;
    logic [dice_pkg::DICE_HW_CTA_SIZE_WIDTH-1:0] hw_cta_size;
  } active_cta_t;


  typedef struct packed {
    logic [DICE_CTA_ID_WIDTH:0] hw_cta_id;
    logic                       is_prefetch;
    logic [DICE_ADDR_WIDTH-1:0] predict_pc;
  } cta_status_t;




  typedef struct packed {
    logic update_with_divergence;  // 0 = no divergence, 1 = with divergence
    logic [DICE_ADDR_WIDTH-1:0]         update_next_pc;  // No divergence: next PC, With divergence: branch taken PC
    // Divergence case inputs (only used when update_with_divergence = 1)
    logic [SIMT_STACK_COUNT*SIMT_STACK_THREAD_WIDTH-1:0] predicate_regs_value;
    logic [DICE_ADDR_WIDTH-1:0] branch_not_taken_pc;
    logic [DICE_ADDR_WIDTH-1:0] branch_reconvergence_pc;
  } simt_stack_update_t;


  /**
   * Pending Branch Info Structure
   * Stores branch metadata when resolution is deferred due to pending eblocks.
   * Used by divergence_monitor to resolve branches once dependencies clear.
   */
  typedef struct packed {
    logic [$clog2(DICE_PR_NUM)-1:0] pred_reg;      // Predicate register index
    logic                           neg_pred;      // Polarity (0=jump if pred=1, 1=jump if pred=0)
    logic [DICE_ADDR_WIDTH-1:0]     taken_pc;      // Branch taken target PC
    logic [DICE_ADDR_WIDTH-1:0]     not_taken_pc;  // Fall-through PC
    logic [DICE_ADDR_WIDTH-1:0]     reconv_pc;     // Reconvergence point PC
  } pending_branch_info_t;


  // =========================================================
  // Interface Support Structures
  // =========================================================

  /**
   * SIMT Stack Status Entry
   * Per-CTA status from the SIMT stack controller.
   */
  typedef struct packed {
    logic                              valid;
    logic [DICE_ADDR_WIDTH-1:0]        next_pc;
    logic [DICE_ADDR_WIDTH-1:0]        reconvergence_pc;
    logic [SIMT_STACK_THREAD_WIDTH-1:0] active_mask;
    logic                              empty;
    logic                              full;
  } simt_stack_status_entry_t;

  /**
   * Branch Control Structure
   * Aggregates divergence and prefetch clearing signals from FDR.
   */
  typedef struct packed {
    logic                            clear_divergence_valid;
    logic [DICE_HW_CTA_ID_WIDTH-1:0] clear_divergence_cta_id;
    logic                            clear_prefetch_valid;
    logic [DICE_HW_CTA_ID_WIDTH-1:0] clear_prefetch_hw_cta_id;
    logic                            predict_miss_flush;
  } branch_control_t;


endpackage
