//NOTES AND POTENTIAL ISSUES:
//circular pointer could result in stalls if it wraps around to an eblock
//that is waiting for a long latency load/store

//the priority schedule was changed so it starts from id+1, ensure that is correct


module cta_scheduler
  import dice_pkg::*;
  import dice_frontend_pkg::*;
#(
    parameter int MAX_EBLOCK = DICE_NUM_MAX_CTA_PER_CORE + 4,
    parameter int THREAD_WIDTH = DICE_NUM_MAX_THREADS_PER_CORE /
                                 DICE_NUM_MAX_CTA_PER_CORE
) (
    input logic clk_i,
    input logic rst_i,
    input logic enable_i, // Enable signal for scheduler operation

    // Active CTA Table
    input active_cta_t [DICE_NUM_MAX_CTA_PER_CORE-1:0]
        active_cta_entries_i,

    //CTA STATUS TABLE
    input cta_status_t [DICE_NUM_MAX_CTA_PER_CORE-1:0]
        cta_status_entries_i,

    //SIMT STACK
    //do i need this: stack_top_valid
    input logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_ADDR_WIDTH-1:0]
        cta_next_pc_i,
    input logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][THREAD_WIDTH-1:0] stack_top_active_mask_i,


    // External interface to invalidate committed e-blocks
    input logic                                      eblock_commit_valid_i,
    input logic [DICE_EBLOCK_ID_WIDTH-1:0] eblock_commit_id_i,

    // Scheduler outputs
    cta_sched_if.master scheduled_eblock
);

  // E-block tracking table
  logic [MAX_EBLOCK-1:0] eblock_live_q;
  logic [DICE_EBLOCK_ID_WIDTH-1:0] eblock_ptr_q;  // Circular pointer for e-block alloc

  // PC history for locality scheduling
  logic [DICE_ADDR_WIDTH-1:0] previous_pc_q;
  logic pc_history_valid_q;

  // Round-robin tracking
  logic [DICE_HW_CTA_ID_WIDTH-1:0] last_dispatched_cta_q;

  // Internal scheduling signals
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] priority_match;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] non_branch_candidates;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] any_valid_candidates;

  logic priority_found;
  logic non_branch_found;
  logic any_valid_found;

  logic [DICE_HW_CTA_ID_WIDTH-1:0] selected_cta_id;
  logic selection_valid;
  logic selected_from_branch_resolving;

  //unpack struct
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] cta_valid, cta_branch_resolving;
  always_comb begin
    for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
      cta_valid[i] = active_cta_entries_i[i].cta_valid;
      cta_branch_resolving[i] = cta_status_entries_i[i].is_prefetch;
    end
  end


  // Priority 1: PC locality matching (next_pc matches previous_pc)
  always_comb begin
    priority_match = '0;
    priority_found = 1'b0;

    if (pc_history_valid_q == 1'b1) begin
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        if ((cta_valid[i] == 1'b1) && (cta_next_pc_i[i] == previous_pc_q)) begin
          priority_match[i] = 1'b1;
          priority_found = 1'b1;
        end
      end
    end
  end


  // Priority 2: Non-branch resolving CTAs (round-robin among valid && !branch_resolving)
  always_comb begin
    non_branch_candidates = cta_valid & ~cta_branch_resolving;
    non_branch_found = (non_branch_candidates != '0);
  end

  // Priority 3: Any valid CTAs (round-robin among all valid)
  always_comb begin
    any_valid_candidates = cta_valid;
    any_valid_found = (any_valid_candidates != '0);
  end

  // Selection logic with priority encoding
  always_comb begin
    // Declare loop variable outside of for loops to prevent latch inference
    logic [DICE_CTA_ID_WIDTH-1:0] check_idx;
    check_idx = '0;  // Default assignment

    selected_cta_id = '0;
    selection_valid = 1'b0;
    selected_from_branch_resolving = 1'b0;

    if (priority_found == 1'b1) begin
      // Priority 1: Select first matching PC locality
      selection_valid = 1'b1;

      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        check_idx = (DICE_CTA_ID_WIDTH)'((last_dispatched_cta_q + 1 + i) &
                                                   (DICE_NUM_MAX_CTA_PER_CORE - 1));

        if (priority_match[check_idx] == 1'b1) begin
          selected_cta_id = check_idx;
          selection_valid = 1'b1;
          selected_from_branch_resolving = cta_branch_resolving[check_idx];
          break;
        end
      end

    end else if (non_branch_found == 1'b1) begin
      // Priority 2: Round-robin among non-branch resolving CTAs
      selection_valid = 1'b1;
      selected_from_branch_resolving = 1'b0;  // By definition, not branch resolving

      // Start from next CTA after last dispatched
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        check_idx = (DICE_CTA_ID_WIDTH)'((last_dispatched_cta_q + 1 + i) &
                                                   (DICE_NUM_MAX_CTA_PER_CORE - 1));
        if (non_branch_candidates[check_idx] == 1'b1) begin
          selected_cta_id = check_idx;
          break;
        end
      end

    end else if (any_valid_found == 1'b1) begin
      // Priority 3: Round-robin among any valid CTAs (including branch resolving)
      selection_valid = 1'b1;

      // Start from next CTA after last dispatched
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        check_idx = (DICE_CTA_ID_WIDTH)'((last_dispatched_cta_q + 1 + i) &
                                                   (DICE_NUM_MAX_CTA_PER_CORE - 1));
        if (any_valid_candidates[check_idx] == 1'b1) begin
          selected_cta_id = check_idx;
          selected_from_branch_resolving = cta_branch_resolving[check_idx];
          break;
        end
      end
    end
  end

  // Output assignments
  always_comb begin
    scheduled_eblock.valid = enable_i && selection_valid && (eblock_live_q[eblock_ptr_q] == 1'b0);
    scheduled_eblock.data.schedule_hw_cta_id = (DICE_HW_CTA_ID_WIDTH)'(selected_cta_id);
    scheduled_eblock.data.schedule_next_pc = (selected_from_branch_resolving == 1'b1) ?
        cta_status_entries_i[selected_cta_id].predict_pc : cta_next_pc_i[selected_cta_id];
    scheduled_eblock.data.schedule_eblock_id = (EBLOCK_ID_WIDTH)'(eblock_ptr_q);
    scheduled_eblock.data.schedule_active_mask = stack_top_active_mask_i[selected_cta_id];
    scheduled_eblock.data.schedule_prefetch_block = selected_from_branch_resolving;
    scheduled_eblock.data.schedule_cta_id = active_cta_entries_i[selected_cta_id].cta_id;
    scheduled_eblock.data.schedule_grid_size = active_cta_entries_i[selected_cta_id].grid_size;
    scheduled_eblock.data.schedule_cta_size = active_cta_entries_i[selected_cta_id].cta_size;
    scheduled_eblock.data.schedule_kernel_id = active_cta_entries_i[selected_cta_id].kernel_id;
    scheduled_eblock.data.schedule_smem_per_cta = active_cta_entries_i[selected_cta_id]
        .smem_per_cta;
    scheduled_eblock.data.schedule_hw_cta_size = active_cta_entries_i[selected_cta_id].hw_cta_size;
  end


  // Sequential logic for state updates
  always_ff @(posedge clk_i) begin
    if (rst_i == 1'b1) begin
      // Reset all state
      eblock_live_q <= '0;
      eblock_ptr_q <= '0;
      previous_pc_q <= '0;
      pc_history_valid_q <= 1'b0;
      last_dispatched_cta_q <= '1;
    end else begin
      // Handle e-block commit (invalidation)
      if (eblock_commit_valid_i == 1'b1) begin
        eblock_live_q[eblock_commit_id_i] <= 1'b0;
      end

      // Handle successful scheduling
      if ((enable_i == 1'b1) && (scheduled_eblock.valid == 1'b1) &&
          (scheduled_eblock.ready == 1'b1)) begin
        // Allocate current e-block and advance pointer
        eblock_live_q[eblock_ptr_q] <= 1'b1;
        if (eblock_ptr_q == MAX_EBLOCK - 1) begin
          eblock_ptr_q <= '0;
        end else begin
          eblock_ptr_q <= eblock_ptr_q + 1;
        end

        // Update PC history for locality tracking
        previous_pc_q <= (selected_from_branch_resolving == 1'b1)
                            ? cta_status_entries_i[selected_cta_id].predict_pc
                            : cta_next_pc_i[selected_cta_id];

        pc_history_valid_q <= 1'b1;

        // Update last dispatched CTA for round-robin
        last_dispatched_cta_q <= selected_cta_id;
      end
    end
  end

endmodule
