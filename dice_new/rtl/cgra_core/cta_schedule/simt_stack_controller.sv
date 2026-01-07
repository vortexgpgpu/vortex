module simt_stack_controller
  import dice_pkg::*;
  import dice_frontend_pkg::*;
(
    input logic clk_i,
    input logic rst_i,

    // CTA configuration
    input logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0] hw_cta_id_i,
    input logic [1:0] hw_cta_size_i,  // 00=1 stack, 01=2 stacks, 11=4 stacks

    // Update request interface (valid/ready handshake) - BRANCH HANDLER
    input logic update_valid_i,
    input logic update_with_divergence_i,  // 0 = no divergence, 1 = with divergence
    input logic [DICE_ADDR_WIDTH-1:0] update_next_pc_i,  // No divergence: next PC

    // Divergence case inputs (only used when update_with_divergence = 1)
    input thread_mask_t predicate_regs_value_i,
    input logic [DICE_ADDR_WIDTH-1:0] branch_not_taken_pc_i,
    input logic [DICE_ADDR_WIDTH-1:0] branch_reconvergence_pc_i,

    output logic update_ready_o,

    // Initialization interface (higher priority) - CTA CONTROLLER
    input logic init_valid_i,
    input logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0] init_hw_cta_id_i,
    input logic [1:0] init_hw_cta_size_i,
    input logic [DICE_ADDR_WIDTH-1:0] init_pc_i,
    input logic [DICE_ADDR_WIDTH-1:0] init_reconvergence_pc_i,
    output logic init_ready_o,

    // Stack top outputs (when not busy) - combined from active stacks
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_top_valid_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_ADDR_WIDTH-1:0] stack_top_next_pc_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_ADDR_WIDTH-1:0] stack_top_reconvergence_pc_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0][DICE_NUM_MAX_THREADS_PER_CORE/DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_top_active_mask_o,

    // Stack status - individual stack status
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_empty_o,
    output logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_full_o
);

  // -------------------------------------------------------------------------
  // Local Parameters (derived from packages)
  // -------------------------------------------------------------------------
  localparam int ThreadWidth = DICE_NUM_MAX_THREADS_PER_CORE / DICE_NUM_MAX_CTA_PER_CORE;
  localparam int MetadataLengthWidth = BITSTREAM_LENGTH_WIDTH;  // From dice_frontend_pkg

  logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE)-1:0] hw_cta_id_q;
  logic [1:0] hw_cta_size_q;

  // Calculate number of active stacks based on CTA size
  // hw_cta_size represents additional stacks beyond the base stack
  logic [2:0] num_active_stacks;
  always_comb begin
    case (hw_cta_size_q)
      2'b00:   num_active_stacks = 3'd1;  // hw_cta_id only (1 stack)
      2'b01:   num_active_stacks = 3'd2;  // hw_cta_id + hw_cta_id+1 (2 stacks)
      2'b11:   num_active_stacks = 3'd4;  // hw_cta_id through hw_cta_id+3 (4 stacks)
      default: num_active_stacks = 3'd1;
    endcase
  end

  // Calculate effective thread width based on CTA size
  logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE*ThreadWidth+1)-1:0] effective_thread_width;
  always_comb begin
    case (hw_cta_size_q)
      2'b00:   effective_thread_width = ThreadWidth;  // 256 threads
      2'b01:   effective_thread_width = 2 * ThreadWidth;  // 512 threads
      2'b11:   effective_thread_width = 4 * ThreadWidth;  // 1024 threads
      default: effective_thread_width = ThreadWidth;
    endcase
  end

  // FSM states
  typedef enum logic [2:0] {
    StateIdle,
    StateReadTop,
    StateModifyTop,
    StatePushFirst,
    StatePushSecond,
    StatePopStack,
    StateInitPush,
    StateFinalRead
  } state_e;

  state_e current_state_q, next_state;

  // Internal registers for multi-cycle operations
  logic update_with_divergence_q;
  logic [DICE_ADDR_WIDTH-1:0] update_next_pc_q;
  thread_mask_t predicate_regs_value_q;
  logic [DICE_ADDR_WIDTH-1:0] branch_not_taken_pc_q;
  logic [DICE_ADDR_WIDTH-1:0] branch_reconvergence_pc_q;

  // Registers for init operation
  logic [DICE_ADDR_WIDTH-1:0] init_pc_q;
  logic [DICE_ADDR_WIDTH-1:0] init_reconvergence_pc_q;

  // Per-stack signals
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_push;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_modify_top;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_pop;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_read_top;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_out_valid;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_empty_individual;
  logic [DICE_NUM_MAX_CTA_PER_CORE-1:0] stack_full_individual;

  logic [DICE_ADDR_WIDTH-1:0] stack_push_next_pc[DICE_NUM_MAX_CTA_PER_CORE];
  logic [DICE_ADDR_WIDTH-1:0] stack_push_reconvergence_pc[DICE_NUM_MAX_CTA_PER_CORE];
  logic [ThreadWidth-1:0] stack_push_active_mask[DICE_NUM_MAX_CTA_PER_CORE];
  logic [DICE_ADDR_WIDTH-1:0] stack_top_next_pc_int[DICE_NUM_MAX_CTA_PER_CORE];
  logic [DICE_ADDR_WIDTH-1:0] stack_top_reconvergence_pc_int[DICE_NUM_MAX_CTA_PER_CORE];
  logic [ThreadWidth-1:0] stack_top_active_mask_int[DICE_NUM_MAX_CTA_PER_CORE];

  // Combined stack signals
  logic combined_stack_out_valid;
  logic [DICE_ADDR_WIDTH-1:0] combined_stack_top_next_pc;
  logic [DICE_ADDR_WIDTH-1:0] combined_stack_top_reconvergence_pc;
  thread_mask_t combined_stack_top_active_mask;

  // Computed values for divergence analysis
  thread_mask_t taken_active_mask;
  thread_mask_t not_taken_active_mask;
  thread_mask_t effective_active_mask;
  logic all_taken, all_not_taken, has_divergence;

  // Operation control signals - combinational _next signals
  logic need_pop_next, need_modify_top_next, need_push_first_next, need_push_second_next;

  // Operation control signals - registered
  logic need_pop_q, need_modify_top_q, need_push_first_q, need_push_second_q;
  logic [DICE_ADDR_WIDTH-1:0] new_top_pc_q;
  logic [DICE_ADDR_WIDTH-1:0] new_top_reconvergence_pc_q;
  thread_mask_t new_top_active_mask_q;
  logic [DICE_ADDR_WIDTH-1:0] push_pc_1_q, push_pc_2_q;
  logic [DICE_ADDR_WIDTH-1:0] push_reconvergence_pc_1_q, push_reconvergence_pc_2_q;
  thread_mask_t push_active_mask_1_q, push_active_mask_2_q;

  // Generate individual SIMT stacks
  genvar i;
  generate
    for (i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin : gen_stacks
      simt_stack stack_inst (
          .clk_i                  (clk_i),
          .rst_i                  (rst_i),
          .push_i                 (stack_push[i]),
          .modify_top_i           (stack_modify_top[i]),
          .push_next_pc_i         (stack_push_next_pc[i]),
          .push_reconvergence_pc_i(stack_push_reconvergence_pc[i]),
          .push_active_mask_i     (stack_push_active_mask[i]),
          .pop_i                  (stack_pop[i]),
          .read_top_i             (stack_read_top[i]),
          .top_next_pc_o          (stack_top_next_pc_int[i]),
          .top_reconvergence_pc_o (stack_top_reconvergence_pc_int[i]),
          .top_active_mask_o      (stack_top_active_mask_int[i]),
          .out_valid_o            (stack_out_valid[i]),
          .stack_empty_o          (stack_empty_individual[i]),
          .stack_full_o           (stack_full_individual[i])
      );
    end
  endgenerate

  // Combine stack outputs based on CTA configuration
  always_comb begin
    logic all_active_valid;
    int   mask_offset;
    mask_offset = 0;
    combined_stack_out_valid = 1'b0;
    combined_stack_top_next_pc = '0;
    combined_stack_top_reconvergence_pc = '0;
    combined_stack_top_active_mask = '0;

    // Check if all active stacks have valid output
    all_active_valid = 1'b1;
    for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
      if (j >= hw_cta_id_q && j < (hw_cta_id_q + num_active_stacks)) begin
        all_active_valid &= stack_out_valid[j];
      end
    end

    if (all_active_valid == 1'b1) begin
      combined_stack_out_valid = 1'b1;
      // Use the PC from the first active stack (they should all be the same for valid operations)
      combined_stack_top_next_pc = stack_top_next_pc_int[hw_cta_id_q];
      combined_stack_top_reconvergence_pc = stack_top_reconvergence_pc_int[hw_cta_id_q];

      // Combine active masks from all active stacks
      for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
        if (j >= hw_cta_id_q && j < (hw_cta_id_q + num_active_stacks)) begin
          mask_offset = (j - hw_cta_id_q) * ThreadWidth;
          combined_stack_top_active_mask[mask_offset+:ThreadWidth] = stack_top_active_mask_int[j];
        end else begin
          mask_offset = j * ThreadWidth;
          combined_stack_top_active_mask[mask_offset+:ThreadWidth] = '0;
        end
      end
    end
  end

  // Output individual stack status directly
  assign stack_empty_o = stack_empty_individual;
  assign stack_full_o  = stack_full_individual;
  // Extract effective active mask based on CTA size
  always_comb begin
    case (hw_cta_size_q)
      2'b00:   effective_active_mask = combined_stack_top_active_mask[ThreadWidth-1:0];
      2'b01:   effective_active_mask = combined_stack_top_active_mask[2*ThreadWidth-1:0];
      2'b11:   effective_active_mask = combined_stack_top_active_mask;
      default: effective_active_mask = combined_stack_top_active_mask[ThreadWidth-1:0];
    endcase
  end

  // FSM next state logic - uses combinational _next signals for immediate decisions
  always_comb begin
    next_state = current_state_q;

    case (current_state_q)
      StateIdle: begin
        if (init_valid_i == 1'b1) begin
          next_state = StateInitPush;
        end else if (update_valid_i == 1'b1) begin
          next_state = StateReadTop;
        end
      end

      StateReadTop: begin
        if (combined_stack_out_valid == 1'b1) begin
          if (need_pop_next == 1'b1) begin
            next_state = StatePopStack;
          end else if (need_modify_top_next == 1'b1) begin
            next_state = StateModifyTop;
          end else if (need_push_first_next == 1'b1) begin
            next_state = StatePushFirst;
          end else begin
            next_state = StateIdle;  // No operation needed
          end
        end
      end

      StateModifyTop: begin
        if (need_push_first_q == 1'b1) begin
          next_state = StatePushFirst;
        end else begin
          next_state = StateFinalRead;
        end
      end

      StatePushFirst: begin
        if (need_push_second_q == 1'b1) begin
          next_state = StatePushSecond;
        end else begin
          next_state = StateFinalRead;
        end
      end

      StatePushSecond: begin
        next_state = StateFinalRead;
      end

      StatePopStack: begin
        next_state = StateFinalRead;
      end

      StateInitPush: begin
        next_state = StateFinalRead;
      end

      StateFinalRead: begin
        if ((combined_stack_out_valid == 1'b1) || (stack_empty_o != '0)) begin
          next_state = StateIdle;
        end
      end

      default: begin
        next_state = StateIdle;
      end
    endcase
  end

  // Divergence analysis logic - operates on effective thread width
  always_comb begin
    thread_mask_t effective_predicate;

    // Extract effective predicate based on CTA size
    case (hw_cta_size_q)
      2'b00:
      effective_predicate = {
        {(DICE_NUM_MAX_CTA_PER_CORE - 1) * ThreadWidth{1'b0}},
        predicate_regs_value_q[ThreadWidth-1:0]
      };
      2'b01:
      effective_predicate = {
        {(DICE_NUM_MAX_CTA_PER_CORE - 2) * ThreadWidth{1'b0}},
        predicate_regs_value_q[2*ThreadWidth-1:0]
      };
      2'b11: effective_predicate = predicate_regs_value_q;
      default:
      effective_predicate = {
        {(DICE_NUM_MAX_CTA_PER_CORE - 1) * ThreadWidth{1'b0}},
        predicate_regs_value_q[ThreadWidth-1:0]
      };
    endcase

    taken_active_mask = effective_active_mask & effective_predicate;
    not_taken_active_mask = effective_active_mask & ~effective_predicate;
    all_taken = (taken_active_mask == effective_active_mask) && (effective_active_mask != '0);
    all_not_taken = (not_taken_active_mask == effective_active_mask) &&
                    (effective_active_mask != '0);
    has_divergence = (all_taken == 1'b0) && (all_not_taken == 1'b0) &&
                     (effective_active_mask != '0);
  end

  // Operation decision logic - combinational for immediate state decisions using _next signals
  always_comb begin
    // Default values
    need_pop_next = 1'b0;
    need_modify_top_next = 1'b0;
    need_push_first_next = 1'b0;
    need_push_second_next = 1'b0;

    if ((current_state_q == StateReadTop) && (combined_stack_out_valid == 1'b1)) begin
      if (update_with_divergence_q == 1'b0) begin
        // Update top without control divergence
        if (update_next_pc_q == combined_stack_top_reconvergence_pc) begin
          need_pop_next = 1'b1;
        end else begin
          need_modify_top_next = 1'b1;
        end

      end else begin
        // Update top with control divergence
        if (all_taken == 1'b1) begin
          // All threads take the branch
          if (update_next_pc_q == combined_stack_top_reconvergence_pc) begin
            need_pop_next = 1'b1;
          end else begin
            need_modify_top_next = 1'b1;
          end

        end else if (all_not_taken == 1'b1) begin
          // All threads don't take the branch
          if (branch_not_taken_pc_q == combined_stack_top_reconvergence_pc) begin
            need_pop_next = 1'b1;
          end else begin
            need_modify_top_next = 1'b1;
          end

        end else if (has_divergence == 1'b1) begin
          // Real divergence cases
          if ((update_next_pc_q != branch_reconvergence_pc_q) &&
              (branch_not_taken_pc_q != branch_reconvergence_pc_q) &&
              (branch_reconvergence_pc_q != combined_stack_top_reconvergence_pc)) begin
            // Case 2: Push two new entries
            need_modify_top_next  = 1'b1;
            need_push_first_next  = 1'b1;
            need_push_second_next = 1'b1;

          end else if ((update_next_pc_q == branch_reconvergence_pc_q) &&
                       (branch_reconvergence_pc_q != combined_stack_top_reconvergence_pc)) begin
            // Case 3: Push one entry for not taken
            need_modify_top_next = 1'b1;
            need_push_first_next = 1'b1;

          end else if ((branch_not_taken_pc_q == branch_reconvergence_pc_q) &&
                       (branch_reconvergence_pc_q == combined_stack_top_reconvergence_pc)) begin
            // Case 4: Update top to taken branch
            need_modify_top_next = 1'b1;

          end else if ((update_next_pc_q == branch_reconvergence_pc_q) &&
                       (branch_reconvergence_pc_q == combined_stack_top_reconvergence_pc)) begin
            // Case 1: Update top to not taken branch
            need_modify_top_next = 1'b1;
          end
        end
      end
    end
  end

  // Distribute signals to individual stacks based on active configuration
  always_comb begin
    int mask_offset;
    mask_offset = 0;
    // Initialize all stacks to inactive
    for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
      stack_push[j] = 1'b0;
      stack_modify_top[j] = 1'b0;
      stack_pop[j] = 1'b0;
      stack_read_top[j] = 1'b0;
      stack_push_next_pc[j] = '0;
      stack_push_reconvergence_pc[j] = '0;
      stack_push_active_mask[j] = '0;
    end

    // Always read from all stacks to keep outputs valid
    for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
      stack_read_top[j] = 1'b1;
    end

    // Activate only the stacks in the current CTA for operations
    for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
      if (j >= hw_cta_id_q && j < (hw_cta_id_q + num_active_stacks)) begin
        case (current_state_q)
          StateModifyTop: begin
            stack_push[j] = 1'b1;
            stack_modify_top[j] = 1'b1;
            stack_push_next_pc[j] = new_top_pc_q;
            stack_push_reconvergence_pc[j] = new_top_reconvergence_pc_q;
            // Distribute active mask across stacks
            mask_offset = (j - hw_cta_id_q) * ThreadWidth;
            stack_push_active_mask[j] = new_top_active_mask_q[mask_offset+:ThreadWidth];
          end

          StatePushFirst: begin
            stack_push[j] = 1'b1;
            stack_push_next_pc[j] = push_pc_1_q;
            stack_push_reconvergence_pc[j] = push_reconvergence_pc_1_q;
            // Distribute active mask across stacks
            mask_offset = (j - hw_cta_id_q) * ThreadWidth;
            stack_push_active_mask[j] = push_active_mask_1_q[mask_offset+:ThreadWidth];
          end

          StatePushSecond: begin
            stack_push[j] = 1'b1;
            stack_push_next_pc[j] = push_pc_2_q;
            stack_push_reconvergence_pc[j] = push_reconvergence_pc_2_q;
            // Distribute active mask across stacks
            mask_offset = (j - hw_cta_id_q) * ThreadWidth;
            stack_push_active_mask[j] = push_active_mask_2_q[mask_offset+:ThreadWidth];
          end

          StatePopStack: begin
            stack_pop[j] = 1'b1;
          end

          StateInitPush: begin
            stack_push[j] = 1'b1;
            stack_modify_top[j] = 1'b0;
            stack_push_next_pc[j] = init_pc_q;
            stack_push_reconvergence_pc[j] = init_reconvergence_pc_q;
            stack_push_active_mask[j] = '1;  // All threads active
          end
          default: ;
        endcase
      end
    end
  end

  // Output assignments - convert unpacked arrays to packed arrays
  assign update_ready_o = (current_state_q == StateIdle) && (init_valid_i == 1'b0);
  assign init_ready_o   = (current_state_q == StateIdle);

  // Convert unpacked arrays to packed arrays for outputs
  // stack_top_valid is always available when stack has data (not dependent on state)
  always_comb begin
    for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
      stack_top_valid_o[j] = stack_out_valid[j] && (stack_empty_individual[j] == 1'b0);
      stack_top_next_pc_o[j] = stack_top_next_pc_int[j];
      stack_top_reconvergence_pc_o[j] = stack_top_reconvergence_pc_int[j];
      stack_top_active_mask_o[j] = stack_top_active_mask_int[j];
    end
  end

  // Sequential logic
  always_ff @(posedge clk_i) begin
    if (rst_i == 1'b1) begin
      current_state_q <= StateIdle;
      update_with_divergence_q <= 1'b0;
      update_next_pc_q <= '0;
      predicate_regs_value_q <= '0;
      branch_not_taken_pc_q <= '0;
      branch_reconvergence_pc_q <= '0;
      hw_cta_id_q <= '0;
      hw_cta_size_q <= '0;
      init_pc_q <= '0;
      init_reconvergence_pc_q <= '0;

      // Reset operation control registers
      need_pop_q <= 1'b0;
      need_modify_top_q <= 1'b0;
      need_push_first_q <= 1'b0;
      need_push_second_q <= 1'b0;
      new_top_pc_q <= '0;
      new_top_reconvergence_pc_q <= '0;
      new_top_active_mask_q <= '0;
      push_pc_1_q <= '0;
      push_pc_2_q <= '0;
      push_reconvergence_pc_1_q <= '0;
      push_reconvergence_pc_2_q <= '0;
      push_active_mask_1_q <= '0;
      push_active_mask_2_q <= '0;

    end else begin
      current_state_q <= next_state;

      // Capture inputs on valid request, init has higher priority
      if ((current_state_q == StateIdle) && (init_valid_i == 1'b1)) begin
        init_pc_q <= init_pc_i;
        init_reconvergence_pc_q <= init_reconvergence_pc_i;
        hw_cta_id_q <= init_hw_cta_id_i;
        hw_cta_size_q <= init_hw_cta_size_i;
      end else if ((current_state_q == StateIdle) && (update_valid_i == 1'b1)) begin
        update_with_divergence_q <= update_with_divergence_i;
        update_next_pc_q <= update_next_pc_i;
        predicate_regs_value_q <= predicate_regs_value_i;
        branch_not_taken_pc_q <= branch_not_taken_pc_i;
        branch_reconvergence_pc_q <= branch_reconvergence_pc_i;
        hw_cta_id_q <= hw_cta_id_i;
        hw_cta_size_q <= hw_cta_size_i;
      end

      // Register operation control values using _next signals
      if ((current_state_q == StateReadTop) && (combined_stack_out_valid == 1'b1)) begin
        // Simply assign _next to _q
        need_pop_q <= need_pop_next;
        need_modify_top_q <= need_modify_top_next;
        need_push_first_q <= need_push_first_next;
        need_push_second_q <= need_push_second_next;

        // Store operation parameters based on what operations are needed
        if (update_with_divergence_q == 1'b0) begin
          // No divergence case
          if (need_modify_top_next == 1'b1) begin
            new_top_pc_q <= update_next_pc_q;
            new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;
            new_top_active_mask_q <= effective_active_mask;
          end

        end else begin
          // With divergence case
          if (all_taken == 1'b1) begin
            if (need_modify_top_next == 1'b1) begin
              new_top_pc_q <= update_next_pc_q;
              new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;
              new_top_active_mask_q <= effective_active_mask;
            end

          end else if (all_not_taken == 1'b1) begin
            if (need_modify_top_next == 1'b1) begin
              new_top_pc_q <= branch_not_taken_pc_q;
              new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;
              new_top_active_mask_q <= effective_active_mask;
            end

          end else if (has_divergence == 1'b1) begin
            if ((update_next_pc_q != branch_reconvergence_pc_q) &&
                            (branch_not_taken_pc_q != branch_reconvergence_pc_q) &&
                            (branch_reconvergence_pc_q !=
                             combined_stack_top_reconvergence_pc)) begin
              // Case 2: Push two new entries
              new_top_pc_q <= branch_reconvergence_pc_q;
              new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;
              new_top_active_mask_q <= effective_active_mask;
              push_pc_1_q <= update_next_pc_q;  // taken target
              push_reconvergence_pc_1_q <= branch_reconvergence_pc_q;
              push_active_mask_1_q <= taken_active_mask;
              push_pc_2_q <= branch_not_taken_pc_q;
              push_reconvergence_pc_2_q <= branch_reconvergence_pc_q;
              push_active_mask_2_q <= not_taken_active_mask;

            end else if ((update_next_pc_q == branch_reconvergence_pc_q) &&
                             (branch_reconvergence_pc_q !=
                              combined_stack_top_reconvergence_pc)) begin
              // Case 3: Push one entry for not taken
              new_top_pc_q <= branch_reconvergence_pc_q;
              new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;
              new_top_active_mask_q <= effective_active_mask;
              push_pc_1_q <= branch_not_taken_pc_q;
              push_reconvergence_pc_1_q <= branch_reconvergence_pc_q;
              push_active_mask_1_q <= not_taken_active_mask;

            end else if ((branch_not_taken_pc_q == branch_reconvergence_pc_q) &&
                                   (branch_reconvergence_pc_q == combined_stack_top_reconvergence_pc)) begin
              // Case 4: Update top to taken branch
              new_top_pc_q <= update_next_pc_q;  // taken target
              new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;

            end else if ((update_next_pc_q == branch_reconvergence_pc_q) &&
                                   (branch_reconvergence_pc_q == combined_stack_top_reconvergence_pc)) begin
              // Case 1: Update top to not taken branch
              new_top_pc_q <= branch_not_taken_pc_q;
              new_top_reconvergence_pc_q <= combined_stack_top_reconvergence_pc;
            end
          end
        end
      end
    end
  end

  // Debug assertions
`ifndef SYNTHESIS
  always @(posedge clk_i) begin
    if (rst_i == 1'b0) begin
      if ((update_valid_i == 1'b1) && (update_ready_o == 1'b1) &&
          (hw_cta_id_i + hw_cta_size_i >= DICE_NUM_MAX_CTA_PER_CORE)) begin
        $error(
            "SIMT Stack Controller: CTA configuration exceeds available stacks " + "(hw_cta_id=%0d, hw_cta_size=%0d, max=%0d)",
            hw_cta_id_i, hw_cta_size_i, DICE_NUM_MAX_CTA_PER_CORE);
      end

      if ((init_valid_i == 1'b1) && (init_ready_o == 1'b1) &&
          (init_hw_cta_id_i + init_hw_cta_size_i >= DICE_NUM_MAX_CTA_PER_CORE)) begin
        $error(
            "SIMT Stack Controller: Init CTA configuration exceeds available stacks " + "(init_hw_cta_id=%0d, init_hw_cta_size=%0d, max=%0d)",
            init_hw_cta_id_i, init_hw_cta_size_i, DICE_NUM_MAX_CTA_PER_CORE);
      end

      // Debug state transitions and operations
      if (current_state_q != next_state) begin
        $display("SIMT Controller: State %0s -> %0s", current_state_q.name(), next_state.name());
      end

      // Debug operation decisions in StateReadTop
      if ((current_state_q == StateReadTop) && (combined_stack_out_valid == 1'b1)) begin
        $display("SIMT Controller: StateReadTop analysis - pop=%b, modify=%b, push1=%b, push2=%b",
                 need_pop_next, need_modify_top_next, need_push_first_next, need_push_second_next);
        if (need_modify_top_next == 1'b1) begin
          $display(
              "SIMT Controller: Will use new_top_pc=0x%h in next cycle",
              (update_with_divergence_q == 1'b0) ? update_next_pc_q : (all_taken == 1'b1) ? update_next_pc_q : (all_not_taken == 1'b1) ? branch_not_taken_pc_q : branch_reconvergence_pc_q);
        end
      end
    end
  end
`endif

endmodule
