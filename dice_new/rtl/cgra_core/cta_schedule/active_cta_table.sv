module active_cta_table
  import dice_pkg::*;
  import dice_frontend_pkg::*;
(
    input logic clk_i,
    input logic rst_i,

    // Add new entry interface (table is slave)
    output logic                                             add_ready_o,
    input  logic                                             add_valid_i,
    input  dice_cta_desc_t                         add_cta_info_i,
    input  logic           [DICE_TID_WIDTH-1:0]    add_cta_size_i,

    // Pop interface
    input  logic                                             pop_valid_i,
    input  logic [DICE_HW_CTA_ID_WIDTH-1:0]        pop_hw_cta_id_i,
    output logic                                             pop_ready_o,

    // Output popped CTA interface (table is master)
    output logic                                             out_valid_o,
    input  logic                                             out_ready_i,
    output dice_cta_id_t                           out_cta_id_o,
    output logic         [DICE_TID_WIDTH-1:0]      out_cta_size_o,
    output logic         [DICE_KERNEL_ID_WIDTH-1:0] out_kernel_id_o,

    // Status outputs
    output active_cta_t [DICE_NUM_MAX_CTA_PER_CORE-1:0]
        active_cta_entries_o,

    // Output flags
    output logic                                             full_o,
    output logic [DICE_HW_CTA_ID_WIDTH-1:0]        next_empty_cta_index_o
);

  // -------------------------------------------------------------------------
  // Local Parameters (derived from packages)
  // -------------------------------------------------------------------------
  localparam int ThreadWidth = DICE_NUM_MAX_THREADS_PER_CORE / DICE_NUM_MAX_CTA_PER_CORE;

  // Calculate number of entries needed for a CTA
  // Optimized for power-of-2 ThreadWidth using bit shifts
  function automatic logic [DICE_HW_CTA_ID_WIDTH:0] calc_entries_needed(
      input logic [DICE_TID_WIDTH-1:0] cta_size);
    // For power-of-2 ThreadWidth, we can use bit shifts
    // entries_needed = ceil(cta_size / ThreadWidth) = (cta_size + ThreadWidth - 1) >> log2(ThreadWidth)
    logic [DICE_TID_WIDTH:0] adjusted_size;
    adjusted_size = (DICE_TID_WIDTH + 1)'(cta_size) + ThreadWidth - 1;
    return (DICE_HW_CTA_ID_WIDTH + 1)'(adjusted_size >>
        (DICE_TID_WIDTH'($clog2(ThreadWidth))));
  endfunction



  // CTA table entry structure
  typedef struct packed {
    logic is_primary;  // True for the first entry of a multi-entry CTA
    logic [$clog2(DICE_NUM_MAX_CTA_PER_CORE):0]
        entries_used;  // Number of entries used by this CTA
    active_cta_t entry_info;
  } cta_entry_t;


  // CTA table storage
  cta_entry_t cta_table_q[DICE_NUM_MAX_CTA_PER_CORE];

  // Output buffer for popped entries (flip-flops)
  logic                                        output_buffer_valid_q;
  dice_cta_id_t                      output_buffer_cta_id_q;
  logic [DICE_TID_WIDTH-1:0]         output_buffer_cta_size_q;
  logic [DICE_KERNEL_ID_WIDTH-1:0]   output_buffer_kernel_id_q;

  // Internal combinational signals
  logic [DICE_HW_CTA_ID_WIDTH-1:0] empty_index;
  logic found_empty;
  logic [DICE_HW_CTA_ID_WIDTH:0] entries_needed;
  logic [DICE_HW_CTA_ID_WIDTH:0] entries_to_clear;

  // Calculate entries needed for incoming CTA
  assign entries_needed = calc_entries_needed(add_cta_size_i);
  assign entries_to_clear = cta_table_q[pop_hw_cta_id_i].entries_used;

  // Find next empty entry - Contiguous Block Search
  always_comb begin
    found_empty = 1'b0;
    empty_index = '0;

    // Search for a contiguous block of 'entries_needed' slots
    for (int i = 0; i <= DICE_NUM_MAX_CTA_PER_CORE - 1; i++) begin
      logic block_valid;
      block_valid = 1'b1;

      // Check if the block fits within the table bounds
      if ((i + 32'(entries_needed)) <= DICE_NUM_MAX_CTA_PER_CORE) begin
        // Check if all slots in the block are empty
        for (int k = 0; k < DICE_NUM_MAX_CTA_PER_CORE; k++) begin
           if (k >= i && k < (i + 32'(entries_needed))) begin
              if (cta_table_q[k].entry_info.cta_valid == 1'b1) begin
                  block_valid = 1'b0;
              end
           end
        end

        if ((block_valid == 1'b1) && (found_empty == 1'b0)) begin
          empty_index = (DICE_HW_CTA_ID_WIDTH)'(i);
          found_empty = 1'b1;
        end
      end
    end
  end

  // Output assignments
  assign full_o = (found_empty == 1'b0);
  assign next_empty_cta_index_o = empty_index;
  assign add_ready_o = found_empty;

  // Output interface
  assign out_valid_o = output_buffer_valid_q;
  assign out_cta_id_o = output_buffer_cta_id_q;
  assign out_cta_size_o = output_buffer_cta_size_q;
  assign out_kernel_id_o = output_buffer_kernel_id_q;

  logic pop_this_cycle;
  logic output_consumed_this_cycle;

  // Pop ready when buffer is empty or being consumed this cycle
  assign pop_ready_o = (output_buffer_valid_q == 1'b0) || (output_consumed_this_cycle == 1'b1);

  assign pop_this_cycle = (pop_valid_i == 1'b1) && (pop_ready_o == 1'b1) &&
                          (cta_table_q[pop_hw_cta_id_i].entry_info.cta_valid == 1'b1);
  assign output_consumed_this_cycle = (out_valid_o == 1'b1) && (out_ready_i == 1'b1);

  // CTA valid outputs and status information - only from primary entries
  always_comb begin
    for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
      if ((cta_table_q[i].entry_info.cta_valid == 1'b1) &&
          (cta_table_q[i].is_primary == 1'b1)) begin
        active_cta_entries_o[i] = cta_table_q[i].entry_info;
      end else begin
        active_cta_entries_o[i] = '0;
      end
    end
  end


  // Main table logic
  always_ff @(posedge clk_i) begin
    if (rst_i == 1'b1) begin
      // Reset all entries
      for (int i = 0; i < DICE_NUM_MAX_CTA_PER_CORE; i++) begin
        cta_table_q[i] <= '0;
      end
      // Reset output buffer
      output_buffer_valid_q <= 1'b0;
      output_buffer_cta_id_q <= '0;
      output_buffer_cta_size_q <= '0;
      output_buffer_kernel_id_q <= '0;

    end else begin
      // Compute entries_to_clear for pop operations
      // entries_to_clear = cta_table_q[pop_hw_cta_id_i].entries_used; // Moved to continuous assignment

      if ((pop_this_cycle == 1'b1) && (output_consumed_this_cycle == 1'b1)) begin
        // Pop and output in same cycle - directly replace buffer contents
        output_buffer_valid_q <= 1'b1;
        output_buffer_cta_id_q <= cta_table_q[pop_hw_cta_id_i].entry_info.cta_id;
        output_buffer_cta_size_q <= (DICE_TID_WIDTH)'(cta_table_q[pop_hw_cta_id_i].entry_info.hw_cta_size);
        output_buffer_kernel_id_q <= cta_table_q[pop_hw_cta_id_i].entry_info.kernel_id;

        // Clear all entries used by this CTA
        for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
          if (j >= 32'(pop_hw_cta_id_i) &&
              j < (32'(pop_hw_cta_id_i) + 32'(entries_to_clear))) begin
            cta_table_q[j] <= '0;
          end
        end

      end else if ((pop_this_cycle == 1'b1) && (output_buffer_valid_q == 1'b0)) begin
        // Pop when buffer is empty - store in buffer
        output_buffer_valid_q <= 1'b1;
        output_buffer_cta_id_q <= cta_table_q[pop_hw_cta_id_i].entry_info.cta_id;
        output_buffer_cta_size_q <= (DICE_TID_WIDTH)'(cta_table_q[pop_hw_cta_id_i].entry_info.hw_cta_size);
        output_buffer_kernel_id_q <= cta_table_q[pop_hw_cta_id_i].entry_info.kernel_id;

        // Clear all entries used by this CTA
        for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
          if (j >= 32'(pop_hw_cta_id_i) &&
              j < (32'(pop_hw_cta_id_i) + 32'(entries_to_clear))) begin
            cta_table_q[j] <= '0;
          end
        end

      end else if (output_consumed_this_cycle == 1'b1) begin
        // Only output buffer consumed - clear buffer
        output_buffer_valid_q <= 1'b0;
        output_buffer_cta_id_q <= '0;
        output_buffer_cta_size_q <= '0;
        output_buffer_kernel_id_q <= '0;
      end
      // If pop_this_cycle && output_buffer_valid && !output_consumed_this_cycle
      // then we can't pop because buffer is full - pop is ignored

      // Handle add operation
      if ((add_valid_i == 1'b1) && (add_ready_o == 1'b1)) begin
        // Allocate consecutive entries for this CTA
        for (int j = 0; j < DICE_NUM_MAX_CTA_PER_CORE; j++) begin
          if (j >= 32'(empty_index) && j < (32'(empty_index) + 32'(entries_needed))) begin
            if (j == 32'(empty_index)) begin
              cta_table_q[j].entry_info.cta_id <= add_cta_info_i.cta_id;
              cta_table_q[j].entry_info.grid_size <= add_cta_info_i.kernel_desc.grid_size;
              cta_table_q[j].entry_info.cta_size <= add_cta_info_i.kernel_desc.cta_size;
              cta_table_q[j].entry_info.kernel_id <= add_cta_info_i.kernel_desc.kernel_id;
              cta_table_q[j].entry_info.smem_per_cta <= add_cta_info_i.kernel_desc.smem_per_cta;
              cta_table_q[j].entry_info.hw_cta_size <=
                  (DICE_HW_CTA_SIZE_WIDTH)'(add_cta_size_i);
            end else begin
              cta_table_q[j] <= '0;
            end
            cta_table_q[j].entry_info.cta_valid <= 1'b1;
            cta_table_q[j].is_primary <= (j == 32'(empty_index));
            cta_table_q[j].entries_used <= entries_needed;
          end
        end
      end
    end
  end



  `ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (rst_i == 1'b0) begin
      if ((add_valid_i == 1'b1) && (add_ready_o == 1'b1)) begin
        assert ((32'(empty_index) + 32'(entries_needed)) <= DICE_NUM_MAX_CTA_PER_CORE)
        else $error("ContiguousAllocation: Allocated block exceeds table bounds");
      end

      if (pop_valid_i == 1'b1) begin
        assert (cta_table_q[pop_hw_cta_id_i].entry_info.cta_valid == 1'b1)
        else $error("PopValidEntry: Popping invalid entry");
      end

      if (out_valid_o == 1'b1) begin
        assert (!$isunknown(out_cta_id_o))
        else $error("OutputKnown: Output ID contains X");
      end
    end
  end
  `endif

endmodule
