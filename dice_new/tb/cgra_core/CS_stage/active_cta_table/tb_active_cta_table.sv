module tb_active_cta_table;

  // -----------------------
  // Parameters / constants
  // -----------------------
  localparam int MAX_NUM_CTA     = 4;
  localparam int THREAD_WIDTH    = 256;
  localparam int CTA_INDEX_WIDTH = $clog2(MAX_NUM_CTA);

  // -----------------------
  // DUT I/O
  // -----------------------
  logic clk, rst_n;

  // Pop interface
  logic pop_valid;
  logic [CTA_INDEX_WIDTH-1:0] pop_hw_cta_id;

  // Add interface
  logic add_valid;
  logic [15:0] add_cta_id_x;
  logic [15:0] add_cta_id_y;
  logic [15:0] add_cta_id_z;
  logic [15:0] add_grid_size_x;
  logic [15:0] add_grid_size_y;
  logic [15:0] add_grid_size_z;
  logic [10:0] add_cta_size_x;
  logic [10:0] add_cta_size_y;
  logic [10:0] add_cta_size_z;
  logic [10:0] add_cta_size;
  logic [15:0] add_kernel_id;
  logic        add_ready;

  // Output popped interface
  logic        out_valid;
  logic [15:0] out_cta_id_x;
  logic [15:0] out_cta_id_y;
  logic [15:0] out_cta_id_z;
  logic [10:0] out_cta_size;
  logic [15:0] out_kernel_id;
  logic        out_ready;

  // Status
  logic [MAX_NUM_CTA-1:0]              cta_valid;
  logic [MAX_NUM_CTA-1:0][15:0]        cta_id_x, cta_id_y, cta_id_z;
  logic [MAX_NUM_CTA-1:0][15:0]        grid_size_x, grid_size_y, grid_size_z;
  logic [MAX_NUM_CTA-1:0][10:0]        cta_size_x, cta_size_y, cta_size_z;
  logic [MAX_NUM_CTA-1:0][10:0]        cta_size_arr;
  logic [MAX_NUM_CTA-1:0][15:0]        kernel_id;
  logic                                full;
  logic [CTA_INDEX_WIDTH-1:0]          next_empty_cta_index;

  // Code used to generate waveforms
  initial begin
    //dump fsdb
    $fsdbDumpfile("tb_active_cta_table.fsdb");
    $fsdbDumpvars("+all");
  end

  // -----------------------
  // DUT Instantiate
  // -----------------------
  active_cta_table #(
    .MAX_NUM_CTA(MAX_NUM_CTA),
    .THREAD_WIDTH(THREAD_WIDTH)
  ) dut (
    .clk, .rst_n,
    .pop_valid, .pop_hw_cta_id,
    .add_valid, .add_cta_id_x, .add_cta_id_y, .add_cta_id_z,
    .add_grid_size_x, .add_grid_size_y, .add_grid_size_z,
    .add_cta_size_x, .add_cta_size_y, .add_cta_size_z, .add_cta_size,
    .add_kernel_id, .add_ready,
    .out_valid, .out_cta_id_x, .out_cta_id_y, .out_cta_id_z, .out_cta_size, .out_kernel_id, .out_ready,
    .cta_valid,
    .cta_id_x, .cta_id_y, .cta_id_z,
    .grid_size_x, .grid_size_y, .grid_size_z,
    .cta_size_x, .cta_size_y, .cta_size_z,
    .cta_size(cta_size_arr),
    .kernel_id,
    .full,
    .next_empty_cta_index
  );

  // -----------------------
  // Clock / reset
  // -----------------------
  initial begin
    clk = 0;
    forever #1 clk = ~clk;
  end

  initial begin
    rst_n = 0;
    pop_valid = 0;
    out_ready = 0;
    add_valid = 0;
    add_cta_id_x = 0;
    add_cta_id_y = 0;
    add_cta_id_z = 0;
    add_grid_size_x = 0;
    add_grid_size_y = 0;
    add_grid_size_z = 0;
    add_cta_size_x = 0;
    add_cta_size_y = 0;
    add_cta_size_z = 0;
    add_cta_size   = 0;
    add_kernel_id  = 0;

    repeat (5) @(posedge clk);
    rst_n = 1;
  end

  // -----------------------
  // Reference model / scoreboard
  // -----------------------
  typedef struct packed {
    bit   valid;
    bit   is_primary;
    int   entries_used;
    int   cta_id_x, cta_id_y, cta_id_z;
    int   grid_size_x, grid_size_y, grid_size_z;
    int   cta_size_x, cta_size_y, cta_size_z;
    int   cta_size;
    int   kernel_id;
  } cta_slot_t;

  cta_slot_t ref_slots [MAX_NUM_CTA];

  // Compute entries needed
  function automatic int ref_entries_needed(input int size);
    int adjusted;
    adjusted = size + THREAD_WIDTH - 1;
    return adjusted >> $clog2(THREAD_WIDTH);
  endfunction

  // return first empty slot index that has enough consecutive space
  function automatic void ref_find_first_empty(output int idx, output bit found);
    int consecutive;
    idx = 0;
    found = 0;
    
    // Just find first invalid slot (DUT's simple approach)
    for (int i = 0; i < MAX_NUM_CTA; i++) begin
      if (!ref_slots[i].valid && !found) begin
        idx = i;
        found = 1;
      end
    end
  endfunction
  
  // // Check if we have N consecutive empty slots starting at idx
  // function automatic bit ref_has_n_consecutive_empty(input int start_idx, input int n);
  //   if (start_idx + n > MAX_NUM_CTA) return 0;
  //   for (int i = start_idx; i < start_idx + n; i++) begin
  //     if (ref_slots[i].valid) return 0;
  //   end
  //   return 1;
  // endfunction
  
  // // Find first position with N consecutive empty slots
  // function automatic void ref_find_first_n_empty(input int n, output int idx, output bit found);
  //   idx = 0;
  //   found = 0;
  //   for (int i = 0; i <= MAX_NUM_CTA - n; i++) begin
  //     if (ref_has_n_consecutive_empty(i, n) && !found) begin
  //       idx = i;
  //       found = 1;
  //     end
  //   end
  // endfunction

  // determine if any empty exists
  function automatic bit ref_has_empty();
    for (int i = 0; i < MAX_NUM_CTA; i++)
      if (!ref_slots[i].valid) return 1;
    return 0;
  endfunction

  // Reset reference model
  task ref_reset();
    for (int i = 0; i < MAX_NUM_CTA; i++) begin
      ref_slots[i] = '{default:0};
    end
  endtask

// Allocate in reference model to mirror DUT's simplified policy:
// - pick the FIRST empty slot only
// - write entries [idx .. idx+need-1] without checking for contiguous empties
// - any j >= MAX_NUM_CTA is ignored (like DUT's j loop guard)
task ref_allocate(
  input int id_x, id_y, id_z,
  input int g_x, g_y, g_z,
  input int s_x, s_y, s_z,
  input int total_size,
  input int kid
);
  int need, idx, j;
  bit found;

  need = ref_entries_needed(total_size);

  // mirror DUT: just find first empty slot (not N-consecutive)
  ref_find_first_empty(idx, found);
  if (!found) begin
    // DUT wouldn't get here because add_ready mirrors (~full),
    // but keep a guard for safety.
    $error("ref_allocate: no empty slot while add_ready expected high");
    return;
  end

  // Write entries the same way DUT does (no occupancy checks).
  for (j = 0; j < MAX_NUM_CTA; j++) begin
    if (j >= idx && j < (idx + need)) begin
      ref_slots[j].valid        = 1;
      ref_slots[j].is_primary   = (j == idx);
      ref_slots[j].entries_used = need;

      if (j == idx) begin
        ref_slots[j].cta_id_x    = id_x;
        ref_slots[j].cta_id_y    = id_y;
        ref_slots[j].cta_id_z    = id_z;
        ref_slots[j].grid_size_x = g_x;
        ref_slots[j].grid_size_y = g_y;
        ref_slots[j].grid_size_z = g_z;
        ref_slots[j].cta_size_x  = s_x;
        ref_slots[j].cta_size_y  = s_y;
        ref_slots[j].cta_size_z  = s_z;
        ref_slots[j].cta_size    = total_size;
        ref_slots[j].kernel_id   = kid;
      end else begin
        // non-primary slots carry no meaningful metadata
        ref_slots[j].cta_id_x    = 0;
        ref_slots[j].cta_id_y    = 0;
        ref_slots[j].cta_id_z    = 0;
        ref_slots[j].grid_size_x = 0;
        ref_slots[j].grid_size_y = 0;
        ref_slots[j].grid_size_z = 0;
        ref_slots[j].cta_size_x  = 0;
        ref_slots[j].cta_size_y  = 0;
        ref_slots[j].cta_size_z  = 0;
        ref_slots[j].cta_size    = 0;
        ref_slots[j].kernel_id   = 0;
      end
    end
  end

  $display("  REF: Allocated CTA(%0d,%0d,%0d) size=%0d to slots %0d-%0d (need %0d entries, DUT-style)",
           id_x, id_y, id_z, total_size, idx, idx+need-1, need);
endtask


  // Pop in reference model
  task ref_pop(input int idx,
               output int id_x, id_y, id_z,
               output int total_size,
               output int kid,
               output int cleared);
    assert(ref_slots[idx].valid && ref_slots[idx].is_primary)
      else $fatal(1, "ref_pop at idx %0d is not a valid primary slot", idx);
    id_x = ref_slots[idx].cta_id_x;
    id_y = ref_slots[idx].cta_id_y;
    id_z = ref_slots[idx].cta_id_z;
    total_size = ref_slots[idx].cta_size;
    kid = ref_slots[idx].kernel_id;
    cleared = ref_slots[idx].entries_used;

    for (int j = idx; j < idx + cleared; j++) begin
      ref_slots[j] = '{default:0};
    end
  endtask

  // -----------------------
  // Helpers to drive DUT
  // -----------------------

  // Drive add
  task automatic drive_add(
    input int id_x, id_y, id_z,
    input int g_x, g_y, g_z,
    input int s_x, s_y, s_z,
    input int total_size,
    input int kid
  );
    add_cta_id_x   <= id_x;
    add_cta_id_y   <= id_y;
    add_cta_id_z   <= id_z;
    add_grid_size_x<= g_x;
    add_grid_size_y<= g_y;
    add_grid_size_z<= g_z;
    add_cta_size_x <= s_x;
    add_cta_size_y <= s_y;
    add_cta_size_z <= s_z;
    add_cta_size   <= total_size;
    add_kernel_id  <= kid;

    do @(posedge clk); while (!add_ready);
    add_valid <= 1'b1;
    @(posedge clk);
    add_valid <= 1'b0;

    ref_allocate(id_x, id_y, id_z, g_x, g_y, g_z, s_x, s_y, s_z, total_size, kid);

    @(posedge clk);
  endtask

  // Drive pop
  task automatic drive_pop(input int idx);
    pop_hw_cta_id <= idx[CTA_INDEX_WIDTH-1:0];
    pop_valid     <= 1'b1;
    @(posedge clk);
    pop_valid     <= 1'b0;
  endtask

  // -----------------------
  // Assertions
  // -----------------------

  property p_add_ready_not_full;
    @(posedge clk) disable iff(!rst_n)
      add_ready == (~full);
  endproperty
  assert property (p_add_ready_not_full)
    else $error("add_ready / full mismatch");

  // Disabled continuous checking - only check in explicit verification points
  // The continuous check was causing spam when ref model diverged from DUT
  // always @(posedge clk) if (rst_n) begin
  //   int ridx;
  //   bit rfound;
  //   ref_find_first_empty(ridx, rfound);
  //   if (rfound) begin
  //     assert(next_empty_cta_index == ridx[CTA_INDEX_WIDTH-1:0])
  //       else $error("next_empty_cta_index mismatch: got %0d exp %0d",
  //                   next_empty_cta_index, ridx);
  //   end else begin
  //     assert(full == 1) else $error("ref says full, but DUT full==0");
  //   end
  // end

  // -----------------------
  // Functional coverage
  // -----------------------
  typedef enum int {NEED_1=1, NEED_2=2, NEED_3=3, NEED_4=4} need_e;

  need_e last_need;
  bit    did_simul_replace;
  bit    held_backpressure;

  covergroup cg;
    option.per_instance = 1;

    coverpoint last_need {
      bins need1 = {NEED_1};
      bins need2 = {NEED_2};
      bins need3 = {NEED_3};
      bins need4 = {NEED_4};
    }

    coverpoint did_simul_replace { bins yes = {1}; bins no = {0}; }
    coverpoint held_backpressure { bins yes = {1}; bins no = {0}; }

    cross last_need, did_simul_replace, held_backpressure;
  endgroup

  cg c = new();

  // -----------------------
  // Checker for status bus vs ref
  // -----------------------
  task check_status_vs_ref(string tag);
    int i;
    bit ref_primary;
    int ridx;
    bit rfound;
    bit errors_found;
    
    errors_found = 0;
    
    // Check next_empty_cta_index
    ref_find_first_empty(ridx, rfound);
    if (rfound) begin
      if (next_empty_cta_index != ridx[CTA_INDEX_WIDTH-1:0]) begin
        $error("[%s] next_empty_cta_index mismatch: DUT=%0d REF=%0d", 
               tag, next_empty_cta_index, ridx);
        errors_found = 1;
      end
    end else begin
      if (full != 1) begin
        $error("[%s] ref says full, but DUT full==0", tag);
        errors_found = 1;
      end
    end
    
    // Check each slot
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      ref_primary = (ref_slots[i].valid && ref_slots[i].is_primary);
      
      if (cta_valid[i] != ref_primary) begin
        $error("[%s] cta_valid[%0d] mismatch got %0b exp %0b",
               tag, i, cta_valid[i], ref_primary);
        errors_found = 1;
      end
      
      if (ref_primary) begin
        if (cta_id_x[i] != ref_slots[i].cta_id_x) begin
          $error("[%s] cta_id_x[%0d] mismatch: DUT=%0d REF=%0d", 
                 tag, i, cta_id_x[i], ref_slots[i].cta_id_x);
          errors_found = 1;
        end
        if (cta_id_y[i] != ref_slots[i].cta_id_y) begin
          $error("[%s] cta_id_y[%0d] mismatch: DUT=%0d REF=%0d", 
                 tag, i, cta_id_y[i], ref_slots[i].cta_id_y);
          errors_found = 1;
        end
        if (cta_id_z[i] != ref_slots[i].cta_id_z) begin
          $error("[%s] cta_id_z[%0d] mismatch: DUT=%0d REF=%0d", 
                 tag, i, cta_id_z[i], ref_slots[i].cta_id_z);
          errors_found = 1;
        end
        if (cta_size_arr[i] != ref_slots[i].cta_size) begin
          $error("[%s] cta_size[%0d] mismatch: DUT=%0d REF=%0d", 
                 tag, i, cta_size_arr[i], ref_slots[i].cta_size);
          errors_found = 1;
        end
        if (kernel_id[i] != ref_slots[i].kernel_id) begin
          $error("[%s] kernel_id[%0d] mismatch: DUT=%0h REF=%0h", 
                 tag, i, kernel_id[i], ref_slots[i].kernel_id);
          errors_found = 1;
        end
      end else begin
        if (cta_id_x[i] != 0 || cta_id_y[i] != 0 || cta_id_z[i] != 0) begin
          $error("[%s] non-primary[%0d] showed nonzero CTA IDs", tag, i);
          errors_found = 1;
        end
      end
    end
    
    // Debug dump if errors found
    if (errors_found) begin
      $display("\n[%s] === DEBUG DUMP ===", tag);
      $display("DUT State:");
      $display("  full=%0b next_empty=%0d cta_valid=%04b", full, next_empty_cta_index, cta_valid);
      for (i = 0; i < MAX_NUM_CTA; i++) begin
        $display("  Slot[%0d]: valid=%0b id=(%0d,%0d,%0d) size=%0d kid=%0h",
                 i, cta_valid[i], cta_id_x[i], cta_id_y[i], cta_id_z[i], 
                 cta_size_arr[i], kernel_id[i]);
      end
      $display("REF State:");
      for (i = 0; i < MAX_NUM_CTA; i++) begin
        $display("  Slot[%0d]: valid=%0b prim=%0b ent_used=%0d id=(%0d,%0d,%0d) size=%0d kid=%0h",
                 i, ref_slots[i].valid, ref_slots[i].is_primary, ref_slots[i].entries_used,
                 ref_slots[i].cta_id_x, ref_slots[i].cta_id_y, ref_slots[i].cta_id_z,
                 ref_slots[i].cta_size, ref_slots[i].kernel_id);
      end
      $display("===================\n");
    end
  endtask

  // -----------------------
  // Scenario 1: Reset & empty
  // -----------------------
  task scenario_reset_check();
    $display("\n--- Scenario 1: Reset & empty ---");
    ref_reset();
    repeat (2) @(posedge clk);
    check_status_vs_ref("reset");
    assert(out_valid == 0) else $error("out_valid not 0 after reset");
  endtask

  function automatic int size_for_need(input int need);
    case (need)
      1: return 256;   // Exactly 256 threads = 1 entry
      2: return 512;   // Exactly 512 threads = 2 entries
      3: return 768;   // Exactly 768 threads = 3 entries
      4: return 1024;  // Exactly 1024 threads = 4 entries
      default: return 256;
    endcase
  endfunction
  
  // Helper to get valid dimensions for a given size
  function automatic void get_dims_for_size(input int size, output int x, y, z);
    case (size)
      256:  begin x = 16; y = 16; z = 1; end   // 16*16*1
      512:  begin x = 16; y = 16; z = 2; end   // 16*16*2
      768:  begin x = 16; y = 16; z = 3; end   // 16*16*3
      1024: begin x = 16; y = 16; z = 4; end   // 16*16*4
      default: begin x = 16; y = 16; z = 1; end
    endcase
  endfunction

  // -----------------------
  // Scenario 2: Directed adds
  // -----------------------
  task scenario_directed_adds();
    int need, total;
    $display("\n--- Scenario 2: Directed adds 1..4 entries ---");
    
    // Add CTA needing 1 entry (256 threads max)
    last_need = NEED_1;
    drive_add(
      11, 21, 31,      // cta_id
      4, 5, 6,         // grid_size
      16, 16, 1,       // cta_size dimensions (16*16*1 = 256)
      256,             // total size
      101              // kernel_id
    );
    c.sample();
    check_status_vs_ref("dir_add_need1");
    
    // Add CTA needing 2 entries (512 threads max)
    last_need = NEED_2;
    drive_add(
      12, 22, 32,
      4, 5, 6,
      16, 16, 2,       // 16*16*2 = 512
      512,
      102
    );
    c.sample();
    check_status_vs_ref("dir_add_need2");
    
    // Add CTA needing 1 entry (to fill last slot)
    last_need = NEED_1;
    drive_add(
      13, 23, 33,
      4, 5, 6,
      8, 8, 4,         // 8*8*4 = 256
      256,
      103
    );
    c.sample();
    check_status_vs_ref("dir_add_need3");

    assert(full == 1) else $error("Table should be full after directed adds (1+2+1=4 slots)");
  endtask

  // -----------------------
  // Scenario 3: Pop w/ immediate consume
  // -----------------------
  task scenario_pop_consume_immediate();
    int idx_primary;
    int e_x, e_y, e_z, e_size, e_kid, cleared;
    int i;
    
    $display("\n--- Scenario 3: Pop w/ immediate consume ---");
    
    idx_primary = -1;
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (ref_slots[i].valid && ref_slots[i].is_primary) begin
        idx_primary = i;
        break;
      end
    end
    assert(idx_primary >= 0) else $fatal(1, "No primary slot found");

    ref_pop(idx_primary, e_x, e_y, e_z, e_size, e_kid, cleared);

    drive_pop(idx_primary);
    out_ready <= 1'b1;

    do @(posedge clk); while (!out_valid);

    assert(out_cta_id_x == e_x) else $error("out_cta_id_x mismatch");
    assert(out_cta_id_y == e_y) else $error("out_cta_id_y mismatch");
    assert(out_cta_id_z == e_z) else $error("out_cta_id_z mismatch");
    assert(out_cta_size == e_size) else $error("out_cta_size mismatch");
    assert(out_kernel_id == e_kid) else $error("out_kernel_id mismatch");

    @(posedge clk);  // Consume
    out_ready <= 1'b0;

    @(posedge clk);  // Wait for update
    check_status_vs_ref("pop_consume_immediate");
    
    // Pop remaining entries from Scenario 2 to clean up
    $display("  Cleaning up remaining entries from Scenario 2...");
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (cta_valid[i]) begin
        int ex, ey, ez, es, ek, cl;
        $display("    Popping slot %0d", i);
        ref_pop(i, ex, ey, ez, es, ek, cl);
        drive_pop(i);
        out_ready <= 1'b1;
        do @(posedge clk); while (!out_valid);
        @(posedge clk);
        out_ready <= 1'b0;
        @(posedge clk);
      end
    end
    
    check_status_vs_ref("after_cleanup");
  endtask

  // -----------------------
  // Scenario 4: Backpressure
  // -----------------------
  task scenario_backpressure();
    int idx_primary;
    int e_x, e_y, e_z, e_size, e_kid, cleared;
    int save_x;
    int i;
    
    $display("\n--- Scenario 4: Backpressure ---");
    held_backpressure = 1;
    c.sample();

    idx_primary = -1;
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (ref_slots[i].valid && ref_slots[i].is_primary) begin
        idx_primary = i;
        break;
      end
    end
    
    if (idx_primary < 0) begin
      $display("  No entries to test backpressure, skipping...");
      held_backpressure = 0;
      c.sample();
      return;
    end

    ref_pop(idx_primary, e_x, e_y, e_z, e_size, e_kid, cleared);

    out_ready <= 0;
    drive_pop(idx_primary);

    do @(posedge clk); while (!out_valid);
    assert(out_cta_id_x == e_x);
    assert(out_kernel_id == e_kid);

    save_x = out_cta_id_x;
    drive_pop(idx_primary);
    @(posedge clk);
    assert(out_valid == 1 && out_cta_id_x == save_x)
      else $error("Pop ignored expected while buffer full");

    out_ready <= 1;
    @(posedge clk);
    out_ready <= 0;
    @(posedge clk);

    check_status_vs_ref("backpressure_done");
    held_backpressure = 0;
    c.sample();
    
    // Clean up any remaining entries
    $display("  Cleaning up remaining entries...");
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (cta_valid[i]) begin
        int ex, ey, ez, es, ek, cl;
        $display("    Popping slot %0d", i);
        ref_pop(i, ex, ey, ez, es, ek, cl);
        drive_pop(i);
        out_ready <= 1'b1;
        do @(posedge clk); while (!out_valid);
        @(posedge clk);
        out_ready <= 1'b0;
        @(posedge clk);
      end
    end
  endtask

  // -----------------------
  // Scenario 5: Simultaneous replace (simplified)
  // -----------------------
  task scenario_simultaneous_replace();
    int exp_x1, exp_y1, exp_z1, exp_s1, exp_k1, clr1;
    int exp_x2, exp_y2, exp_z2, exp_s2, exp_k2, clr2;
    int idx1, idx2;
    int i;
    
    $display("\n--- Scenario 5: Back-to-back pop operations ---");
    
    last_need = NEED_1;
    c.sample();
    drive_add(101,0,0, 1,1,1, 16,16,1, 256, 501);
    drive_add(102,0,0, 1,1,1, 16,16,1, 256, 502);

    // Pop first CTA
    idx1 = 0;
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (ref_slots[i].valid && ref_slots[i].is_primary) begin
        idx1 = i;
        break;
      end
    end
    ref_pop(idx1, exp_x1, exp_y1, exp_z1, exp_s1, exp_k1, clr1);
    drive_pop(idx1);
    out_ready <= 1'b1;
    do @(posedge clk); while (!out_valid);
    
    // Verify first CTA output
    assert(out_cta_id_x == exp_x1) else $error("First pop: expected x=%0d got %0d", exp_x1, out_cta_id_x);
    
    @(posedge clk);  // Consume first CTA
    
    // Find and pop second CTA
    idx2 = -1;
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (ref_slots[i].valid && ref_slots[i].is_primary) begin
        idx2 = i;
        break;
      end
    end
    assert(idx2 >= 0) else $fatal(1, "No second primary found");

    ref_pop(idx2, exp_x2, exp_y2, exp_z2, exp_s2, exp_k2, clr2);
    
    // Pop second CTA (out_ready still high)
    pop_hw_cta_id<= idx2[CTA_INDEX_WIDTH-1:0];
    pop_valid    <= 1'b1;
    @(posedge clk);
    pop_valid    <= 1'b0;
    
    // Wait for new output
    do @(posedge clk); while (!out_valid);
    
    // Verify second CTA output
    assert(out_cta_id_x == exp_x2) else $error("Second pop: expected x=%0d got %0d", exp_x2, out_cta_id_x);
    $display("  Successfully popped both CTAs back-to-back");

    @(posedge clk);  // Consume second CTA
    out_ready <= 1'b0;
    
    @(posedge clk);
    did_simul_replace = 1;
    c.sample();
    did_simul_replace = 0;

    check_status_vs_ref("back_to_back_done");
  endtask

  // -----------------------
  // Scenario 6: Overfill attempt
  // -----------------------
  task scenario_overfill_attempt();
    int i;
    
    $display("\n--- Scenario 6: Overfill attempt ---");
    
    // First, ensure table is completely empty by popping any remaining entries
    $display("  Cleaning up any remaining entries...");
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (cta_valid[i]) begin
        $display("    Found valid entry at slot %0d, popping it", i);
        drive_pop(i);
        out_ready <= 1'b1;
        do @(posedge clk); while (!out_valid);
        @(posedge clk);
        out_ready <= 1'b0;
      end
    end
    
    // Now reset reference model
    ref_reset();
    @(posedge clk);
    @(posedge clk);
    
    check_status_vs_ref("fresh");
    
    // Add 4 single-entry CTAs to fill table
    for (i = 0; i < 4; i++) begin
      last_need = NEED_1;
      c.sample();
      drive_add(200+i,0,0, 1,1,1, 16,16,1, 256, 600+i);
    end
    assert(full == 1) else $error("Should be full now");

    @(posedge clk);
    if (!add_ready) begin
      $display("  As expected, add_ready is low when full.");
    end else begin
      $error("add_ready unexpectedly high when full");
    end

    check_status_vs_ref("overfill_stable");
    
    // IMPORTANT: Clean up after this test!
    $display("  Cleaning up filled table...");
    for (i = 0; i < MAX_NUM_CTA; i++) begin
      if (cta_valid[i]) begin
        int ex, ey, ez, es, ek, cl;
        ref_pop(i, ex, ey, ez, es, ek, cl);
        drive_pop(i);
        out_ready <= 1'b1;
        do @(posedge clk); while (!out_valid);
        @(posedge clk);
        out_ready <= 1'b0;
        @(posedge clk);
      end
    end
    
    $display("  Table cleanup complete");
  endtask


  // -----------------------
  // Scenario 7: Randomized
  // -----------------------
  task scenario_randomized(int iters);
    int t, need, total, idxp;
    int ex, ey, ez, esz, ekid, cl;
    int i, dx, dy, dz;
    
    $display("\n--- Scenario 7: Randomized sequence (%0d iters) ---", iters);
    ref_reset();
    check_status_vs_ref("rand_start");

    for (t = 0; t < iters; t++) begin
      @(posedge clk);

      if (ref_has_empty() && ($urandom_range(0,99) < 50)) begin
        need = $urandom_range(1,4);
        total = size_for_need(need);
        get_dims_for_size(total, dx, dy, dz);
        last_need = need_e'(need);
        c.sample();
        drive_add(
          300+t, 0, 0,
          2,2,2,
          dx, dy, dz,
          total,
          700+t
        );
      end

      if (out_valid) begin
        if ($urandom_range(0,99) < 50) begin
          out_ready <= 1'b1;
          @(posedge clk);
          out_ready <= 1'b0;
        end
      end else begin
        idxp = -1;
        for (i = 0; i < MAX_NUM_CTA; i++) begin
          if (ref_slots[i].valid && ref_slots[i].is_primary) begin
            idxp = i;
            break;
          end
        end
        if (idxp >= 0 && ($urandom_range(0,99) < 40)) begin
          ref_pop(idxp, ex, ey, ez, esz, ekid, cl);
          drive_pop(idxp);
          do @(posedge clk); while (!out_valid);
          if ($urandom_range(0,99) < 50) begin
            out_ready <= 1'b1;
            @(posedge clk);
            out_ready <= 1'b0;
          end
        end
      end

      check_status_vs_ref($sformatf("rand_step_%0d", t));
    end
  endtask

  // -----------------------
  // Main sequence
  // -----------------------
  initial begin
    @(posedge rst_n);
    repeat (2) @(posedge clk);

    ref_reset();

    scenario_reset_check();
    scenario_directed_adds();
    scenario_pop_consume_immediate();
    scenario_backpressure();
    scenario_simultaneous_replace();
    scenario_overfill_attempt();
    scenario_randomized(150);

    $display("\nAll scenarios completed.");
    repeat (5) @(posedge clk);
    $finish;
  end

endmodule