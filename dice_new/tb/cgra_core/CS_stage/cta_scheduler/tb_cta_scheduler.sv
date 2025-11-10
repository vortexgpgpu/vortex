module tb_cta_scheduler;

    // Parameters
    parameter MAX_NUM_CTA = 4;
    parameter PC_WIDTH = 64;
    parameter CTA_ID_WIDTH = $clog2(MAX_NUM_CTA);
    parameter EBLOCK_ID_WIDTH = $clog2(MAX_NUM_CTA + 4);
    parameter MAX_EBLOCK = MAX_NUM_CTA + 4;
    parameter CLK_PERIOD = 2.5; // 400MHz = 2.5ns period
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // DUT signals
    logic enable;
    logic [MAX_NUM_CTA-1:0] cta_valid;
    logic [MAX_NUM_CTA-1:0] cta_branch_resolving;
    logic [MAX_NUM_CTA-1:0][PC_WIDTH-1:0] cta_next_pc;
    logic eblock_commit_valid;
    logic [EBLOCK_ID_WIDTH-1:0] eblock_commit_id;
    logic schedule_valid;
    logic [CTA_ID_WIDTH-1:0] schedule_hw_cta_id;
    logic [PC_WIDTH-1:0] schedule_next_pc;
    logic [EBLOCK_ID_WIDTH-1:0] schedule_eblock_id;
    logic schedule_cta_predicted;
    logic schedule_ready;
    
    // Test control signals
    int test_count;
    int pass_count;
    int fail_count;

    // Used for generating waveforms
    initial begin
        //dump fsdb
        $fsdbDumpfile("tb_cta_scheduler.fsdb");
        $fsdbDumpvars("+all");
    end

    // Test data for CTA states
    logic [PC_WIDTH-1:0] test_pcs[MAX_NUM_CTA] = '{
        64'h1000, 64'h2000, 64'h3000, 64'h4000
    };
    
    // DUT instantiation
    cta_scheduler #(
        .MAX_NUM_CTA(MAX_NUM_CTA),
        .PC_WIDTH(PC_WIDTH),
        .MAX_EBLOCK(MAX_EBLOCK)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .cta_valid(cta_valid),
        .cta_branch_resolving(cta_branch_resolving),
        .cta_next_pc(cta_next_pc),
        .eblock_commit_valid(eblock_commit_valid),
        .eblock_commit_id(eblock_commit_id),
        .schedule_valid(schedule_valid),
        .schedule_hw_cta_id(schedule_hw_cta_id),
        .schedule_next_pc(schedule_next_pc),
        .schedule_eblock_id(schedule_eblock_id),
        .schedule_cta_predicted(schedule_cta_predicted),
        .schedule_ready(schedule_ready)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Task to initialize signals
    task init_signals();
        enable = 1; // Default enabled
        cta_valid = '0;
        cta_branch_resolving = '0;
        cta_next_pc = '0;
        eblock_commit_valid = 0;
        eblock_commit_id = '0;
        schedule_ready = 1; // Default ready
    endtask
    
    // Task to reset DUT between tests
    task reset_dut();
        @(negedge clk);
        rst_n = 0;
        init_signals(); // Reset all inputs
        repeat(2) @(negedge clk);
        rst_n = 1;
        repeat(2) @(negedge clk);
        $display("--- DUT Reset Complete ---");
    endtask
    
    // Task to set CTA state
    task set_cta_state(input int cta_id, input logic valid, input logic branch_resolving, input logic [PC_WIDTH-1:0] pc);
        cta_valid[cta_id] = valid;
        cta_branch_resolving[cta_id] = branch_resolving;
        cta_next_pc[cta_id] = pc;
    endtask
    
    // Task to commit e-block
    task commit_eblock(input logic [EBLOCK_ID_WIDTH-1:0] eid);
        @(negedge clk);
        eblock_commit_valid = 1;
        eblock_commit_id = eid;
        @(negedge clk);
        eblock_commit_valid = 0;
        $display("Committed e-block %0d", eid);
    endtask
    
    // Task to expect scheduling result
    task expect_schedule(input logic exp_valid, input logic [CTA_ID_WIDTH-1:0] exp_cta_id, 
                        input logic [PC_WIDTH-1:0] exp_pc, input logic exp_predicted, input string test_name);
        if (schedule_valid == exp_valid) begin
            if (!exp_valid) begin
                $display("✓ %s: No schedule as expected", test_name);
                pass_count++;
            end else if (schedule_hw_cta_id == exp_cta_id && schedule_next_pc == exp_pc && schedule_cta_predicted == exp_predicted) begin
                $display("✓ %s: Schedule match - CTA %0d, PC=0x%h, predicted=%b, e-block=%0d", 
                         test_name, schedule_hw_cta_id, schedule_next_pc, schedule_cta_predicted, schedule_eblock_id);
                pass_count++;
            end else begin
                $display("✗ %s: Schedule mismatch - Expected CTA %0d, PC=0x%h, predicted=%b, Got CTA %0d, PC=0x%h, predicted=%b",
                         test_name, exp_cta_id, exp_pc, exp_predicted, schedule_hw_cta_id, schedule_next_pc, schedule_cta_predicted);
                fail_count++;
            end
        end else begin
            $display("✗ %s: Schedule valid mismatch - Expected %b, Got %b", test_name, exp_valid, schedule_valid);
            fail_count++;
        end
        test_count++;
    endtask

    
    // Task to block scheduling
    task block_schedule();
        @(negedge clk);
        schedule_ready = 0;
        $display("Blocking schedule acceptance");
    endtask
    
    // Test cases
    initial begin
        $display("=== CTA Scheduler Testbench Started ===");
        
        // VCD dump for waveform viewing
        `ifdef XCELIUM
            $recordfile("cta_scheduler.trn");
            $recordvars("");
        `endif

        $dumpfile("cta_scheduler.vcd");
        $dumpvars(0, tb_cta_scheduler);
        
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Initialize
        init_signals();
        rst_n = 0;
        repeat(3) @(negedge clk);
        rst_n = 1;
        repeat(2) @(negedge clk);
        
        // Test 1: No CTAs available
        $display("\n--- Test 1: No CTAs Available ---");
        @(negedge clk);
        expect_schedule(1'b0, 'x, 'x, 'x, "No CTAs");
        
        // Test 2: Single CTA scheduling
        $display("\n--- Test 2: Single CTA ---");
        reset_dut();
        @(negedge clk);
        enable = 1;
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]);
        @(negedge clk);
        expect_schedule(1'b1, 0, test_pcs[0], 1'b0, "Single CTA");
        enable = 0;
        // Test 3: Round-robin among non-branch CTAs
        @(negedge clk);
        $display("\n--- Test 3: Round-Robin Non-Branch ---");
        reset_dut();
        enable = 0;
        @(negedge clk);
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]);
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]);
        set_cta_state(2, 1'b1, 1'b0, test_pcs[2]);
        
        // Should pick CTA 0 (first valid after reset)
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 0, test_pcs[0], 1'b0, "Round-Robin 1st");
        @(negedge clk);
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]+32);
        enable = 0;

        @(negedge clk);
        enable = 1;
        // Should pick CTA 1 (next after last dispatched CTA 0)
        @(posedge clk);
        expect_schedule(1'b1, 1, test_pcs[1], 1'b0, "Round-Robin 2nd");
        @(negedge clk);
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]+32);
        enable = 0;
        
        // Should pick CTA 2 (next after last dispatched CTA 1)
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 2, test_pcs[2], 1'b0, "Round-Robin 3rd");
        @(negedge clk);
        set_cta_state(2, 1'b1, 1'b0, test_pcs[2]+32);
        enable = 0;
        @(negedge clk);
        // Test 4: PC locality priority
        $display("\n--- Test 4: PC Locality Priority ---");
        @(negedge clk);
        reset_dut();
        // First schedule a CTA to establish PC history
        @(negedge clk);
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]);
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 0, test_pcs[0], 1'b0, "Establish PC History");
        @(negedge clk);
        enable = 0;
        
        // Set CTA 3 to have same PC as previous (should get priority)
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]+32);
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]);
        set_cta_state(2, 1'b1, 1'b0, test_pcs[2]);
        set_cta_state(3, 1'b1, 1'b0, test_pcs[0]); // Same as previous PC
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 3, test_pcs[0], 1'b0, "PC Locality Priority");
        //accept_schedule();
        @(negedge clk);
        enable = 0;
        // Test 5: Branch resolving CTAs
        $display("\n--- Test 5: Branch Resolving CTAs ---");
        @(negedge clk);
        reset_dut();
        @(negedge clk);
        // Make all CTAs branch resolving
        set_cta_state(0, 1'b1, 1'b1, test_pcs[0]);
        set_cta_state(1, 1'b1, 1'b1, test_pcs[1]);
        set_cta_state(2, 1'b1, 1'b1, test_pcs[2]);
        set_cta_state(3, 1'b1, 1'b1, test_pcs[3]);
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        // Should pick CTA 0 (first valid), predicted=1
        expect_schedule(1'b1, 0, test_pcs[0], 1'b1, "Branch Resolving");
        @(negedge clk);
        enable = 0;
        
        // Test 6: Mixed branch/non-branch (non-branch has priority)
        $display("\n--- Test 6: Mixed Branch States ---");
        @(negedge clk);
        reset_dut();
        @(negedge clk);
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]); // Non-branch
        set_cta_state(2, 1'b1, 1'b1, test_pcs[2]); // Branch resolving

        @(negedge clk);
        enable = 1;
        @(posedge clk);
        // Should pick non-branch CTA 1, predicted=0
        expect_schedule(1'b1, 1, test_pcs[1], 1'b0, "Non-Branch Priority");
        //accept_schedule();
        @(negedge clk);
        enable = 0;
        // Test 7: PC locality with branch resolving
        $display("\n--- Test 7: PC Locality with Branch Resolving ---");
        @(negedge clk);
        reset_dut();
        // First establish PC history

        @(negedge clk);
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]);
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 1, test_pcs[1], 1'b0, "Establish PC History");
        @(negedge clk);
        enable = 0;
        // Set CTA 2 to have same PC as previous, but it's branch resolving
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]+32); // update
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]); // Non-branch, different PC
        set_cta_state(2, 1'b1, 1'b1, test_pcs[1]); // Same PC, but branch resolving
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 2, test_pcs[1], 1'b1, "PC Locality Branch");
        @(negedge clk);
        enable = 0;

        // Test 8: E-block exhaustion
        $display("\n--- Test 8: E-Block Exhaustion ---");
        @(negedge clk);
        reset_dut();
        // Fill all e-blocks by scheduling without committing
        block_schedule();
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]);
        
        // Schedule until e-blocks are full
        for (int i = 0; i < MAX_EBLOCK; i++) begin
            @(negedge clk);
            schedule_ready = 1;
            if (schedule_valid) begin
                @(negedge clk);
                schedule_ready = 0; // Don't commit, just allocate
                $display("Allocated e-block %0d", schedule_eblock_id);
            end
        end
        
        // Now should not schedule due to e-block exhaustion
        @(negedge clk);
        schedule_ready = 1;
        expect_schedule(1'b0, 'x, 'x, 'x, "E-Block Exhaustion");
        @(negedge clk);
        enable = 0;
        // Test 9: E-block commit and reuse
        $display("\n--- Test 9: E-Block Commit ---");
        commit_eblock(0);
        enable = 0;
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 0, test_pcs[0], 1'b0, "After E-Block Commit");
        //accept_schedule();
        @(negedge clk);
        enable = 0;
        // Test 10: Schedule ready backpressure
        $display("\n--- Test 10: Schedule Ready Backpressure ---");
        @(negedge clk);
        reset_dut();
        @(negedge clk);
        enable = 0;
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]);
        block_schedule();
        @(negedge clk);
        enable = 1;
        // Should have valid but not advance until ready
        @(posedge clk);
        if (schedule_valid && !schedule_ready) begin
            $display("✓ Schedule valid but blocked by ready");
            pass_count++;
        end else begin
            $display("✗ Schedule not properly blocked");
            fail_count++;
        end
        test_count++;
        
        // Accept and verify advancement
        @(negedge clk);
        schedule_ready = 1;
        expect_schedule(1'b1, 1, test_pcs[1], 1'b0, "After Ready");
        //accept_schedule();
        @(negedge clk);
        enable = 0;
        // Test 11: Complex priority interaction
        $display("\n--- Test 11: Complex Priority Test ---");
        @(negedge clk);
        reset_dut();
        @(negedge clk);
        enable = 0;
        // First establish PC history
        set_cta_state(1, 1'b1, 1'b0, test_pcs[1]);
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 1, test_pcs[1], 1'b0, "Establish PC History");
        @(negedge clk);
        enable = 0;
        
        // Setup: CTA 0 (non-branch), CTA 1 (branch, PC locality), CTA 2 (non-branch)
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]);
        set_cta_state(1, 1'b1, 1'b1, test_pcs[1]); // Same as previous PC (locality)
        set_cta_state(2, 1'b1, 1'b0, test_pcs[2]);
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        // PC locality should win over non-branch priority
        expect_schedule(1'b1, 1, test_pcs[1], 1'b1, "PC Locality Wins");
        //accept_schedule();
        @(negedge clk);
        enable = 0;
        // Test 12: Enable/Disable functionality
        $display("\n--- Test 12: Enable/Disable ---");
        reset_dut();
        @(negedge clk);
        enable = 0;
        set_cta_state(0, 1'b1, 1'b0, test_pcs[0]);
        
        // Disable scheduler
        @(negedge clk);
        enable = 0;
        @(posedge clk);
        expect_schedule(1'b0, 'x, 'x, 'x, "Scheduler Disabled");
        
        // Re-enable scheduler
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b1, 0, test_pcs[0], 1'b0, "Scheduler Re-enabled");
        //accept_schedule();
        
        // Test 13: All CTAs invalid
        $display("\n--- Test 13: All CTAs Invalid ---");
        reset_dut();
        @(negedge clk);
        enable = 0;
        set_cta_state(0, 1'b0, 1'b0, test_pcs[0]);
        set_cta_state(1, 1'b0, 1'b0, test_pcs[1]);
        set_cta_state(2, 1'b0, 1'b0, test_pcs[2]);
        set_cta_state(3, 1'b0, 1'b0, test_pcs[3]);
        @(negedge clk);
        enable = 1;
        @(posedge clk);
        expect_schedule(1'b0, 'x, 'x, 'x, "All Invalid");
        
        // Summary
        $display("\n=== Test Summary ===");
        $display("Total Tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("🎉 ALL TESTS PASSED!");
        end else begin
            $display("❌ %0d TESTS FAILED", fail_count);
        end
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #(CLK_PERIOD * 2000);
        $display("⚠️  TIMEOUT: Test took too long");
        $finish;
    end
    
    // Monitor critical signals
    always @(posedge clk) begin
        if (rst_n) begin
            $display("Time=%0t: valid=%b, ready=%b, cta_id=%0d, pc=0x%h, predicted=%b, eblock=%0d", 
                     $time, schedule_valid, schedule_ready, schedule_hw_cta_id, schedule_next_pc, schedule_cta_predicted, schedule_eblock_id);
        end
    end

endmodule