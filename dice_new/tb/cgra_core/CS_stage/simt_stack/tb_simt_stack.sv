module tb_simt_stack;

    // Parameters
    parameter STACK_DEPTH = 32;
    parameter PC_WIDTH = 64;
    parameter THREAD_WIDTH = 256;
    parameter CLK_PERIOD = 2.5; // 400MHz = 2.5ns period
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // DUT signals
    logic push;
    logic modify_top;
    logic [PC_WIDTH-1:0] push_next_pc;
    logic [PC_WIDTH-1:0] push_reconvergence_pc;
    logic [THREAD_WIDTH-1:0] push_active_mask;
    logic pop;
    logic read_top;
    logic [PC_WIDTH-1:0] top_next_pc;
    logic [PC_WIDTH-1:0] top_reconvergence_pc;
    logic [THREAD_WIDTH-1:0] top_active_mask;
    logic out_valid;
    logic stack_empty;
    logic stack_full;
    
    // Test control signals
    int test_count;
    int pass_count;
    int fail_count;
    
    // Test data structure
    typedef struct {
        logic [PC_WIDTH-1:0] next_pc;
        logic [PC_WIDTH-1:0] reconvergence_pc;
        logic [THREAD_WIDTH-1:0] active_mask;
    } stack_entry_t;
    
    // Test data array
    stack_entry_t test_entries[STACK_DEPTH + 5]; // Extra entries for overflow testing
    
    initial begin
        //dump fsdb
        $fsdbDumpfile("tb_simt_stack.fsdb");
        $fsdbDumpvars("+all");
    end

    // DUT instantiation
    simt_stack #(
        .STACK_DEPTH(STACK_DEPTH),
        .PC_WIDTH(PC_WIDTH),
        .THREAD_WIDTH(THREAD_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .push(push),
        .modify_top(modify_top),
        .push_next_pc(push_next_pc),
        .push_reconvergence_pc(push_reconvergence_pc),
        .push_active_mask(push_active_mask),
        .pop(pop),
        .read_top(read_top),
        .top_next_pc(top_next_pc),
        .top_reconvergence_pc(top_reconvergence_pc),
        .top_active_mask(top_active_mask),
        .out_valid(out_valid),
        .stack_empty(stack_empty),
        .stack_full(stack_full)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Initialize test data
    initial begin
        for (int i = 0; i < STACK_DEPTH + 5; i++) begin
            test_entries[i] = '{
                64'h1000 + (i * 64'h10), 
                64'h2000 + (i * 64'h10), 
                256'h1 << (i % 256)  // Rotating bit pattern for 256-bit mask
            };
        end
    end
    
    // Task to initialize signals
    task init_signals();
        push = 0;
        modify_top = 0;
        push_next_pc = 0;
        push_reconvergence_pc = 0;
        push_active_mask = 0;
        pop = 0;
        read_top = 0;
    endtask
    
    // Task to push entry to stack
    task push_entry(input int index);
        @(negedge clk);
        push = 1;
        modify_top = 0;
        push_next_pc = test_entries[index].next_pc;
        push_reconvergence_pc = test_entries[index].reconvergence_pc;
        push_active_mask = test_entries[index].active_mask;
        @(negedge clk);
        push = 0;
        $display("Pushed entry %0d: PC=0x%h, ReconvPC=0x%h, Mask=0x%h", 
                 index, test_entries[index].next_pc, test_entries[index].reconvergence_pc, 
                 test_entries[index].active_mask);
    endtask
    
    // Task to modify top of stack
    task modify_top_entry(input int index);
        @(negedge clk);
        push = 1;
        modify_top = 1;
        push_next_pc = test_entries[index].next_pc;
        push_reconvergence_pc = test_entries[index].reconvergence_pc;
        push_active_mask = test_entries[index].active_mask;
        @(negedge clk);
        push = 0;
        modify_top = 0;
        $display("Modified top with entry %0d: PC=0x%h, ReconvPC=0x%h, Mask=0x%h", 
                 index, test_entries[index].next_pc, test_entries[index].reconvergence_pc, 
                 test_entries[index].active_mask);
    endtask
    
    // Task to pop from stack
    task pop_entry();
        @(negedge clk);
        pop = 1;
        @(negedge clk);
        pop = 0;
        $display("Popped entry");
    endtask
    
    // Task to read and check top of stack
    task read_and_check_top(input stack_entry_t expected, input string test_name);
        automatic logic match;
        
        @(negedge clk);
        read_top = 1;
        @(negedge clk);
        read_top = 0;
        
        // Wait for out_valid and check
        wait (out_valid);
        @(negedge clk);
        
        match = (top_next_pc == expected.next_pc) &&
                (top_reconvergence_pc == expected.reconvergence_pc) &&
                (top_active_mask == expected.active_mask);
        
        if (match) begin
            $display("✓ %s: TOP MATCH - PC=0x%h, ReconvPC=0x%h, Mask=0x%h", 
                     test_name, top_next_pc, top_reconvergence_pc, top_active_mask);
            pass_count++;
        end else begin
            $display("✗ %s: TOP MISMATCH - Expected PC=0x%h, ReconvPC=0x%h, Mask=0x%h, Got PC=0x%h, ReconvPC=0x%h, Mask=0x%h",
                     test_name, expected.next_pc, expected.reconvergence_pc, expected.active_mask,
                     top_next_pc, top_reconvergence_pc, top_active_mask);
            fail_count++;
        end
        test_count++;
    endtask
    
    // Task to check stack status
    task check_status(input logic exp_empty, input logic exp_full, input string test_name);
        @(negedge clk);
        
        if ((stack_empty == exp_empty) && (stack_full == exp_full)) begin
            $display("✓ %s: STATUS OK - empty=%b, full=%b", 
                     test_name, stack_empty, stack_full);
            pass_count++;
        end else begin
            $display("✗ %s: STATUS ERROR - Expected empty=%b, full=%b, Got empty=%b, full=%b",
                     test_name, exp_empty, exp_full, stack_empty, stack_full);
            fail_count++;
        end
        test_count++;
    endtask
    
    // Test cases
    initial begin
        $display("=== SIMT Stack Testbench Started ===");
        
        // VCD dump for waveform viewing
        `ifdef XCELIUM
            $recordfile("simt_stack.trn");
            $recordvars("");
        `else
            $dumpfile("simt_stack.vcd");
            $dumpvars(0, tb_simt_stack);
        `endif
        
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Initialize
        init_signals();
        rst_n = 0;
        repeat(3) @(negedge clk);
        rst_n = 1;
        repeat(2) @(negedge clk);
        
        // Test 1: Reset state
        $display("\n--- Test 1: Reset State ---");
        check_status(1'b1, 1'b0, "Reset State");
        if (!out_valid) begin
            $display("✓ Reset top outputs are zero and out_valid is false");
            pass_count++;
        end else begin
            $display("✗ Reset top outputs not zero or out_valid not false");
            fail_count++;
        end
        test_count++;
        
        // Test 2: Single push
        $display("\n--- Test 2: Single Push ---");
        push_entry(0);
        check_status(1'b0, 1'b0, "After Single Push");
        read_and_check_top(test_entries[0], "Single Push Top");
        
        // Test 3: Modify top
        $display("\n--- Test 3: Modify Top ---");
        modify_top_entry(1);
        check_status(1'b0, 1'b0, "After Modify Top");
        read_and_check_top(test_entries[1], "Modified Top");
        
        // Test 4: Single pop
        $display("\n--- Test 4: Single Pop ---");
        pop_entry();
        check_status(1'b1, 1'b0, "After Single Pop");
        
        // Test 5: Fill stack completely
        $display("\n--- Test 5: Fill Stack Completely ---");
        for (int i = 0; i < STACK_DEPTH; i++) begin 
            push_entry(i);
        end
        check_status(1'b0, 1'b1, "Full Stack");
        read_and_check_top(test_entries[STACK_DEPTH - 1], "Full Stack Top");
        
        // Test 6: Try to push when full
        $display("\n--- Test 6: Push When Full ---");
        @(negedge clk);
        push = 1;
        push_next_pc = test_entries[STACK_DEPTH + 2].next_pc;
        push_reconvergence_pc = test_entries[STACK_DEPTH + 2].reconvergence_pc;
        push_active_mask = test_entries[STACK_DEPTH + 2].active_mask;
        @(negedge clk);
        push = 0;
        // Should still have same top and be full
        check_status(1'b0, 1'b1, "Push When Full");
        read_and_check_top(test_entries[STACK_DEPTH - 1], "Push When Full Top");
        
        // Test 7: Pop from full stack
        $display("\n--- Test 7: Pop From Full Stack ---");
        pop_entry();
        check_status(1'b0, 1'b0, "After Pop From Full");
        read_and_check_top(test_entries[STACK_DEPTH - 2], "Pop From Full Top");
        
        // Test 8: Pop all entries
        $display("\n--- Test 8: Pop All Entries ---");
        for (int i = STACK_DEPTH - 2; i >= 0; i--) begin // Start from STACK_DEPTH-2 since we already popped once
            if (i > 0) begin
                read_and_check_top(test_entries[i], $sformatf("Before Pop %0d", i));
            end
            pop_entry();
            if (i == 0) begin
                check_status(1'b1, 1'b0, "After Pop All");
            end else begin
                check_status(1'b0, 1'b0, $sformatf("After Pop %0d", STACK_DEPTH - 1 - i));
                read_and_check_top(test_entries[i - 1], $sformatf("After Pop %0d Top", STACK_DEPTH - 1 - i));
            end
        end
        
        // Test 9: Push/Pop sequence
        $display("\n--- Test 9: Push/Pop Sequence ---");
        push_entry(0);
        push_entry(1);
        check_status(1'b0, 1'b0, "Push Two");
        read_and_check_top(test_entries[1], "Push Two Top");
        
        pop_entry();
        check_status(1'b0, 1'b0, "Pop One");
        read_and_check_top(test_entries[0], "Pop One Top");
        
        pop_entry();
        check_status(1'b1, 1'b0, "Pop All");
        
        // Test 10: Modify top on empty stack (should be ignored)
        $display("\n--- Test 10: Modify Top on Empty Stack ---");
        modify_top_entry(5);
        check_status(1'b1, 1'b0, "Modify Empty Stack");
        
        // Test 11: Complex push/modify/pop sequence
        $display("\n--- Test 11: Complex Sequence ---");
        push_entry(0);
        push_entry(1);
        modify_top_entry(2);
        read_and_check_top(test_entries[2], "Complex Modify Top");
        push_entry(3);
        read_and_check_top(test_entries[3], "Complex Push Top");
        check_status(1'b0, 1'b0, "Complex Status");
        
        // Pop and verify order
        pop_entry();
        read_and_check_top(test_entries[2], "Complex Pop 1");
        
        pop_entry();
        read_and_check_top(test_entries[0], "Complex Pop 2");
        
        pop_entry();
        check_status(1'b1, 1'b0, "Complex Final");
        
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
            $display("Time=%0t: empty=%b, full=%b, top_pc=0x%h", 
                     $time, stack_empty, stack_full, top_next_pc);
        end
    end

endmodule