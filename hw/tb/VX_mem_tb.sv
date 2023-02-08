/*
    socet115 / zlagpaca@purdue.edu

    testbench for vortex, simulating memory interface
*/

`include "VX_define.vh"

`timescale 1 ns / 1 ps

module vortex_tb ();

    // testbench signals
    parameter PERIOD = 1;
    logic clk = 0, reset;

    // clock gen
    always #(PERIOD/2) CLK++;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
	// memory interfacing signals:

    // Memory request:
    // vortex outputs
    logic                               mem_req_valid;
    logic                               mem_req_rw;    
    logic [`VX_MEM_BYTEEN_WIDTH-1:0]    mem_req_byteen;    
    logic [`VX_MEM_ADDR_WIDTH-1:0]      mem_req_addr;
    logic [`VX_MEM_DATA_WIDTH-1:0]      mem_req_data;
    logic [`VX_MEM_TAG_WIDTH-1:0]       mem_req_tag;
    // vortex inputs
    logic                               mem_req_ready;

    // Memory response:
    // vortex inputs
    logic                               mem_rsp_valid;        
    logic [`VX_MEM_DATA_WIDTH-1:0]      mem_rsp_data;
    logic [`VX_MEM_TAG_WIDTH-1:0]       mem_rsp_tag;
    // vortex outputs
    logic                               mem_rsp_ready;

    // Status:
    // vortex outputs
    logic                               busy;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    // test program
	test #(.PERIOD(PERIOD)) PROG (
        .clk            (clk),
        .reset          (reset),

        .mem_req_valid  (mem_req_valid),
        .mem_req_rw     (mem_req_rw),
        .mem_req_byteen (mem_req_byteen),
        .mem_req_addr   (mem_req_addr),
        .mem_req_data   (mem_req_data),
        .mem_req_tag    (mem_req_tag),
        .mem_req_ready  (mem_req_ready),

        .mem_rsp_valid  (mem_rsp_valid),
        .mem_rsp_data   (mem_rsp_data),
        .mem_rsp_tag    (mem_rsp_tag),
        .mem_rsp_ready  (mem_rsp_ready),

        .busy           (busy)
	);
	
	// DUT
	Vortex DUT (
        .clk            (clk),
        .reset          (reset),

        .mem_req_valid  (mem_req_valid),
        .mem_req_rw     (mem_req_rw),
        .mem_req_byteen (mem_req_byteen),
        .mem_req_addr   (mem_req_addr),
        .mem_req_data   (mem_req_data),
        .mem_req_tag    (mem_req_tag),
        .mem_req_ready  (mem_req_ready),

        .mem_rsp_valid  (mem_rsp_valid),
        .mem_rsp_data   (mem_rsp_data),
        .mem_rsp_tag    (mem_rsp_tag),
        .mem_rsp_ready  (mem_rsp_ready),

        .busy           (busy)
    );
    
endmodule

program test
(
    input clk,
    output logic reset,

    input   mem_req_valid,
            mem_req_rw,
            mem_req_byteen,
            mem_req_addr,
            mem_req_data,
            mem_req_tag,
    
    output  mem_req_ready,

    output  mem_rsp_valid,
            mem_rsp_data,
            mem_rsp_tag,

    input   mem_rsp_ready,

    input   busy
);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
	// test signals:
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	// import types
    import cpu_types_pkg::*;

	// tb signals
	parameter PERIOD 		= 1;
	integer test_num 		= 0;
	string test_string 		= "start";
	string task_string		= "no task";
	logic testing 			= 1'b0;
	logic error				= 1'b0;
	integer num_errors		= 0;

    // tb expected signals
    // Memory request:
    logic                               expected_mem_req_valid;
    logic                               expected_mem_req_rw;
    logic [`VX_MEM_BYTEEN_WIDTH-1:0]    expected_mem_req_byteen;  
    logic [`VX_MEM_ADDR_WIDTH-1:0]      expected_mem_req_addr;
    logic [`VX_MEM_DATA_WIDTH-1:0]      expected_mem_req_data;
    logic [`VX_MEM_TAG_WIDTH-1:0]       expected_mem_req_tag;
    // Memory response:
    logic                               expected_mem_rsp_ready;
    // Status:
    logic                               expected_busy;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
	// tasks:
	///////////////////////////////////////////////////////////////////////////////////////////////////////

    task check_outputs;
    begin
        testing = 1'b1;

        // check for good output
		assert (
            mem_req_valid === expected_mem_req_valid &
            mem_req_rw === expected_mem_req_rw &
            mem_req_byteen === expected_mem_req_byteen &
            mem_req_addr === expected_mem_req_addr &
            mem_req_data === expected_mem_req_data &
            mem_req_tag === expected_mem_req_tag &
            mem_rsp_ready === expected_mem_rsp_read &
            busy === expected_busy
            )
		begin
			$display("Correct outputs");
		end
        // otherwise, error
        else
        begin
            error = 1'b1;
            
            // check for specific errors:
            if (mem_req_valid !== expected_mem_req_valid)
            begin
                num_errors++;
                $display("\tmem_req_valid:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_valid, mem_req_valid);
            end
            
            // check for specific errors:
            if (mem_req_rw !== expected_mem_req_rw)
            begin
                num_errors++;
                $display("\tmem_req_rw:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_rw, mem_req_rw);
            end

            // check for specific errors:
            if (mem_req_byteen !== expected_mem_req_byteen)
            begin
                num_errors++;
                $display("\tmem_req_byteen:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_byteen, mem_req_byteen);
            end

            // check for specific errors:
            if (mem_req_addr !== expected_mem_req_addr)
            begin
                num_errors++;
                $display("\tmem_req_addr:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_addr, mem_req_addr);
            end

            // check for specific errors:
            if (mem_req_data !== expected_mem_req_data)
            begin
                num_errors++;
                $display("\tmem_req_data:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_data, mem_req_data);
            end

            // check for specific errors:
            if (mem_req_tag !== expected_mem_req_tag)
            begin
                num_errors++;
                $display("\tmem_req_tag:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_tag, mem_req_tag);
            end

            // check for specific errors:
            if (mem_req_read !== expected_mem_req_read)
            begin
                num_errors++;
                $display("\tmem_req_read:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_mem_req_read, mem_req_read);
            end

            // check for specific errors:
            if (busy !== expected_busy)
            begin
                num_errors++;
                $display("\tbusy:");
                $display("\texpected: 0x%h\n\t  output: 0x%h", 
                expected_busy, busy);
            end
        end

        #(0.01);
        testing = 1'b0;
        error = 1'b0;
    end
    endtask

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
	// tb:
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	
	initial
	begin
		// init valules
		error = 1'b0;
		num_errors = 0;
		test_num = 0;
		test_string = "";
        task_string = "";
		$display("init");
        $display("");

        ////////////////////
		// reset testing: //
		////////////////////
		@(negedge CLK);
        test_num++;
        test_string = "reset testing";
		$display("reset testing");
		begin
            task_string = "assert reset";
            $display("\n-> testing %s", task_string);

            // input stimuli:
            mem_req_ready   = 1'b0;
            mem_rsp_valid   = 1'b0;
            mem_rsp_data    = '0;
            mem_rsp_tag     = '0;
            mem_rsp_ready   = 1'b0;

            reset = 1'b1;

            // expected outputs:
            expected_mem_req_valid;
            expected_mem_req_rw;
            expected_mem_req_byteen;  
            expected_mem_req_addr;
            expected_mem_req_data;
            expected_mem_req_tag;
            expected_mem_rsp_ready;
            expected_busy;
            
            check_outputs();

            #(PERIOD);
            @(negedge CLK);
            task_string = "deassert nRST";
            $display("\n-> testing %s", task_string);

            // input stimuli:
            mem_req_ready   = 1'b0;
            mem_rsp_valid   = 1'b0;
            mem_rsp_data    = '0;
            mem_rsp_tag     = '0;
            mem_rsp_ready   = 1'b0;

            reset = 1'b0;

            // expected outputs:
            expected_mem_req_valid;
            expected_mem_req_rw;
            expected_mem_req_byteen;  
            expected_mem_req_addr;
            expected_mem_req_data;
            expected_mem_req_tag;
            expected_mem_rsp_ready;
            expected_busy;
            
            check_outputs();
		end
        $display("");

        

        //////////////////////
		// testing results: //
		//////////////////////
        @(negedge CLK);
		test_num 			= 0;
		test_string 		= "testing results";
		$display("");
		$display("//////////////////////");
		$display("// testing results: //");
		$display("//////////////////////");
		$display("");
		begin
			#(PERIOD);

			// check for errors
			if (num_errors)
			begin
				$display("UNSUCCESSFUL VERIFICATION\n%d error(s)", num_errors);
			end
			else
			begin
				$display("SUCCESSFUL VERIFICATION\n\tno errors");
			end
		end
		$display("");

        $finish();
    end

endprogram
