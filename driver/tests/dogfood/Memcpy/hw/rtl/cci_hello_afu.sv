//
// Copyright (c) 2017, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// Neither the name of the Intel Corporation nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


// Read from the memory locations first and then write to the memory locations

`include "platform_if.vh"
`include "afu_json_info.vh"


module ccip_std_afu
   (
    // CCI-P Clocks and Resets
    input           logic             pClk,              // 400MHz - CCI-P clock domain. Primary interface clock
    input           logic             pClkDiv2,          // 200MHz - CCI-P clock domain.
    input           logic             pClkDiv4,          // 100MHz - CCI-P clock domain.
    input           logic             uClk_usr,          // User clock domain. Refer to clock programming guide  ** Currently provides fixed 300MHz clock **
    input           logic             uClk_usrDiv2,      // User clock domain. Half the programmed frequency  ** Currently provides fixed 150MHz clock **
    input           logic             pck_cp2af_softReset,      // CCI-P ACTIVE HIGH Soft Reset
    input           logic [1:0]       pck_cp2af_pwrState,       // CCI-P AFU Power State
    input           logic             pck_cp2af_error,          // CCI-P Protocol Error Detected

    // Interface structures
    input           t_if_ccip_Rx      pck_cp2af_sRx,        // CCI-P Rx Port
    output          t_if_ccip_Tx      pck_af2cp_sTx         // CCI-P Tx Port
    );


    //
    // Run the entire design at the standard CCI-P frequency (400 MHz).
    //
    logic clk;
    assign clk = pClk;

    logic reset;
    assign reset = pck_cp2af_softReset;

    logic [511:0] wr_data;
    logic [511:0] rd_data;

    logic do_update;
    logic start_read;
    logic start_write;
    logic wr_addr_next_valid;
    logic addr_next_valid;
    logic rd_end_of_list;
    logic rd_needed;
    logic wr_needed;
    logic read_req;
    logic write_req;
    logic [15:0] cnt_list_length;
    t_ccip_clAddr rd_addr;
    t_ccip_clAddr wr_addr;
    t_ccip_clAddr addr_next;
    t_ccip_clAddr wr_addr_next;

    // =========================================================================
    //
    //   Register requests.
    //
    // =========================================================================

    //
    // The incoming pck_cp2af_sRx and outgoing pck_af2cp_sTx must both be
    // registered.  Here we register pck_cp2af_sRx and assign it to sRx.
    // We also assign pck_af2cp_sTx to sTx here but don't register it.
    // The code below never uses combinational logic to write sTx.
    //

    t_if_ccip_Rx sRx;
    always_ff @(posedge clk)
    begin
        sRx <= pck_cp2af_sRx;
    end

    t_if_ccip_Tx sTx;
    assign pck_af2cp_sTx = sTx;


    // =========================================================================
    //
    //   CSR (MMIO) handling.
    //
    // =========================================================================

    // The AFU ID is a unique ID for a given program.  Here we generated
    // one with the "uuidgen" program and stored it in the AFU's JSON file.
    // ASE and synthesis setup scripts automatically invoke afu_json_mgr
    // to extract the UUID into afu_json_info.vh.
    logic [127:0] afu_id = `AFU_ACCEL_UUID;

    //
    // A valid AFU must implement a device feature list, starting at MMIO
    // address 0.  Every entry in the feature list begins with 5 64-bit
    // words: a device feature header, two AFU UUID words and two reserved
    // words.
    //

    // Is a CSR read request active this cycle?
    logic is_csr_read;
    assign is_csr_read = sRx.c0.mmioRdValid;

    // Is a CSR write request active this cycle?
    logic is_csr_write;
    assign is_csr_write = sRx.c0.mmioWrValid;

    // The MMIO request header is overlayed on the normal c0 memory read
    // response data structure.  Cast the c0Rx header to an MMIO request
    // header.
    t_ccip_c0_ReqMmioHdr mmio_req_hdr;
    assign mmio_req_hdr = t_ccip_c0_ReqMmioHdr'(sRx.c0.hdr);


    //
    // Implement the device feature list by responding to MMIO reads.
    //

    always_ff @(posedge clk)
    begin
        if (reset)
        begin
            sTx.c2.mmioRdValid <= 1'b0;
        end
        else
        begin
            // Always respond with something for every read request
            sTx.c2.mmioRdValid <= is_csr_read;

            // The unique transaction ID matches responses to requests
            sTx.c2.hdr.tid <= mmio_req_hdr.tid;

            // Addresses are of 32-bit objects in MMIO space.  Addresses
            // of 64-bit objects are thus multiples of 2.
            case (mmio_req_hdr.address)
              0: // AFU DFH (device feature header)
                begin
                    // Here we define a trivial feature list.  In this
                    // example, our AFU is the only entry in this list.
                    sTx.c2.data <= t_ccip_mmioData'(0);
                    // Feature type is AFU
                    sTx.c2.data[63:60] <= 4'h1;
                    // End of list (last entry in list)
                    sTx.c2.data[40] <= 1'b1;
                end

              // AFU_ID_L
              2: sTx.c2.data <= afu_id[63:0];

              // AFU_ID_H
              4: sTx.c2.data <= afu_id[127:64];

              // DFH_RSVD0
              6: sTx.c2.data <= t_ccip_mmioData'(0);

              // DFH_RSVD1
              8: sTx.c2.data <= t_ccip_mmioData'(0);

	      // Updated by apurve to check fpgaReadMMIO
              10: sTx.c2.data <= t_ccip_mmioData'(start_read);

              default: sTx.c2.data <= t_ccip_mmioData'(0);
            endcase
        end
    end


    //
    // CSR write handling.  Host software must tell the AFU the memory address
    // to which it should be writing.  The address is set by writing a CSR.
    //

    // We use MMIO address 0 to set the memory address.  The read and
    // write MMIO spaces are logically separate so we are free to use
    // whatever we like.  This may not be good practice for cleanly
    // organizing the MMIO address space, but it is legal.
    logic is_mem_addr_csr_write;
    assign is_mem_addr_csr_write = is_csr_write &&
                                   (mmio_req_hdr.address == t_ccip_mmioAddr'(0));

    // Memory address to which this AFU will write.
    t_ccip_clAddr write_mem_addr;

    always_ff @(posedge clk)
    begin
        if (reset)
        begin
	    start_write <= 1'b0;
        end
	else if (is_mem_addr_csr_write)
        begin
            write_mem_addr <= t_ccip_clAddr'(sRx.c0.data);
	    start_write <= 1'b1;
            //$display("Write mem address is 0x%x", t_ccip_clAddr'(write_mem_addr));
        end
    end
    

    // We use MMIO address 8 to set the memory address for reading data.
    logic is_mem_addr_csr_read;
    assign is_mem_addr_csr_read = is_csr_write &&
                                   (mmio_req_hdr.address == t_ccip_mmioAddr'(2));

    // Memory address from which this AFU will read.
    t_ccip_clAddr read_mem_addr;

    //logic start_traversal = 'b0;
    //t_ccip_clAddr start_traversal_addr;

    always_ff @(posedge clk)
    begin
        if (reset)
        begin
	    start_read <= 1'b0;
        end
        else if (is_mem_addr_csr_read)
        begin
            read_mem_addr <= t_ccip_clAddr'(sRx.c0.data);
	    start_read <= 1'b1;
            //$display("Read mem address is 0x%x", t_ccip_clAddr'(read_mem_addr));
        end
    end


    // =========================================================================
    //
    //   Main AFU logic
    //
    // =========================================================================

    //
    // States in our simple example.
    //
    //typedef enum logic [0:0]
    typedef enum logic [1:0]
    {
	STATE_IDLE,
        STATE_READ,
        STATE_UPDATE,
        STATE_WRITE
    }
    t_state;

    t_state state;

    //
    // State machine
    //
    always_ff @(posedge clk)
    begin
        if (reset)
        begin
            state <= STATE_IDLE;
	    rd_end_of_list <= 1'b0;
        end
        else
        begin
            case (state)
              STATE_IDLE:
                begin
                    // Traversal begins when CSR 1 is written
                    if (start_read)
                    begin
                        state <= STATE_READ;
                        $display("AFU starting traversal at 0x%x", t_ccip_clAddr'(read_mem_addr));
                    end
                end

              STATE_READ:
                begin
                    $display("AFU in READ...");
                    $display("do_update is %d...",do_update);
                    $display("addr_next_valid is %d...",addr_next_valid);
                    $display("rd_needed is %d...",rd_needed);
                    if (!rd_needed && do_update)
                    begin
		    	state <= STATE_UPDATE;
                        $display("AFU moving to UPDATE...");
                    end
                end

              STATE_UPDATE:
                begin
		    // Update the read value to be written back
                    $display("AFU in UPDATE...");
                    if (!do_update)
		    begin
		    	state <= STATE_WRITE;
			wr_needed <= 1'b1; 
                        $display("AFU moving to WRITE...");
		    end
                end

              STATE_WRITE:
                begin
		    // Write the updated value to the address
		    // Point to new address after that
		    // if done then point to IDLE; else read new values 
                    $display("AFU in WRITE...");
                    if (rd_end_of_list)
		    begin
			state <= STATE_IDLE;
			$display("AFU done...");
		    end
                    else if (!wr_needed)
		    begin
			state <= STATE_READ;
			$display("AFU moving to READ from WRITE...");
		    	start_write <= 1'b0;
			write_req <= 1'b0;
		    end
                end
            endcase
        end
    end


    // =========================================================================
    //
    //   Read logic.
    //
    // =========================================================================

    //
    // READ REQUEST
    //

    // Did a write response just arrive

    // Next read address

    always_ff @(posedge clk)
    begin
	// Next read address is valid when we have got the write response back
	if (sRx.c1.rspValid)
    	begin
            addr_next_valid <= sRx.c1.rspValid;

	    //if (state == STATE_READ && !rd_needed)
    	    //begin
                // Apurve: Next address is current address plus address length
                //addr_next <= addr_next + addr_size;
            addr_next <= (addr_next_valid ? rd_addr + 0 : rd_addr);

                // End of list reached if we have read 5 times
            rd_end_of_list <= (cnt_list_length == 'h5);
    	    //end
    	end	
    end

    //
    // Since back pressure may prevent an immediate read request, we must
    // record whether a read is needed and hold it until the request can
    // be sent to the FIU.
    //

    always_ff @(posedge clk)
    begin
        if (reset)
        begin
            rd_needed <= 1'b0;
        end
        else
        begin
            // If reads are allowed this cycle then we can safely clear
            // any previously requested reads.  This simple AFU has only
            // one read in flight at a time since it is walking a pointer
            // chain.
            if (rd_needed)
            begin
                //rd_needed <= sRx.c0TxAlmFull;
                //rd_needed <= (!sRx.c0TxAlmFull && !sRx.c0.rspValid);
                rd_needed <= !sRx.c0.rspValid;
            end
            else if (state == STATE_READ)
            begin
                // Need a read under two conditions:
                //   - Starting a new walk
                //   - A read response just arrived from a line containing
                //     a next pointer.
                rd_needed <= (start_read || (!sRx.c0TxAlmFull && (addr_next_valid && ! rd_end_of_list)));
                rd_addr <= (start_read ? read_mem_addr : addr_next);
            	//$display("rd_addr is 0x%x",  t_ccip_clAddr'(rd_addr));
            	//$display("read mem addr is 0x%x",  t_ccip_clAddr'(read_mem_addr));
            	//$display("start read is %d", start_read);
            end
        end
    end

    //
    // Emit read requests to the FIU.
    //

    // Read header defines the request to the FIU
    t_ccip_c0_ReqMemHdr rd_hdr;

    always_comb
    begin
        rd_hdr = t_ccip_c0_ReqMemHdr'(0);

        // Read request type (No intention to cache)
        //rd_hdr.req_type = 4'h0;

        // Virtual address (MPF virtual addressing is enabled)
        rd_hdr.address = rd_addr;

        // Read over channel VA 
        //rd_hdr.vc_sel = 2'h0;

        // Read one cache line (64 bytes) 
        //rd_hdr.cl_len = 2'h0;
    end

    // Send read requests to the FIU
    always_ff @(posedge clk)
    begin
        if (reset)
        begin
            sTx.c0.valid <= 1'b0;
            cnt_list_length <= 0;
	    read_req <= 1'b0;
        end
        else
        begin
            // Generate a read request when needed and the FIU isn't full
	    if (state == STATE_READ)
            begin
            	sTx.c0.valid <= (rd_needed && !sRx.c0TxAlmFull && !read_req);

            	if (rd_needed && !sRx.c0TxAlmFull && !read_req)
            	begin
	    	    sTx.c0.hdr <= rd_hdr;
            	    cnt_list_length <= cnt_list_length + 1;
		    read_req <= 1'b1;
            	    $display("Incrementing read count...%d",cnt_list_length);
            	    $display("Read address is 0x%x...",rd_hdr.address);
		    addr_next_valid <= 1'b0;
		    // Apurve: Add something to stop read once this section has been accessed
		    //rd_needed <= 1'b0; 
            	end
            end
        end
    end

    //
    // READ RESPONSE HANDLING
    //

    //
    // Receive data (read responses).
    //
    always_ff @(posedge clk)
    begin
	if (reset)
	begin
            do_update <= 1'b0;
        end
	else
	begin
	    if (!do_update && sRx.c0.rspValid)
	    begin
                rd_data <= sRx.c0.data;
                do_update <= 1'b1;
	        $display("rd data is %d...",rd_data);
            end

	    if ((state == STATE_UPDATE) && (do_update == 1'b1))
	    begin
	        // Update the read data and put it in the write data to be written
                wr_data <= rd_data + 2;
                do_update <= 1'b0;
		read_req <= 1'b0;
	        $display("write data is %d...",wr_data);

		// First read done. Next reads should be from the updated addresses
		start_read <= 1'b0; 
            end
        end
    end


    // =========================================================================
    //
    //   Write logic.
    //
    // =========================================================================


    //
    // WRITE REQUEST
    //

    // Did a write response just arrive

    // Next write address

    always_ff @(posedge clk)
    begin
	if (sRx.c0.rspValid)
    	begin
            // Next write address is valid when we have got the read response back
            wr_addr_next_valid <= sRx.c0.rspValid;
            //wr_addr_next_valid <= (!start_write && sRx.c0.rspValid);

	    //if (state == STATE_WRITE && !wr_needed)
	    //begin
                // Apurve: Next address is current address plus address length
                //wr_addr_next <= wr_addr + 0;
                wr_addr_next <= (wr_addr_next_valid ? wr_addr + 0 : wr_addr);
	    //end
	end
    end

    //
    // Since back pressure may prevent an immediate write request, we must
    // record whether a write is needed and hold it until the request can
    // be sent to the FIU.
    //

    always_ff @(posedge clk)
    begin
        if (reset)
        begin
            wr_needed <= 1'b0;
        end
        else
        begin
            // If writes are allowed this cycle then we can safely clear
            // any previously requested writes.  This simple AFU has only
            // one write in flight at a time since it is walking a pointer
            // chain.
            if (wr_needed)
            begin
                //wr_needed <= sRx.c1TxAlmFull;
                //wr_needed <= (!sRx.c1TxAlmFull && !sRx.c1.rspValid);
                wr_needed <= !sRx.c1.rspValid;
            end
            else
            begin
                // Need a write under two conditions:
                //   - Starting a new walk
                //   - A write response just arrived from a line containing
                //     a next pointer.
                wr_needed <= (start_write || (!sRx.c1TxAlmFull && wr_addr_next_valid));
                wr_addr <= (start_write ? write_mem_addr : wr_addr_next);
            	//$display("Write mem address later is 0x%x", t_ccip_clAddr'(write_mem_addr));
            end
        end
    end

    //
    // Emit write requests to the FIU.
    //

    // Write header defines the request to the FIU
    t_ccip_c1_ReqMemHdr wr_hdr;

    always_comb
    begin
        wr_hdr = t_ccip_c1_ReqMemHdr'(0);

        // Write request type
        //wr_hdr.req_type = 4'h0;

        // Virtual address (MPF virtual addressing is enabled)
        wr_hdr.address = wr_addr;

        // Let the FIU pick the channel
        //wr_hdr.vc_sel = 2'h2;

        // Write 1 cache line (64 bytes) 
        //wr_hdr.cl_len = 2'h0;

        // Start of packet is true (single line write)
        wr_hdr.sop = 1'b1;
    end

    // Send write requests to the FIU
    always_ff @(posedge clk)
    begin
        if (reset)
        begin
            sTx.c1.valid <= 1'b0;
            write_req <= 1'b0;
        end
        else
        begin
            // Generate a write request when needed and the FIU isn't full
	    if (state == STATE_WRITE)
            begin
            	sTx.c1.valid <= (wr_needed && !sRx.c1TxAlmFull && !write_req);
		if (wr_needed && !sRx.c1TxAlmFull && !write_req)
		begin
            	    sTx.c1.hdr <= wr_hdr;
	    	    sTx.c1.data <= t_ccip_clData'(wr_data);
		    write_req <= 1'b1;
		    wr_addr_next_valid <= 1'b0;
		    $display("Write address is 0x%x...", wr_hdr.address);
            	end
            end
        end
    end


    //
    // WRITE RESPONSE HANDLING
    //

    // Apurve: Check if a signal is to be sent to read to start reading in case
    // write response does not work
    //
    // Send data (write requests).
    //
    //always_ff @(posedge clk)
    //begin
    //    if (state == STATE_WRITE)
    //    begin
    //        rd_data <= sRx.c0.data;
    //    end
    //    if (state == STATE_UPDATE)
    //    begin
    //        // Update the write data and put it in the write data to be written
    //        wr_data <= rd_data + 1;
    //    end
    //end

endmodule
