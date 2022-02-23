`include "VX_platform.vh"

`TRACING_OFF
module VX_mem_streamer #(
    parameter NUM_REQS = 1
	parameter TAG_WIDTH = 32
	parameter QUEUE_SIZE = 16
	parameter PARTIAL_RESPONSE = 0
) (
    input  wire clk,
    input  wire reset,

    /* DCache interface */
	VX_dcache_req_if.master dcache_req_if,
	VX_dcache_rsp_if.slave dcache_rsp_if,

	/* Inputs */
	VX_msu_req_if.slave msu_req_if,

	/* Outputs */
	VX_msu_rsp_if.master msu_rsp_if
  );

	localparam QUEUE_ADDRW = `CLOG2(QUEUE_SIZE);
	localparam RSP_DATAW = 1 + NUM_REQS + NUM_REQS*32 + TAG_WIDTH;

	/* Detect duplicate addresses 
	 */
	wire [NUM_REQS-2:0] addr_matches
	for(genvar i = 0; i < NUM_REQS-1; i++) begin
		assign addr_matches[i] = (msu_req_if.addr[i+1] == msu_req_if.addr[0]) || ~msu_req_if.valid[i+1]
	end
	wire req_dup = msu_req_if.valid[0] && (& addr_matches);
	/* If all threads request the same address, make sure that only a single request is sent to the cache
	 *
	 * i.e. only a single request among a batch of requests is valid
	 */
	wire [NUM_REQS-1:0] req_dup_mask = msu_req_if.mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};

	/* Pending queue */
	wire [QUEUE_ADDRW-1:0] 	pq_waddr;
	wire 					pq_push;
	wire [QUEUE_ADDRW-1:0] 	pq_raddr;
	wire 					pq_pop;
	wire 					pq_empty;
	wire 					pq_full;

	/* Incoming and outgoing requests */
	wire 									pq_req_valid, 	req_valid;
	wire 									pq_req_rw, 		req_rw;
	wire [NUM_REQS-1:0] 					pq_req_mask, 	req_mask;
	wire [WORD_SIZE-1:0] 					pq_req_byteen, 	req_byteen;
	wire [NUM_REQS-1:0][31:0] 				pq_req_addr, 	req_addr;
	wire [NUM_REQS-1:0][31:0] 				pq_req_data, 	req_data;
	wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] 	pq_req_tag, 	req_tag;

	/* Address queue */
	wire [QUEUE_ADDRW-1:0] 	aq_raddr;
	wire 					aq_empty;
	wire 					aq_full;

	/* DCache request */
	wire [NUM_REQS-1:0] dcache_req_fire;
	wire 				dcache_req_ready;
	reg  [NUM_REQS-1:0] req_sent_mask;
	wire [NUM_REQS-1:0] req_sent_mask_n

	/* DCache response */
	wire 								dcache_rsp_fire;
	reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0]	rsp_rem_mask;
	wire [NUM_THREADS-1:0] 				rsp_rem_mask_n

	/* Response gather
	 *
	 * If partial response is enabled, we send responses to the pipe register as we get them
	 *
	 * If partial response is disabled, we wait until we get the entire batch of responses
	 *
	 */
	wire 						rg_waddr;
	wire 						rg_push;
	wire 						rg_raddr;
	wire 						rg_rsp_valid;
	wire [NUM_REQS-1:0] 		rg_rsp_mask;
	wire [NUM_REQS-1:0][31:0] 	rg_rsp_data;
	wire 						rg_pop;
	wire 						rg_full;
	wire						rg_empty;

	wire req_en;
	wire rsp_en;
				
	/* Save incoming requests into a pending queue.
	 *
	 * Each entry in the PQ stores a batch of requests.
	 * Even invalid requests are stored in the PQ. This is because we need the valid bit to check whether the entry
	 * retrieved using read_addr is valid or not. 
	 *
	 * @write_addr is an output from the index buffer. It is the address in the PQ where an entry is written to.
	 * It is asserted the clock cycle following the write operation.
	 * 
	 * @read_addr is an input to the PQ. It's the address in the PQ where we want to read from.
	 * It is selected between an output from the dcache response and an output from the address queue.
	 * We need to check if the address coming from the address queue is valid before trying 
	 * to access the entry corresponding to that address.

	 * @release_slot - In the LSU, mbuf_pop is asserted when we receive a response from the dcache.
	 * But what I'm doing is releasing an entry not when we get a response, but rather when all entries are sent.
	 * I don't think this is correct. An entry should be released only after rsp_mask = req_mask
	 *
	 */
	VX_index_buffer #(
		.DATAW	(1 + 1 + NUM_REQS + WORD_SIZE + (NUM_REQS * 32) + (NUM_REQS * 32) + TAG_WIDTH),
		.SIZE	(QUEUE_SIZE)
	) pending_queue (
		.clk			(clk),
		.reset			(reset),
		.write_addr		(pq_waddr),
		.acquire_slot	(pq_push),
		.read_addr		(pq_raddr),
		.write_data		(msu_req_if.valid, msu_req_if.rw, msu_req_if.mask, msu_req_if.byteen, msu_req_if.addr, msu_req_if.data, msu_req_if.tag),
		.read_data		(pq_req_valid, 	   pq_req_rw,     pq_req_mask, 	   pq_req_byteen,     pq_req_addr,     pq_req_data,     pq_req_tag),
		.release_addr	(pq_raddr),
		.release_slot	(pq_pop),
		.full			(pq_full),
		.empty			(pq_empty)
	);

	/* Clear entry from the pending queue and address queue
	 * 
	 * An entry is cleared if all the responses come back
	 * i.e. no response is remaining
	 *
	 * Also cleared if the entry is for a store and not a load
	 */
	assign pq_pop = dcache_rsp_fire & (0 == rsp_rem_mask_n) & req_rw;
	/* DCache response has higher priority over address queue
	 *
	 * This causes a delay in the processing of new requests
	 * Because if we aren't reading a new request from the pending queue
	 * (but rather reading a response from the cache), then we aren't sending a new request
	 * to the dcache in the next cycle.
	 * Rather, we use the next cycle to check whether the dcache response is complete
	 * If it is, we clear the entry from the PQ and AQ
	 */
	assign pq_raddr = dcache_rsp_fire ? dcache_rsp_if.tag : aq_raddr;
	/* Write an entry into the pending queue if it isn't full */
	assign pq_push = ~pq_full && ~aq_full;

	/* Save pending queue addresses into an address queue */
	VX_fifo_queue #(
		.DATAW	(QUEUE_ADDRW),
		.SIZE	(QUEUE_SIZE)
	) addr_queue (
		.clk		(clk),
		.reset		(reset),
		.push		(pq_push),
		.pop		(pq_pop),
		.data_in	(pq_waddr),
		.data_out	(aq_raddr),
		.empty		(aq_empty),
		.full		(aq_full),
		`UNUSED_PIN(alm_full),
		`UNUSED_PIN(alm_empty),
		`UNUSED_PIN(size)
	);

	/* DCache response */
	/* Can we accept a new cache response? */
	assign dcache_rsp_if.ready = ~rg_full;
	assign dcache_rsp_fire = dcache_rsp_if.valid && dcache_rsp_if.ready;
	assign rsp_rem_mask_n = rsp_rem_mask[pq_raddr] & ~dcache_rsp_if.tmask;

	always @(posedge clk) begin
		if (pq_push) begin
			rsp_rem_mask[pq_waddr] <= req_dup_mask;
		end
		if (dcache_rsp_fire) begin
			rsp_rem_mask[pq_raddr]	<= rsp_rem_mask_n;
		end
	end

	/* Response gather */
	VX_index_buffer #(
		.DATAW	(1 + NUM_REQS + (NUM_REQS * 32)),
		.SIZE	(QUEUE_SIZE)
	) rsp_gather (
		.clk			(clk),
		.reset			(reset),
		.write_addr		(rg_waddr),
		.acquire_slot	(rg_push),
		.read_addr		(rg_raddr),
		.write_data		(dcache_rsp_if.valid, dcache_rsp_if.mask dcache_rsp_if.data),
		.read_data		(rg_rsp_valid,        rg_rsp_mask,       rg_rsp_data),
		.release_addr	(rg_raddr),
		.release_slot	(rg_pop),
		.full			(rg_full),
		.empty			(rg_empty)
	);

	assign rg_raddr = dcache_rsp_if.tag;
	assign rg_push = dcache_rsp_fire & ~or_full;
	assign rg_pop
	
	/* Stall pipeline
	 *
	 * Stall when:
	 * 1. Invalid request
	 * 2. All responses have come back - Since we need to clear the current entry in the pending 
	 * queue and move on to the next entry.
	 *
	 * Note that when rsp_pipe_reg is enabled, req_pipe_reg is disabled.
	 * req_pipe_reg is activated the next clock cycle, when the pending queue outputs a different entry
	 */
	assign req_en = pq_req_valid || (0 == rsp_rem_mask_n);
	assign rsp_en = PARTIAL_RESPONSE ? dcache_rsp_fire : (0 == rsp_rem_mask_n);

	/* Store the output of the pending queue in a pipe register
	 * The pipe register only needs to contain information that needs to be sent to the dcache
	 * i.e. rw, byteen, addr, data, tag
	 * We store the value of mask to check whether the cache has taken all of our requests or not
	 * What should we do if the output of the pending queue is an invalid request?
	 *
	 * 
	 */
	VX_pipe_register #(
		.DATAW	(1 + NUM_REQS + WORD_SIZE + (NUM_REQS * 32) + (NUM_REQS * 32) + QUEUE_ADDRW),
		.RESETW (1)
	) req_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(req_en),
		.data_in	(pq_req_rw, pq_req_mask, pq_req_byteen, pq_req_addr, pq_req_data, pq_raddr),
		.data_out	(req_rw,    req_mask,    req_byteen,    req_addr,    req_data,    req_tag)
	);

	/* Store the output of response gather in another pipe register
	 */
	VX_pipe_register #(
		.DATAW	(1 + (NUM_REQS * 32)),
		.RESETW (1)
	) rsp_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(rsp_en),
		.data_in	(rg_rsp_valid,     rg_rsp_mask,     rg_rsp_data,     pq_req_tag),
		.data_out	(msu_rsp_if.valid, msu_rsp_if.mask, msu_rsp_if.data, msu_rsp_if.tag)
	);

	/* DCache request
	 * Sending a request to the dcache is all about asserting valid and checking ready
	 * No support yet for writing into an offset address
	 * Only writing into word aligned addresses for now

	 * @valid - request is valid and hasn't been sent out yet.
	 */
	assign dcache_req_if.valid = req_mask & ~req_sent_mask;
	assign dcache_req_if.rw = {NUM_REQS{req_rw}};
	assign dcache_req_if.addr = req_addr[31:2];
	assign dcache_req_if.byteen = {NUM_REQS{req_byteen}};
	assign dcache_req_if.data = req_data;
	assign dcache_req_if.tag = {NUM_REQS{req_tag}};

	assign [NUM_REQS-1:0] dcache_req_fire = dcache_req_if.valid & dcache_req_if.ready;
	/* All of the requests in a batch have been sent to the cache */
    assign dcache_req_ready = &(dcache_req_if.ready | req_sent_mask | ~req_dup_mask);
	assign [NUM_REQS-1:0] req_sent_mask_n = req_sent_mask | dcache_req_fire;

	/* Check whether request has been sent to the cache or not
	 * Should be cleared on reset and when the tag changes
	 */
	always @(posedge clk) begin
		if (reset) begin
			req_sent_mask <= 0;
		end else begin
			if(dcache_req_ready) begin
				req_sent_mask <= 0;
			end
			else begin
				req_sent_mask <= req_sent_mask_n;
			end
		end
	end

endmodule
`TRACING_ON