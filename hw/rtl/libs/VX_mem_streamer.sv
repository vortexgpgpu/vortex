`include "VX_platform.vh"

// Test memory stream reader step by step

module VX_mem_streamer #(
    parameter NUM_REQS = 4,
	parameter ADDRW = 32,	
	parameter DATAW = 32,
	parameter TAGW = 32,
	parameter WORD_SIZE = 4,
	parameter QUEUE_SIZE = 16,
	parameter QUEUE_ADDRW = `CLOG2(QUEUE_SIZE),
	parameter PARTIAL_RESPONSE = 1
) (
    input  wire clk,
    input  wire reset,

	// Input request
	input wire 								req_valid,
	input wire 								req_rw,
	input wire [NUM_REQS-1:0] 				req_mask,
	input wire [WORD_SIZE-1:0] 				req_byteen,
	input wire [NUM_REQS-1:0][ADDRW-1:0] 	req_addr,
	input wire [NUM_REQS-1:0][DATAW-1:0] 	req_data,
	input wire [TAGW-1:0]					req_tag,
	output wire 							req_ready,

	// Output request
	output wire [NUM_REQS-1:0] 					mem_req_valid,
	output wire [NUM_REQS-1:0] 					mem_req_rw,
	output wire [NUM_REQS-1:0][WORD_SIZE-1:0] 	mem_req_byteen,
	output wire [NUM_REQS-1:0][ADDRW-1:0] 		mem_req_addr,
	output wire [NUM_REQS-1:0][DATAW-1:0] 		mem_req_data,
	output wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] mem_req_tag,
	input wire 	[NUM_REQS-1:0]					mem_req_ready,

	// Input response
	input wire 								mem_rsp_valid,
	input wire [NUM_REQS-1:0] 				mem_rsp_mask,
	input wire [NUM_REQS-1:0][DATAW-1:0] 	mem_rsp_data,
	input wire [QUEUE_ADDRW-1:0] 			mem_rsp_tag,
	output wire 							mem_rsp_ready,

	// Output response
	output wire 							rsp_valid,
	output wire [NUM_REQS-1:0] 				rsp_mask,
	output wire [NUM_REQS-1:0][DATAW-1:0] 	rsp_data,
	output wire [TAGW-1:0] 					rsp_tag,
	input wire 								rsp_ready
  );

	localparam RSPW = 1 + NUM_REQS + (NUM_REQS * DATAW);

	// Detect duplicate addresses
	wire [NUM_REQS-2:0] addr_matches;
	wire req_dup;
	wire [NUM_REQS-1:0] req_dup_mask;

	// Pending queue
	wire [QUEUE_ADDRW-1:0] 	pq_waddr;
	wire 					pq_push;
	wire 					pq_pop;
	wire [QUEUE_ADDRW-1:0] 	pq_raddr;
	wire 					pq_full;
	wire 					pq_empty;

	wire 							pq_req_rw;	
	wire [NUM_REQS-1:0] 			pq_req_mask; 	
	wire [WORD_SIZE-1:0] 			pq_req_byteen; 	
	wire [NUM_REQS-1:0][ADDRW-1:0] 	pq_req_addr; 	
	wire [NUM_REQS-1:0][DATAW-1:0] 	pq_req_data; 	
	wire [TAGW-1:0] 				pq_req_tag;

	// Index queue
	wire [QUEUE_ADDRW-1:0] 	iq_raddr;
	wire 					iq_push;
	wire 					iq_pop;
	wire 					iq_full;
	wire 					iq_empty;

	// Memory request
	wire 									mreq_en;	
	wire [NUM_REQS-1:0] 					mreq_valid;
	wire [NUM_REQS-1:0] 					mreq_rw;
	wire [NUM_REQS-1:0][WORD_SIZE-1:0] 		mreq_byteen;
	wire [NUM_REQS-1:0][ADDRW-1:0] 			mreq_addr;
	wire [NUM_REQS-1:0][DATAW-1:0] 			mreq_data;
	wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] 	mreq_tag;

	wire [NUM_REQS-1:0] 					mem_req_fire;
	wire [NUM_REQS-1:0] 					mem_req_mask;
	reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0] 	req_sent_mask;
	wire [NUM_REQS-1:0] 					req_sent_mask_n;
	wire 									req_sent_all;

	// Memory response
	wire 									rsp_en;
	reg  [QUEUE_SIZE-1:0][RSPW-1:0]			rsp;
	wire [RSPW-1:0] 						rsp_n;
	wire 									mem_rsp_fire;
	reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0]		rsp_rem_mask;
	wire [NUM_REQS-1:0] 					rsp_rem_mask_n;

	//////////////////////////////////////////////////////////////////

	// Detect duplicate addresses
	for(genvar i = 0; i < NUM_REQS-1; i++) begin
		assign addr_matches[i] = (req_addr[i+1] == req_addr[0]) || ~req_mask[i+1];
	end

	assign req_dup = req_mask[0] && (& addr_matches);
	assign req_dup_mask = req_mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};

	//////////////////////////////////////////////////////////////////

	// Save incoming requests into a pending queue

	// Select entry in PQ
	assign pq_raddr = mem_rsp_fire ? mem_rsp_tag : iq_raddr;
	assign pq_push = req_valid && ~pq_full && ~iq_full;
	assign pq_pop = rsp_en && ~pq_empty;
	assign req_ready = ~pq_full;

	VX_index_buffer #(
		.DATAW	(1 + NUM_REQS + WORD_SIZE + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + TAGW),
		.SIZE	(QUEUE_SIZE)
	) pending_queue (
		.clk			(clk),
		.reset			(reset),
		.write_addr		(pq_waddr),
		.acquire_slot	(pq_push),
		.read_addr		(pq_raddr),
		.write_data		({req_rw,     req_dup_mask,  req_byteen,    req_addr,    req_data,    req_tag}),
		.read_data		({pq_req_rw,  pq_req_mask,   pq_req_byteen, pq_req_addr, pq_req_data, pq_req_tag}),
		.release_addr	(pq_raddr),
		.release_slot	(pq_pop),
		.full			(pq_full),
		.empty			(pq_empty)
	);

	//////////////////////////////////////////////////////////////////

	// Save write address of pending queue into an index queue

	assign iq_push = req_valid && ~iq_full;

	VX_fifo_queue #(
		.DATAW	(QUEUE_ADDRW),
		.SIZE	(QUEUE_SIZE)
	) idx_queue (
		.clk		(clk),
		.reset		(reset),
		.push		(iq_push),
		.pop		(iq_pop),
		.data_in	(pq_waddr),
		.data_out	(iq_raddr),
		.full		(iq_full),
		.empty 		(iq_empty),
		`UNUSED_PIN (alm_full),
		`UNUSED_PIN (alm_empty),
		`UNUSED_PIN (size)
	);

	//////////////////////////////////////////////////////////////////

	// Memory response
	assign mem_rsp_ready = 1'b1;
	assign mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
	assign rsp_rem_mask_n = rsp_rem_mask[pq_waddr] & ~mem_rsp_mask;

	// Evaluate remaining responses
	always @(posedge clk) begin
		if (reset) begin
			rsp_rem_mask <= 0;
		end
		if (pq_push) begin
			rsp_rem_mask[pq_waddr] <= req_dup_mask;
		end
		if (mem_rsp_fire) begin
			rsp_rem_mask[pq_raddr] <= rsp_rem_mask_n;
		end
	end

	// Store response till ready to send
	assign rsp_n = rsp[pq_waddr] | {mem_rsp_valid, mem_rsp_mask, mem_rsp_data};

	always @(posedge clk) begin
		if (reset) begin
			rsp <= 0;
		end
		if(mem_rsp_fire) begin
			rsp[pq_raddr] <= rsp_n;
		end
	end

	assign rsp_en = ((PARTIAL_RESPONSE) ? rsp_n[RSPW-1] : (0 == rsp_rem_mask_n)) && rsp_ready;

	VX_pipe_register #(
		.DATAW	(1 + NUM_REQS + (NUM_REQS * DATAW) + TAGW ),
		.RESETW (1)
	) rsp_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(rsp_en),
		.data_in	({rsp_n, pq_req_tag}),
		.data_out	({rsp_valid, rsp_mask, rsp_data, rsp_tag})
	);

	//////////////////////////////////////////////////////////////////

	// Memory request
	assign mreq_valid 	= pq_req_mask & ~req_sent_mask[pq_raddr];
	assign mreq_rw 	    = {NUM_REQS{pq_req_rw}};
	assign mreq_byteen  = {NUM_REQS{pq_req_byteen}};
	assign mreq_addr 	= pq_req_addr;
	assign mreq_data 	= pq_req_data;
	assign mreq_tag 	= {NUM_REQS{pq_raddr}};

	assign mreq_en = (| mreq_valid);

	VX_pipe_register #(
		.DATAW	(NUM_REQS + NUM_REQS + (NUM_REQS * WORD_SIZE) + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + (NUM_REQS * QUEUE_ADDRW) + NUM_REQS + 1),
		.RESETW (1)
	) req_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(mreq_en),
		.data_in	({mreq_valid,    mreq_rw,    mreq_byteen,    mreq_addr,    mreq_data,    mreq_tag,    pq_req_mask,  req_sent_all & ~iq_empty}),
		.data_out	({mem_req_valid, mem_req_rw, mem_req_byteen, mem_req_addr, mem_req_data, mem_req_tag, mem_req_mask, iq_pop})
	);

	assign mem_req_fire 	= mem_req_valid & mem_req_ready;
	assign req_sent_mask_n 	= req_sent_mask[mem_req_tag[0 +: QUEUE_ADDRW]] | mem_req_fire;
	assign req_sent_all 	= (| mem_req_fire) && (req_sent_mask_n == mem_req_mask);

	always @(posedge clk) begin
		if (reset) begin
			req_sent_mask <= 0;
		end else begin
			if (req_sent_all) begin
				req_sent_mask[mem_req_tag[0 +: QUEUE_ADDRW]] <= 0;
			end else begin
				req_sent_mask[mem_req_tag[0 +: QUEUE_ADDRW]] <= req_sent_mask_n;
			end
		end
	end

	//////////////////////////////////////////////////////////////////

endmodule