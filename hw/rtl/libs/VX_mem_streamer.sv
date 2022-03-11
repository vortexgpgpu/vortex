`include "VX_platform.vh"

module VX_mem_streamer #(
    parameter NUM_REQS = 4,
	parameter ADDRW = 32,	
	parameter DATAW = 32,
	parameter TAGW = 32,
	parameter WORD_SIZE = 4,
	parameter QUEUE_SIZE = 16,
	parameter QUEUE_ADDRW = `CLOG2(QUEUE_SIZE),
	parameter PARTIAL_RESPONSE = 0
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

	localparam RSPW = QUEUE_ADDRW + NUM_REQS + (NUM_REQS * DATAW) + 1;

	// Detect duplicate addresses
	wire [NUM_REQS-2:0] addr_matches;
	wire req_dup;
	wire [NUM_REQS-1:0] req_dup_mask;

	// Pending queue
	wire 								pq_rw;
	wire [NUM_REQS-1:0] 				pq_mask;
	wire [WORD_SIZE-1:0] 				pq_byteen;
	wire [NUM_REQS-1:0][ADDRW-1:0] 		pq_addr;
	wire [NUM_REQS-1:0][DATAW-1:0] 		pq_data;
	wire [TAGW-1:0] 					pq_tag;
	wire [QUEUE_ADDRW-1:0] 				mem_tag;

	wire 					sreq_push;
	wire [QUEUE_ADDRW-1:0] 	sreq_waddr;
	wire [QUEUE_ADDRW-1:0]	sreq_raddr;
	wire [QUEUE_ADDRW-1:0]	sreq_pop_addr;
	wire 					sreq_pop;
	reg  					sreq_pop_r;
	wire 					sreq_full;
	wire 					sreq_empty;

	wire 					sidx_push;
	wire 					sidx_pop;
	reg  					sidx_pop_r;
	wire [QUEUE_ADDRW-1:0] 	sidx_din;
	wire [QUEUE_ADDRW-1:0] 	sidx_dout;
	wire 					sidx_full;
	wire 					sidx_empty;

	// Memory request
	wire 									mreq_en;	
	wire [NUM_REQS-1:0] 					mreq_valid;
	wire [NUM_REQS-1:0] 					mreq_rw;
	wire [NUM_REQS-1:0][WORD_SIZE-1:0] 		mreq_byteen;
	wire [NUM_REQS-1:0][ADDRW-1:0] 			mreq_addr;
	wire [NUM_REQS-1:0][DATAW-1:0] 			mreq_data;
	wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] 	mreq_tag;

	wire [NUM_REQS-1:0] 					mem_req_fire;
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

	assign sreq_push = req_valid && !sreq_full && !sidx_full;
	assign sreq_raddr = mem_rsp_fire ? mem_rsp_tag : sidx_dout;
	assign req_ready = !sreq_full && !sidx_full;

	VX_index_buffer #(
		.DATAW	(1 + NUM_REQS + WORD_SIZE + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + TAGW + QUEUE_ADDRW),
		.SIZE	(QUEUE_SIZE)
	) store_req (
		.clk			(clk),
		.reset			(reset),
		.write_addr		(sreq_waddr),
		.acquire_slot	(sreq_push),
		.read_addr		(sreq_raddr),
		.write_data		({req_rw, req_dup_mask, req_byteen, req_addr, req_data, req_tag, sreq_waddr}),
		.read_data		({pq_rw,  pq_mask,      pq_byteen,  pq_addr,  pq_data,  pq_tag,  mem_tag}),
		.release_addr	(sreq_pop_addr),
		.release_slot	(sreq_pop),
		.full			(sreq_full),
		.empty			(sreq_empty)
	);

	wire sidx_push = sreq_push;
	wire sidx_din = sreq_waddr;

	VX_fifo_queue #(
		.DATAW	(QUEUE_ADDRW),
		.SIZE	(QUEUE_SIZE)
	) store_idx (
		.clk		(clk),
		.reset		(reset),
		.push		(sidx_push),
		.pop		(sidx_pop),
		.data_in	(sidx_din),
		.data_out	(sidx_dout),
		.full		(sidx_full),
		.empty 		(sidx_empty),
		`UNUSED_PIN (alm_full),
		`UNUSED_PIN (alm_empty),
		`UNUSED_PIN (size)
	);

	//////////////////////////////////////////////////////////////////

	// Memory response
	assign mem_rsp_ready = 1'b1;
	assign mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
	assign rsp_rem_mask_n = rsp_rem_mask[sreq_raddr] & ~mem_rsp_mask;

	// Evaluate remaining responses
	always @(posedge clk) begin
		if (reset) begin
			rsp_rem_mask <= 0;
		end
		if (sreq_push) begin
			rsp_rem_mask[sreq_waddr] <= req_dup_mask;
		end
		if (mem_rsp_fire) begin
			rsp_rem_mask[sreq_raddr] <= rsp_rem_mask_n;
		end
	end

	// Store response till ready to send
	assign rsp_n = rsp[sreq_raddr] | {mem_rsp_tag, mem_rsp_mask, mem_rsp_data, mem_rsp_valid};

	always @(posedge clk) begin
		if (reset) begin
			rsp <= 0;
		end
		if(mem_rsp_fire) begin
			rsp[sreq_raddr] <= rsp_n;
		end
	end

	assign rsp_en = ((PARTIAL_RESPONSE) ? rsp_n[0] : (0 == rsp_rem_mask_n)) && rsp_ready;

	// Assert sreq_pop for only one clk cycle
	assign sreq_pop = sreq_pop_r;
	always @(posedge clk) begin
		if (reset)
			sreq_pop_r <= 1'b0;
		else begin
			if (sreq_pop)
				sreq_pop_r <= 1'b0;
			else
				sreq_pop_r <= (0 == rsp_rem_mask_n) && mem_rsp_fire && ~sreq_empty;
		end
	end

	wire is_send;

	VX_pipe_register #(
		.DATAW	(QUEUE_ADDRW + NUM_REQS + (NUM_REQS * DATAW) + 1 + TAGW + 1),
		.RESETW (1)
	) rsp_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(rsp_en),
		.data_in	({rsp_n,                                        pq_tag, mem_rsp_fire}),
		.data_out	({sreq_pop_addr, rsp_mask, rsp_data, rsp_valid, rsp_tag, is_send})
	);

	//////////////////////////////////////////////////////////////////

	// Memory request
	assign mreq_valid 	= pq_mask & ~req_sent_mask[mem_tag] & {NUM_REQS{!is_send}};
	assign mreq_rw 	    = {NUM_REQS{pq_rw}};
	assign mreq_byteen  = {NUM_REQS{pq_byteen}};
	assign mreq_addr 	= pq_addr;
	assign mreq_data 	= pq_data;
	assign mreq_tag 	= {NUM_REQS{mem_tag}};
	assign mreq_en 		= (| mreq_valid);

	assign mem_req_fire 	= mreq_valid & mem_req_ready;
	assign req_sent_mask_n 	= req_sent_mask[mem_tag] | mem_req_fire;
	assign req_sent_all 	= (req_sent_mask_n == pq_mask);

	always @(posedge clk) begin
		if (reset)
			req_sent_mask <= 0;
		else begin
			if (req_sent_all)
				req_sent_mask[mem_tag] <= 0;
			else
				req_sent_mask[mem_tag] <= req_sent_mask_n;
		end
	end

	// Assert sidx_pop for only one clk cycle
	assign sidx_pop = sidx_pop_r;
	always @(posedge clk) begin
		if (reset)
			sidx_pop_r <= 1'b0;
		else begin
			if (sidx_pop)
				sidx_pop_r <= 1'b0;
			else
				sidx_pop_r <= (| mem_req_fire) && (req_sent_all) &&  ~sidx_empty;
		end
	end

	VX_pipe_register #(
		.DATAW	(NUM_REQS + NUM_REQS + (NUM_REQS * WORD_SIZE) + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + (NUM_REQS * QUEUE_ADDRW)),
		.RESETW (1)
	) req_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(mreq_en),
		.data_in	({mreq_valid,    mreq_rw,    mreq_byteen,    mreq_addr,    mreq_data,    mreq_tag}),
		.data_out	({mem_req_valid, mem_req_rw, mem_req_byteen, mem_req_addr, mem_req_data, mem_req_tag})
	);

	//////////////////////////////////////////////////////////////////

endmodule