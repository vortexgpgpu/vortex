`include "VX_platform.vh"

`TRACING_OFF
module VX_mem_streamer #(
    parameter NUM_REQS = 1
	parameter ADDRW = 32	
	parameter DATAW = 32
	parameter TAGW = 32
	parameter WORD_SIZE = 4
	parameter QUEUE_SIZE = 16
	parameter QUEUE_ADDRW = `CLOG2(QUEUE_SIZE)
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
	input wire 									mem_req_ready,

	// Input response
	input wire 								mem_rsp_valid
	input wire [NUM_REQS-1:0] 				mem_rsp_mask,
	input wire [NUM_REQS-1:0][DATAW-1:0] 	mem_rsp_data,
	input wire [QUEUE_ADDRW-1:0] 			mem_rsp_tag
	output wire 							mem_rsp_ready ;

	// Output response
	output wire 							rsp_valid,
	output wire [NUM_REQS-1:0] 				rsp_mask,
	output wire [NUM_REQS-1:0][DATAW-1:0] 	rsp_data,
	output wire [TAGW-1:0] 					rsp_tag,
	input wire 								rsp_ready
  );

	localparam RSPW = 1 + NUM_REQS + DATAW + TAGW;

	// Detect duplicate addresses
	wire [NUM_REQS-2:0] addr_matches;
	wire req_dup;
	wire [NUM_REQS-1:0] req_dup_mask;

	// Pending queue
	wire [QUEUE_ADDRW-1:0] 	pq_waddr;
	wire 					pq_push;
	wire [QUEUE_ADDRW-1:0] 	pq_raddr;
	wire 					pq_pop;
	wire 					pq_empty;
	wire 					pq_full;

	wire 									pq_req_valid;
	wire 									pq_req_rw, 		reqq_rw;
	wire [NUM_REQS-1:0] 					pq_req_mask, 	reqq_mask;
	wire [WORD_SIZE-1:0] 					pq_req_byteen, 	reqq_byteen;
	wire [NUM_REQS-1:0][31:0] 				pq_req_addr, 	reqq_addr;
	wire [NUM_REQS-1:0][31:0] 				pq_req_data, 	reqq_data;
	wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] 	pq_req_tag, 	reqq_tag;

	// Index queue
	wire [QUEUE_ADDRW-1:0] 	iq_raddr;
	wire 					iq_pop;
	wire 					iq_empty;
	wire 					iq_full;

	// Memory request
	wire [NUM_REQS-1:0] mem_req_fire;
	wire 				mem_req_ready;
	reg  [NUM_REQS-1:0] req_sent_mask;
	wire [NUM_REQS-1:0] req_sent_mask_n;
	wire req_sent_all;

	// Memory response
	wire 								mem_rsp_fire;
	reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0]	rsp_rem_mask;
	wire [NUM_THREADS-1:0] 				rsp_rem_mask_n

	// Response gather
	wire [QUEUE_ADDRW-1:0] rg_waddr;
	wire [QUEUE_ADDRW-1:0] rg_raddr;
	reg  [QUEUE_SIZE-1:0][RSPW-1:0] rg_rsp;
	wire [RSPW-1:0] rg_rsp_n;

	wire req_en;
	wire rsp_en;

	// Detect duplicate addresses
	for(genvar i = 0; i < NUM_REQS-1; i++) begin
		assign addr_matches[i] = (req_addr[i+1] == req_addr[0]) || ~req_valid[i+1]
	end

	assign req_dup = req_valid[0] && (& addr_matches);
	assign req_dup_mask = req_mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};

	// Clear entry in PQ when all responses come back
	assign pq_pop = (mem_rsp_fire && (0 == rsp_rem_mask_n)) || req_rw;

	// Select entry in PQ
	assign pq_raddr = mem_rsp_fire ? mem_rsp_tag : iq_raddr;

	assign pq_push = ~pq_full;

	assign req_ready = pq_push;
				
	// Save incoming requests into a PQ
	VX_index_buffer #(
		.DATAW	(1 + 1 + NUM_REQS + WORD_SIZE + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + TAGW),
		.SIZE	(QUEUE_SIZE)
	) pending_queue (
		.clk			(clk),
		.reset			(reset),
		.write_addr		(pq_waddr),
		.acquire_slot	(pq_push),
		.read_addr		(pq_raddr),
		.write_data		(req_valid,    req_rw,    req_dup_mask, req_byteen,    req_addr,    req_data,    req_tag),
		.read_data		(pq_req_valid, pq_req_rw, pq_req_mask,  pq_req_byteen, pq_req_addr, pq_req_data, pq_req_tag),
		.release_addr	(pq_raddr),
		.release_slot	(pq_pop),
		.full			(pq_full),
		.empty			(pq_empty)
	);

	// Clear entry from IQ when all requests have been sent
	assign iq_pop = req_sent_all;

	// Save PQ addresses into an IQ
	VX_fifo_queue #(
		.DATAW	(QUEUE_ADDRW),
		.SIZE	(QUEUE_SIZE)
	) idx_queue (
		.clk		(clk),
		.reset		(reset),
		.push		(pq_push),
		.pop		(iq_pop),
		.data_in	(pq_waddr),
		.data_out	(iq_raddr),
		.empty		(iq_empty),
		.full		(iq_full),
		`UNUSED_PIN(alm_full),
		`UNUSED_PIN(alm_empty),
		`UNUSED_PIN(size)
	);

	// Memory response
	assign mem_rsp_ready = ~rg_full;
	assign mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
	assign rsp_rem_mask_n = rsp_rem_mask[pq_raddr] & ~mem_rsp_mask;
	assign rg_rsp_n = rg_rsp[pq_raddr];

	always @(posedge clk) begin
		if (pq_push) begin
			rsp_rem_mask[pq_waddr] <= req_dup_mask;
			rg_rsp[pq_waddr] <= 0;
		end
		if (mem_rsp_fire) begin
			rsp_rem_mask[pq_raddr]	<= rsp_rem_mask_n;
			rg_rsp[pq_raddr] <=  rg_rsp_n | {mem_rsp_valid, mem_rsp_mask, mem_rsp_data, pq_req_tag};
		end
	end

	// Stall request pipeline in case of an invalid request
	assign req_en = pq_req_valid;

	// Partial response
	assign rsp_en = PARTIAL_RESPONSE ? mem_rsp_fire : (0 == rsp_rem_mask_n);

	VX_pipe_register #(
		.DATAW	(1 + NUM_REQS + WORD_SIZE + (NUM_REQS * 32) + (NUM_REQS * 32) + QUEUE_ADDRW),
		.RESETW (1)
	) req_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(req_en),
		.data_in	(pq_req_rw, pq_req_mask, pq_req_byteen, pq_req_addr, pq_req_data, pq_raddr),
		.data_out	(reqq_rw,    reqq_mask,    reqq_byteen,    reqq_addr,    reqq_data,    reqq_tag)
	);

	VX_pipe_register #(
		.DATAW	(1 + (NUM_REQS * 32)),
		.RESETW (1)
	) rsp_pipe_reg (
		.clk		(clk),
		.reset		(reset),
		.enable		(rsp_en),
		.data_in	(rg_rsp),
		.data_out	({rsp_valid, rsp_mask, rsp_data, rsp_tag})
	);

	// Memory request
	assign mem_req_valid 	= req_mask && ~req_sent_mask && {NUM_REQS{req_sent_all}};
	assign mem_req_rw 		= {NUM_REQS{req_rw}};
	assign mem_req_addr 	= req_addr[31:2];
	assign mem_req_byteen 	= {NUM_REQS{req_byteen}};
	assign mem_req_data 	= req_data;
	assign mem_req_tag 		= {NUM_REQS{req_tag}};

	assign mem_req_fire = mem_req_valid & mem_req_ready;
    assign mem_req_new = &(mem_req_ready | req_sent_mask | ~req_mask);
	assign req_sent_mask_n = req_sent_mask | mem_req_fire;
	assign req_sent_all = (req_mask == req_sent_mask);

	always @(posedge clk) begin
		if (reset) begin
			req_sent_mask <= 0;
		end else begin
			if(mem_req_new) begin
				req_sent_mask <= 0;
			end
			else begin
				req_sent_mask <= req_sent_mask_n;
			end
		end
	end


endmodule
`TRACING_ON