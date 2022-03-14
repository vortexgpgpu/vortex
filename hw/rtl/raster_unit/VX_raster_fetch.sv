`include "VX_raster_define.vh"

// Module for primitive fetch
//  Descrption: Performs strided fetch
//  of primitive data from the buffer

module VX_raster_fetch #(  
    parameter RASTER_SLICE_NUM = 1,
    parameter RASTER_PRIM_REQUEST_SIZE = 5,
) (
    input logic clk,
    input logic reset,
    input logic input_valid,
    input logic [`RASTER_DCR_DATA_BITS-1:0] pbuf_addr,
    input logic ['RASTER_DCR_DATA_BITS-1:0] pbuf_stride,
);

    `STATIC_ASSERT(RASTER_SLICE_NUM > 1, ("invalid parameter"))

    logic [RASTER_SLICE_NUM-1:0][`RASTER_DCR_DATA_BITS-1:0] p_addr;

    for(genvar i = 0; i < RASTER_SLICE_NUM; ++i) begin
        p_addr[i] = pbuf_addr + i*pbuf_stride;
    end

VX_mem_streamer #(
    .NUM_REQS(3),
	.ADDRW(`RASTER_DCR_DATA_BITS),	
	.DATAW(`RASTER_PRIMITIVE_DATA_BITS)
) primitive_streamer (
    .clk(clk),
    .reset(reset),

	// Input request
	.req_valid(input_valid),
	.req_rw(0),
	.req_mask(RASTER_SLICE(1'b1)),
	.req_byteen(WORD_SIZE(1'b1)),
	.req_addr(p_addr),
	`UNUSED_VAR (req_data),
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

endmodule