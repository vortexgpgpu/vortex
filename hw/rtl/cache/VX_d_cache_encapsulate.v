`include "VX_define.vh"

`define NUM_WORDS_PER_BLOCK 4

module VX_d_cache_encapsulate (
		clk,
		rst,

		i_p_initial_request,
		i_p_addr,
		i_p_writedata,
		i_p_read_or_write,
		i_p_valid,

		o_p_readdata,
		o_p_readdata_valid,
		o_p_waitrequest,

		o_m_addr,
		o_m_writedata,
		o_m_read_or_write,
		o_m_valid,

		i_m_readdata,
		i_m_ready
);

    parameter NUMBER_BANKS = 8;




    //parameter cache_entry = 9;
    input  wire            clk, rst;

    input  wire        i_p_valid[`NUM_THREADS-1:0];
    input  wire [31:0] i_p_addr[`NUM_THREADS-1:0];
    input  wire        i_p_initial_request;
    input  wire [31:0] i_p_writedata[`NUM_THREADS-1:0];
    input  wire        i_p_read_or_write;

    input  wire [31:0] i_m_readdata[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0];
    input  wire        i_m_ready;

    output reg [31:0]  o_p_readdata[`NUM_THREADS-1:0];
    output reg         o_p_readdata_valid[`NUM_THREADS-1:0] ;
    output reg         o_p_waitrequest;

    output reg [31:0]  o_m_addr;
    output reg         o_m_valid;
    output reg [31:0]  o_m_writedata[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0];
    output reg         o_m_read_or_write;


    // Inter
    wire [`NUM_THREADS-1:0]        i_p_valid_inter;
    wire [`NUM_THREADS-1:0][31:0]  i_p_addr_inter;
    wire [`NUM_THREADS-1:0][31:0]  i_p_writedata_inter;

    reg [`NUM_THREADS-1:0][31:0]   o_p_readdata_inter;
    reg [`NUM_THREADS-1:0]         o_p_readdata_valid_inter;

    reg[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0][31:0]  o_m_writedata_inter;
    wire[NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0][31:0] i_m_readdata_inter;


    genvar curr_thraed, curr_bank, curr_word;
	 generate
    for (curr_thraed = 0; curr_thraed < `NUM_THREADS; curr_thraed = curr_thraed + 1) begin : threads
    	assign i_p_valid_inter[curr_thraed]                = i_p_valid[curr_thraed];
    	assign i_p_addr_inter[curr_thraed]                 = i_p_addr[curr_thraed];
    	assign i_p_writedata_inter[curr_thraed]            = i_p_writedata[curr_thraed];
    	assign o_p_readdata[curr_thraed]       = o_p_readdata_inter[curr_thraed];
    	assign o_p_readdata_valid[curr_thraed] = o_p_readdata_valid_inter[curr_thraed];
    end
	 
    for (curr_bank = 0; curr_bank < NUMBER_BANKS; curr_bank = curr_bank + 1) begin : banks
    	for (curr_word = 0; curr_word < `NUM_WORDS_PER_BLOCK; curr_word = curr_word + 1) begin : words

    		assign o_m_writedata[curr_bank][curr_word] = o_m_writedata_inter[curr_bank][curr_word];
    		assign i_m_readdata_inter[curr_bank][curr_word]        = i_m_readdata[curr_bank][curr_word];

    	end
    end
	 endgenerate

VX_d_cache dcache(
	.clk                (clk),
	.rst                (rst),
	.i_p_valid          (i_p_valid_inter),
	.i_p_addr           (i_p_addr_inter),
	.i_p_initial_request(i_p_initial_request),
	.i_p_writedata      (i_p_writedata_inter),
	.i_p_read_or_write  (i_p_read_or_write),
	.o_p_readdata       (o_p_readdata_inter),
	.o_p_readdata_valid (o_p_readdata_valid_inter),
	.o_p_waitrequest    (o_p_waitrequest),
	.o_m_addr           (o_m_addr),
	.o_m_valid          (o_m_valid),
	.o_m_writedata      (o_m_writedata_inter),
	.o_m_read_or_write  (o_m_read_or_write),
	.i_m_readdata       (i_m_readdata_inter),
	.i_m_ready          (i_m_ready)
	);


endmodule








