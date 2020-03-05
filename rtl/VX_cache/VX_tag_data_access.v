`include "VX_cache_config.v"

module VX_tag_data_access (
	input  wire                            clk,
	input  wire                            reset,
	input  wire                            stall,

	// Initial Reading
	input  wire[31:0]                      readaddr_st10, 

	// Write/Read Logic
	input  wire                            valid_req_st1e,
	input  wire                            writefill_st1e,
	input  wire[31:0]                      writeaddr_st1e,
	input  wire[31:0]                      writeword_st1e,
	input  wire[`BANK_LINE_SIZE_RNG][31:0] writedata_st1e,
	input  wire[2:0]                       mem_write_st1e,
	input  wire[2:0]                       mem_read_st1e, 

	output wire[31:0]                      readword_st1e,
	output wire[`BANK_LINE_SIZE_RNG][31:0] readdata_st1e,
	output wire[`TAG_SELECT_SIZE_RNG]      readtag_st1e,
	output wire                            miss_st1e,
	output wire                            dirty_st1e,
	output wire                            fill_saw_dirty_st1e
	
);


	reg[`BANK_LINE_SIZE_RNG][31:0] readdata_st[`STAGE_1_CYCLES-1:0];

	reg                            read_valid_st1c[`STAGE_1_CYCLES-1:0];
	reg                            read_dirty_st1c[`STAGE_1_CYCLES-1:0];
	reg[`TAG_SELECT_SIZE_RNG]      read_tag_st1c  [`STAGE_1_CYCLES-1:0];
	reg[`BANK_LINE_SIZE_RNG][31:0] read_data_st1c [`STAGE_1_CYCLES-1:0];


	wire                            qual_read_valid_st1;
	wire                            qual_read_dirty_st1;
	wire[`TAG_SELECT_SIZE_RNG]      qual_read_tag_st1;
	wire[`BANK_LINE_SIZE_RNG][31:0] qual_read_data_st1;

	wire                            use_read_valid_st1e;
	wire                            use_read_dirty_st1e;
	wire[`TAG_SELECT_SIZE_RNG]      use_read_tag_st1e;
	wire[`BANK_LINE_SIZE_RNG][31:0] use_read_data_st1e;
	wire[`BANK_LINE_SIZE_RNG][3:0]  use_write_enable;
	wire[`BANK_LINE_SIZE_RNG][31:0] use_write_data;


	wire fill_sent;
	VX_tag_data_structure VX_tag_data_structure(
		.clk         (clk),
		.reset       (reset),

		.read_addr   (readaddr_st10),
		.read_valid  (qual_read_valid_st1),
		.read_dirty  (qual_read_dirty_st1),
		.read_tag    (qual_read_tag_st1),
		.read_data   (qual_read_data_st1),

		.write_enable(use_write_enable),
		.write_fill  (writefill_st1e),
		.write_addr  (writeaddr_st1e),
		.write_data  (use_write_data),
		.fill_sent   (fill_sent)
		);

	VX_generic_register #(.N( 1 + 1 + `TAG_SELECT_NUM_BITS + (`BANK_LINE_SIZE_WORDS*32) )) s0_1_c0 (
		.clk  (clk),
		.reset(reset),
		.stall(stall),
		.flush(0),
		.in   ({qual_read_valid_st1, qual_read_dirty_st1, qual_read_tag_st1, qual_read_data_st1}),
		.out  ({read_valid_st1c[0] , read_dirty_st1c[0] , read_tag_st1c[0] , read_data_st1c[0]})
	);

	genvar curr_stage;
	generate
		for (curr_stage = 1; curr_stage < `STAGE_1_CYCLES; curr_stage = curr_stage + 1) begin
			VX_generic_register #(.N( 1 + 1 + `TAG_SELECT_NUM_BITS + (`BANK_LINE_SIZE_WORDS*32) )) s0_1_cc (
				.clk  (clk),
				.reset(reset),
				.stall(stall),
				.flush(0),
				.in   ({read_valid_st1c[curr_stage-1] , read_dirty_st1c[curr_stage-1] , read_tag_st1c[curr_stage-1] , read_data_st1c[curr_stage-1]}),
				.out  ({read_valid_st1c[curr_stage]   , read_dirty_st1c[curr_stage]   , read_tag_st1c[curr_stage]   , read_data_st1c[curr_stage]  })
			);
		end
	endgenerate


	assign use_read_valid_st1e = read_valid_st1c[`STAGE_1_CYCLES-1];
	assign use_read_dirty_st1e = read_dirty_st1c[`STAGE_1_CYCLES-1];
	assign use_read_tag_st1e   = read_tag_st1c  [`STAGE_1_CYCLES-1];

	genvar curr_w;
	for (curr_w = 0; curr_w < `BANK_LINE_SIZE_WORDS; curr_w = curr_w+1) assign use_read_data_st1e[curr_w][31:0]  = read_data_st1c[`STAGE_1_CYCLES-1][curr_w][31:0];
	// assign use_read_data_st1e  = read_data_st1c [`STAGE_1_CYCLES-1];

/////////////////////// LOAD LOGIC ///////////////////

	wire[`OFFSET_SIZE_RNG]      byte_select  = writeaddr_st1e[`OFFSET_ADDR_RNG];
	wire[`WORD_SELECT_SIZE_RNG] block_offset = writeaddr_st1e[`WORD_SELECT_ADDR_RNG];

    wire lw  = (mem_read_st1e == `LW_MEM_READ);
    wire lb  = (mem_read_st1e == `LB_MEM_READ);
    wire lh  = (mem_read_st1e == `LH_MEM_READ);
    wire lhu = (mem_read_st1e == `LHU_MEM_READ);
    wire lbu = (mem_read_st1e == `LBU_MEM_READ);

    wire b0 = (byte_select == 0);
    wire b1 = (byte_select == 1);
    wire b2 = (byte_select == 2);
    wire b3 = (byte_select == 3);

    wire[31:0] w0 = read_data_st1c[`STAGE_1_CYCLES-1][0][31:0];
    wire[31:0] w1 = read_data_st1c[`STAGE_1_CYCLES-1][1][31:0];
    wire[31:0] w2 = read_data_st1c[`STAGE_1_CYCLES-1][2][31:0];
    wire[31:0] w3 = read_data_st1c[`STAGE_1_CYCLES-1][3][31:0];

    wire[31:0] data_unmod  = read_data_st1c[`STAGE_1_CYCLES-1][block_offset][31:0];

    wire[31:0] data_unQual = (b0 || lw) ? (data_unmod) :
                             b1 ? (data_unmod >> 8)    :
                             b2 ? (data_unmod >> 16)   :
                             (data_unmod >> 24);


    wire[31:0] lb_data     = (data_unQual[7] ) ? (data_unQual | 32'hFFFFFF00) : (data_unQual & 32'hFF);
    wire[31:0] lh_data     = (data_unQual[15]) ? (data_unQual | 32'hFFFF0000) : (data_unQual & 32'hFFFF);
    wire[31:0] lbu_data    = (data_unQual & 32'hFF);
    wire[31:0] lhu_data    = (data_unQual & 32'hFFFF);
    wire[31:0] lw_data     = (data_unQual);


    wire[31:0] sw_data     = writeword_st1e;

    wire[31:0] sb_data     = b1 ? {{16{1'b0}}, writeword_st1e[7:0], { 8{1'b0}}} :
                             b2 ? {{ 8{1'b0}}, writeword_st1e[7:0], {16{1'b0}}} :
                             b3 ? {{ 0{1'b0}}, writeword_st1e[7:0], {24{1'b0}}} :
                             writeword_st1e;

    wire[31:0] sh_data     = b2 ? {writeword_st1e[15:0], {16{1'b0}}} : writeword_st1e;



    wire[31:0] use_write_dat   = sb ? sb_data :
                                 sh ? sh_data :
                                 sw_data;


    wire[31:0] data_Qual   = lb  ? lb_data  :
                             lh  ? lh_data  :
                             lhu ? lhu_data :
                             lbu ? lbu_data :
                             lw_data;

/////////////////////// STORE LOGIC ///////////////////

    wire sw  = (mem_write_st1e == `SW_MEM_WRITE);
    wire sb  = (mem_write_st1e == `SB_MEM_WRITE);
    wire sh =  (mem_write_st1e == `SH_MEM_WRITE);

    wire[3:0] sb_mask = (b0 ? 4'b0001 : (b1 ? 4'b0010 : (b2 ? 4'b0100 : 4'b1000)));
    wire[3:0] sh_mask = (b0 ? 4'b0011 : 4'b1100);

    wire should_write = (sw || sb || sh) && valid_req_st1e && use_read_valid_st1e && !miss_st1e;
    wire force_write  = writefill_st1e && valid_req_st1e && miss_st1e;

    wire[`BANK_LINE_SIZE_RNG][3:0]  we;
    wire[`BANK_LINE_SIZE_RNG][31:0] data_write;
	genvar g; 
	generate
		for (g = 0; g < `BANK_LINE_SIZE_WORDS; g = g + 1) begin : write_enables
		    wire normal_write = (block_offset == g) && should_write;

		    assign we[g]      = (force_write)        ? 4'b1111  : 
		                        (normal_write && sw) ? 4'b1111  :
		                        (normal_write && sb) ? sb_mask  :
		                        (normal_write && sh) ? sh_mask  :
		                        4'b0000;

		    assign data_write[g] = force_write ? writedata_st1e[g] : use_write_dat;
		end
	endgenerate

	assign use_write_enable = we;
	assign use_write_data   = data_write;

///////////////////////

	assign readword_st1e       = data_Qual;
	assign miss_st1e           = (valid_req_st1e && !use_read_valid_st1e) || (valid_req_st1e && use_read_valid_st1e && !writefill_st1e && (writeaddr_st1e[`TAG_SELECT_ADDR_RNG] != use_read_tag_st1e));
	assign dirty_st1e          = valid_req_st1e && use_read_valid_st1e && use_read_dirty_st1e;
	assign readdata_st1e       = use_read_data_st1e;
	assign readtag_st1e        = use_read_tag_st1e;
	assign fill_sent           = miss_st1e;
	assign fill_saw_dirty_st1e = force_write && dirty_st1e;

endmodule




