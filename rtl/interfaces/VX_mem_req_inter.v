interface VX_mem_req_inter ();

	wire[`NT_M1:0][31:0] alu_result;
	wire[2:0]            mem_read; 
	wire[2:0]            mem_write;
	wire[4:0]            rd;
	wire[1:0]            wb;
	wire[4:0]            rs1;
	wire[4:0]            rs2;
	wire[`NT_M1:0][31:0] rd2;
	wire[31:0]           PC_next;
	wire[31:0]           curr_PC;
	wire[31:0]           branch_offset;
	wire[2:0]            branch_type; 
	wire[`NT_M1:0]       valid;
	wire[`NW_M1:0]       warp_num;


	modport snk (
		input alu_result,
		input mem_read, 
		input mem_write,
		input rd,
		input wb,
		input rs1,
		input rs2,
		input rd2,
		input PC_next,
		input curr_PC,
		input branch_offset,
		input branch_type, 
		input valid,
		input warp_num
	);


	modport src (
		output alu_result,
		output mem_read, 
		output mem_write,
		output rd,
		output wb,
		output rs1,
		output rs2,
		output rd2,
		output PC_next,
		output curr_PC,
		output branch_offset,
		output branch_type, 
		output valid,
		output warp_num
	);


endinterface