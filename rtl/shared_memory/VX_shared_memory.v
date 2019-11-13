`include "../VX_define.v"

module VX_shared_memory
	#(
		parameter NB            = 4,
		parameter BITS_PER_BANK = 3
	)
	(
	//INPUTS
	input wire clk,
	input wire reset,
	input wire[`NT_M1:0] in_valid,
	input wire[`NT_M1:0][31:0] in_address,
	input wire[`NT_M1:0][31:0] in_data,
	input wire[2:0] mem_read,
	input wire[2:0] mem_write,
	//OUTPUTS
	output wire[`NT_M1:0] out_valid,
	output wire[`NT_M1:0][31:0] out_data,
	output wire stall
	);

reg[NB:0][31:0] temp_address;
reg[NB:0][31:0] temp_in_data;
reg[NB:0] temp_in_valid;

reg[`NT_M1:0] temp_out_valid;
reg[`NT_M1:0][31:0] temp_out_data;

reg [NB:0][6:0] block_addr;
reg [NB:0][3:0][31:0] block_wdata;
reg [NB:0][3:0][31:0] block_rdata;
reg [NB:0][1:0] block_we;

wire send_data;

reg[NB:0][1:0] req_num;

wire [`NT_M1:0] orig_in_valid;


genvar f;
	generate
		for(f = 0; f < `NT; f = f+1) begin
			assign orig_in_valid[f] = in_valid[f];
		end

		assign out_valid  = send_data ? temp_out_valid : 0;
		assign out_data   = send_data ? temp_out_data : 0;
	endgenerate


VX_priority_encoder_sm #(.NB(NB), .BITS_PER_BANK(BITS_PER_BANK)) vx_priority_encoder_sm(
	.clk(clk),
	.reset(reset),
	.in_valid(orig_in_valid),
	.in_address(in_address),
	.in_data(in_data),

	.out_valid(temp_in_valid),
	.out_address(temp_address),
	.out_data(temp_in_data),

	.req_num(req_num),
	.stall(stall),
	.send_data(send_data)
	);


genvar j;
integer i;
generate
for(j=0; j<= NB; j=j+1) begin : sm_mem_block

	wire shm_write = (mem_write != `NO_MEM_WRITE) && temp_in_valid[j];

	VX_shared_memory_block vx_shared_memory_block(
		.clk      (clk),
		.reset    (reset),
		.addr     (block_addr[j]),
		.wdata    (block_wdata[j]),
		.we       (block_we[j]),
		.shm_write(shm_write),
		.data_out (block_rdata[j])
	);
end	


always @(*) begin
		block_addr = 0;
		block_we = 0;
		block_wdata = 0;
	for(i = 0; i <= NB; i = i+1) begin
		if(temp_in_valid[i] == 1'b1) begin
			//1. Check if the request is actually to the shared memory
			if((temp_address[i][31:24]) == 8'hFF) begin
			// STORES
				if(mem_write != `NO_MEM_WRITE) begin 
					if(mem_write == `SB_MEM_WRITE) begin
						//TODO
					end
					else if(mem_write == `SH_MEM_WRITE) begin
						//TODO
					end
					else if(mem_write == `SW_MEM_WRITE) begin
						block_addr[i] = temp_address[i][13:7];
						block_we[i] = temp_address[i][6:5];
						block_wdata[i][temp_address[i][6:5]] = temp_in_data[i];
					end
				end
				//LOADS
				else if(mem_read != `NO_MEM_READ) begin 
					if(mem_read == `LB_MEM_READ) begin
						//TODO
					end
					else if (mem_read == `LH_MEM_READ)
					begin
						//TODO
					end
					else if (mem_read == `LW_MEM_READ)
					begin
						block_addr[i] = temp_address[i][13:7];
						temp_out_data[req_num[i]] = block_rdata[i][temp_address[i][6:5]];
						temp_out_valid[req_num[i]] = 1'b1;
					end
					else if (mem_read == `LBU_MEM_READ)
					begin
						//TODO
					end
					else if (mem_read == `LHU_MEM_READ)
					begin
						//TODO
					end
				end
			end
		end
	end
end

endgenerate


endmodule