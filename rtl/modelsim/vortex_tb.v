// `include "../VX_define.v"
// `include "../Vortex.v"

`timescale 1ns/1ps

module vortex_tb (
	
);

	reg           clk;
	reg           reset;
	reg[31:0] icache_response_instruction;
	reg[31:0] icache_request_pc_address;
	// IO
	reg        io_valid;
	reg[31:0]  io_data;
	// Req
    reg [31:0]  o_m_read_addr;
    reg [31:0]  o_m_evict_addr;
    reg         o_m_valid;
    reg [31:0]  o_m_writedata[8 - 1:0][4-1:0];
    reg         o_m_read_or_write;

    // Rsp
    reg [31:0] i_m_readdata[8 - 1:0][4-1:0];
    reg        i_m_ready;
	reg        out_ebreak;

	integer temp;
	initial begin
		// $dumpfile("vortex2.vcd");
		$dumpvars(0, vortex);
		for (temp = 0; temp < 10; temp=temp+1)
		begin

			if (temp == 0) begin
				icache_response_instruction = 32'h00000513;
				$display("Cycle 1");
			end if (temp == 1) begin
				$display("Cycle 2",);
				icache_response_instruction = 32'h0005006b;
			end
		end
	end

	Vortex vortex(
		.clk                        (clk),
		.reset                      (reset),
		.icache_response_instruction(icache_response_instruction),
		.icache_request_pc_address  (icache_request_pc_address),
		.io_valid                   (io_valid),
		.io_data                    (io_data),
		.o_m_read_addr              (o_m_read_addr),
		.o_m_evict_addr             (o_m_evict_addr),
		.o_m_valid                  (o_m_valid),
		.o_m_writedata              (o_m_writedata),
		.o_m_read_or_write          (o_m_read_or_write),
		.i_m_readdata               (i_m_readdata),
		.i_m_ready                  (i_m_ready),
		.out_ebreak                 (out_ebreak)
		);


	always @(clk) #5 clk <= ~clk;

endmodule