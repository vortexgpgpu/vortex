`include "VX_define.vh"

module VX_writeback (
	input wire clk,
	input wire reset,
	// Mem WB info
	VX_inst_mem_wb_if     vx_mem_wb,
	// EXEC Unit WB info
	VX_inst_exec_wb_if    vx_inst_exec_wb,
	// CSR Unit WB info
	VX_csr_wb_if          vx_csr_wb,

	// Actual WB to GPR
	VX_wb_if              vx_writeback_if,
	output wire              no_slot_mem,
	output wire 			 no_slot_exec,
	output wire              no_slot_csr
);

	VX_wb_if              vx_writeback_tempp();

	wire exec_wb = (vx_inst_exec_wb.wb != 0) && (|vx_inst_exec_wb.wb_valid);
	wire mem_wb  = (vx_mem_wb.wb       != 0) && (|vx_mem_wb.wb_valid);
	wire csr_wb  = (vx_csr_wb.wb       != 0) && (|vx_csr_wb.valid);


	assign no_slot_mem = mem_wb && (exec_wb || csr_wb);
	assign no_slot_csr = csr_wb && (exec_wb);
	assign no_slot_exec = 0;

	assign vx_writeback_tempp.write_data  = exec_wb ? vx_inst_exec_wb.alu_result :
											csr_wb  ? vx_csr_wb.csr_result       :
											mem_wb  ? vx_mem_wb.loaded_data      :
											0;


	assign vx_writeback_tempp.wb_valid    = exec_wb ? vx_inst_exec_wb.wb_valid :
											csr_wb  ? vx_csr_wb.valid          :
											mem_wb  ? vx_mem_wb.wb_valid       :
											0;    

	assign vx_writeback_tempp.rd          = exec_wb ? vx_inst_exec_wb.rd :
											csr_wb  ? vx_csr_wb.rd       :
											mem_wb  ? vx_mem_wb.rd       :
											0;

	assign vx_writeback_tempp.wb          = exec_wb ? vx_inst_exec_wb.wb :
											csr_wb  ? vx_csr_wb.wb       :
											mem_wb  ? vx_mem_wb.wb       :
											0;   

	assign vx_writeback_tempp.wb_warp_num = exec_wb ? vx_inst_exec_wb.wb_warp_num :
											csr_wb  ? vx_csr_wb.warp_num          :
											mem_wb  ? vx_mem_wb.wb_warp_num       :
											0;    		



	assign vx_writeback_tempp.wb_pc       = exec_wb ? vx_inst_exec_wb.exec_wb_pc  :
											csr_wb  ? 32'hdeadbeef                :
											mem_wb  ? vx_mem_wb.mem_wb_pc         :
											32'hdeadbeef;


	wire zero = 0;

	wire[`NUM_THREADS-1:0][31:0] use_wb_data;

	VX_generic_register #(.N(39 + `NW_BITS-1 + 1 + `NUM_THREADS*33)) wb_register(
		.clk  (clk),
		.reset(reset),
		.stall(zero),
		.flush(zero),
		.in   ({vx_writeback_tempp.write_data, vx_writeback_tempp.wb_valid, vx_writeback_tempp.rd, vx_writeback_tempp.wb, vx_writeback_tempp.wb_warp_num, vx_writeback_tempp.wb_pc}),
		.out  ({use_wb_data                  , vx_writeback_if.wb_valid, vx_writeback_if.rd, vx_writeback_if.wb, vx_writeback_if.wb_warp_num, vx_writeback_if.wb_pc})
		);


	reg[31:0] last_data_wb /* verilator public */ ;
	always @(posedge clk) begin
		if ((|vx_writeback_if.wb_valid) && (vx_writeback_if.wb != 0) && (vx_writeback_if.rd == 28)) begin
			last_data_wb <= use_wb_data[0];
		end
	end

	assign vx_writeback_if.write_data = use_wb_data;

endmodule : VX_writeback







