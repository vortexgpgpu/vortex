`include "VX_define.vh"

module VX_scope #(     
	parameter DATAW  = 64, 
	parameter BUSW   = 64, 
	parameter SIZE   = 16,
	parameter UPDW   = 1,
	parameter DELTAW = 16
) ( 
	input wire clk,
	input wire reset,
	input wire start,
	input wire stop,
	input wire changed,
	input wire [DATAW-1:0] data_in,
	input wire [BUSW-1:0]  bus_in,
	output reg [BUSW-1:0]  bus_out,	
	input wire bus_write,
	input wire bus_read
);
	localparam DELTA_ENABLE = (UPDW != 0);
	localparam MAX_DELTA    = (1**DELTAW)-1;

	typedef enum logic[2:0] { 		
		CMD_GET_VALID,
		CMD_GET_DATA,
		CMD_GET_WIDTH,
		CMD_GET_COUNT,
		CMD_SET_DELAY,
		CMD_SET_STOP,	
		CMD_RESERVED1,
		CMD_RESERVED2
	} cmd_t;

	typedef enum logic[1:0] { 		
		GET_VALID,
		GET_DATA,
		GET_WIDTH,
		GET_COUNT
	} cmd_get_t;

	reg [DATAW-1:0] data_store [SIZE-1:0];
	reg [DELTAW-1:0] delta_store [SIZE-1:0];
	reg [UPDW-1:0] prev_id;
	reg [DELTAW-1:0] delta;

	reg [`CLOG2(SIZE)-1:0] raddr, waddr, waddr_end;

	reg [`LOG2UP(DATAW)-1:0] read_offset;

	reg start_wait, recording, data_valid, read_delta;

	reg [BUSW-3:0] delay_val, delay_cntr;

	reg [1:0] out_cmd;

	wire [2:0] cmd_type;
	wire [BUSW-4:0] cmd_data;
	assign {cmd_data, cmd_type} = bus_in;

	wire [UPDW-1:0] trigger_id = data_in[UPDW-1:0];

	always @(posedge clk) begin
		if (reset) begin
			raddr      	<= 0;	
			waddr      	<= 0;	
			start_wait 	<= 0;
			recording   <= 0;
			delay_cntr 	<= 0;
			read_offset	<= 0;
			data_valid  <= 0;
			out_cmd     <= $bits(out_cmd)'(CMD_GET_VALID);
			delay_val   <= 0;
			waddr_end   <= $bits(waddr)'(SIZE-1);
			delta       <= 0;
			read_delta  <= 0;
		end else begin

			if (bus_write) begin
				case (cmd_type)
					CMD_GET_VALID, 
					CMD_GET_DATA, 
					CMD_GET_WIDTH, 
				    CMD_GET_COUNT:    out_cmd <= $bits(out_cmd)'(cmd_type); 
					CMD_SET_DELAY:  delay_val <= $bits(delay_val)'(cmd_data);
		            CMD_SET_STOP:	waddr_end <= $bits(waddr)'(cmd_data);
				default:;
				endcase				
			end

			if (start) begin		
				waddr <= 0;
				if (0 == delay_val) begin					
					start_wait <= 0;
					recording  <= 1;
					delay_cntr <= 0;	
					delta      <= MAX_DELTA;						
				end else begin
					start_wait <= 1;
					recording  <= 0;
					delay_cntr <= delay_val;							
				end
			end

			if (start_wait) begin				
				delay_cntr <= delay_cntr - 1;
				if (1 == delay_cntr) begin				
					start_wait <= 0;
					recording  <= 1;
					delta      <= MAX_DELTA;
				end 
			end

			if (recording) begin
				if (DELTA_ENABLE) begin
					if (changed
					 || (delta == MAX_DELTA)
					 || (trigger_id != prev_id)) begin
						data_store[waddr]  <= data_in;
						delta_store[waddr] <= delta;
						waddr <= waddr + 1;
						delta <= 0;
					end else begin
						delta <= delta + 1;
					end
					prev_id <= trigger_id;
				end else begin
					data_store[waddr] <= data_in;
					waddr <= waddr + 1;
				end

				if (stop 
				 || (waddr >= waddr_end)) begin
					waddr      <= waddr;  // keep last written address
					recording  <= 0;
					data_valid <= 1;
					read_delta <= DELTA_ENABLE;					
				end
			end

			if (bus_read 			 
			 && (out_cmd == GET_DATA)
			 && data_valid)  begin
				if (read_delta) begin
					read_delta <= 0; 
				end else begin
					if (DATAW > BUSW) begin
						if (read_offset < $bits(read_offset)'(DATAW-BUSW)) begin
							read_offset <= read_offset + $bits(read_offset)'(BUSW);
						end else begin
							raddr       <= raddr + 1;
							read_offset <= 0;							
							read_delta  <= DELTA_ENABLE; 
							if (raddr == waddr) begin
								data_valid <= 0;
							end
						end					
					end else begin
						raddr <= raddr + 1;					
						read_delta <= DELTA_ENABLE; 
						if (raddr == waddr) begin
							data_valid <= 0;
						end
					end
				end				
			end
		end		
	end

	always @(*) begin
		case (out_cmd)
			GET_VALID : bus_out = BUSW'(data_valid);
			GET_WIDTH : bus_out = BUSW'(DATAW);
			GET_COUNT : bus_out = BUSW'(waddr) + BUSW'(1);
			default   : bus_out = read_delta ? BUSW'(delta_store[raddr]) : BUSW'(data_store[raddr] >> read_offset);
		endcase
	end

`ifdef DBG_PRINT_SCOPE
	always_ff @(posedge clk) begin
		if (bus_read) begin
			$display("%t: scope-read: cmd=%0d, out=0x%0h, addr=%0d, off=%0d", $time, out_cmd, bus_out, raddr, read_offset);
		end
		if (bus_write) begin
			$display("%t: scope-write: cmd=%0d, value=%0d", $time, cmd_type, cmd_data);
		end
	end
`endif

endmodule