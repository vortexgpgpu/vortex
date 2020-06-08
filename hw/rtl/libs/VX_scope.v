`include "VX_define.vh"

module VX_scope #(     
	parameter DATAW = 64, 
	parameter BUSW  = 64, 
	parameter SIZE  = 256,
	parameter IDW   = 1
) ( 
	input wire clk,
	input wire reset,
	input wire start,
	input wire [DATAW-1:0] data_in,
	input wire [BUSW-1:0]  bus_in,
	output reg [BUSW-1:0]  bus_out,	
	input wire bus_write,
	input wire bus_read
);
	localparam DELTA_ENABLE = (IDW != 0);

	typedef enum logic[2:0] { 		
		CMD_GET_VALID,
		CMD_GET_DATA,
		CMD_GET_WIDTH,
		CMD_GET_DEPTH,
		CMD_SET_DELAY,
		CMD_SET_DURATION,
		CMD_SET_RESERVED1,		
		CMD_SET_RESERVED2
	} cmd_t;

	typedef enum logic[1:0] { 		
		GET_VALID,
		GET_DATA,
		GET_WIDTH,
		GET_DEPTH
	} cmd_get_t;

	reg [DATAW-1:0] data_store [SIZE-1:0];

	reg [63:0] delta_store [SIZE-1:0];
	reg [IDW-1:0] prev_id;
	reg [63:0] delta;

	reg [`CLOG2(SIZE)-1:0] raddr, waddr, waddr_end;

	reg [`LOG2UP(DATAW)-1:0] read_offset;

	reg start_wait, recording, data_valid, read_delta;

	reg [BUSW-3:0] delay_val, delay_cntr;

	reg [1:0] out_cmd;

	wire [2:0] cmd_type;
	wire [BUSW-4:0] cmd_data;
	assign {cmd_data, cmd_type} = bus_in;

	wire [IDW-1:0] trigger_id = data_in[DATAW-1:DATAW-IDW];

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
				    CMD_GET_DEPTH:     out_cmd  <= $bits(out_cmd)'(cmd_type); 
					CMD_SET_DELAY:    delay_val <= $bits(delay_val)'(cmd_data);
		            CMD_SET_DURATION: waddr_end <= $bits(waddr)'(cmd_data);
				default:;
				endcase				
			end

			if (start) begin		
				waddr <= 0;
				if (0 == delay_val) begin					
					start_wait <= 0;
					recording  <= 1;
					delay_cntr <= 0;	
					delta      <= 0;						
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
					delta      <= 0;
				end 
			end

			if (recording) begin
				if (DELTA_ENABLE) begin
					if (0 == waddr 
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

				if (waddr == waddr_end) begin
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
							if (raddr == waddr_end) begin
								data_valid <= 0;
							end
						end					
					end else begin
						raddr <= raddr + 1;					
						read_delta <= DELTA_ENABLE; 
						if (raddr == waddr_end) begin
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
			GET_DEPTH : bus_out = BUSW'(waddr_end) + BUSW'(1);
			default   : bus_out = read_delta ? (BUSW)'(delta_store[raddr]) : (BUSW)'(data_store[raddr] >> read_offset);
		endcase
	end

`ifdef DBG_PRINT_SCOPE
	always_ff @(posedge clk) begin
		if (bus_read) begin
			$display("%t: scope-read: cmd=%0d, out=0x%0h, addr=%0d, off=%0d", $time, out_cmd, bus_out, raddr, read_offset);
		end
		if (DELTA_ENABLE && recording && (trigger_id != prev_id) && (delta != 0)) begin
			$display("%t: scope-write: waddr=%0d, delta=%0d", $time, waddr, delta);
		end
	end
`endif

endmodule