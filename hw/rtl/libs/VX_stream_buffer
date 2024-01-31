// Copyright 2024 blaise
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A stream elastic buffer operates at full-bandwidth where push and pop can happen simultaneously
// It has the following benefits:
// + full-bandwidth throughput
// + ready_in and ready_out are decoupled
// + data_out can be fully registered
// It has the following limitations:
// - requires two registers for storage

`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_buffer #(
    parameter DATAW    = 1,
	parameter OUT_REG  = 0,
    parameter PASSTHRU = 0
) ( 
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    output wire             ready_in,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
); 
    if (PASSTHRU != 0) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;        
        assign data_out  = data_in;
    end else begin
		if (OUT_REG != 0) begin

			reg [DATAW-1:0] data_out_r;
			reg [DATAW-1:0] buffer;
			reg             valid_out_r;
			reg             use_buffer;
			
			wire push = valid_in && ready_in;
			wire stall_out = valid_out_r && ~ready_out;
			
			always @(posedge clk) begin
				if (reset) begin
					valid_out_r <= 0; 
					use_buffer  <= 0;
				end else begin
					if (ready_out) begin
						use_buffer <= 0;
					end else if (valid_in && valid_out) begin
						use_buffer <= 1;
					end
					if (~stall_out) begin
						valid_out_r <= valid_in || use_buffer;
					end
				end
			end

			always @(posedge clk) begin
				if (push) begin
					buffer <= data_in;
				end
				if (~stall_out) begin
					data_out_r <= use_buffer ? buffer : data_in;
				end
			end

			assign ready_in  = ~use_buffer;
			assign valid_out = valid_out_r;
			assign data_out  = data_out_r;

		end else begin

			reg [1:0][DATAW-1:0] shift_reg;
			reg valid_out_r, ready_in_r, rd_ptr_r;

			wire push = valid_in && ready_in;
			wire pop = valid_out_r && ready_out;

			always @(posedge clk) begin
				if (reset) begin
					valid_out_r <= 0;
					ready_in_r  <= 1;
					rd_ptr_r    <= 1;
				end else begin
					if (push) begin
						if (!pop) begin                            
							ready_in_r  <= rd_ptr_r;
							valid_out_r <= 1;
						end
					end else if (pop) begin
						ready_in_r  <= 1;
						valid_out_r <= rd_ptr_r;
					end
					rd_ptr_r <= rd_ptr_r ^ (push ^ pop);
				end                   
			end

			always @(posedge clk) begin
				if (push) begin
					shift_reg[1] <= shift_reg[0];
					shift_reg[0] <= data_in;
				end
			end

			assign ready_in  = ready_in_r;
			assign valid_out = valid_out_r;
			assign data_out  = shift_reg[rd_ptr_r];
		end
    end

endmodule
`TRACING_ON

