// Copyright 2019-2023
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

// A stream elastic buffer operates at full-bandwidth where fire_in and fire_out can happen simultaneously
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
    if (PASSTHRU != 0) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;
        assign data_out  = data_in;

	end else if (OUT_REG != 0) begin : g_out_reg

		reg [DATAW-1:0] data_out_r;
		reg [DATAW-1:0] buffer;
		reg             valid_out_r;
		reg             no_buffer;

		wire fire_in = valid_in && ready_in;
		wire flow_out = ready_out || ~valid_out;

		always @(posedge clk) begin
			if (reset) begin
				valid_out_r <= 0;
				no_buffer  <= 1;
			end else begin
				if (flow_out) begin
					no_buffer <= 1;
				end else if (valid_in) begin
					no_buffer <= 0;
				end
				if (flow_out) begin
					valid_out_r <= valid_in || ~no_buffer;
				end
			end
		end

		always @(posedge clk) begin
			if (fire_in) begin
				buffer <= data_in;
			end
			if (flow_out) begin
				data_out_r <= no_buffer ? data_in : buffer;
			end
		end

		assign ready_in  = no_buffer;
		assign valid_out = valid_out_r;
		assign data_out  = data_out_r;

	end else begin : g_no_out_reg

		reg [1:0][DATAW-1:0] shift_reg;
		reg [1:0] fifo_state, fifo_state_n;

		wire fire_in = valid_in && ready_in;
		wire fire_out = valid_out && ready_out;

		always @(*) begin
			case ({fire_in, fire_out})
			2'b10:	 fifo_state_n = {fifo_state[0], 1'b1}; // 00 -> 01, 01 -> 10
			2'b01:	 fifo_state_n = {1'b0, fifo_state[1]}; // 10 -> 01, 01 -> 00
			default: fifo_state_n = fifo_state;
			endcase
		end

		always @(posedge clk) begin
			if (reset) begin
				fifo_state <= 2'b00;
			end else begin
				fifo_state <= fifo_state_n;
			end
		end

		always @(posedge clk) begin
			if (fire_in) begin
				shift_reg[1] <= shift_reg[0];
				shift_reg[0] <= data_in;
			end
		end

		assign ready_in  = ~fifo_state[1];
		assign valid_out = fifo_state[0];
		assign data_out  = shift_reg[fifo_state[1]];

	end

endmodule
`TRACING_ON
