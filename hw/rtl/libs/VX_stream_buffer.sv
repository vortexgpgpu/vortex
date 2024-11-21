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

// A stream elastic buffer_r operates at full-bandwidth where fire_in and fire_out can happen simultaneously
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

	end else begin : g_buffer

		reg [DATAW-1:0] data_out_r, buffer_r;
		reg valid_out_r, valid_in_r;

		wire fire_in = valid_in && ready_in;
		wire flow_out = ready_out || ~valid_out;

		always @(posedge clk) begin
			if (reset) begin
				valid_in_r <= 1'b1;
			end else if (valid_in || flow_out) begin
				valid_in_r <= flow_out;
			end
		end

		always @(posedge clk) begin
			if (reset) begin
				valid_out_r <= 1'b0;
			end else if (flow_out) begin
				valid_out_r <= valid_in || ~valid_in_r;
			end
		end

		if (OUT_REG != 0) begin : g_out_reg

			always @(posedge clk) begin
				if (fire_in) begin
					buffer_r <= data_in;
				end
			end

			always @(posedge clk) begin
				if (flow_out) begin
					data_out_r <= valid_in_r ? data_in : buffer_r;
				end
			end

			assign data_out = data_out_r;

		end else begin : g_no_out_reg

			always @(posedge clk) begin
				if (fire_in) begin
					data_out_r <= data_in;
				end
			end

			always @(posedge clk) begin
				if (fire_in) begin
					buffer_r <= data_out_r;
				end
			end

			assign data_out  = valid_in_r ? data_out_r : buffer_r;

		end

		assign valid_out = valid_out_r;
		assign ready_in  = valid_in_r;

	end

endmodule
`TRACING_ON
