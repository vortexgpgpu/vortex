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

// Elastic stream buffer: full-bandwidth (fire_in and fire_out simultaneously),
// decoupled ready_in/ready_out, fully registered data_out.
// Uses two registers for storage.

`include "VX_platform.vh"

`TRACING_OFF
module VX_stream_buffer #(
    parameter DATAW    = 1,
	parameter OUT_REG  = 0,
    parameter PASSTHRU = 0,
    // ASIC: data holds while no valid enters; 0 for a consumer that reads
    // data_out without valid_out
    parameter DATA_GATE = 1
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
        `UNUSED_PARAM (DATA_GATE)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;
        assign data_out  = data_in;

	end else begin : g_buffer

		reg [DATAW-1:0] data_out_r, buffer_r;
		reg valid_out_r, valid_in_r;

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

			// The skid register is read only once the input side closes, which
			// takes an accepted input; while open it may follow the input bus,
			// so its wide enable does not wait on valid_in.
		`ifdef ASIC
			// Load only what will be read: the skid slot when an input is
			// held back, the output when it becomes valid.
			wire buffer_en = valid_in_r && (valid_in || (DATA_GATE == 0));
			wire data_out_en = flow_out && (valid_in || ~valid_in_r || (DATA_GATE == 0));
			wire buffer_clk, data_out_clk;
			if (DATAW >= 4) begin : g_cg
				VX_clockgate buffer_cg (.clk_in (clk), .en (buffer_en), .clk_out (buffer_clk));
				VX_clockgate data_out_cg (.clk_in (clk), .en (data_out_en), .clk_out (data_out_clk));
			end else begin : g_clk
				assign buffer_clk = clk;
				assign data_out_clk = clk;
			end

			always @(posedge buffer_clk) begin
				if (buffer_en) begin
					buffer_r <= data_in;
				end
			end

			always @(posedge data_out_clk) begin
				if (data_out_en) begin
					data_out_r <= valid_in_r ? data_in : buffer_r;
				end
			end
		`else
			`UNUSED_PARAM (DATA_GATE)
			always @(posedge clk) begin
				if (valid_in_r) begin
					buffer_r <= data_in;
				end
			end

			always @(posedge clk) begin
				if (flow_out) begin
					data_out_r <= valid_in_r ? data_in : buffer_r;
				end
			end
		`endif

			assign data_out = data_out_r;

		end else begin : g_no_out_reg
			`UNUSED_PARAM (DATA_GATE)

			wire fire_in = valid_in && ready_in;

		`ifdef ASIC
			wire fire_clk;
			if (DATAW >= 2) begin : g_cg
				VX_clockgate fire_cg (.clk_in (clk), .en (fire_in), .clk_out (fire_clk));
			end else begin : g_clk
				assign fire_clk = clk;
			end

			always @(posedge fire_clk) begin
				if (fire_in) begin
					data_out_r <= data_in;
				end
			end

			always @(posedge fire_clk) begin
				if (fire_in) begin
					buffer_r <= data_out_r;
				end
			end
		`else
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
		`endif

			assign data_out  = valid_in_r ? data_out_r : buffer_r;

		end

		assign valid_out = valid_out_r;
		assign ready_in  = valid_in_r;

	end

endmodule
`TRACING_ON
