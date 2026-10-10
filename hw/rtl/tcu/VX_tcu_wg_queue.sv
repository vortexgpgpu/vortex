// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_platform.vh"

// Two-entry skid queue with a registered head, timed exactly like
// VX_stream_buffer #(.OUT_REG(1)), that also exposes the entry behind the
// head. The WGMMA tile buffers evaluate the uop that will be at the head next
// cycle (the next entry when the head fires this cycle), so the TCU's operand
// ready can be registered instead of closing a same-cycle loop through fire.

`TRACING_OFF
module VX_tcu_wg_queue #(
    parameter DATAW = 1
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    output wire             ready_in,
    input  wire [DATAW-1:0] data_in,
    output wire             valid_out,
    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             next_valid,
    output wire [DATAW-1:0] next_data
);
    reg [DATAW-1:0] data_out_r, buffer_r;
    reg valid_out_r, empty_r;

    wire fire_in  = valid_in && ready_in;
    wire flow_out = ready_out || ~valid_out_r;

    always @(posedge clk) begin
        if (reset) begin
            empty_r <= 1'b1;
        end else if (valid_in || flow_out) begin
            empty_r <= flow_out;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_out_r <= 1'b0;
        end else if (flow_out) begin
            valid_out_r <= valid_in || ~empty_r;
        end
    end

    always @(posedge clk) begin
        if (fire_in) begin
            buffer_r <= data_in;
        end
    end

    always @(posedge clk) begin
        if (flow_out) begin
            data_out_r <= empty_r ? data_in : buffer_r;
        end
    end

    assign valid_out  = valid_out_r;
    assign data_out   = data_out_r;
    assign ready_in   = empty_r;
    // The skid register holds the entry behind the head while the queue is full.
    assign next_valid = valid_out_r && ~empty_r;
    assign next_data  = buffer_r;

endmodule
`TRACING_ON
