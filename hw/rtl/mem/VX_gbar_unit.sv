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

`include "VX_define.vh"

module VX_gbar_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    VX_gbar_bus_if.slave gbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // A core sends one request per barrier generation, so counting requests
    // tracks the participants without a per-core mask over the whole device.
    reg [`VX_CFG_NUM_BARRIERS-1:0][CORE_ID_WIDTH-1:0] arrive_counts;
    wire [NB_WIDTH-1:0] req_id = gbar_bus_if.req_data.id;
    wire [CORE_ID_WIDTH-1:0] curr_arrive_count = arrive_counts[req_id];
    `UNUSED_VAR (gbar_bus_if.req_data.core_id)

    reg rsp_valid;
    reg [NB_WIDTH-1:0] rsp_bar_id;

    wire req_fire  = gbar_bus_if.req_valid && gbar_bus_if.req_ready;
    wire rsp_fire  = gbar_bus_if.rsp_valid && gbar_bus_if.rsp_ready;
    wire rsp_stall = gbar_bus_if.rsp_valid && ~gbar_bus_if.rsp_ready;

    always @(posedge clk) begin
        if (reset) begin
            arrive_counts <= '0;
            rsp_valid <= 0;
        end else begin
            if (rsp_fire) begin
                rsp_valid <= 0;
            end
            if (req_fire) begin
                if (curr_arrive_count == gbar_bus_if.req_data.size_m1) begin
                    arrive_counts[req_id] <= '0;
                    rsp_valid  <= 1;
                    rsp_bar_id <= req_id;
                end else begin
                    arrive_counts[req_id] <= curr_arrive_count + CORE_ID_WIDTH'(1);
                end
            end
        end
    end

    // request ack
    assign gbar_bus_if.req_ready = ~rsp_stall;

    // response broadcast
    assign gbar_bus_if.rsp_valid = rsp_valid;
    assign gbar_bus_if.rsp_data.id = rsp_bar_id;

`ifdef DBG_TRACE_GBAR
    always @(posedge clk) begin
        if (req_fire) begin
            `TRACE(2, ("%t: %s arrive: bar_id=%0d, size=%0d, core_id=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1, gbar_bus_if.req_data.core_id))
        end
        if (rsp_fire) begin
            `TRACE(2, ("%t: %s resume: bar_id=%0d\n", $time, INSTANCE_ID, gbar_bus_if.rsp_data.id))
        end
    end
`endif

endmodule
