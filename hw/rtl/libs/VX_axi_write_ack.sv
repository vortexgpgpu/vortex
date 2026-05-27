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

`TRACING_OFF
module VX_axi_write_ack (
    input wire  clk,
    input wire  reset,
    input wire  awvalid,
    input wire  awready,
    input wire  wvalid,
    input wire  wready,
    output wire aw_ack,
    output wire w_ack,
    output wire tx_ack,
    output wire tx_rdy
);
    reg aw_fired;
    reg w_fired;

    wire aw_fire = awvalid && awready;
    wire w_fire = wvalid && wready;

    always @(posedge clk) begin
        if (reset) begin
            aw_fired <= 0;
            w_fired <= 0;
        end else begin
            if (aw_fire) begin
                aw_fired <= 1;
            end
            if (w_fire) begin
                w_fired <= 1;
            end
            if (tx_ack) begin
                aw_fired <= 0;
                w_fired <= 0;
            end
        end
    end

    assign aw_ack = aw_fired;
    assign w_ack = w_fired;

    assign tx_ack = (aw_fire || aw_fired) && (w_fire || w_fired);
    assign tx_rdy = (awready || aw_fired) && (wready || w_fired);

endmodule
`TRACING_ON
