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
    reg awfired;
    reg wfired;

    wire awfire = awvalid && awready;
    wire wfire = wvalid && wready;

    always @(posedge clk) begin
        if (reset) begin
            awfired <= 0;
            wfired <= 0;
        end else begin
            if (awfire) begin
                awfired <= 1;
            end
            if (wfire) begin
                wfired <= 1;
            end
            if (tx_ack) begin
                awfired <= 0;
                wfired <= 0;
            end
        end
    end

    assign aw_ack = awfired;
    assign w_ack = wfired;

    assign tx_ack = (awfire || awfired) && (wfire || wfired);
    assign tx_rdy = (awready || awfired) && (wready || wfired);

endmodule
`TRACING_ON
