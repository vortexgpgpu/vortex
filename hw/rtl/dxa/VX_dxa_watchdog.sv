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

// DXA progress watchdog: fires RUNTIME_ASSERT if no progress for STALL_TIMEOUT cycles.

`include "VX_define.vh"

module VX_dxa_watchdog import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // Progress signals (any high = progress).
    input wire transfer_active,
    input wire gmem_req_fire,
    input wire gmem_rsp_valid,
    input wire smem_req_fire,

    // Context for error message.
    input wire [NC_WIDTH-1:0]    active_core_id,
    input wire [NW_WIDTH-1:0]    active_wid,
    input wire [BAR_ADDR_W-1:0]  active_bar_addr
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (active_core_id)
    `UNUSED_VAR (active_wid)
    `UNUSED_VAR (active_bar_addr)

    reg [31:0] stall_ctr_r;

    wire no_progress = transfer_active
                    && ~gmem_req_fire
                    && ~gmem_rsp_valid
                    && ~smem_req_fire;

    always @(posedge clk) begin
        if (reset || ~no_progress) begin
            stall_ctr_r <= '0;
        end else begin
            stall_ctr_r <= stall_ctr_r + 32'd1;
        end
    end

    `RUNTIME_ASSERT(stall_ctr_r < STALL_TIMEOUT, (
        "*** %s worker no-progress: core=%0d, wid=%0d, bar=%0d",
        INSTANCE_ID, active_core_id, active_wid, active_bar_addr))

endmodule
