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

module VX_kmu_top import VX_gpu_pkg::*;
(
    input wire                              clk,
    input wire                              reset,

    // DCR request interface
    input  wire                             dcr_req_valid,
    input  wire                             dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]     dcr_req_data,

    // Start trigger: pulse high for one cycle to begin CTA dispatch
    input  wire                             start
);

    VX_kmu_bus_if kmu_bus_if();

    // Tie off ready so KMU advances through CTAs immediately
    assign kmu_bus_if.req_ready = 1'b1;

    /* verilator lint_off UNUSED */
    logic kmu_req_valid_unused;
    assign kmu_req_valid_unused = kmu_bus_if.req_valid;
    /* verilator lint_on UNUSED */

    VX_kmu kmu (
        .clk           (clk),
        .reset         (reset),
        .dcr_req_valid (dcr_req_valid),
        .dcr_req_rw    (dcr_req_rw),
        .dcr_req_addr  (dcr_req_addr),
        .dcr_req_data  (dcr_req_data),
        .start         (start),
        .kmu_bus_if    (kmu_bus_if)
    );

endmodule

