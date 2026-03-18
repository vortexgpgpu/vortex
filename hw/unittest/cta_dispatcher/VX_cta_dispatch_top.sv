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

module VX_cta_dispatch_top import VX_gpu_pkg::*;
(
    input wire                      clk,
    input wire                      reset,

    // KMU bus input (struct fields flattened for C++ testbench driving)
    input  wire                     task_in_req_valid,
    output wire                     task_in_req_ready,
    input  wire [31:0]              num_warps,
    input  wire [`XLEN-1:0]         start_pc,
    input  wire [`XLEN-1:0]         input_param,
    input  wire [31:0]              input_cta_x,
    input  wire [31:0]              input_cta_y,
    input  wire [31:0]              input_cta_z,
    input  wire [31:0]              input_cta_id,
    input  wire [`NUM_THREADS-1:0]  input_remain_mask,

    input  wire [`NUM_WARPS-1:0]    active_warps,

    // Scheduler dispatch outputs
    output wire                     cta_sched_fire,
    output wire [NW_WIDTH-1:0]      cta_sched_wid,
    output wire [PC_BITS-1:0]       cta_sched_PC,
    output wire [`NUM_THREADS-1:0]  cta_sched_tmask,

    // CSR outputs
    output wire                     cta_csr_valid,
    output wire [NW_WIDTH-1:0]      cta_csr_wid
);

    VX_kmu_bus_if kmu_bus();

    assign kmu_bus.req_valid        = task_in_req_valid;
    assign task_in_req_ready        = kmu_bus.req_ready;
    assign kmu_bus.req_data.PC      = start_pc;
    assign kmu_bus.req_data.num_warps = num_warps;
    assign kmu_bus.req_data.tmask   = input_remain_mask;
    assign kmu_bus.req_data.cta_id  = input_cta_id;
    assign kmu_bus.req_data.cta_x   = input_cta_x;
    assign kmu_bus.req_data.cta_y   = input_cta_y;
    assign kmu_bus.req_data.cta_z   = input_cta_z;
    assign kmu_bus.req_data.param   = input_param;

    /* verilator lint_off UNUSED */
    cta_csrs_t cta_csr_data_w;
    /* verilator lint_on UNUSED */

    VX_cta_dispatch cta_dispatch (
        .clk             (clk),
        .reset           (reset),
        .kmu_bus_if      (kmu_bus),
        .active_warps    (active_warps),
        .cta_sched_fire  (cta_sched_fire),
        .cta_sched_wid   (cta_sched_wid),
        .cta_sched_PC    (cta_sched_PC),
        .cta_sched_tmask (cta_sched_tmask),
        .cta_csr_valid   (cta_csr_valid),
        .cta_csr_wid     (cta_csr_wid),
        .cta_csr_data    (cta_csr_data_w)
    );

endmodule

