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

`ifdef VX_CFG_EXT_DXA_S2G_ENABLE

// S2G data half. Setup and address generation are shared with G2S by the
// enclosing worker; this block only gathers LMEM words and emits stores.
module VX_dxa_s2g_data import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_LINE_SIZE = `VX_CFG_L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE),
    parameter GMEM_OFF_BITS = `CLOG2(GMEM_LINE_SIZE)
) (
    input wire clk,
    input wire reset,
    input wire transfer_active,
    input wire pipeline_start,
    // Empty descriptors still complete as one source-consumed operation, but
    // must not enter the normal zero-length cache-line path.
    input wire zero_length,

    input wire                       ag_valid,
    output wire                      ag_ready,
    input wire [GMEM_ADDR_WIDTH-1:0] ag_cl_addr,
    input wire [DXA_SMEM_ADDR_W-1:0] ag_smem_byte_addr,
    input wire [GMEM_OFF_BITS-1:0]   ag_byte_offset,
    input wire [GMEM_OFF_BITS:0]     ag_valid_length,
    input wire                       ag_oob,
    input wire                       ag_last,

    input wire [NC_WIDTH-1:0]        active_core_id,
    input wire [UUID_WIDTH-1:0]      active_uuid,
    input wire [NW_WIDTH-1:0]        active_wid,
    input wire [DXA_GROUP_ID_W-1:0] active_group_id,

    VX_mem_bus_if.master gmem_bus_if,
    VX_mem_bus_if.master smem_bus_if,
    VX_dxa_group_completion_if.master completion_if,

    output wire transfer_done,
    output wire [31:0] sent_count,
    output wire [31:0] read_count
);
    localparam GMEM_BYTES = GMEM_LINE_SIZE;
    localparam GMEM_DATAW = GMEM_BYTES * 8;

    wire cl_valid;
    wire cl_ready;
    wire [GMEM_ADDR_WIDTH-1:0] cl_addr;
    wire [GMEM_DATAW-1:0] cl_data;
    wire [GMEM_BYTES-1:0] cl_byteen;
    wire cl_last;
    wire read_outstanding;
    wire source_done;
    wire source_done_has_store;

    VX_dxa_smem2cl #(
        .GMEM_LINE_SIZE  (GMEM_BYTES),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH)
    ) smem2cl (
        .clk                   (clk),
        .reset                 (reset),
        .in_valid              (ag_valid),
        .in_ready              (ag_ready),
        .in_cl_addr            (ag_cl_addr),
        .in_smem_byte_addr     (ag_smem_byte_addr),
        .in_byte_offset        (ag_byte_offset),
        .in_valid_length       (ag_valid_length),
        .in_oob                (ag_oob),
        .in_last               (ag_last),
        .active_core_id        (active_core_id),
        .active_uuid           (active_uuid),
        .out_valid             (cl_valid),
        .out_ready             (cl_ready),
        .out_cl_addr           (cl_addr),
        .out_data              (cl_data),
        .out_byteen            (cl_byteen),
        .out_last              (cl_last),
        .smem_bus_if           (smem_bus_if),
        .read_outstanding      (read_outstanding),
        .read_count            (read_count),
        .source_done           (source_done),
        .source_done_has_store (source_done_has_store)
    );

    assign gmem_bus_if.req_valid = cl_valid;
    assign gmem_bus_if.req_data.rw = 1'b1;
    assign gmem_bus_if.req_data.addr = cl_addr;
    assign gmem_bus_if.req_data.data = cl_data;
    assign gmem_bus_if.req_data.byteen = cl_byteen;
    assign gmem_bus_if.req_data.attr = '0;
    assign gmem_bus_if.req_data.tag.uuid = active_uuid;
    assign gmem_bus_if.req_data.tag.value = '0;
    assign cl_ready = gmem_bus_if.req_ready;
    assign gmem_bus_if.rsp_ready = 1'b0;

    wire store_fire = gmem_bus_if.req_valid && gmem_bus_if.req_ready;
    wire last_store_fire = store_fire && cl_last;

    reg completion_pending_r;
    reg completion_sent_r;
    reg destination_done_r;
    reg [31:0] sent_count_r;

    assign completion_if.valid = completion_pending_r && !pipeline_start;
    assign completion_if.core_id = active_core_id;
    assign completion_if.data.wid = active_wid;
    assign completion_if.data.group_id = active_group_id;
    wire completion_fire = completion_if.valid && completion_if.ready;

    wire source_event_done = completion_sent_r || completion_fire;
    wire destination_event_done = destination_done_r || last_store_fire
                               || (source_done && !source_done_has_store);
    assign transfer_done = transfer_active && !pipeline_start
                        && source_event_done && destination_event_done;
    assign sent_count = sent_count_r;

    always @(posedge clk) begin
        if (reset) begin
            completion_pending_r <= 1'b0;
            completion_sent_r <= 1'b0;
            destination_done_r <= 1'b0;
            sent_count_r <= '0;
        end else if (pipeline_start) begin
            completion_pending_r <= 1'b0;
            completion_sent_r <= 1'b0;
            destination_done_r <= zero_length;
            sent_count_r <= '0;
            if (zero_length)
                completion_pending_r <= 1'b1;
        end else begin
            if (source_done) begin
                completion_pending_r <= 1'b1;
                if (!source_done_has_store)
                    destination_done_r <= 1'b1;
            end
            if (completion_fire) begin
                completion_pending_r <= 1'b0;
                completion_sent_r <= 1'b1;
            end
            if (store_fire)
                sent_count_r <= sent_count_r + 1'b1;
            if (last_store_fire)
                destination_done_r <= 1'b1;
        end
    end

    `RUNTIME_ASSERT(!gmem_bus_if.rsp_valid,
        ("unexpected response to ordinary S2G store"))
    `RUNTIME_ASSERT(!source_done || !read_outstanding,
        ("S2G READ completion fired with an LMEM read outstanding"))

endmodule

`endif
