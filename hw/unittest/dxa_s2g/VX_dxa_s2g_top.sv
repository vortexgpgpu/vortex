// Copyright © 2019-2023
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

`include "VX_define.vh"

module VX_dxa_s2g_top import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input wire clk,
    input wire reset,
    input wire issue_allowed,

    input wire req_valid,
    output wire req_ready,
    input wire [NC_WIDTH-1:0] req_core_id,
    input wire [UUID_WIDTH-1:0] req_uuid,
    input wire [NW_WIDTH-1:0] req_wid,
    input wire [DXA_GROUP_EPOCH_W-1:0] req_epoch,
    input wire [DXA_GROUP_SEQ_W-1:0] req_group_seq,
    input wire [DXA_GROUP_OPID_W-1:0] req_op_id,
    input wire [DXA_SMEM_ADDR_W-1:0] req_smem_addr,
    input wire [31:0] req_meta,
    input wire [4:0][31:0] req_coords,
    input wire [`VX_CFG_NUM_WARPS-1:0] req_cta_mask,

    input wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] desc_base_addr,
    input wire [31:0] desc_meta,
    input wire [31:0] desc_tile01,
    input wire [31:0] desc_tile23,
    input wire [31:0] desc_tile4,
    input wire [31:0] desc_size0,
    input wire [31:0] desc_size1,
    input wire [31:0] desc_size2,
    input wire [31:0] desc_size3,
    input wire [31:0] desc_size4,
    input wire [31:0] desc_stride0,
    input wire [31:0] desc_stride1,
    input wire [31:0] desc_stride2,
    input wire [31:0] desc_stride3,

    output wire lmem_req_valid,
    output wire lmem_req_rw,
    input wire lmem_req_ready,
    output wire [DXA_LMEM_ADDR_W-1:0] lmem_req_addr,
    output wire [DXA_LMEM_TAG_W-1:0] lmem_req_tag,
    input wire lmem_rsp_valid,
    output wire lmem_rsp_ready,
    input wire [DXA_LMEM_WORD_SIZE*8-1:0] lmem_rsp_data,
    input wire [DXA_LMEM_TAG_W-1:0] lmem_rsp_tag,

    output wire gmem_req_valid,
    output wire gmem_req_rw,
    input wire gmem_req_ready,
    output wire [GMEM_ADDR_WIDTH-1:0] gmem_req_addr,
    output wire [GMEM_LINE_SIZE*8-1:0] gmem_req_data,
    output wire [GMEM_LINE_SIZE-1:0] gmem_req_byteen,
    output wire [GMEM_TAG_WIDTH-1:0] gmem_req_tag,
    input wire gmem_rsp_valid,
    output wire gmem_rsp_ready,
    input wire [GMEM_TAG_WIDTH-1:0] gmem_rsp_tag,

    output wire completion_valid,
    input wire completion_ready,
    output wire [NC_WIDTH-1:0] completion_core_id,
    output wire [NW_WIDTH-1:0] completion_wid,
    output wire [DXA_GROUP_EPOCH_W-1:0] completion_epoch,
    output wire [DXA_GROUP_SEQ_W-1:0] completion_group_seq,
    output wire [DXA_GROUP_OPID_W-1:0] completion_op_id,

    output wire busy
);
    localparam GMEM_LINE_SIZE = `VX_CFG_L1_LINE_SIZE;
    localparam GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE);
    localparam GMEM_TAG_WIDTH = L1_MEM_ARB_TAG_WIDTH;
    wire unused_package_constants = |32'(DXA_DESC_SLOT_W)
                                  | |32'(DXA_DEST_BLOCKMAJOR);
    `UNUSED_VAR (unused_package_constants)

    VX_dxa_worker_req_if req_if();
    dxa_req_data_t req_data;
    dxa_desc_t desc_data;

    always_comb begin
        req_data = '0;
        req_data.core_id = req_core_id;
        req_data.uuid = req_uuid;
        req_data.wid = req_wid;
        req_data.dir = DXA_DIR_S2G;
        req_data.epoch = req_epoch;
        req_data.group_seq = req_group_seq;
        req_data.op_id = req_op_id;
        req_data.smem_addr = req_smem_addr;
        req_data.meta = req_meta;
        req_data.coords = req_coords;
        req_data.cta_mask = req_cta_mask;

        desc_data = '0;
        desc_data.base_addr = desc_base_addr;
        desc_data.meta = desc_meta;
        desc_data.tile01 = desc_tile01;
        desc_data.tile23 = desc_tile23;
        desc_data.tile4 = desc_tile4;
        desc_data.size0 = desc_size0;
        desc_data.size1 = desc_size1;
        desc_data.size2 = desc_size2;
        desc_data.size3 = desc_size3;
        desc_data.size4 = desc_size4;
        desc_data.stride0 = desc_stride0;
        desc_data.stride1 = desc_stride1;
        desc_data.stride2 = desc_stride2;
        desc_data.stride3 = desc_stride3;
    end

    assign req_if.valid = req_valid && issue_allowed;
    assign req_if.req_data = req_data;
    assign req_if.desc_data = desc_data;
    assign req_ready = issue_allowed && req_if.ready;

    VX_mem_bus_if #(
        .DATA_SIZE  (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH  (DXA_LMEM_TAG_W),
        .ATTR_WIDTH (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH (DXA_LMEM_ADDR_W)
    ) lmem_bus_if();

    assign lmem_req_valid = lmem_bus_if.req_valid;
    assign lmem_req_rw = lmem_bus_if.req_data.rw;
    assign lmem_req_addr = lmem_bus_if.req_data.addr;
    assign lmem_req_tag = lmem_bus_if.req_data.tag;
    assign lmem_bus_if.req_ready = lmem_req_ready;
    assign lmem_bus_if.rsp_valid = lmem_rsp_valid;
    assign lmem_bus_if.rsp_data.data = lmem_rsp_data;
    assign lmem_bus_if.rsp_data.tag = lmem_rsp_tag;
    assign lmem_rsp_ready = lmem_bus_if.rsp_ready;
    wire unused_lmem_request_fields = |lmem_bus_if.req_data;
    `UNUSED_VAR (unused_lmem_request_fields)

    VX_mem_bus_if #(
        .DATA_SIZE (GMEM_LINE_SIZE),
        .TAG_WIDTH (GMEM_TAG_WIDTH)
    ) gmem_bus_if();

    assign gmem_req_valid = gmem_bus_if.req_valid;
    assign gmem_req_rw = gmem_bus_if.req_data.rw;
    assign gmem_req_addr = gmem_bus_if.req_data.addr;
    assign gmem_req_data = gmem_bus_if.req_data.data;
    assign gmem_req_byteen = gmem_bus_if.req_data.byteen;
    assign gmem_req_tag = gmem_bus_if.req_data.tag;
    assign gmem_bus_if.req_ready = gmem_req_ready;
    assign gmem_bus_if.rsp_valid = gmem_rsp_valid;
    assign gmem_bus_if.rsp_data.data = '0;
    assign gmem_bus_if.rsp_data.tag = gmem_rsp_tag;
    assign gmem_rsp_ready = gmem_bus_if.rsp_ready;
    wire unused_gmem_request_fields = |gmem_bus_if.req_data;
    `UNUSED_VAR (unused_gmem_request_fields)

    VX_dxa_group_completion_if completion_if();
    assign completion_valid = completion_if.valid;
    assign completion_core_id = completion_if.core_id;
    assign completion_wid = completion_if.data.wid;
    assign completion_epoch = completion_if.data.epoch;
    assign completion_group_seq = completion_if.data.group_seq;
    assign completion_op_id = completion_if.data.op_id;
    assign completion_if.ready = completion_ready;

    VX_dxa_worker #(
        .INSTANCE_ID ("dxa-s2g-test"),
        .WORKER_ID (0),
        .GMEM_TAG_WIDTH (GMEM_TAG_WIDTH)
    ) dut (
        .clk          (clk),
        .reset        (reset),
        .req_if       (req_if),
        .gmem_bus_if  (gmem_bus_if),
        .smem_bus_if  (lmem_bus_if),
        .completion_if(completion_if),
        .busy         (busy)
    );

endmodule
