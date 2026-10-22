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

// Merges the SFU blocks' requests onto the core's single DXA, TEX and RTU buses.
module VX_sfu_shared import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.slave  block_dxa_req_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_dxa_req_bus_if.master dxa_req_bus_if,
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.slave     block_tex_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.slave     block_rtu_bus_if [`VX_CFG_NUM_SFU_BLOCKS],
    VX_rtu_bus_if.master    rtu_bus_if,
`endif

    input wire              clk,
    input wire              reset
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NB = `VX_CFG_NUM_SFU_BLOCKS;
    // the merged request leaves the core: register it when it is a real merge
    localparam REQ_OUT_BUF = (NB > 1) ? 3 : 0;

`ifndef VX_CFG_EXT_DXA_ENABLE
`ifndef VX_CFG_EXT_TEX_ENABLE
`ifndef VX_CFG_EXT_RTU_ENABLE
    `UNUSED_VAR ({clk, reset})
    `UNUSED_PARAM (REQ_OUT_BUF)
`endif
`endif
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if dxa_arb_out_if[1]();

    VX_dxa_req_arb #(
        .NUM_INPUTS  (NB),
        .NUM_OUTPUTS (1),
        .OUT_BUF     (REQ_OUT_BUF),
        .ARBITER     ("R")
    ) dxa_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (block_dxa_req_bus_if),
        .bus_out_if (dxa_arb_out_if)
    );

    assign dxa_req_bus_if.req_valid     = dxa_arb_out_if[0].req_valid;
    assign dxa_req_bus_if.req_data      = dxa_arb_out_if[0].req_data;
    assign dxa_arb_out_if[0].req_ready  = dxa_req_bus_if.req_ready;
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (TEX_CORE_TAG_WIDTH)
    ) tex_arb_out_if[1]();

    VX_tex_bus_arb #(
        .NUM_INPUTS  (NB),
        .NUM_OUTPUTS (1),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH   (TEX_REQ_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_REQ (REQ_OUT_BUF),
        .OUT_BUF_RSP (0)
    ) tex_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (block_tex_bus_if),
        .bus_out_if (tex_arb_out_if)
    );

    assign tex_bus_if.req_valid         = tex_arb_out_if[0].req_valid;
    assign tex_bus_if.req_data          = tex_arb_out_if[0].req_data;
    assign tex_arb_out_if[0].req_ready  = tex_bus_if.req_ready;
    assign tex_arb_out_if[0].rsp_valid  = tex_bus_if.rsp_valid;
    assign tex_arb_out_if[0].rsp_data   = tex_bus_if.rsp_data;
    assign tex_bus_if.rsp_ready         = tex_arb_out_if[0].rsp_ready;
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    localparam RTU_SEL_W = `UP(`ARB_SEL_BITS(NB, 1));

    VX_rtu_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (RTU_CORE_TAG_WIDTH),
        .SRC_WIDTH (RTU_SEL_W)
    ) rtu_arb_out_if[1]();

    VX_rtu_bus_arb #(
        .NUM_INPUTS  (NB),
        .NUM_OUTPUTS (1),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH   (RTU_REQ_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_ARM (REQ_OUT_BUF),
        .OUT_BUF_REQ (REQ_OUT_BUF),
        .OUT_BUF_WIN (0)
    ) rtu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (block_rtu_bus_if),
        .bus_out_if (rtu_arb_out_if)
    );

    // A warp lives in exactly one SFU block, so {wid} alone names a beat's owner
    // on the core's bus; the socket re-derives the core index.
    `UNUSED_VAR (rtu_arb_out_if[0].arm_data.src)
    `UNUSED_VAR (rtu_arb_out_if[0].req_data.src)

    assign rtu_bus_if.arm_valid             = rtu_arb_out_if[0].arm_valid;
    assign rtu_bus_if.arm_data.src          = 1'b0;
    assign rtu_bus_if.arm_data.wid          = rtu_arb_out_if[0].arm_data.wid;
    assign rtu_bus_if.arm_data.mask         = rtu_arb_out_if[0].arm_data.mask;
    assign rtu_bus_if.arm_data.scene_base   = rtu_arb_out_if[0].arm_data.scene_base;
    assign rtu_bus_if.arm_data.flags        = rtu_arb_out_if[0].arm_data.flags;
    assign rtu_bus_if.arm_data.cull_mask    = rtu_arb_out_if[0].arm_data.cull_mask;
    assign rtu_bus_if.arm_data.payload_ptr  = rtu_arb_out_if[0].arm_data.payload_ptr;
    assign rtu_bus_if.arm_data.tag          = rtu_arb_out_if[0].arm_data.tag;
    assign rtu_arb_out_if[0].arm_ready      = rtu_bus_if.arm_ready;

    assign rtu_bus_if.req_valid             = rtu_arb_out_if[0].req_valid;
    assign rtu_bus_if.req_data.kind         = rtu_arb_out_if[0].req_data.kind;
    assign rtu_bus_if.req_data.src          = 1'b0;
    assign rtu_bus_if.req_data.wid          = rtu_arb_out_if[0].req_data.wid;
    assign rtu_bus_if.req_data.data         = rtu_arb_out_if[0].req_data.data;
    assign rtu_bus_if.req_data.cb_action    = rtu_arb_out_if[0].req_data.cb_action;
    assign rtu_arb_out_if[0].req_ready      = rtu_bus_if.req_ready;

    assign rtu_arb_out_if[0].win_valid      = rtu_bus_if.win_valid;
    assign rtu_arb_out_if[0].win_data       = rtu_bus_if.win_data;
    assign rtu_bus_if.win_ready             = rtu_arb_out_if[0].win_ready;
`endif

endmodule
