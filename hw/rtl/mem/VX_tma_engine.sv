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

module VX_tma_engine import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter ENGINE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_dcr_bus_if.slave     dcr_bus_if,
    VX_tma_bus_if.slave     tma_bus_if,

    VX_mem_bus_if.master    gmem_bus_if,
    VX_mem_bus_if.master    smem_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam TMA_OP_ISSUE   = 3'd4;

    localparam TMA_REQ_DATAW      = NC_WIDTH + UUID_WIDTH + NW_WIDTH + 3 + (2 * `XLEN);
    localparam TMA_RSP_DATAW      = NC_WIDTH + UUID_WIDTH + NW_WIDTH + BAR_ADDR_W + 2;
    localparam TMA_CTX_COUNT      = `NUM_CORES * `NUM_WARPS;
    localparam TMA_CTX_BITS       = `UP(`CLOG2(TMA_CTX_COUNT));
    localparam TMA_DESC_SLOT_BITS = `CLOG2(`VX_DCR_TMA_DESC_COUNT);
    localparam TMA_DESC_SLOT_W    = `UP(TMA_DESC_SLOT_BITS);
    localparam TMA_DESC_WORD_BITS = `CLOG2(`VX_DCR_TMA_DESC_STRIDE);
    localparam TMA_DESC_WORD_W    = `UP(TMA_DESC_WORD_BITS);

    localparam GMEM_BYTES      = `L2_LINE_SIZE;
    localparam GMEM_DATAW      = GMEM_BYTES * 8;
    localparam GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(GMEM_BYTES);
    localparam GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES);

    localparam SMEM_BYTES      = LSU_WORD_SIZE;
    localparam SMEM_DATAW      = SMEM_BYTES * 8;
    localparam SMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(SMEM_BYTES);
    localparam SMEM_OFF_BITS   = `CLOG2(SMEM_BYTES);

    localparam TMA_SMEM_ENGINE_BITS = `CLOG2(`NUM_TMA_UNITS);
    localparam TMA_SMEM_ENGINE_W    = `UP(TMA_SMEM_ENGINE_BITS);
    localparam UUID_W               = `UP(UUID_WIDTH);
    localparam SMEM_TAG_TOTALW      = LMEM_TAG_WIDTH;
    localparam SMEM_TAG_VALUEW      = SMEM_TAG_TOTALW - UUID_W;
    localparam SMEM_TAG_ROUTEW      = TMA_SMEM_ENGINE_W + NC_WIDTH;

    `STATIC_ASSERT(SMEM_TAG_TOTALW >= SMEM_TAG_ROUTEW, ("invalid parameter"))
    wire req_valid;
    wire [TMA_REQ_DATAW-1:0] req_data_raw;
    wire req_accept;
    wire req_fire;
    wire [NC_WIDTH-1:0] req_core_id;
    wire [UUID_WIDTH-1:0] req_uuid;
    wire [NW_WIDTH-1:0] req_wid;
    wire [2:0] req_op;
    wire [`XLEN-1:0] req_rs1;
    wire [`XLEN-1:0] req_rs2;

    wire [TMA_CTX_BITS-1:0] req_ctx_idx;
    wire [TMA_CTX_BITS-1:0] req_core_ofs;
    wire [BAR_ADDR_W-1:0] req_bar_addr;

    wire [BAR_ADDR_W-1:0] issue_bar_addr;
    wire [TMA_DESC_SLOT_W-1:0] issue_desc_slot;
    wire [`XLEN-1:0] issue_smem_addr;
    wire [`XLEN-1:0] issue_flags;
    wire [4:0][`XLEN-1:0] issue_coords;

    wire [`MEM_ADDR_WIDTH-1:0] issue_base_addr;
    wire [31:0] issue_desc_meta;
    wire [31:0] issue_desc_tile01;
    wire [31:0] issue_desc_tile23;
    wire [31:0] issue_desc_tile4;
    wire [31:0] issue_desc_cfill;
    wire [31:0] issue_size0_raw;
    wire [31:0] issue_size1_raw;
    wire [31:0] issue_stride0_raw;

    tma_issue_dec_t issue_dec;
    tma_xfer_state_t xfer_state_r;
    tma_xfer_math_t xfer_math_w;
    tma_xfer_evt_t xfer_evt;

    wire                       done_rsp_valid_r;
    wire [TMA_RSP_DATAW-1:0]   done_rsp_data_r;

    VX_elastic_buffer #(
        .DATAW (TMA_REQ_DATAW),
        .SIZE  (8)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tma_bus_if.req_valid),
        .ready_in  (tma_bus_if.req_ready),
        .data_in   (tma_bus_if.req_data),
        .valid_out (req_valid),
        .ready_out (req_accept),
        .data_out  (req_data_raw)
    );

    assign {req_core_id, req_uuid, req_wid, req_op, req_rs1, req_rs2} = req_data_raw;
    assign req_accept = ~req_valid
                     || (req_op != TMA_OP_ISSUE)
                     || (~xfer_state_r.active && ~done_rsp_valid_r);
    assign req_fire = req_valid && req_accept;

    assign req_core_ofs = TMA_CTX_BITS'(req_core_id) * TMA_CTX_BITS'(`NUM_WARPS);
    assign req_ctx_idx = req_core_ofs + TMA_CTX_BITS'(req_wid);

    if (`NUM_WARPS > 1) begin : g_req_bar_addr_w
        assign req_bar_addr = {req_rs2[NW_BITS-1:0], req_rs2[16 +: NB_BITS]};
    end else begin : g_req_bar_addr_wo
        assign req_bar_addr = req_rs2[16 +: NB_BITS];
    end

    VX_tma_issue_state #(
        .TMA_CTX_COUNT     (TMA_CTX_COUNT),
        .TMA_CTX_BITS      (TMA_CTX_BITS),
        .TMA_DESC_SLOT_BITS(TMA_DESC_SLOT_BITS),
        .TMA_DESC_SLOT_W   (TMA_DESC_SLOT_W),
        .TMA_DESC_WORD_BITS(TMA_DESC_WORD_BITS),
        .TMA_DESC_WORD_W   (TMA_DESC_WORD_W)
    ) issue_state (
        .clk            (clk),
        .reset          (reset),
        .req_fire       (req_fire),
        .req_op         (req_op),
        .req_ctx_idx    (req_ctx_idx),
        .req_rs1        (req_rs1),
        .req_rs2        (req_rs2),
        .req_bar_addr   (req_bar_addr),
        .dcr_bus_if     (dcr_bus_if),
        .issue_bar_addr (issue_bar_addr),
        .issue_desc_slot(issue_desc_slot),
        .issue_smem_addr(issue_smem_addr),
        .issue_flags    (issue_flags),
        .issue_coords   (issue_coords),
        .issue_base_addr(issue_base_addr),
        .issue_desc_meta(issue_desc_meta),
        .issue_desc_tile01(issue_desc_tile01),
        .issue_desc_tile23(issue_desc_tile23),
        .issue_desc_tile4(issue_desc_tile4),
        .issue_desc_cfill(issue_desc_cfill),
        .issue_size0    (issue_size0_raw),
        .issue_size1    (issue_size1_raw),
        .issue_stride0  (issue_stride0_raw)
    );

    VX_tma_issue_decode #(
        .GMEM_BYTES(GMEM_BYTES),
        .SMEM_BYTES(SMEM_BYTES)
    ) issue_decode (
        .issue_flags      (issue_flags),
        .issue_desc_meta  (issue_desc_meta),
        .issue_desc_tile01(issue_desc_tile01),
        .issue_size0_raw  (issue_size0_raw),
        .issue_size1_raw  (issue_size1_raw),
        .issue_stride0_raw(issue_stride0_raw),
        .issue_dec        (issue_dec)
    );

    VX_tma_xfer_math xfer_math_i (
        .xfer_state  (xfer_state_r),
        .gmem_rsp_data(gmem_bus_if.rsp_data.data),
        .smem_rsp_data(smem_bus_if.rsp_data.data),
        .xfer_math   (xfer_math_w)
    );

    wire wr_issue_gmem = xfer_math_w.state_wait_wr && xfer_state_r.write_to_gmem;
    wire wr_issue_smem = xfer_math_w.state_wait_wr && ~xfer_state_r.write_to_gmem;
    wire rd_wait_gmem = xfer_math_w.state_wait_rd && xfer_state_r.wait_rsp_from_gmem;
    wire rd_wait_smem = xfer_math_w.state_wait_rd && ~xfer_state_r.wait_rsp_from_gmem;

    assign xfer_evt.gmem_rd_req_fire = xfer_math_w.cur_read_from_gmem && gmem_bus_if.req_ready;
    assign xfer_evt.smem_rd_req_fire = xfer_math_w.cur_read_from_smem && smem_bus_if.req_ready;
    assign xfer_evt.gmem_wr_req_fire = wr_issue_gmem && gmem_bus_if.req_ready;
    assign xfer_evt.smem_wr_req_fire = wr_issue_smem && smem_bus_if.req_ready;
    assign xfer_evt.gmem_rsp_fire = rd_wait_gmem && gmem_bus_if.rsp_valid;
    assign xfer_evt.smem_rsp_fire = rd_wait_smem && smem_bus_if.rsp_valid;

    VX_tma_xfer_ctrl #(
        .TMA_RSP_DATAW(TMA_RSP_DATAW)
    ) xfer_ctrl (
        .clk                   (clk),
        .reset                 (reset),
        .req_fire              (req_fire),
        .req_op                (req_op),
        .req_core_id           (req_core_id),
        .req_uuid              (req_uuid),
        .req_wid               (req_wid),
        .issue_bar_addr        (issue_bar_addr),
        .issue_dec             (issue_dec),
        .issue_base_addr       (issue_base_addr),
        .issue_smem_addr       (issue_smem_addr),
        .issue_coords          (issue_coords),
        .issue_desc_cfill      (issue_desc_cfill),
        .xfer_math             (xfer_math_w),
        .xfer_evt              (xfer_evt),
        .tma_rsp_ready         (tma_bus_if.rsp_ready),
        .xfer_state_r          (xfer_state_r),
        .done_rsp_valid_r      (done_rsp_valid_r),
        .done_rsp_data_r       (done_rsp_data_r)
    );

    wire [SMEM_TAG_TOTALW-1:0] smem_tag_route;
    wire [UUID_W-1:0] smem_tag_uuid;
    wire [SMEM_TAG_VALUEW-1:0] smem_tag_value;
    assign smem_tag_route = {
        {(SMEM_TAG_TOTALW - SMEM_TAG_ROUTEW){1'b0}},
        NC_WIDTH'(xfer_state_r.core_id),
        TMA_SMEM_ENGINE_W'(ENGINE_ID)
    };
    assign smem_tag_uuid  = UUID_W'(smem_tag_route[SMEM_TAG_TOTALW-1 -: UUID_W]);
    assign smem_tag_value = SMEM_TAG_VALUEW'(smem_tag_route[SMEM_TAG_VALUEW-1:0]);

    assign tma_bus_if.rsp_valid = done_rsp_valid_r;
    assign tma_bus_if.rsp_data  = done_rsp_data_r;

    assign gmem_bus_if.req_valid = xfer_math_w.cur_read_from_gmem || wr_issue_gmem;
    assign gmem_bus_if.req_data.rw = wr_issue_gmem;
    assign gmem_bus_if.req_data.addr = wr_issue_gmem
        ? GMEM_ADDR_WIDTH'(xfer_state_r.pending_wr_byte_addr >> GMEM_OFF_BITS)
        : GMEM_ADDR_WIDTH'(xfer_math_w.cur_gmem_byte_addr >> GMEM_OFF_BITS);
    assign gmem_bus_if.req_data.data = wr_issue_gmem
        ? GMEM_DATAW'(xfer_math_w.pending_gmem_wr_data_shifted)
        : '0;
    assign gmem_bus_if.req_data.byteen = wr_issue_gmem
        ? GMEM_BYTES'(xfer_math_w.pending_gmem_byteen)
        : {GMEM_BYTES{1'b1}};
    assign gmem_bus_if.req_data.flags = '0;
    assign gmem_bus_if.req_data.tag.uuid = xfer_state_r.uuid;
    assign gmem_bus_if.req_data.tag.value = '0;
    assign gmem_bus_if.rsp_ready = rd_wait_gmem;

    assign smem_bus_if.req_valid = xfer_math_w.cur_read_from_smem || wr_issue_smem;
    assign smem_bus_if.req_data.rw = wr_issue_smem;
    assign smem_bus_if.req_data.addr = wr_issue_smem
        ? SMEM_ADDR_WIDTH'(xfer_state_r.pending_wr_byte_addr >> SMEM_OFF_BITS)
        : SMEM_ADDR_WIDTH'(xfer_math_w.cur_smem_byte_addr >> SMEM_OFF_BITS);
    assign smem_bus_if.req_data.data = wr_issue_smem
        ? SMEM_DATAW'(xfer_math_w.pending_smem_wr_data_shifted)
        : '0;
    assign smem_bus_if.req_data.byteen = wr_issue_smem
        ? SMEM_BYTES'(xfer_math_w.pending_smem_byteen)
        : {SMEM_BYTES{1'b1}};
    assign smem_bus_if.req_data.flags = '0;
    assign smem_bus_if.req_data.tag.uuid = smem_tag_uuid;
    assign smem_bus_if.req_data.tag.value = smem_tag_value;
    assign smem_bus_if.rsp_ready = rd_wait_smem;

    `UNUSED_VAR (dcr_bus_if.write_addr)
    `UNUSED_VAR (xfer_state_r.pending_rd_byte_addr[`MEM_ADDR_WIDTH-1:6])
    `UNUSED_VAR (xfer_state_r.tile1)
    `UNUSED_VAR (xfer_math_w.state_idle)
    `UNUSED_VAR (xfer_math_w.cur_gmem_byteen)
    `UNUSED_VAR (xfer_math_w.cur_smem_byteen)
    `UNUSED_VAR (xfer_math_w.cur_need_read)
    `UNUSED_VAR (issue_desc_slot)
    `UNUSED_VAR (issue_desc_tile23)
    `UNUSED_VAR (issue_desc_tile4)

endmodule
