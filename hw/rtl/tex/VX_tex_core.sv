//!/bin/bash

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

`include "VX_tex_define.vh"

module VX_tex_core import VX_gpu_pkg::*; import VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) (
    `SCOPE_IO_DECL

    input wire  clk,
    input wire  reset,

    // PERF
`ifdef PERF_ENABLE
    VX_tex_perf_if.master   perf_tex_if,
`endif

    VX_mem_bus_if.master    cache_bus_if [TCACHE_NUM_REQS],

    VX_dcr_bus_if.slave     dcr_bus_if,

    VX_tex_bus_if.slave     tex_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BLEND_FRAC_W = (TEX_NUM_LEVELS * 2 * NUM_LANES * `TEX_BLEND_FRAC);
    localparam LODFRAC_W = (NUM_LANES * `VX_TEX_LOD_FRAC_BITS);
    localparam BORDER_W = (TEX_NUM_LEVELS * 4 * NUM_LANES);
    localparam FLIP_W   = (NUM_LANES * 2);
    // Per lane: gather enable and the component it returns.
    localparam GATHER_W = (NUM_LANES * 3);
    // The stage's reduction rides with the gather fields to the sampler.
    localparam RED_W = 2;
    // vx_tex_cmp: its flag, compare function and the slot holding its per-lane
    // references (REF_SLOTS deep; the references themselves stay in the slot,
    // off the request path).
    localparam REF_SLOTS = `VX_CFG_TEX_MEM_QUEUE_SIZE;
    localparam SLOT_W = `LOG2UP(REF_SLOTS);
    localparam CMP_W = 1 + TEX_CMP_BITS + SLOT_W;
    localparam INFO_W = TAG_WIDTH + CMP_W + RED_W + GATHER_W + TEX_FORMAT_BITS + 32;
    localparam W_ADDR_BITS = `TEX_ADDR_BITS + 6;

    // DCRs

    tex_dcrs_t tex_dcrs;

    VX_tex_dcr #(
        .INSTANCE_ID ($sformatf("%s-dcr", INSTANCE_ID)),
        .NUM_STAGES  (`VX_TEX_STAGE_COUNT)
    ) tex_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .stage      (tex_bus_if.req_data.stage),
        .tex_dcrs   (tex_dcrs)
    );

    // Texture stage select

    wire                                        req_valid;
    wire [NUM_LANES-1:0]                        req_mask;
    wire [NUM_LANES-1:0][TEX_FILTER_BITS-1:0]   req_filter, sel_filter;
    wire [TEX_FORMAT_BITS-1:0]                 req_format;
    wire [RED_W-1:0]                            req_reduction;
    wire [1:0][TEX_WRAP_BITS-1:0]              req_wraps;
    wire [1:0][TEX_DIM_W-1:0]               req_dims;
    wire [NUM_LANES-1:0][W_ADDR_BITS-1:0]       req_baseaddr, sel_baseaddr;
    wire [NUM_LANES-1:0][2:0]                   req_gather, sel_gather;
    wire [31:0]                                 req_bordercolor;
    wire [1:0][NUM_LANES-1:0][31:0]             req_coords;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][TEX_LOD_BITS-1:0] req_miplevel, sel_miplevel;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][`TEX_MIPOFF_BITS-1:0] req_mipoff, sel_mipoff;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] req_lodfrac, sel_lodfrac;
    wire [TAG_WIDTH-1:0]                        req_tag;
    wire [CMP_W-1:0]                            req_cmpinfo;
    wire                                        req_ready;

    // vx_tex_cmp: the reference rides in the lod lane, so the lod fields come
    // from above the coordinates, which are sign-extended from their top bit.
    wire req_cmp = tex_bus_if.req_data.cmp;
    wire [1:0][NUM_LANES-1:0][31:0] sel_coords;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cmp_coords
        for (genvar k = 0; k < 2; ++k) begin : g_k
            wire [31:0] c = tex_bus_if.req_data.coords[k][i];
            assign sel_coords[k][i] = req_cmp ? 32'($signed(c[`VX_TEX_CMP_COORD_BITS-1:0])) : c;
        end
    end

    // Both levels a sample can read are selected here and fetched by one
    // request, so nothing downstream has to pair two responses. A lane may carry
    // its own filter -- the shader's min/mag decision and whether it blends
    // levels -- and its array layer in the operand's upper fields; otherwise the
    // stage's FILTER applies and the layer is 0.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mip_sel
        wire [31:0] cu = tex_bus_if.req_data.coords[0][i];
        wire [31:0] cv = tex_bus_if.req_data.coords[1][i];
        wire [TEX_ARG_BITS-1:0] arg_cmp =
              TEX_ARG_BITS'(cv[`VX_TEX_CMP_V_LEVEL_SHIFT +: TEX_LOD_BITS])
            | (TEX_ARG_BITS'(cv[`VX_TEX_CMP_V_LANE_FILTER]) << `VX_TEX_ARG_LANE_FILTER)
            | (TEX_ARG_BITS'(cv[`VX_TEX_CMP_V_BILINEAR]) << `VX_TEX_ARG_BILINEAR)
            | (TEX_ARG_BITS'(cu[`VX_TEX_CMP_U_LAYER_SHIFT +: `VX_TEX_CMP_LAYER_BITS]) << `VX_TEX_ARG_LAYER_SHIFT);
        `UNUSED_VAR (cu[`VX_TEX_CMP_U_LAYER_SHIFT-1:0])
        `UNUSED_VAR (cv[`VX_TEX_CMP_V_LEVEL_SHIFT-1:0])
        wire [TEX_ARG_BITS-1:0] arg = req_cmp ? arg_cmp : tex_bus_if.req_data.lod[i];
        // A gather reads the bilinear footprint of one level, whatever the filter.
        wire gather = arg[`VX_TEX_ARG_GATHER];
        assign sel_gather[i] = {gather, arg[`VX_TEX_ARG_GATHER_COMP +: 2]};
        // A compare samples one level; the shader blends two for MIP_LINEAR.
        wire [TEX_FILTER_BITS-1:0] lane_filter = gather ? TEX_FILTER_BITS'(`VX_TEX_FILTER_BILINEAR)
                             : arg[`VX_TEX_ARG_LANE_FILTER]
                             ? {arg[`VX_TEX_ARG_MIP_LINEAR], arg[`VX_TEX_ARG_BILINEAR]}
                             : tex_dcrs.filter;
        assign sel_filter[i] = {lane_filter[TEX_FILTER_BITS-1] & ~req_cmp, lane_filter[TEX_FILTER_BITS-2:0]};
        wire mip_linear = sel_filter[i][TEX_FILTER_BITS-1];
        wire [`VX_TEX_ARG_LAYER_BITS-1:0] layer = arg[`VX_TEX_ARG_LAYER_SHIFT +: `VX_TEX_ARG_LAYER_BITS];
        assign sel_baseaddr[i] = {tex_dcrs.baseaddr, 6'b0}
                               + W_ADDR_BITS'(layer * W_ADDR_BITS'(tex_dcrs.layer_stride));
        `UNUSED_VAR (arg[`VX_TEX_ARG_LANE_FILTER-1:`VX_TEX_ARG_GATHER+1])
        `UNUSED_VAR (arg[`VX_TEX_ARG_LAYER_SHIFT-1:`VX_TEX_ARG_MIP_LINEAR+1])
        // The lod operand names an integer level, except under mip-linear where
        // it is fixed-point: the level in the high bits and the weight between
        // it and the next above in the low ones. A caller that never asks for a
        // blend therefore never has to know about the fractional form.
        wire [TEX_LOD_BITS-1:0] level_lo = mip_linear
            ? arg[`VX_TEX_LOD_FRAC_BITS +: TEX_LOD_BITS]
            : arg[TEX_LOD_BITS-1:0];
        // The top of the chain has no level above it, so the upper set repeats
        // the lower one and the weight then blends a value with itself. The
        // increment is as wide as the table index, so left alone it would wrap
        // to entry zero and blend the smallest level with the largest.
        wire at_top = (level_lo == TEX_LOD_BITS'(`VX_TEX_LOD_MAX));
        assign sel_miplevel[i][0] = level_lo;
        assign sel_miplevel[i][1] = at_top ? level_lo : (level_lo + TEX_LOD_BITS'(1));
        assign sel_mipoff[i][0] = tex_dcrs.mipoff[sel_miplevel[i][0]];
        assign sel_mipoff[i][1] = tex_dcrs.mipoff[sel_miplevel[i][1]];
        assign sel_lodfrac[i] = mip_linear ? arg[0 +: `VX_TEX_LOD_FRAC_BITS]
                                           : `VX_TEX_LOD_FRAC_BITS'(0);
    end

    // A compare's references wait in a slot until its taps reach the sampler,
    // which reads them there. A compare is held while every slot is taken.
    wire [SLOT_W-1:0] ref_waddr, ref_raddr;
    wire [NUM_LANES-1:0][31:0] ref_rdata;
    wire ref_full, ref_release;
    wire in_valid, in_ready;
    assign in_valid = tex_bus_if.req_valid && ~(req_cmp && ref_full);
    assign tex_bus_if.req_ready = in_ready && ~(req_cmp && ref_full);
    wire ref_acquire = in_valid && in_ready && req_cmp;

    VX_index_buffer #(
        .DATAW (NUM_LANES * 32),
        .SIZE  (REF_SLOTS)
    ) ref_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (ref_acquire),
        .write_addr   (ref_waddr),
        .write_data   (tex_bus_if.req_data.lod),
        .read_data    (ref_rdata),
        .read_addr    (ref_raddr),
        .release_en   (ref_release),
        .full         (ref_full),
        `UNUSED_PIN (empty)
    );

    VX_elastic_buffer #(
        .DATAW   (NUM_LANES  + CMP_W + RED_W + TEX_FORMAT_BITS + 2 * TEX_WRAP_BITS + 2 * TEX_DIM_W + 32 + GATHER_W + NUM_LANES * (TEX_FILTER_BITS + W_ADDR_BITS + 2 * 32 + `VX_TEX_LOD_FRAC_BITS + TEX_NUM_LEVELS * (TEX_LOD_BITS + `TEX_MIPOFF_BITS)) + TAG_WIDTH),
        .OUT_REG (1)
    ) pipe_reg (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (in_valid),
        .ready_in  (in_ready),
        .data_in   ({tex_bus_if.req_data.mask, sel_filter, sel_gather, {req_cmp, tex_dcrs.compare, ref_waddr}, tex_dcrs.reduction, tex_dcrs.format, tex_dcrs.wraps, tex_dcrs.dims, sel_baseaddr, tex_dcrs.border, sel_coords, sel_miplevel, sel_mipoff, sel_lodfrac, tex_bus_if.req_data.tag}),
        .data_out  ({req_mask, req_filter, req_gather, req_cmpinfo, req_reduction, req_format, req_wraps, req_dims, req_baseaddr, req_bordercolor, req_coords, req_miplevel, req_mipoff, req_lodfrac, req_tag}),
        .valid_out (req_valid),
        .ready_out (req_ready)
    );

    // address generation

    wire mem_req_valid;
    wire [NUM_LANES-1:0] mem_req_mask;
    wire [NUM_LANES-1:0][TEX_FILTER_BITS-1:0] mem_req_filter;
    wire [`TEX_LGSTRIDE_BITS-1:0] mem_req_lgstride;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][1:0][`TEX_BLEND_FRAC-1:0] mem_req_blends;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_FRAC_BITS-1:0] mem_req_lodfrac;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0] mem_req_border;
    wire [NUM_LANES-1:0][1:0] mem_req_flip;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] mem_req_addr;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] mem_req_baseaddr;
    wire [INFO_W-1:0] mem_req_info;
    wire mem_req_ready;

    // The border colour rides the request rather than being read at the
    // sampler, for the same reason the format does: the state a sample merges
    // under is the state it was issued under.
    VX_tex_addr #(
        .INSTANCE_ID ($sformatf("%s-addr", INSTANCE_ID)),
        .REQ_TAGW    (INFO_W),
        .NUM_LANES   (NUM_LANES)
    ) tex_addr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .req_valid  (req_valid),
        .req_mask   (req_mask),
        .req_coords (req_coords),
        .req_format (req_format),
        .req_filter (req_filter),
        .req_wraps  (req_wraps),
        .req_baseaddr(req_baseaddr),
        .req_miplevel(req_miplevel),
        .req_mipoff (req_mipoff),
        .req_lodfrac(req_lodfrac),
        .req_dims   (req_dims),
        .req_tag    ({req_tag, req_cmpinfo, req_reduction, req_gather, req_format, req_bordercolor}),
        .req_ready  (req_ready),

        // outputs
        .rsp_valid  (mem_req_valid),
        .rsp_mask   (mem_req_mask),
        .rsp_filter (mem_req_filter),
        .rsp_lgstride(mem_req_lgstride),
        .rsp_baseaddr(mem_req_baseaddr),
        .rsp_addr   (mem_req_addr),
        .rsp_blends (mem_req_blends),
        .rsp_border (mem_req_border),
        .rsp_flip   (mem_req_flip),
        .rsp_lodfrac(mem_req_lodfrac),
        .rsp_tag    (mem_req_info),
        .rsp_ready  (mem_req_ready)
    );

    // A tap whose blend weight is zero cannot change the result, so it is not
    // fetched: the high-u taps when alpha is 0, the high-v taps when beta is 0,
    // and the upper level when the level weight is 0.
    // A gather returns all four taps, so none of its taps is skipped.
    wire [NUM_LANES-1:0][2:0] mem_req_gather = mem_req_info[(TEX_FORMAT_BITS + 32) +: GATHER_W];
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0] mem_req_skip_u, mem_req_skip_v;
    wire [NUM_LANES-1:0] mem_req_skip_l1;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_skip
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            assign mem_req_skip_u[i][k] = (mem_req_blends[i][k][0] == '0) && ~mem_req_gather[i][2];
            assign mem_req_skip_v[i][k] = (mem_req_blends[i][k][1] == '0) && ~mem_req_gather[i][2];
        end
        assign mem_req_skip_l1[i] = (mem_req_lodfrac[i] == '0);
    end

    // retrieve texel values from memory

    wire mem_rsp_valid;
    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] mem_rsp_data;
    wire [(FLIP_W + INFO_W + BLEND_FRAC_W + BORDER_W + LODFRAC_W)-1:0] mem_rsp_info;
    wire mem_rsp_ready;

    VX_tex_mem #(
        .INSTANCE_ID ($sformatf("%s-mem", INSTANCE_ID)),
        .REQ_TAGW    (FLIP_W + INFO_W + BLEND_FRAC_W + BORDER_W + LODFRAC_W),
        .NUM_LANES   (NUM_LANES)
    ) tex_mem (
        .clk       (clk),
        .reset     (reset),

        // memory interface
        .cache_bus_if (cache_bus_if),

        // inputs
        .req_valid (mem_req_valid),
        .req_mask  (mem_req_mask),
        .req_filter(mem_req_filter),
        .req_skip_u(mem_req_skip_u),
        .req_skip_v(mem_req_skip_v),
        .req_skip_l1(mem_req_skip_l1),
        .req_lgstride(mem_req_lgstride),
        .req_baseaddr(mem_req_baseaddr),
        .req_addr  (mem_req_addr),
        .req_tag   ({mem_req_flip, mem_req_info, mem_req_blends, mem_req_border, mem_req_lodfrac}),
        .req_ready (mem_req_ready),

        // outputs
        .rsp_valid (mem_rsp_valid),
        .rsp_data  (mem_rsp_data),
        .rsp_tag   (mem_rsp_info),
        .rsp_ready (mem_rsp_ready)
    );

    // apply sampler

    wire sampler_rsp_valid;
    wire [NUM_LANES-1:0][31:0] sampler_rsp_data;
    wire [TAG_WIDTH-1:0] sampler_rsp_info;
    wire sampler_rsp_ready;

    wire [LODFRAC_W-1:0] mem_rsp_lodfrac = mem_rsp_info[0 +: LODFRAC_W];
    wire [BORDER_W-1:0] mem_rsp_border = mem_rsp_info[LODFRAC_W +: BORDER_W];
    wire [BLEND_FRAC_W-1:0] mem_rsp_blends = mem_rsp_info[(LODFRAC_W + BORDER_W) +: BLEND_FRAC_W];
    wire [31:0] mem_rsp_bordercolor = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W) +: 32];
    wire [TEX_FORMAT_BITS-1:0] mem_rsp_format = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W + 32) +: TEX_FORMAT_BITS];
    wire [GATHER_W-1:0] mem_rsp_gather = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W + 32 + TEX_FORMAT_BITS) +: GATHER_W];
    wire [RED_W-1:0] mem_rsp_reduction = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W + 32 + TEX_FORMAT_BITS + GATHER_W) +: RED_W];
    wire [CMP_W-1:0] mem_rsp_cmpinfo = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W + 32 + TEX_FORMAT_BITS + GATHER_W + RED_W) +: CMP_W];
    wire [TAG_WIDTH-1:0] mem_rsp_tag = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W + 32 + TEX_FORMAT_BITS + GATHER_W + RED_W + CMP_W) +: TAG_WIDTH];
    wire [FLIP_W-1:0] mem_rsp_flip = mem_rsp_info[(LODFRAC_W + BORDER_W + BLEND_FRAC_W + INFO_W) +: FLIP_W];

    wire mem_rsp_cmp = mem_rsp_cmpinfo[CMP_W-1];
    wire [TEX_CMP_BITS-1:0] mem_rsp_compare = mem_rsp_cmpinfo[SLOT_W +: TEX_CMP_BITS];
    assign ref_raddr = mem_rsp_cmpinfo[0 +: SLOT_W];
    assign ref_release = mem_rsp_valid && mem_rsp_ready && mem_rsp_cmp;

    VX_tex_sampler #(
        .INSTANCE_ID ($sformatf("%s-sampler", INSTANCE_ID)),
        .REQ_TAGW    (TAG_WIDTH),
        .NUM_LANES   (NUM_LANES)
    ) tex_sampler (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .req_valid       (mem_rsp_valid),
        .req_data        (mem_rsp_data),
        .req_blends      (mem_rsp_blends),
        .req_border      (mem_rsp_border),
        .req_bordercolor (mem_rsp_bordercolor),
        .req_lodfrac     (mem_rsp_lodfrac),
        .req_format      (mem_rsp_format),
        .req_gather      (mem_rsp_gather),
        .req_flip        (mem_rsp_flip),
        .req_reduction   (mem_rsp_reduction),
        .req_cmp         (mem_rsp_cmp),
        .req_compare     (mem_rsp_compare),
        .req_ref         (ref_rdata),
        .req_tag         (mem_rsp_tag),
        .req_ready       (mem_rsp_ready),

        // outputs
        .rsp_valid       (sampler_rsp_valid),
        .rsp_data        (sampler_rsp_data),
        .rsp_tag         (sampler_rsp_info),
        .rsp_ready       (sampler_rsp_ready)
    );

    VX_elastic_buffer #(
        .DATAW   (NUM_LANES * 32 + TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (2) // external bus should be registered
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (sampler_rsp_valid),
        .ready_in  (sampler_rsp_ready),
        .data_in   ({sampler_rsp_data, sampler_rsp_info}),
        .data_out  ({tex_bus_if.rsp_data.texels, tex_bus_if.rsp_data.tag}),
        .valid_out (tex_bus_if.rsp_valid),
        .ready_out (tex_bus_if.rsp_ready)
    );

`ifdef SCOPE
`ifdef DBG_SCOPE_TEX
    `SCOPE_IO_SWITCH (1);
    wire cache_bus_req_fire_0 = cache_bus_if[0].req_valid && cache_bus_if[0].req_ready;
    wire cache_bus_rsp_fire_0 = cache_bus_if[0].rsp_valid && cache_bus_if[0].rsp_ready;
    wire [TCACHE_TAG_WIDTH-UUID_WIDTH-1:0] cache_bus_req_tag  = cache_bus_if[0].req_data.tag[TCACHE_TAG_WIDTH-UUID_WIDTH-1:0];
    wire [UUID_WIDTH-1:0] cache_bus_req_uuid = cache_bus_if[0].req_data.tag[TCACHE_TAG_WIDTH-1 -: UUID_WIDTH];
    wire [TCACHE_TAG_WIDTH-UUID_WIDTH-1:0] cache_bus_rsp_tag  = cache_bus_if[0].rsp_data.tag[TCACHE_TAG_WIDTH-UUID_WIDTH-1:0];
    wire [UUID_WIDTH-1:0] cache_bus_rsp_uuid = cache_bus_if[0].rsp_data.tag[TCACHE_TAG_WIDTH-1 -: UUID_WIDTH];
    wire [TEX_REQ_ARB1_TAG_WIDTH-UUID_WIDTH-1:0] tex_bus_req_tag = tex_bus_if.req_data.tag[TEX_REQ_ARB1_TAG_WIDTH-UUID_WIDTH-1:0];
    wire [UUID_WIDTH-1:0] tex_bus_req_uuid = tex_bus_if.req_data.tag[TEX_REQ_ARB1_TAG_WIDTH-1 -: UUID_WIDTH];
    wire [TEX_REQ_ARB1_TAG_WIDTH-UUID_WIDTH-1:0] tex_bus_rsp_tag = tex_bus_if.rsp_data.tag[TEX_REQ_ARB1_TAG_WIDTH-UUID_WIDTH-1:0];
    wire [UUID_WIDTH-1:0] tex_bus_rsp_uuid = tex_bus_if.rsp_data.tag[TEX_REQ_ARB1_TAG_WIDTH-1 -: UUID_WIDTH];
    wire tex_bus_req_fire = tex_bus_if.req_valid && tex_bus_if.req_ready;
    wire tex_bus_rsp_fire = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 5, 8, 5, (
            TCACHE_ADDR_WIDTH + 1 + TCACHE_TAG_WIDTH +
            (TCACHE_WORD_SIZE * 8) + TCACHE_TAG_WIDTH +
            VX_DCR_ADDR_WIDTH + VX_DCR_DATA_WIDTH +
            1 * (1 + 2 * 32 + TEX_ARG_BITS) + TEX_STAGE_BITS + TAG_WIDTH +
            1 * 32 + TAG_WIDTH +
            `TEX_ADDR_BITS + `TEX_MIPOFF_BITS + 2 * TEX_DIM_W
        ), {
            cache_bus_if[0].req_valid,
            cache_bus_if[0].req_ready,
            cache_bus_if[0].rsp_valid,
            cache_bus_if[0].rsp_ready,
            tex_bus_if.req_valid,
            tex_bus_if.req_ready,
            tex_bus_if.rsp_valid,
            tex_bus_if.rsp_ready
        }, {
            cache_bus_req_fire_0,
            cache_bus_rsp_fire_0,
            dcr_bus_if.write_valid,
            tex_bus_req_fire,
            tex_bus_rsp_fire
        },{
            cache_bus_if[0].req_data.addr,
            cache_bus_if[0].req_data.rw,
            cache_bus_req_tag,
            cache_bus_req_uuid,
            cache_bus_if[0].rsp_data.data,
            cache_bus_rsp_tag,
            cache_bus_rsp_uuid,
            dcr_bus_if.write_addr,
            dcr_bus_if.write_data,
            tex_bus_if.req_data.mask[0],
            tex_bus_if.req_data.coords[0][0],
            tex_bus_if.req_data.coords[1][0],
            tex_bus_if.req_data.lod[0],
            tex_bus_if.req_data.stage,
            tex_bus_req_tag,
            tex_bus_req_uuid,
            tex_bus_if.rsp_data.texels[0],
            tex_bus_rsp_tag,
            tex_bus_rsp_uuid,
            tex_dcrs.baseaddr,
            tex_dcrs.mipoff[0],
            tex_dcrs.dims[0],
            tex_dcrs.dims[1]
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED()
`endif
`endif
`ifdef CHIPSCOPE
    ila_tex ila_tex_inst (
        .clk    (clk),
        .probe0 ({cache_bus_if[0].rsp_data.data, cache_bus_if[0].rsp_data.tag, cache_bus_if[0].rsp_ready, cache_bus_if[0].rsp_valid, cache_bus_if[0].req_data.tag, cache_bus_if[0].req_data.addr, cache_bus_if[0].req_data.rw, cache_bus_if[0].req_valid, cache_bus_if[0].req_ready}),
        .probe1 ({dcr_bus_if.write_valid, dcr_bus_if.write_addr, dcr_bus_if.write_data}),
        .probe2 ({tex_bus_if.req_valid, tex_bus_if.req_data, tex_bus_if.req_ready}),
        .probe3 ({tex_bus_if.rsp_valid, tex_bus_if.rsp_data, tex_bus_if.rsp_ready})
    );
`endif

`ifdef PERF_ENABLE
    wire [`CLOG2(TCACHE_NUM_REQS+1)-1:0] perf_mem_req_per_cycle;
    wire [`CLOG2(TCACHE_NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;
    wire [`CLOG2(TCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [TCACHE_NUM_REQS-1:0] perf_mem_req_fire;
    for (genvar i = 0; i < TCACHE_NUM_REQS; ++i) begin : g_perf_mem_req_fire
        assign perf_mem_req_fire[i] = cache_bus_if[i].req_valid && cache_bus_if[i].req_ready;
    end

    wire [TCACHE_NUM_REQS-1:0] perf_mem_rsp_fire;
    for (genvar i = 0; i < TCACHE_NUM_REQS; ++i) begin : g_perf_mem_rsp_fire
        assign perf_mem_rsp_fire[i] = cache_bus_if[i].rsp_valid && cache_bus_if[i].rsp_ready;
    end

    `POP_COUNT(perf_mem_req_per_cycle, perf_mem_req_fire);
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_fire);

    reg [PERF_CTR_BITS-1:0] perf_pending_reads;
    assign perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = tex_bus_if.req_valid & ~tex_bus_if.req_ready;

    reg [PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads + PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency + PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_tex_if.mem_reads    = perf_mem_reads;
    assign perf_tex_if.mem_latency  = perf_mem_latency;
    assign perf_tex_if.stall_cycles = perf_stall_cycles;
`endif

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_bus_if.req_valid && tex_bus_if.req_ready) begin
            `TRACE(1, ("%d: %s-req: valid=%b, stage=%0d, lod=0x%0h, u=",
                    $time, INSTANCE_ID, tex_bus_if.req_data.mask, tex_bus_if.req_data.stage, tex_bus_if.req_data.lod))
            `TRACE_ARRAY1D(1, "0x%0h", tex_bus_if.req_data.coords[0], NUM_LANES)
            `TRACE(1, (", v="))
            `TRACE_ARRAY1D(1, "0x%0h", tex_bus_if.req_data.coords[1], NUM_LANES)
            `TRACE(1, (", tag=0x%0h (#%0d)\n", tex_bus_if.req_data.tag, tex_bus_if.req_data.tag[TAG_WIDTH-1 -: UUID_WIDTH]))
        end
        if (tex_bus_if.rsp_valid && tex_bus_if.rsp_ready) begin
            `TRACE(1, ("%d: %s-rsp: texels=", $time, INSTANCE_ID))
            `TRACE_ARRAY1D(1, "0x%0h", tex_bus_if.rsp_data.texels, NUM_LANES)
            `TRACE(1, (", tag=0x%0h (#%0d)\n", tex_bus_if.rsp_data.tag, tex_bus_if.rsp_data.tag[TAG_WIDTH-1 -: UUID_WIDTH]))
        end
    end
`endif

endmodule
