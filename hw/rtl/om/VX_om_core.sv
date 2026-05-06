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

`include "VX_om_define.vh"

module VX_om_core import VX_gpu_pkg::*; import VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 4
) (
    `SCOPE_IO_DECL

    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_om_perf_if.master   perf_om_if,
`endif

    // Memory interface
    VX_mem_bus_if.master    cache_bus_if [OCACHE_NUM_REQS],

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,
    VX_om_bus_if.slave      om_bus_if
);
    localparam MEM_TAG_WIDTH   = UUID_WIDTH + NUM_LANES * (`VX_OM_DIM_BITS + `VX_OM_DIM_BITS + 32 + `VX_OM_DEPTH_BITS + 1);
    localparam DS_TAG_WIDTH    = UUID_WIDTH + NUM_LANES * (`VX_OM_DIM_BITS + `VX_OM_DIM_BITS + 1 + 1 + 32);
    localparam BLEND_TAG_WIDTH = UUID_WIDTH + NUM_LANES * (`VX_OM_DIM_BITS + `VX_OM_DIM_BITS + 1);

    // DCRs

    om_dcrs_t om_dcrs;

    VX_om_dcr #(
        .INSTANCE_ID ($sformatf("%s-dcr", INSTANCE_ID))
    ) om_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .om_dcrs    (om_dcrs)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                                    mem_req_valid, mem_req_valid_r;
    wire [NUM_LANES-1:0]                    mem_req_ds_mask, mem_req_ds_mask_r;
    wire [NUM_LANES-1:0]                    mem_req_c_mask, mem_req_c_mask_r;
    wire                                    mem_req_rw, mem_req_rw_r;
    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] mem_req_pos_x, mem_req_pos_x_r;
    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] mem_req_pos_y, mem_req_pos_y_r;
    om_color_t [NUM_LANES-1:0]              mem_req_color, mem_req_color_r;
    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0] mem_req_depth, mem_req_depth_r;
    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] mem_req_stencil, mem_req_stencil_r;
    wire [NUM_LANES-1:0]                    mem_req_face, mem_req_face_r;
    wire [MEM_TAG_WIDTH-1:0]                mem_req_tag, mem_req_tag_r;
    wire                                    mem_req_ready, mem_req_ready_r;

    wire                                    mem_rsp_valid;
    wire [NUM_LANES-1:0]                    mem_rsp_mask;
    om_color_t [NUM_LANES-1:0]              mem_rsp_color;
    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0] mem_rsp_depth;
    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] mem_rsp_stencil;
    wire [MEM_TAG_WIDTH-1:0]                mem_rsp_tag;
    wire                                    mem_rsp_ready;
    wire                                    mem_write_notify;

    VX_om_mem #(
        .INSTANCE_ID ($sformatf("%s-mem", INSTANCE_ID)),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (MEM_TAG_WIDTH)
    ) om_mem (
        .clk            (clk),
        .reset          (reset),

        .dcrs           (om_dcrs),

        .cache_bus_if   (cache_bus_if),

        .req_valid      (mem_req_valid_r),
        .req_ds_mask    (mem_req_ds_mask_r),
        .req_c_mask     (mem_req_c_mask_r),
        .req_rw         (mem_req_rw_r),
        .req_pos_x      (mem_req_pos_x_r),
        .req_pos_y      (mem_req_pos_y_r),
        .req_color      (mem_req_color_r),
        .req_depth      (mem_req_depth_r),
        .req_stencil    (mem_req_stencil_r),
        .req_face       (mem_req_face_r),
        .req_tag        (mem_req_tag_r),
        .req_ready      (mem_req_ready_r),
        .write_notify   (mem_write_notify),

        .rsp_valid      (mem_rsp_valid),
        .rsp_mask       (mem_rsp_mask),
        .rsp_color      (mem_rsp_color),
        .rsp_depth      (mem_rsp_depth),
        .rsp_stencil    (mem_rsp_stencil),
        .rsp_tag        (mem_rsp_tag),
        .rsp_ready      (mem_rsp_ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                    ds_valid_in;
    wire [DS_TAG_WIDTH-1:0] ds_tag_in;
    wire                    ds_ready_in;
    wire                    ds_valid_out;
    wire [DS_TAG_WIDTH-1:0] ds_tag_out;
    wire                    ds_ready_out;

    wire [NUM_LANES-1:0]    ds_face;

    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]    ds_depth_ref;
    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]    ds_depth_val;
    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0]  ds_stencil_val;

    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0]    ds_depth_out;
    wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0]  ds_stencil_out;
    wire [NUM_LANES-1:0]                           ds_pass_out;

    VX_om_ds #(
        .INSTANCE_ID ($sformatf("%s-ds", INSTANCE_ID)),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (DS_TAG_WIDTH)
    ) om_ds (
        .clk            (clk),
        .reset          (reset),

        .dcrs           (om_dcrs),

        .valid_in       (ds_valid_in),
        .tag_in         (ds_tag_in),
        .ready_in       (ds_ready_in),

        .valid_out      (ds_valid_out),
        .tag_out        (ds_tag_out),
        .ready_out      (ds_ready_out),

        .face           (ds_face),
        .depth_ref      (ds_depth_ref),
        .depth_val      (ds_depth_val),
        .stencil_val    (ds_stencil_val),

        .depth_out      (ds_depth_out),
        .stencil_out    (ds_stencil_out),
        .pass_out       (ds_pass_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire                    blend_valid_in;
    wire [BLEND_TAG_WIDTH-1:0] blend_tag_in;
    wire                    blend_ready_in;
    wire                    blend_valid_out;
    wire [BLEND_TAG_WIDTH-1:0] blend_tag_out;
    wire                    blend_ready_out;

    om_color_t [NUM_LANES-1:0]  blend_src_color;
    om_color_t [NUM_LANES-1:0]  blend_dst_color;
    om_color_t [NUM_LANES-1:0]  blend_color_out;

    VX_om_blend #(
        .INSTANCE_ID ($sformatf("%s-blend", INSTANCE_ID)),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (BLEND_TAG_WIDTH)
    ) om_blend (
        .clk            (clk),
        .reset          (reset),

        .dcrs           (om_dcrs),

        .valid_in       (blend_valid_in),
        .tag_in         (blend_tag_in),
        .ready_in       (blend_ready_in),

        .valid_out      (blend_valid_out),
        .tag_out        (blend_tag_out),
        .ready_out      (blend_ready_out),

        .src_color      (blend_src_color),
        .dst_color      (blend_dst_color),
        .color_out      (blend_color_out)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire color_writeen = (om_dcrs.cbuf_writemask != 0);

    wire depth_enable  = om_dcrs.depth_enable;
    wire depth_writeen = om_dcrs.depth_enable && (om_dcrs.depth_writemask != 0);

    wire stencil_enable  = (| om_dcrs.stencil_enable);
    wire stencil_writeen = (om_dcrs.stencil_enable[0] && (om_dcrs.stencil_writemask[0] != 0))
                        || (om_dcrs.stencil_enable[1] && (om_dcrs.stencil_writemask[1] != 0));

    wire ds_enable  = depth_enable || stencil_enable;
    wire ds_writeen = depth_writeen || stencil_writeen;

    wire blend_enable  = om_dcrs.blend_enable;
    wire blend_writeen = om_dcrs.blend_enable && color_writeen;

    wire ds_color_writeen = ds_writeen || (ds_enable && color_writeen);

    wire mem_readen = ds_color_writeen || blend_writeen;

    wire write_bypass = ~ds_enable && ~blend_enable && color_writeen;

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] mem_rsp_pos_x, mem_rsp_pos_y;
    wire [UUID_WIDTH-1:0] mem_rsp_uuid;
    `UNUSED_VAR (mem_rsp_uuid)

    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] ds_write_pos_x, ds_write_pos_y;
    wire [NUM_LANES-1:0] ds_write_face, ds_rsp_mask;
    om_color_t [NUM_LANES-1:0] ds_write_color;
    wire [UUID_WIDTH-1:0] ds_write_uuid;

    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] blend_write_pos_x, blend_write_pos_y;
    wire [NUM_LANES-1:0] blend_rsp_mask;
    wire [UUID_WIDTH-1:0] blend_write_uuid;

    wire [MEM_TAG_WIDTH-1:0] def_mem_req_tag, ds_write_tag, blend_write_tag;

    wire pending_reads_full;

    assign def_mem_req_tag = {om_bus_if.req_data.uuid, om_bus_if.req_data.pos_x, om_bus_if.req_data.pos_y, om_bus_if.req_data.color, om_bus_if.req_data.depth, om_bus_if.req_data.face};
    assign {mem_rsp_uuid, mem_rsp_pos_x, mem_rsp_pos_y, blend_src_color, ds_depth_ref, ds_face} = mem_rsp_tag;

    assign ds_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask, ds_face, blend_src_color, mem_rsp_uuid};
    assign {ds_write_pos_x, ds_write_pos_y, ds_rsp_mask, ds_write_face, ds_write_color, ds_write_uuid} = ds_tag_out;
    assign ds_write_tag = {ds_write_uuid, (MEM_TAG_WIDTH-UUID_WIDTH)'(0)};

    assign blend_tag_in = {mem_rsp_pos_x, mem_rsp_pos_y, mem_rsp_mask, mem_rsp_uuid};
    assign {blend_write_pos_x, blend_write_pos_y, blend_rsp_mask, blend_write_uuid} = blend_tag_out;
    assign blend_write_tag = {blend_write_uuid, (MEM_TAG_WIDTH-UUID_WIDTH)'(0)};

    wire color_write = om_bus_if.req_valid && write_bypass;

    wire ds_blend_read = om_bus_if.req_valid && mem_readen && ~pending_reads_full;

    wire ds_write = ds_color_writeen && ds_valid_out;

    wire blend_write = blend_writeen && blend_valid_out;

    wire ds_blend_write_any = ds_write || blend_write;

    wire ds_blend_write_sync = (ds_color_writeen && blend_writeen) ? (ds_valid_out && blend_valid_out) : ds_blend_write_any;

    wire [NUM_LANES-1:0] ds_read_mask, ds_write_mask;
    wire [NUM_LANES-1:0] blend_read_mask, blend_write_mask;
    wire [NUM_LANES-1:0] color_bypass_mask, ds_color_write_mask;

    for (genvar i = 0;  i < NUM_LANES; ++i) begin : g_masks
        assign ds_read_mask[i]        = om_bus_if.req_data.mask[i] && ds_enable;
        assign blend_read_mask[i]     = om_bus_if.req_data.mask[i] && blend_writeen;
        assign ds_write_mask[i]       = ds_rsp_mask[i] && (stencil_writeen || (depth_writeen && ds_pass_out[i]));
        assign blend_write_mask[i]    = blend_rsp_mask[i] && blend_writeen && (~ds_enable || ds_pass_out[i]);
        assign color_bypass_mask[i]   = om_bus_if.req_data.mask[i] && color_writeen;
        assign ds_color_write_mask[i] = ds_rsp_mask[i] && ds_pass_out[i];
    end

    assign mem_req_valid    = ds_blend_write_sync || ds_blend_read || color_write;
    assign mem_req_ds_mask  = ds_valid_out ? ds_write_mask : ds_read_mask;
    assign mem_req_c_mask   = write_bypass ? color_bypass_mask : (blend_valid_out ? blend_write_mask : (ds_valid_out ? ds_color_write_mask : blend_read_mask));
    assign mem_req_rw       = ds_blend_write_any || write_bypass;
    assign mem_req_face     = ds_write_face;
    assign mem_req_pos_x    = ds_valid_out ? ds_write_pos_x : (blend_valid_out ? blend_write_pos_x : om_bus_if.req_data.pos_x);
    assign mem_req_pos_y    = ds_valid_out ? ds_write_pos_y : (blend_valid_out ? blend_write_pos_y : om_bus_if.req_data.pos_y);
    assign mem_req_color    = blend_enable ? blend_color_out : (ds_enable ? ds_write_color : om_bus_if.req_data.color);
    assign mem_req_depth    = ds_depth_out;
    assign mem_req_stencil  = ds_stencil_out;
    assign mem_req_tag      = ds_valid_out ? ds_write_tag : (blend_valid_out ? blend_write_tag : def_mem_req_tag);

    assign om_bus_if.req_ready = mem_req_ready && ~ds_blend_write_any && ~(mem_readen && pending_reads_full);
    assign ds_ready_out     = mem_req_ready && (~blend_writeen || blend_valid_out);
    assign blend_ready_out  = mem_req_ready && (~ds_color_writeen || ds_valid_out);

    assign ds_valid_in      = ds_enable && mem_rsp_valid && (~blend_enable || blend_ready_in);
    assign blend_valid_in   = blend_enable && mem_rsp_valid && (~ds_enable || ds_ready_in);
    assign blend_dst_color  = mem_rsp_color;

    assign ds_depth_val     = mem_rsp_depth;
    assign ds_stencil_val   = mem_rsp_stencil;
    assign mem_rsp_ready    = (ds_enable && blend_enable) ? (ds_ready_in && blend_ready_in) :
                                (ds_enable ? ds_ready_in :
                                    (blend_enable ? blend_ready_in :
                                        1'b0));

    wire mem_req_fire = mem_req_valid && mem_req_ready;

    wire write_req_canceled;

    // We need to ensure that read responses can be processed without stalls
    // otherwise we get into potential read/write deadlock.
    // ensure the memory scheduler's queue doesn't fill up
    VX_pending_size #(
        .SIZE (`OM_MEM_QUEUE_SIZE)
    ) pending_reads (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_req_fire && ~mem_req_rw && (ds_color_writeen || blend_writeen)),
        .decr  ((mem_write_notify || write_req_canceled) && (ds_color_writeen || blend_writeen)),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (alm_empty),
        .full  (pending_reads_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    wire mem_req_valid_unqual_r;

    VX_elastic_buffer #(
        .DATAW   (1 + NUM_LANES * (1 + 1 + 2 * `VX_OM_DIM_BITS + $bits(om_color_t) + `VX_OM_DEPTH_BITS + `VX_OM_STENCIL_BITS + 1) + MEM_TAG_WIDTH),
        .OUT_REG (1)
    ) mem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_valid),
        .ready_in  (mem_req_ready),
        .data_in   ({mem_req_rw, mem_req_ds_mask, mem_req_c_mask, mem_req_pos_x, mem_req_pos_y, mem_req_color, mem_req_depth, mem_req_stencil, mem_req_face, mem_req_tag}),
        .data_out  ({mem_req_rw_r, mem_req_ds_mask_r, mem_req_c_mask_r, mem_req_pos_x_r, mem_req_pos_y_r, mem_req_color_r, mem_req_depth_r, mem_req_stencil_r, mem_req_face_r, mem_req_tag_r}),
        .valid_out (mem_req_valid_unqual_r),
        .ready_out (mem_req_ready_r)
    );

    wire is_degenerate_req = (mem_req_ds_mask_r | mem_req_c_mask_r) == 0;

    assign mem_req_valid_r = mem_req_valid_unqual_r && ~is_degenerate_req;

    assign write_req_canceled = mem_req_valid_unqual_r && mem_req_rw_r && is_degenerate_req && mem_req_ready_r;

`ifdef SCOPE
`ifdef DBG_SCOPE_OM
    `SCOPE_IO_SWITCH (1);
    wire cache_bus_req_fire_0 = cache_bus_if[0].req_valid && cache_bus_if[0].req_ready;
    wire cache_bus_rsp_fire_0 = cache_bus_if[0].rsp_valid && cache_bus_if[0].rsp_ready;
    wire [OCACHE_TAG_WIDTH-UUID_WIDTH-1:0] cache_bus_req_tag  = cache_bus_if[0].req_data.tag[OCACHE_TAG_WIDTH-UUID_WIDTH-1:0];
    wire [UUID_WIDTH-1:0] cache_bus_req_uuid = cache_bus_if[0].req_data.tag[OCACHE_TAG_WIDTH-1 -: UUID_WIDTH];
    wire [OCACHE_TAG_WIDTH-UUID_WIDTH-1:0] cache_bus_rsp_tag  = cache_bus_if[0].rsp_data.tag[OCACHE_TAG_WIDTH-UUID_WIDTH-1:0];
    wire [UUID_WIDTH-1:0] cache_bus_rsp_uuid = cache_bus_if[0].rsp_data.tag[OCACHE_TAG_WIDTH-1 -: UUID_WIDTH];
    wire om_bus_fire = om_bus_if.req_valid && om_bus_if.req_ready;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 6, 6, 4, (
            OCACHE_ADDR_WIDTH + 1 + OCACHE_TAG_WIDTH +
            (OCACHE_WORD_SIZE * 8) + OCACHE_TAG_WIDTH +
            VX_DCR_ADDR_WIDTH + VX_DCR_DATA_WIDTH +
            1 * (1 + `VX_OM_DIM_BITS + `VX_OM_DIM_BITS + $bits(om_color_t) + `VX_OM_DEPTH_BITS + 1) +
            `OM_ADDR_BITS + `VX_OM_PITCH_BITS + `OM_ADDR_BITS + `VX_OM_PITCH_BITS + UUID_WIDTH
        ), {
            cache_bus_if[0].req_valid,
            cache_bus_if[0].req_ready,
            cache_bus_if[0].rsp_valid,
            cache_bus_if[0].rsp_ready,
            om_bus_if.req_valid,
            om_bus_if.req_ready
        }, {
            cache_bus_req_fire_0,
            cache_bus_rsp_fire_0,
            dcr_bus_if.write_valid,
            om_bus_fire
        }, {
            cache_bus_if[0].req_data.addr,
            cache_bus_if[0].req_data.rw,
            cache_bus_req_tag,
            cache_bus_req_uuid,
            cache_bus_if[0].rsp_data.data,
            cache_bus_rsp_tag,
            cache_bus_rsp_uuid,
            dcr_bus_if.write_addr,
            dcr_bus_if.write_data,
            om_bus_if.req_data.mask[0],
            om_bus_if.req_data.pos_x[0],
            om_bus_if.req_data.pos_y[0],
            om_bus_if.req_data.color[0],
            om_bus_if.req_data.depth[0],
            om_bus_if.req_data.face[0],
            om_bus_if.req_data.uuid,
            om_dcrs.cbuf_addr,
            om_dcrs.cbuf_pitch,
            om_dcrs.zbuf_addr,
            om_dcrs.zbuf_pitch
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED()
`endif
`endif
`ifdef CHIPSCOPE
    ila_om ila_om_inst (
        .clk    (clk),
        .probe0 ({cache_bus_if[0].rsp_data.data, cache_bus_if[0].rsp_data.tag, cache_bus_if[0].rsp_ready, cache_bus_if[0].rsp_valid, cache_bus_if[0].req_data.tag, cache_bus_if[0].req_data.addr, cache_bus_if[0].req_data.rw, cache_bus_if[0].req_valid, cache_bus_if[0].req_ready}),
        .probe1 ({dcr_bus_if.write_valid, dcr_bus_if.write_addr, dcr_bus_if.write_data}),
        .probe2 ({om_bus_if.req_valid, om_bus_if.req_data, om_bus_if.req_ready})
    );
`endif

`ifdef PERF_ENABLE

    wire [`CLOG2(OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_req_per_cycle;
    wire [`CLOG2(OCACHE_NUM_REQS+1)-1:0] perf_mem_wr_req_per_cycle;
    wire [`CLOG2(OCACHE_NUM_REQS+1)-1:0] perf_mem_rd_rsp_per_cycle;
    wire [`CLOG2(OCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [OCACHE_NUM_REQS-1:0] perf_mem_rd_req_fire;
    for (genvar i = 0; i < OCACHE_NUM_REQS; ++i) begin : g_perf_mem_rd_req_fire
        assign perf_mem_rd_req_fire[i] = cache_bus_if[i].req_valid && ~cache_bus_if[i].req_data.rw && cache_bus_if[i].req_ready;
    end

    wire [OCACHE_NUM_REQS-1:0] perf_mem_wr_req_fire;
    for (genvar i = 0; i < OCACHE_NUM_REQS; ++i) begin : g_perf_mem_wr_req_fire
        assign perf_mem_wr_req_fire[i] = cache_bus_if[i].rsp_valid && cache_bus_if[i].req_data.rw && cache_bus_if[i].rsp_ready;
    end

    wire [OCACHE_NUM_REQS-1:0] perf_mem_rd_rsp_fire;
    for (genvar i = 0; i < OCACHE_NUM_REQS; ++i) begin : g_perf_mem_rd_rsp_fire
        assign perf_mem_rd_rsp_fire[i] = cache_bus_if[i].rsp_valid && cache_bus_if[i].rsp_ready;
    end

    `POP_COUNT(perf_mem_rd_req_per_cycle, perf_mem_rd_req_fire);
    `POP_COUNT(perf_mem_wr_req_per_cycle, perf_mem_wr_req_fire);
    `POP_COUNT(perf_mem_rd_rsp_per_cycle, perf_mem_rd_rsp_fire);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;
    assign perf_pending_reads_cycle = perf_mem_rd_req_per_cycle - perf_mem_rd_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = om_bus_if.req_valid & ~om_bus_if.req_ready;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_writes;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_writes   <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads    + `PERF_CTR_BITS'(perf_mem_rd_req_per_cycle);
            perf_mem_writes   <= perf_mem_writes   + `PERF_CTR_BITS'(perf_mem_wr_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency  + `PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_om_if.mem_reads    = perf_mem_reads;
    assign perf_om_if.mem_writes   = perf_mem_writes;
    assign perf_om_if.mem_latency  = perf_mem_latency;
    assign perf_om_if.stall_cycles = perf_stall_cycles;

`endif

endmodule
