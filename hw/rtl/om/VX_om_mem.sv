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

module VX_om_mem import VX_gpu_pkg::*; import VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 4,
    parameter TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // Device configuration. rt_dcrs is the attachment state already selected for
    // the request in flight, so the address pipeline never sees the index.
    input om_dcrs_t    dcrs,
    input om_rt_dcrs_t rt_dcrs,

    // Memory interface
    VX_mem_bus_if.master                            cache_bus_if [OCACHE_NUM_REQS],

    // Request interface
    input wire                                      req_valid,
    input wire [NUM_LANES-1:0]                      req_ds_mask,
    input wire [NUM_LANES-1:0]                      req_c_mask,
    input wire                                      req_rw,
    input wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] req_pos_x,
    input wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0] req_pos_y,
    input om_color_t [NUM_LANES-1:0]                req_color,
    input wire [NUM_LANES-1:0][`VX_OM_DEPTH_BUS_BITS-1:0] req_depth,
    input wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] req_stencil,
    input wire [NUM_LANES-1:0]                      req_face,
    input wire [TAG_WIDTH-1:0]                      req_tag,
    output wire                                     req_ready,
    output wire                                     write_notify,
    // High while any request is inside the address pipeline or the memory
    // scheduler — the unit's drain indicator (requests spend LATENCY_IMUL
    // cycles in the address pipe before the scheduler can see them).
    output wire                                     busy,

    // Response interface
    output wire                                     rsp_valid,
    output wire [NUM_LANES-1:0]                     rsp_mask,
    output om_color_t [NUM_LANES-1:0]               rsp_color,
    output wire [NUM_LANES-1:0][`VX_OM_DEPTH_BUS_BITS-1:0] rsp_depth,
    output wire [NUM_LANES-1:0][`VX_OM_STENCIL_BITS-1:0] rsp_stencil,
    output wire [TAG_WIDTH-1:0]                     rsp_tag,
    input wire                                      rsp_ready
);

    localparam NUM_REQS    = OM_MEM_REQS;
    localparam W_ADDR_BITS = (`OM_ADDR_BITS + 6) - 2;
    localparam B_ADDR_BITS = `OM_ADDR_BITS + 6;
    // A colour read returns words; the format and each lane's byte offset ride
    // with the request so the response can be shifted down and unpacked.
    // A D16 read likewise carries which half of its word each lane's texel is.
    localparam MTAG_WIDTH  = TAG_WIDTH + OM_CFMT_BITS + NUM_LANES + 2 * NUM_LANES;

    // The merger blends in ARGB8888. R8 and RG8 hold its R (then G) byte;
    // R5G6B5 and A1R5G5B5 round its channels, x/255 taken as (x+1+(x>>8))>>8,
    // which is exact over the range.
    function automatic [7:0] widen5(input [4:0] v);
        widen5 = {v, v[4:2]};
    endfunction
    function automatic [7:0] widen6(input [5:0] v);
        widen6 = {v, v[5:4]};
    endfunction
    function automatic [5:0] narrow(input [7:0] v, input [5:0] max);
        reg [13:0] x;
        begin
            x = 14'(v) * 14'(max) + 14'd127;
            narrow = 6'((x + 14'd1 + (x >> 8)) >> 8);
        end
    endfunction
    function automatic [31:0] cfmt_unpack(input [OM_CFMT_BITS-1:0] fmt, input [31:0] raw);
        case (fmt)
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_R8):
                cfmt_unpack = {8'hff, raw[7:0], 16'h0};
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_RG8):
                cfmt_unpack = {8'hff, raw[7:0], raw[15:8], 8'h0};
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_R5G6B5):
                cfmt_unpack = {8'hff, widen5(raw[15:11]), widen6(raw[10:5]), widen5(raw[4:0])};
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_A1R5G5B5):
                cfmt_unpack = {{8{raw[15]}}, widen5(raw[14:10]), widen5(raw[9:5]), widen5(raw[4:0])};
            default:
                cfmt_unpack = raw;
        endcase
    endfunction
    function automatic [31:0] cfmt_pack(input [OM_CFMT_BITS-1:0] fmt, input [31:0] argb);
        case (fmt)
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_R8):
                cfmt_pack = {24'h0, argb[23:16]};
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_RG8):
                cfmt_pack = {16'h0, argb[15:8], argb[23:16]};
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_R5G6B5):
                cfmt_pack = {16'h0, 5'(narrow(argb[23:16], 6'd31)), narrow(argb[15:8], 6'd63),
                             5'(narrow(argb[7:0], 6'd31))};
            OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_A1R5G5B5):
                cfmt_pack = {16'h0, argb[31], 5'(narrow(argb[23:16], 6'd31)),
                             5'(narrow(argb[15:8], 6'd31)), 5'(narrow(argb[7:0], 6'd31))};
            default:
                cfmt_pack = argb;
        endcase
    endfunction

    // The memory scheduler cannot fill more memory channels than it has core
    // requests, so a build with more ocache banks than OM memory ports clamps
    // the channel count; the leftover cache ports keep the uniform ocache
    // attach geometry and are tied off below.
    localparam MEM_CHANNELS = `MIN(NUM_REQS, OCACHE_NUM_REQS);

    wire                        mreq_valid, mreq_valid_r;
    wire                        mreq_rw, mreq_rw_r;
    wire [NUM_REQS-1:0]         mreq_mask, mreq_mask_r;
    wire [NUM_REQS-1:0][OCACHE_ADDR_WIDTH-1:0] mreq_addr, mreq_addr_r;
    wire [NUM_REQS-1:0][31:0]   mreq_data, mreq_data_r;
    wire [NUM_REQS-1:0][3:0]    mreq_byteen, mreq_byteen_r;
    wire [TAG_WIDTH-1:0]        mreq_tag;
    wire [MTAG_WIDTH-1:0]       mreq_mtag, mreq_mtag_r;
    wire                        mreq_ready_r;

    wire                        req_queue_empty_w;
    wire                        mrsp_valid;
    wire [NUM_REQS-1:0]         mrsp_mask;
    wire [NUM_REQS-1:0][31:0]   mrsp_data;
    wire [MTAG_WIDTH-1:0]       mrsp_mtag;
    wire                        mrsp_ready;

    `UNUSED_VAR (dcrs)
    `UNUSED_VAR (rt_dcrs)

    `STATIC_ASSERT((`VX_OM_DEPTH_BUS_BITS == 32) && (`VX_OM_DEPTH_BITS + `VX_OM_STENCIL_BITS == 32),
        ("a depth/stencil word is one 32-bit word"))

    // A D24S8 word is {stencil, depth[23:0]}; a D32F word is the depth alone.
    // A D32F_S8 texel is two words, the key then the stencil, so its stencil
    // takes a request slot of its own: slots [0, L) are depth/stencil words,
    // [L, 2L) colour, [2L, 3L) D32F_S8 stencil words.
    `STATIC_ASSERT(NUM_REQS == 3 * NUM_LANES, ("one depth, colour and stencil slot per lane"))
    wire [3:0] color_byteen = rt_dcrs.cbuf_writemask;
    wire [2:0] depth_byteen = {3{dcrs.depth_writemask}};
    wire [NUM_LANES-1:0] stencil_byteen;
    for (genvar i = 0;  i < NUM_LANES; ++i) begin : g_stencil_byteen
        assign stencil_byteen[i] = (dcrs.stencil_writemask[req_face[i]] != 0);
    end

    // A request enters the fixed-latency address pipe only with a reserved
    // slot in the landing queue below, so the pipe never stalls mid-flight and
    // no enable network spans it: multipliers and shift registers free-run.
    wire req_fire = req_valid && req_ready;

    // depth/stencil values submission. A D16 texel is half a word; its row
    // pitch is a multiple of 4 bytes, so the half is the x position's low bit.
    wire [NUM_LANES-1:0] z_half;
    for (genvar i = 0;  i < NUM_LANES; ++i) begin : g_DS
        wire [31:0] m_y_pitch;
        `UNUSED_VAR (m_y_pitch)

        VX_multiplier #(
            .A_WIDTH (`VX_OM_DIM_BITS),
            .B_WIDTH (OM_PITCH_BITS),
            .R_WIDTH (32),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (1'b1),
            .dataa  (req_pos_y[i]),
            .datab  (dcrs.zbuf_pitch),
            .result (m_y_pitch)
        );

        // A D32F_S8 texel is two words wide, a D16 one half a word.
        wire [W_ADDR_BITS-1:0] xoff = dcrs.depth_stencil_word ? (W_ADDR_BITS'(req_pos_x[i]) << 1)
                                    : dcrs.depth_d16 ? (W_ADDR_BITS'(req_pos_x[i]) >> 1)
                                    : W_ADDR_BITS'(req_pos_x[i]);
        wire [W_ADDR_BITS-1:0] baddr, baddr_s;
        assign baddr = {dcrs.zbuf_addr, 4'b0} + xoff;

        wire half = dcrs.depth_d16 && req_pos_x[i][0];
        wire [3:0] d16_byteen = half ? {{2{dcrs.depth_writemask}}, 2'b00}
                                     : {2'b00, {2{dcrs.depth_writemask}}};
        wire [3:0] byteen = req_rw ? (dcrs.depth_float ? {4{dcrs.depth_writemask}}
                                    : dcrs.depth_d16 ? d16_byteen
                                    : {stencil_byteen[i], depth_byteen})
                                   : 4'b1111;
        wire [31:0] data = dcrs.depth_float ? req_depth[i]
                         : dcrs.depth_d16 ? {2{req_depth[i][`VX_OM_DEPTH_BITS-1 -: 16]}}
                         : {req_stencil[i], req_depth[i][`VX_OM_DEPTH_BITS-1:0]};
        // A D32F_S8 write touches only the words whose aspect is written.
        wire mask = req_ds_mask[i] && ~(req_rw && dcrs.depth_stencil_word && ~dcrs.depth_writemask);

        wire [3:0]  s_byteen = req_rw ? {3'b0, stencil_byteen[i]} : 4'b1111;
        wire [31:0] s_data   = 32'(req_stencil[i]);
        wire        s_mask   = req_ds_mask[i] && dcrs.depth_stencil_word && ~(req_rw && ~stencil_byteen[i]);

        VX_shift_register #(
            .DATAW (1 + 4 + 1 + 4 + W_ADDR_BITS + 32 + 32 + 1),
            .DEPTH (`LATENCY_IMUL)
        ) shift_reg (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (1'b1),
            .data_in  ({mask,         byteen,         s_mask,
                        s_byteen,                     baddr,   data,         s_data,
                        half}),
            .data_out ({mreq_mask[i], mreq_byteen[i], mreq_mask[2 * NUM_LANES + i],
                        mreq_byteen[2 * NUM_LANES + i], baddr_s, mreq_data[i],
                        mreq_data[2 * NUM_LANES + i], z_half[i]})
        );

        wire [W_ADDR_BITS-1:0] addr = baddr_s + W_ADDR_BITS'(m_y_pitch[31:2]);
        assign mreq_addr[i] = OCACHE_ADDR_WIDTH'(addr);
        assign mreq_addr[2 * NUM_LANES + i] = OCACHE_ADDR_WIDTH'(addr + W_ADDR_BITS'(1));
    end

    // blend color submission
    wire [OM_CFMT_BITS-1:0] cfmt = rt_dcrs.cbuf_format;
    // log2 of the texel size: R8 1 byte; RG8, R5G6B5, A1R5G5B5 2; A8R8G8B8 4.
    wire [1:0] c_lgbpp = (cfmt == OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_R8)) ? 2'd0
                       : (cfmt == OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_RG8)
                       || cfmt == OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_R5G6B5)
                       || cfmt == OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_A1R5G5B5)) ? 2'd1 : 2'd2;
    // The texel's bytes the writemask opens: R8 holds R (mask bit 2), RG8 R
    // then G. A packed 16-bit texel is written whole: a channel straddles its
    // bytes, so a mask opening only some of its channels is unsupported.
    wire [3:0] c_texel_byteen = (c_lgbpp == 2'd0) ? {3'b0, color_byteen[2]}
                              : (cfmt == OM_CFMT_BITS'(`VX_OM_COLOR_FORMAT_RG8)) ? {2'b0, color_byteen[1], color_byteen[2]}
                              : (c_lgbpp == 2'd1) ? {2'b0, {2{| color_byteen}}}
                              : color_byteen;
    wire [OM_CFMT_BITS-1:0] cfmt_s;
    wire [NUM_LANES-1:0][1:0] c_off;

    for (genvar i = NUM_LANES; i < 2 * NUM_LANES; ++i) begin : g_color
        wire [31:0] m_y_pitch;

        VX_multiplier #(
            .A_WIDTH (`VX_OM_DIM_BITS),
            .B_WIDTH (OM_PITCH_BITS),
            .R_WIDTH (32),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (1'b1),
            .dataa  (req_pos_y[i - NUM_LANES]),
            .datab  (rt_dcrs.cbuf_pitch),
            .result (m_y_pitch)
        );

        wire [B_ADDR_BITS-1:0] baddr, baddr_s;
        assign baddr = {rt_dcrs.cbuf_addr, 6'b0} + (B_ADDR_BITS'(req_pos_x[i - NUM_LANES]) << c_lgbpp);

        wire [3:0]  byteen = req_rw ? c_texel_byteen : 4'b1111;
        wire [31:0] data = cfmt_pack(cfmt, req_color[i - NUM_LANES]);
        wire mask = req_c_mask[i - NUM_LANES];
        wire [3:0]  byteen_s;
        wire [31:0] data_s;

        VX_shift_register #(
            .DATAW (1 + 4 + B_ADDR_BITS + 32),
            .DEPTH (`LATENCY_IMUL)
        ) shift_reg (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (1'b1),
            .data_in  ({mask,         byteen,   baddr,    data}),
            .data_out ({mreq_mask[i], byteen_s, baddr_s,  data_s})
        );

        wire [B_ADDR_BITS-1:0] addr = baddr_s + B_ADDR_BITS'(m_y_pitch);
        wire [1:0] off = addr[1:0];
        assign c_off[i - NUM_LANES] = off;
        assign mreq_addr[i]   = OCACHE_ADDR_WIDTH'(addr[B_ADDR_BITS-1:2]);
        assign mreq_byteen[i] = byteen_s << off;
        assign mreq_data[i]   = data_s << {off, 3'b0};
        `UNUSED_VAR (m_y_pitch)
    end

    VX_shift_register #(
        .DATAW (OM_CFMT_BITS),
        .DEPTH (`LATENCY_IMUL)
    ) cfmt_reg (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (1'b1),
        .data_in  (cfmt),
        .data_out (cfmt_s)
    );

    assign mreq_mtag = {mreq_tag, cfmt_s, z_half, c_off};

    VX_shift_register #(
        .DATAW  (1 + 1 + TAG_WIDTH),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({req_fire,   req_rw,  req_tag}),
        .data_out ({mreq_valid, mreq_rw, mreq_tag})
    );

    // Landing-slot reservation: a credit is taken at admission and released
    // when the landed request leaves for the memory scheduler, so the queue
    // can absorb everything the free-running pipe delivers and back-pressure
    // never has to reach inside it.
    localparam LANDQ_SIZE = 1 << `CLOG2(`LATENCY_IMUL + 2);

    wire pipe_empty;
    wire credits_full;
    wire landq_ready;

    assign req_ready = ~credits_full;

    VX_pending_size #(
        .SIZE (LANDQ_SIZE)
    ) landq_credits (
        .clk   (clk),
        .reset (reset),
        .incr  (req_fire),
        .decr  (mreq_valid_r && mreq_ready_r),
        .empty (pipe_empty),
        `UNUSED_PIN (alm_empty),
        .full  (credits_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    VX_elastic_buffer #(
        .DATAW   (1 + NUM_REQS * (1 + 4 + OCACHE_ADDR_WIDTH + 32) + MTAG_WIDTH),
        .SIZE    (LANDQ_SIZE),
        .OUT_REG (1),
        .LUTRAM  (1)
    ) landing_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mreq_valid),
        .ready_in  (landq_ready),
        .data_in   ({mreq_rw,   mreq_mask,   mreq_byteen,   mreq_addr,   mreq_data,   mreq_mtag}),
        .data_out  ({mreq_rw_r, mreq_mask_r, mreq_byteen_r, mreq_addr_r, mreq_data_r, mreq_mtag_r}),
        .valid_out (mreq_valid_r),
        .ready_out (mreq_ready_r)
    );

    `RUNTIME_ASSERT(~(mreq_valid && ~landq_ready),
        ("%t: *** %s: landing queue overflow — the credit reservation is broken", $time, INSTANCE_ID))
    `UNUSED_VAR (landq_ready)

    assign busy = ~pipe_empty || ~req_queue_empty_w;

    VX_lsu_mem_if #(
        .NUM_LANES (MEM_CHANNELS),
        .DATA_SIZE (4),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) lsu_mem_if();

    VX_mem_scheduler #(
        .INSTANCE_ID  ($sformatf("%s-memsched", INSTANCE_ID)),
        .CORE_REQS    (NUM_REQS),
        .MEM_CHANNELS (MEM_CHANNELS),
        .WORD_SIZE    (4),
        .ADDR_WIDTH   (OCACHE_ADDR_WIDTH),
        .USER_WIDTH   (0),
        .TAG_WIDTH    (MTAG_WIDTH),
        .CORE_QUEUE_SIZE(`VX_CFG_OM_MEM_QUEUE_SIZE),
        .UUID_WIDTH   (UUID_WIDTH),
        .RSP_PARTIAL  (0),
        .MEM_OUT_BUF  (3), // fully register cache-request output (SLR-crossing skid)
        .CORE_OUT_BUF (3)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .core_req_valid (mreq_valid_r),
        .core_req_rw    (mreq_rw_r),
        .core_req_mask  (mreq_mask_r),
        .core_req_byteen(mreq_byteen_r),
        .core_req_addr  (mreq_addr_r),
        .core_req_user  ('0),
        .core_req_data  (mreq_data_r),
        .core_req_tag   (mreq_mtag_r),
        .core_req_ready (mreq_ready_r),
        .req_queue_rw_notify (write_notify),
        .req_queue_empty (req_queue_empty_w),

        // Output response
        .core_rsp_valid (mrsp_valid),
        .core_rsp_mask  (mrsp_mask),
        .core_rsp_data  (mrsp_data),
        .core_rsp_tag   (mrsp_mtag),
        `UNUSED_PIN (core_rsp_sop),
        `UNUSED_PIN (core_rsp_eop),
        .core_rsp_ready (mrsp_ready),

        // Memory request
        .mem_req_valid  (lsu_mem_if.req_valid),
        .mem_req_rw     (lsu_mem_if.req_data.rw),
        .mem_req_mask   (lsu_mem_if.req_data.mask),
        .mem_req_byteen (lsu_mem_if.req_data.byteen),
        .mem_req_addr   (lsu_mem_if.req_data.addr),
        `UNUSED_PIN (mem_req_user),
        .mem_req_data   (lsu_mem_if.req_data.data),
        .mem_req_tag    (lsu_mem_if.req_data.tag),
        .mem_req_ready  (lsu_mem_if.req_ready),

        // Memory response
        .mem_rsp_valid  (lsu_mem_if.rsp_valid),
        .mem_rsp_mask   (lsu_mem_if.rsp_data.mask),
        .mem_rsp_data   (lsu_mem_if.rsp_data.data),
        .mem_rsp_tag    (lsu_mem_if.rsp_data.tag),
        .mem_rsp_ready  (lsu_mem_if.rsp_ready)
    );

    // OM never sets any memory attr; tie off the scheduler-driven LSU bus.
    assign lsu_mem_if.req_data.user = '0;

    VX_mem_bus_if #(
        .DATA_SIZE (4),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) sched_bus_if [MEM_CHANNELS]();

    VX_lsu_adapter #(
        .NUM_LANES    (MEM_CHANNELS),
        .DATA_SIZE    (4),
        .TAG_WIDTH    (OCACHE_TAG_WIDTH),
        .TAG_SEL_BITS (OCACHE_TAG_WIDTH - UUID_WIDTH),
        .REQ_OUT_BUF  (0),
        .RSP_OUT_BUF  (0)
    ) lsu_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (lsu_mem_if),
        .mem_bus_if (sched_bus_if)
    );

    for (genvar i = 0; i < OCACHE_NUM_REQS; ++i) begin : g_cache_bus_if
        if (i < MEM_CHANNELS) begin : g_active
            `ASSIGN_VX_MEM_BUS_IF (cache_bus_if[i], sched_bus_if[i]);
        end else begin : g_tieoff
            assign cache_bus_if[i].req_valid = 1'b0;
            assign cache_bus_if[i].req_data  = '0;
            assign cache_bus_if[i].rsp_ready = 1'b0;
            `UNUSED_VAR (cache_bus_if[i].req_ready)
            `UNUSED_VAR (cache_bus_if[i].rsp_valid)
            `UNUSED_VAR (cache_bus_if[i].rsp_data)
        end
    end

    assign rsp_valid = mrsp_valid;

    assign rsp_mask = mrsp_mask[0 +: NUM_LANES] | mrsp_mask[NUM_LANES +: NUM_LANES]
                    | mrsp_mask[2 * NUM_LANES +: NUM_LANES];

    wire [NUM_LANES-1:0] rsp_z_half = mrsp_mtag[2 * NUM_LANES +: NUM_LANES];
    for (genvar i = 0;  i < NUM_LANES; ++i) begin : g_rsp_depth
        wire [15:0] d16 = rsp_z_half[i] ? mrsp_data[i][31:16] : mrsp_data[i][15:0];
        assign rsp_depth[i]   = dcrs.depth_float ? mrsp_data[i]
                              : dcrs.depth_d16 ? 32'({d16, d16[15:8]})
                              : (mrsp_data[i] & 32'(OM_DEPTH_MASK));
    end

    for (genvar i = 0;  i < NUM_LANES; ++i) begin : g_rsp_stencil
        assign rsp_stencil[i] = dcrs.depth_stencil_word ? `VX_OM_STENCIL_BITS'(mrsp_data[2 * NUM_LANES + i])
                              : (dcrs.depth_float || dcrs.depth_d16) ? '0
                              : `VX_OM_STENCIL_BITS'(mrsp_data[i] >> `VX_OM_DEPTH_BITS);
    end

    wire [OM_CFMT_BITS-1:0] rsp_cfmt = mrsp_mtag[3 * NUM_LANES +: OM_CFMT_BITS];
    for (genvar i = NUM_LANES; i < 2 * NUM_LANES; ++i) begin : g_rsp_color
        wire [1:0] off = mrsp_mtag[2 * (i - NUM_LANES) +: 2];
        assign rsp_color[i - NUM_LANES] = cfmt_unpack(rsp_cfmt, mrsp_data[i] >> {off, 3'b0});
    end

    assign rsp_tag = mrsp_mtag[MTAG_WIDTH-1 -: TAG_WIDTH];

    assign mrsp_ready = rsp_ready;

endmodule
