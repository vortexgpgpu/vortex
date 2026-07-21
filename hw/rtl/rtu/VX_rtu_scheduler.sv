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

// VX_rtu_scheduler — context-pool ray-traversal control, one walker for both
// scene formats: the CW-BVH walk (RTU_BVH_WIDTH > 0) and the flat triangle-list
// walk (RTU_BVH_WIDTH == 0), which runs as a degenerate one-leaf scene through
// the same triangle loop and instance loop.
//
// Per-context state lives in a context-indexed RAM (the context store), not in
// flip-flops: selecting a context is an ADDRESS, not a NUM_CTX:1 mux. The
// walker is a three-stage pipeline over the store —
//
//   SELECT  pick a ready context (an arbiter over event-set wake bits) and
//           issue every context-indexed RAM read for it.
//   ALIGN   capture the RAM outputs into the stage snapshot; precompute the
//           absolute structure address.
//   EXEC    byte-align + decode the fetched image, advance the context FSM,
//           drive the PEs and the memory port, and write the whole context
//           word back.
//
// The stages overlap across DIFFERENT contexts, so the walker retires one
// micro-step per cycle. A context never occupies two stages at once: its wake
// bit is cleared at SELECT and can only be set again by its own EXEC or, once
// parked, by the one event it waits for — so a context-store read can never
// race its own write, and every RAM here is a plain write-first BRAM with no
// fabric bypass.
//
// Contexts talk to the shared units through EVENTS, each carrying the context
// id as a tag: a memory response, a tri/box/xform/reciprocal result, a ray
// landing at launch. The event sets the context's wake bit and, where there is
// data, writes it into a context-indexed result RAM that the next SELECT reads
// back. A structural conflict (memory port busy, a full queue) re-arms the
// wake bit and retries — there is no stall network.
//
// Results leave through the window store: a {slot, word} row RAM holding the
// committed-hit and candidate records one field-row (all lanes) per word, so
// the core's record write-back is a row-address counter. A commit engine
// serializes hit commits and candidate stagings into it; a barrier walker runs
// the per-slot finalise (CHS/MISS staging) and resume (candidate accept)
// copies one row at a time.

`include "VX_define.vh"

module VX_rtu_scheduler import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_SLOTS = 1,
    parameter NUM_CTX   = 4,
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8,
    parameter CTX_TAG_W = `LOG2UP(NUM_CTX)
) (
    input  wire clk,
    input  wire reset,

    // ── slot launch (from the core's fill engine) ─────────────────────
    // slot_start arms the slot (mask + warp-uniform ray scalars) before any of
    // its rays land; each lane's ray then arrives on ray_wr, which is also that
    // context's launch: it starts traversing while later lanes still stream in.
    input  wire                              slot_start_valid,
    input  wire [`LOG2UP(NUM_SLOTS)-1:0]     slot_start_slot,
    input  wire [(NUM_CTX/NUM_SLOTS)-1:0]    slot_start_mask,
    input  wire [15:0]                       slot_start_flags,
    input  wire [15:0]                       slot_start_cull,
    input  wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] slot_start_scene,

    input  wire                              ray_wr_valid,
    input  wire [CTX_TAG_W-1:0]              ray_wr_ctx,
    input  wire [RTU_RAY_BEATS*32-1:0]       ray_wr_data,

    // ── traversal status ──────────────────────────────────────────────
    output wire [NUM_SLOTS-1:0]              busy,
    output wire [NUM_SLOTS-1:0]              done,
    output wire [NUM_SLOTS-1:0]              yield,

    // per-context result flags (the record walk's status/mask inputs)
    output wire [NUM_CTX-1:0]                       hit_bits,
    output wire [NUM_CTX-1:0]                       yld_bits,
    output wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0] cb_types,
    output wire [NUM_CTX-1:0]                       attr_vld,

    // callback resume: the warp's per-lane actions, held stable by the core
    input  wire [NUM_SLOTS-1:0]                       resume,
    input  wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] action,

    // ── window store access (core side; always accepted) ──────────────
    input  wire                                           win_wr_valid,
    input  wire [`LOG2UP(NUM_SLOTS)+RTU_WS_WORD_BITS-1:0] win_wr_addr,
    input  wire [(NUM_CTX/NUM_SLOTS)-1:0]                 win_wr_wren,
    input  wire [(NUM_CTX/NUM_SLOTS)*32-1:0]              win_wr_data,
    input  wire                                           win_rd_valid,
    input  wire [`LOG2UP(NUM_SLOTS)+RTU_WS_WORD_BITS-1:0] win_rd_addr,
    output wire [(NUM_CTX/NUM_SLOTS)*32-1:0]              win_rd_data,

    // ── node/leaf fetch (context-id tagged) ───────────────────────────
    output wire                              mem_req_valid,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [CTX_TAG_W-1:0]              mem_req_tag,
    input  wire                              mem_req_ready,
    input  wire                              mem_rsp_valid,
    input  wire [LINE_BITS-1:0]              mem_rsp_data,
    input  wire [CTX_TAG_W-1:0]              mem_rsp_tag,
    output wire                              mem_rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam FLAT      = (RTU_BVH_WIDTH == 0);
    localparam NUM_LANES = NUM_CTX / NUM_SLOTS;
    localparam SLOT_W    = `LOG2UP(NUM_SLOTS);
    localparam LINES     = FLAT ? RTU_FLAT_LINES : RTU_NODE_LINES;
    localparam LB        = `CLOG2(LINES + 1);
    localparam BUF_BITS  = LINES * LINE_BITS;
    localparam IDXW      = `CLOG2(`UP(RTU_BVH_WIDTH));
    localparam NODE_W    = `UP(RTU_BVH_WIDTH);
    localparam ADDRW     = `VX_CFG_MEM_ADDR_WIDTH;

    `STATIC_ASSERT(((NUM_LANES * NUM_SLOTS) == NUM_CTX),
        ("RTU_NUM_CTX must be a whole multiple of RTU_NUM_SLOTS"))

`ifdef VX_CFG_RTU_TLAS_ENABLE
    localparam FLAT_TLAS = FLAT ? 1 : 0;
`else
    localparam FLAT_TLAS = 0;
`endif

    localparam RECIP_LAT = (`VX_CFG_RTU_RECIP_DSP_SEED != 0) ? 5 : RTU_FDIV_LAT;

    localparam RTU_RESTART_CAP = 8;
    localparam RST_CNTW        = `CLOG2(RTU_RESTART_CAP + 1);
    localparam STK_IDXW        = `CLOG2(RTU_STACK_DEPTH);

    // box collections in flight; sized so the collector never caps the node
    // rate the pipelined front end can sustain over the box-PE latency.
    localparam COLL_SIZE = 16;
    localparam COLL_IDW  = `CLOG2(COLL_SIZE);

    // ── per-context FSM states ────────────────────────────────────────
    localparam [4:0] CS_DONE         = 5'd0,
                     CS_SETUP        = 5'd1,   // issue one 1/dir axis per pass
                     CS_SETUP_WT     = 5'd2,   // park: last reciprocal result
                     CS_HDR_REQ      = 5'd3,
                     CS_HDR_WAIT     = 5'd4,
                     CS_REQ0         = 5'd5,
                     CS_RSP0         = 5'd6,
                     CS_REQN         = 5'd7,
                     CS_RSPN         = 5'd8,
                     CS_DISPATCH     = 5'd9,
                     CS_FEED         = 5'd10,  // one box child per pass
                     CS_WAIT         = 5'd11,  // park: box collection
                     CS_PUSH         = 5'd12,  // one sibling push per pass
                     CS_POP          = 5'd13,
                     CS_PROC_WAIT    = 5'd14,  // park: procedural AABB result
                     CS_LTRI_REQ0    = 5'd15,  // triangle-record fetch loop
                     CS_LTRI_RSP0    = 5'd16,
                     CS_LTRI_REQN    = 5'd17,
                     CS_LTRI_RSPN    = 5'd18,
                     CS_TRI_WAIT     = 5'd19,  // park: tri PE result
                     CS_INST_REQ     = 5'd20,  // instance-record fetch loop
                     CS_INST_RSP0    = 5'd21,
                     CS_INST_REQN    = 5'd22,
                     CS_INST_RSPN    = 5'd23,
                     CS_XFORM_WT     = 5'd24,  // park: object ray
                     CS_OBJ_SETUP    = 5'd25,
                     CS_OBJ_SETUP_WT = 5'd26,
                     CS_INST_NEXT    = 5'd27,
                     CS_BHDR_REQ     = 5'd28,  // flat TLAS: BLAS header fetch
                     CS_BHDR_WAIT    = 5'd29;

    // ── the context word: everything only the walker's EXEC touches ───
    // One row of the context store. State an async producer writes (fetched
    // lines, PE results) lives in its own result RAM; state a parallel
    // observer needs (sp, f_slot, the hit/yield flags) stays in flip-flops.
    typedef struct packed {
        logic [4:0]                 cstate;
        logic [1:0]                 setup_axis;
        logic [2:0][31:0]           inv_d;
        logic [31:0]                best_t;
        logic [31:0]                yld_t;       // staged candidate's t (compare copy)
        logic [31:0]                cur_off;
        logic [LB-1:0]              f_idx;
        logic [LB-1:0]              f_total;
        logic [RTU_CHILD_BITS-1:0]  feed_idx;
        logic [RTU_CHILD_BITS-1:0]  push_ptr;
        logic [COLL_IDW-1:0]        coll_id;
        logic [31:0]                tri_i;       // triangle loop (fat leaf / flat scan)
        logic [31:0]                tri_n;
        logic [31:0]                prim_base;
        logic [31:0]                geom_r;
        logic [31:0]                tri_flags;   // current record's flag word
        logic [RTU_CB_SBT_BITS-1:0] proc_sbt;
        logic [31:0]                inst_idx;    // TLAS instance loop
        logic [31:0]                inst_cnt;
        logic [31:0]                inst_base;
        logic [31:0]                blas_root;
        logic [31:0]                inst_id;
        logic [31:0]                inst_cust;
        logic [7:0]                 inst_flags;
        logic [31:0]                root_off;
        logic                       ovf_w;
        logic                       ovf_o;
        logic [RST_CNTW-1:0]        rst_w;
        logic [RST_CNTW-1:0]        rst_o;
        logic [2:0][31:0]           obj_o;
        logic [2:0][31:0]           obj_d;
        logic [2:0][31:0]           obj_inv_d;
        logic [RTU_STACK_BITS-1:0]  blas_floor;
        logic                       in_blas;
    } ctx_state_t;
    localparam CTXW = $bits(ctx_state_t);

    // per-lane ray row (matches the fill engine's beat order)
    typedef struct packed {
        logic [31:0]      t_max;
        logic [31:0]      t_min;
        logic [2:0][31:0] dir;
        logic [2:0][31:0] origin;
    } lane_ray_t;

    // ── hot per-context flip-flops ────────────────────────────────────
    reg [NUM_CTX-1:0]                       rdy_set;   // wake bits (event-driven)
    reg [NUM_CTX-1:0]                       fresh_set; // first pass runs the launch init
    reg [NUM_CTX-1:0]                       done_q;
    reg [NUM_CTX-1:0]                       mask_q;
    reg [NUM_CTX-1:0]                       hit_q;
    reg [NUM_CTX-1:0]                       yld_q;
    reg [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0] cbtype_q;
    reg [NUM_CTX-1:0]                       attr_q;
    reg [NUM_CTX-1:0][RTU_STACK_BITS-1:0]   sp_q_arr;
    reg [NUM_CTX-1:0][LB-1:0]               f_slot_q;
    reg [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] act_q;

    // ── per-slot state ────────────────────────────────────────────────
    reg [NUM_SLOTS-1:0]            running;
    reg [NUM_SLOTS-1:0]            finalised;
    reg [NUM_SLOTS-1:0]            done_r;
    reg [NUM_SLOTS-1:0]            pend_resume;
    reg [NUM_SLOTS-1:0][15:0]      slot_flags;
    reg [NUM_SLOTS-1:0][15:0]      slot_cull;
    reg [NUM_SLOTS-1:0][ADDRW-1:0] slot_scene;

    // ═══════════════════════ SELECT ═══════════════════════════════════
    wire [CTX_TAG_W-1:0] grant_idx;
    wire                 grant_valid;
    VX_rr_arbiter #(
        .NUM_REQS (NUM_CTX)
    ) ctx_arbiter (
        .clk          (clk),
        .reset        (reset),
        .requests     (rdy_set),
        .grant_index  (grant_idx),
        `UNUSED_PIN (grant_onehot),
        .grant_valid  (grant_valid),
        .grant_ready  (1'b1)
    );

    reg                 s1_valid;
    reg [CTX_TAG_W-1:0] s1_sel;
    reg                 s1_fresh;
    wire [SLOT_W-1:0]   s1_slot = SLOT_W'(32'(s1_sel) / NUM_LANES);

    // ═══════════════════════ ALIGN snapshot ═══════════════════════════
    reg                      x_valid;
    reg [CTX_TAG_W-1:0]      sel_q;
    reg                      fresh_q;
    ctx_state_t              word_q;
    lane_ray_t               ray_q;
    reg [BUF_BITS-1:0]       fbuf_q;
    reg [31:0]               stacktop_q;
    reg [ADDRW-1:0]          structaddr_q;
    reg [RTU_STACK_BITS-1:0] sp_q;
    reg [15:0]               flags_q;
    reg [15:0]               cull_q;
    reg                      trihit_q, triback_q;
    reg [31:0]               trit_q, triu_q, triv_q;
    reg [2:0][31:0]          xfo_q, xfd_q;
    reg [2:0][31:0]          recip_q;
    // collector head sampled at ALIGN: EXEC reads these registers instead of
    // re-muxing the 16-entry collector file through word_q.coll_id (the 300 MHz
    // critical path). The entry is frozen before its context is woken, so the
    // one-cycle-earlier sample is the same value the live read would see.
    reg                      coll_hit_q;
    reg [31:0]               coll_t0_q;
    reg [RTU_CHILD_BITS-1:0] coll_cnt_q;

    // ── the context store ─────────────────────────────────────────────
    // Read at SELECT, written whole by EXEC; reader and writer always name
    // different contexts (a context's wake bit cannot re-arm while it is in
    // the pipeline), so read-during-write data is never consumed.
    wire            cs_wr;
    ctx_state_t     cs_wdata;
    wire [CTXW-1:0] cs_rdata;
    VX_dp_ram #(
        .DATAW    (CTXW),
        .SIZE     (NUM_CTX),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) ctx_store (
        .clk   (clk),
        .reset (reset),
        .read  (grant_valid),
        .write (cs_wr),
        .wren  (1'b1),
        .waddr (sel_q),
        .wdata (CTXW'(cs_wdata)),
        .raddr (grant_idx),
        .rdata (cs_rdata)
    );
    ctx_state_t cs_word;
    assign cs_word = ctx_state_t'(cs_rdata);

    // ── the ray store: one row per context, written at launch ─────────
    wire [RTU_RAY_BEATS*32-1:0] ray_rdata;
    VX_dp_ram #(
        .DATAW    (RTU_RAY_BEATS * 32),
        .SIZE     (NUM_CTX),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) ray_store (
        .clk   (clk),
        .reset (reset),
        .read  (grant_valid),
        .write (ray_wr_valid),
        .wren  (1'b1),
        .waddr (ray_wr_ctx),
        .wdata (ray_wr_data),
        .raddr (grant_idx),
        .rdata (ray_rdata)
    );

    // ── the fetched-line buffer: LINES line-RAMs per context ──────────
    wire [BUF_BITS-1:0] fbuf;
    for (genvar s = 0; s < LINES; ++s) begin : g_fbuf_ram
        wire [LINE_BITS-1:0] line_rd;
        VX_dp_ram #(
            .DATAW    (LINE_BITS),
            .SIZE     (NUM_CTX),
            .OUT_REG  (1),
            .RDW_MODE ("W")
        ) fbuf_ram (
            .clk   (clk),
            .reset (reset),
            .read  (grant_valid),
            .write (mem_rsp_valid && (f_slot_q[mem_rsp_tag] == LB'(s))),
            .wren  (1'b1),
            .waddr (mem_rsp_tag),
            .wdata (mem_rsp_data),
            .raddr (grant_idx),
            .rdata (line_rd)
        );
        assign fbuf[s*LINE_BITS +: LINE_BITS] = line_rd;
    end
    assign mem_rsp_ready = 1'b1;

    // ── PE result RAMs (context-indexed, written by the tagged results) ──
    wire                 tri_valid_out, tri_hit, tri_back;
    wire [CTX_TAG_W-1:0] tri_tag_out;
    wire [31:0]          tri_t, tri_u, tri_v;
    wire [97:0]          trires_rdata;
    VX_dp_ram #(
        .DATAW    (98),
        .SIZE     (NUM_CTX),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) trires_ram (
        .clk   (clk),
        .reset (reset),
        .read  (grant_valid),
        .write (tri_valid_out),
        .wren  (1'b1),
        .waddr (tri_tag_out),
        .wdata ({tri_hit, tri_back, tri_t, tri_u, tri_v}),
        .raddr (grant_idx),
        .rdata (trires_rdata)
    );

    wire                 xform_valid_out;
    wire [CTX_TAG_W-1:0] xform_tag_out;
    wire [2:0][31:0]     xform_obj_o, xform_obj_d;
    wire [191:0]         xfres_rdata;
    VX_dp_ram #(
        .DATAW    (192),
        .SIZE     (NUM_CTX),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) xfres_ram (
        .clk   (clk),
        .reset (reset),
        .read  (grant_valid),
        .write (xform_valid_out),
        .wren  (1'b1),
        .waddr (xform_tag_out),
        .wdata ({xform_obj_o, xform_obj_d}),
        .raddr (grant_idx),
        .rdata (xfres_rdata)
    );

    // ── reciprocal (BVH only: the flat walk runs no box tests) ────────
    wire                 recip_issue;
    wire                 recip_obj;
    wire                 recip_valid_out, recip_last_out;
    wire [CTX_TAG_W-1:0] recip_tag_out;
    wire [1:0]           recip_axis_out;
    wire [31:0]          recip_result;
    wire [95:0]          recip_rdata;

    // ── short stack (BVH only) ────────────────────────────────────────
    wire        stk_wr;
    wire [31:0] stk_wdata;
    wire [31:0] stk_rdata;
    if (!FLAT) begin : g_stack
        VX_dp_ram #(
            .DATAW    (32),
            .SIZE     (NUM_CTX << STK_IDXW),
            .OUT_REG  (1),
            .RDW_MODE ("W")
        ) node_stack_ram (
            .clk   (clk),
            .reset (reset),
            .read  (grant_valid),
            .write (stk_wr),
            .wren  (1'b1),
            .waddr ({sel_q, sp_q[STK_IDXW-1:0]}),
            .wdata (stk_wdata),
            .raddr ({grant_idx, STK_IDXW'(sp_q_arr[grant_idx] - RTU_STACK_BITS'(1))}),
            .rdata (stk_rdata)
        );
    end else begin : g_no_stack
        assign stk_rdata = '0;
        `UNUSED_VAR ({stk_wr, stk_wdata, sp_q_arr})
    end

    // ═══════════════════════ stage advance ════════════════════════════
    always_ff @(posedge clk) begin
        if (reset) begin
            s1_valid <= 1'b0;
            x_valid  <= 1'b0;
        end else begin
            s1_valid <= grant_valid;
            s1_sel   <= grant_idx;
            s1_fresh <= grant_valid && fresh_set[grant_idx];
            x_valid  <= s1_valid;
            if (s1_valid) begin
                sel_q        <= s1_sel;
                fresh_q      <= s1_fresh;
                word_q       <= cs_word;
                ray_q        <= lane_ray_t'(ray_rdata);
                fbuf_q       <= fbuf;
                stacktop_q   <= stk_rdata;
                // a fresh context's store row is stale: its walk starts at the
                // scene base (the init template's cur_off is 0)
                structaddr_q <= slot_scene[s1_slot]
                              + (s1_fresh ? ADDRW'(0) : ADDRW'(cs_word.cur_off));
                sp_q         <= sp_q_arr[s1_sel];
                flags_q      <= slot_flags[s1_slot];
                cull_q       <= slot_cull[s1_slot];
                {trihit_q, triback_q, trit_q, triu_q, triv_q} <= trires_rdata;
                {xfo_q, xfd_q} <= xfres_rdata;
                recip_q      <= recip_rdata;
                coll_hit_q   <= coll_prochit[cs_word.coll_id];
                coll_t0_q    <= coll_ordt   [cs_word.coll_id][0];
                coll_cnt_q   <= coll_ordcnt [cs_word.coll_id];
            end
        end
    end

    // ═══════════════════════ EXEC decode ══════════════════════════════
    wire [RTU_LINE_SEL_BITS-1:0] f_off   = structaddr_q[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0] f_shift = {f_off, 3'b000};
    wire [BUF_BITS-1:0] f_aligned = fbuf_q >> f_shift;

    wire [31:0] f_off32 = 32'(f_off);
    wire [LB-1:0] tri_rec_lines =
        LB'(((f_off32 + RTU_TRI_STRIDE - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [LB-1:0] inst_lines =
        LB'(((f_off32 + RTU_INST_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [LB-1:0] node_lines =
        LB'(((f_off32 + RTU_NODE_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [LB-1:0] leaf_lines =
        LB'(((f_off32 + RTU_LEAF_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);

    // node decode (BVH only)
    wire [7:0] node_kind;
    rtu_node_t node;
    if (!FLAT) begin : g_node_decode
        wire [RTU_NODE_IMG_BITS-1:0] node_img = f_aligned[RTU_NODE_IMG_BITS-1:0];
        VX_rtu_node_decode #(
            .IMG_BITS (RTU_NODE_IMG_BITS)
        ) decode (
            .line (node_img),
            .kind (node_kind),
            .node (node)
        );
    end else begin : g_no_node_decode
        assign node_kind = '0;
        assign node = '0;
    end

    // leaf header + procedural AABB corners
    wire [2:0][31:0] leaf_v0, leaf_v1;
    for (genvar a = 0; a < 3; ++a) begin : g_leaf_v
        assign leaf_v0[a] = f_aligned[(RTU_TRI_OFF_V0 + 4*a)*8 +: 32];
        assign leaf_v1[a] = f_aligned[(RTU_TRI_OFF_V1 + 4*a)*8 +: 32];
    end
    wire [31:0] leaf_geom  = f_aligned[RTU_LEAF_OFF_GEOM*8 +: 32];
    wire [31:0] leaf_prim  = f_aligned[RTU_LEAF_OFF_PRIM*8 +: 32];
    wire [31:0] leaf_flags = f_aligned[RTU_LEAF_OFF_FLAGS*8 +: 32];
    wire [7:0]  leaf_count = f_aligned[15:8];
    wire [31:0] hdr_count  = f_aligned[31:0];

    // triangle record (fat-leaf and flat records share one layout)
    wire [2:0][31:0] ltri_v0, ltri_v1, ltri_v2;
    for (genvar a2 = 0; a2 < 3; ++a2) begin : g_ltri_v
        assign ltri_v0[a2] = f_aligned[(RTU_FLAT_OFF_V0 + 4*a2)*8 +: 32];
        assign ltri_v1[a2] = f_aligned[(RTU_FLAT_OFF_V1 + 4*a2)*8 +: 32];
        assign ltri_v2[a2] = f_aligned[(RTU_FLAT_OFF_V2 + 4*a2)*8 +: 32];
    end
    wire [31:0] ltri_flags = f_aligned[RTU_FLAT_OFF_FLAGS*8 +: 32];

    // instance record (the BVH and flat TLAS variants differ in two fields)
    wire [31:0] inst_blas   = f_aligned[RTU_INST_OFF_BLAS*8   +: 32];
    wire [31:0] inst_custom = f_aligned[RTU_INST_OFF_CUSTOM*8 +: 32];
    wire [31:0] inst_cull_w = FLAT ? f_aligned[RTU_INST_OFF_CULL_FLAT*8 +: 32]
                                   : f_aligned[RTU_INST_OFF_CULL_BVH*8  +: 32];
    wire [31:0] inst_id_w   = f_aligned[RTU_INST_OFF_ID_BVH*8 +: 32];
    wire [7:0]  inst_fl_w   = inst_cull_w[RTU_INST_FLAGS_SHIFT +: 8];
    wire [11:0][31:0] inst_xform_w;
    for (genvar xk = 0; xk < 12; ++xk) begin : g_inst_xform
        assign inst_xform_w[xk] = f_aligned[(RTU_INST_OFF_XFORM + 4*xk)*8 +: 32];
    end
    wire inst_culled = ((inst_cull_w & 32'(cull_q) & 32'hff) == 32'd0);

    // ── triangle classification (opacity / face culling / SBT) ────────
    wire [31:0] rflags = 32'(flags_q);
    wire cull_back    = (rflags & 32'(`VX_RT_FLAG_CULL_BACK_FACING))       != 0;
    wire cull_front   = (rflags & 32'(`VX_RT_FLAG_CULL_FRONT_FACING))      != 0;
    wire skip_tris    = (rflags & 32'(`VX_RT_FLAG_SKIP_TRIANGLES))         != 0;
    wire ray_opaque   = (rflags & 32'(`VX_RT_FLAG_OPAQUE))                 != 0;
    wire ray_noopaque = (rflags & 32'(`VX_RT_FLAG_NO_OPAQUE))              != 0;
    wire cull_opaque  = (rflags & 32'(`VX_RT_FLAG_CULL_OPAQUE))            != 0;
    wire cull_noopq   = (rflags & 32'(`VX_RT_FLAG_CULL_NO_OPAQUE))         != 0;
    wire term_first   = (rflags & 32'(`VX_RT_FLAG_TERMINATE_ON_FIRST_HIT)) != 0;

    wire [7:0] cur_iflags = word_q.in_blas ? word_q.inst_flags : 8'd0;
    wire inst_flip    = (cur_iflags & RTU_INST_FLAG_TRI_FLIP)     != 0;
    wire inst_culldis = (cur_iflags & RTU_INST_FLAG_TRI_CULL_DIS) != 0;
    wire inst_fopq    = (cur_iflags & RTU_INST_FLAG_FORCE_OPAQUE) != 0;
    wire inst_fnopq   = (cur_iflags & RTU_INST_FLAG_FORCE_NO_OPQ) != 0;
    wire eff_back     = triback_q ^ inst_flip;
    wire tri_opaque = ray_opaque   ? 1'b1
                    : ray_noopaque ? 1'b0
                    : inst_fopq    ? 1'b1
                    : inst_fnopq   ? 1'b0
                    : ((word_q.tri_flags & RTU_TRI_FLAG_OPAQUE) != 0);
    wire cls_cull = (tri_opaque && cull_opaque) || (!tri_opaque && cull_noopq);
    wire [RTU_CB_SBT_BITS-1:0] cls_sbt =
        RTU_CB_SBT_BITS'((word_q.tri_flags >> RTU_TRI_SBT_IDX_SHIFT) & RTU_TRI_SBT_IDX_MASK);
    // the flat walk classifies a procedural record by its flag word; the BVH
    // walk yields IS candidates from procedural leaves instead
    wire [RTU_CB_TYPE_BITS-1:0] cls_cbtype =
        (FLAT && ((word_q.tri_flags & RTU_TRI_FLAG_PROC) != 0))
            ? RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC)
            : RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_ANYHIT);
    wire tri_pass = trihit_q
                  && !skip_tris
                  && (inst_culldis || !(eff_back  && cull_back))
                  && (inst_culldis || !(!eff_back && cull_front))
                  && !cls_cull;
    wire tri_committable = tri_pass && (trit_q < word_q.best_t);

    // BLAS traversal runs the object-space ray
    wire [2:0][31:0] walk_ro    = word_q.in_blas ? word_q.obj_o     : ray_q.origin;
    wire [2:0][31:0] walk_rd    = word_q.in_blas ? word_q.obj_d     : ray_q.dir;
    wire [2:0][31:0] walk_inv_d = word_q.in_blas ? word_q.obj_inv_d : word_q.inv_d;

    // ═══════════════════════ box collector ════════════════════════════
    // The one read-modify-write an async producer performs: box results stream
    // back one child per cycle and are insertion-sorted (t-ascending) into a
    // small tagged entry pool instead of a per-context array.
    reg [COLL_SIZE-1:0]                     coll_busy;
    reg [COLL_SIZE-1:0]                     coll_proc;
    reg [COLL_SIZE-1:0][CTX_TAG_W-1:0]      coll_ctx;
    reg [COLL_SIZE-1:0][RTU_CHILD_BITS-1:0] coll_cnt;
    reg [COLL_SIZE-1:0][RTU_CHILD_BITS-1:0] coll_last;
    reg [COLL_SIZE-1:0][RTU_CHILD_BITS-1:0] coll_ordcnt;
    reg [COLL_SIZE-1:0][NODE_W-1:0][31:0]   coll_ordoff;
    reg [COLL_SIZE-1:0][NODE_W-1:0][31:0]   coll_ordt;
    reg [COLL_SIZE-1:0]                     coll_prochit;

    wire [COLL_IDW-1:0] coll_alloc_idx;
    wire                coll_alloc_ok;
    VX_priority_encoder #(
        .N (COLL_SIZE)
    ) coll_alloc_enc (
        .data_in    (~coll_busy),
        `UNUSED_PIN (onehot_out),
        .index_out  (coll_alloc_idx),
        .valid_out  (coll_alloc_ok)
    );

    // ── box PE (BVH only) ─────────────────────────────────────────────
    wire        box_feed, box_feed_raw;
    wire        box_valid_out, box_hit;
    wire [31:0] box_t_near;
    wire [COLL_IDW+32-1:0] box_tag_out;
    wire [COLL_IDW-1:0]    box_coll     = box_tag_out[COLL_IDW+32-1 : 32];
    wire [31:0]            box_childoff = box_tag_out[31:0];
    wire [IDXW-1:0]        feed_ci      = word_q.feed_idx[IDXW-1:0];

    if (!FLAT) begin : g_box_pe
        VX_rtu_box_pe #(
            .TAG_WIDTH (COLL_IDW + 32)
        ) box_pe (
            .clk       (clk),
            .reset     (reset),
            .enable    (1'b1),
            .valid_in  (box_feed || box_feed_raw),
            .tag_in    ({box_feed_raw ? coll_alloc_idx : word_q.coll_id,
                         box_feed_raw ? 32'd0 : node.child_off[feed_ci]}),
            .origin    (node.origin),
            .exp       (node.exp),
            .qmin      (node.qmin[feed_ci]),
            .qmax      (node.qmax[feed_ci]),
            .raw       (box_feed_raw),
            .raw_min   (leaf_v0),
            .raw_max   (leaf_v1),
            .ro        (walk_ro),
            .inv_d     (walk_inv_d),
            .t_min     (ray_q.t_min),
            .t_max     (word_q.best_t),
            .valid_out (box_valid_out),
            .tag_out   (box_tag_out),
            .hit       (box_hit),
            .t_near    (box_t_near)
        );
    end else begin : g_no_box_pe
        assign box_valid_out = 1'b0;
        assign box_hit       = 1'b0;
        assign box_t_near    = '0;
        assign box_tag_out   = '0;
        `UNUSED_VAR ({box_feed, box_feed_raw, feed_ci, walk_inv_d})
        `UNUSED_VAR ({leaf_v0, leaf_v1, leaf_geom, leaf_prim, leaf_flags, leaf_count})
        `UNUSED_VAR ({node, node_kind, node_lines, leaf_lines, stacktop_q})
    end

    // insertion index: collected entries with t <= the incoming child's
    reg [RTU_CHILD_BITS-1:0] ins_pos;
    always @(*) begin
        ins_pos = RTU_CHILD_BITS'(0);
        for (integer oc = 0; oc < RTU_BVH_WIDTH; oc = oc + 1) begin
            if ((RTU_CHILD_BITS'(oc) < coll_ordcnt[box_coll])
             && (coll_ordt[box_coll][oc[IDXW-1:0]] <= box_t_near)) begin
                ins_pos = ins_pos + RTU_CHILD_BITS'(1);
            end
        end
    end
    wire coll_pushable = box_hit && (box_childoff != 32'd0);

    // ── tri PE ────────────────────────────────────────────────────────
    wire tri_feed;
    VX_rtu_tri_pe #(
        .TAG_WIDTH (CTX_TAG_W)
    ) tri_pe (
        .clk         (clk),
        .reset       (reset),
        .enable      (1'b1),
        .valid_in    (tri_feed),
        .tag_in      (sel_q),
        .origin      (walk_ro),
        .dir         (walk_rd),
        .v0          (ltri_v0),
        .v1          (ltri_v1),
        .v2          (ltri_v2),
        .t_min       (ray_q.t_min),
        .t_max       (word_q.best_t),
        .valid_out   (tri_valid_out),
        .tag_out     (tri_tag_out),
        .hit         (tri_hit),
        .t           (tri_t),
        .u           (tri_u),
        .v           (tri_v),
        .back_facing (tri_back)
    );

    // ── xform PE: world -> object ray, fed straight off the record ────
    wire xform_feed;
    VX_rtu_xform #(
        .TAG_WIDTH (CTX_TAG_W)
    ) xform_pe (
        .clk       (clk),
        .reset     (reset),
        .enable    (1'b1),
        .valid_in  (xform_feed),
        .tag_in    (sel_q),
        .xform     (inst_xform_w),
        .ro        (ray_q.origin),
        .rd        (ray_q.dir),
        .valid_out (xform_valid_out),
        .tag_out   (xform_tag_out),
        .obj_ro    (xform_obj_o),
        .obj_rd    (xform_obj_d)
    );

    // ── reciprocal datapath: pipelined, one axis per issue ────────────
    wire [1:0]  recip_axis;
    wire [31:0] recip_din = recip_obj ? word_q.obj_d[recip_axis] : ray_q.dir[recip_axis];
    if (!FLAT) begin : g_recip_pe
        VX_rtu_recip #(
            .LATENCY  (RTU_FDIV_LAT),
            .DSP_SEED (`VX_CFG_RTU_RECIP_DSP_SEED)
        ) recip (
            .clk    (clk),
            .reset  (reset),
            .enable (1'b1),
            .mask   (1'b1),
            .x      (recip_din),
            .result (recip_result)
        );
        VX_shift_register #(
            .DATAW  (1 + CTX_TAG_W + 2 + 1),
            .RESETW (1),
            .DEPTH  (RECIP_LAT)
        ) recip_tags (
            .clk      (clk),
            .reset    (reset),
            .enable   (1'b1),
            .data_in  ({recip_issue, sel_q, recip_axis, recip_axis == 2'd2}),
            .data_out ({recip_valid_out, recip_tag_out, recip_axis_out, recip_last_out})
        );
        VX_dp_ram #(
            .DATAW    (96),
            .SIZE     (NUM_CTX),
            .WRENW    (3),
            .OUT_REG  (1),
            .RDW_MODE ("W")
        ) recipres_ram (
            .clk   (clk),
            .reset (reset),
            .read  (grant_valid),
            .write (recip_valid_out),
            .wren  (3'(1) << recip_axis_out),
            .waddr (recip_tag_out),
            .wdata ({3{recip_result}}),
            .raddr (grant_idx),
            .rdata (recip_rdata)
        );
    end else begin : g_no_recip_pe
        assign recip_valid_out = 1'b0;
        assign recip_last_out  = 1'b0;
        assign recip_tag_out   = '0;
        assign recip_axis_out  = '0;
        assign recip_result    = '0;
        assign recip_rdata     = '0;
        `UNUSED_VAR ({recip_issue, recip_obj, recip_din, recip_q})
    end

    // ═══════════════════════ commit engine ════════════════════════════
    // Serializes hit commits and candidate stagings into the window store:
    // one field-row per cycle, per-lane write enable.
    localparam [1:0] CK_HIT  = 2'd0,  // committed opaque hit: the 7 hit rows
                     CK_YLDA = 2'd1,  // any-hit candidate: 7 yld rows + sbt
                     CK_YLDP = 2'd2;  // IS candidate: t/u/v/prim/geom + sbt

    typedef struct packed {
        logic [1:0]                 kind;
        logic [CTX_TAG_W-1:0]       ctx;
        logic [RTU_CB_SBT_BITS-1:0] sbt;
        logic [31:0]                t, u, v, prim, inst, geom, cust;
    } commit_t;

    wire     cf_push, cf_pop, cf_empty, cf_full;
    commit_t cf_din;
    wire [$bits(commit_t)-1:0] cf_dout_w;
    commit_t cf_dout;
    assign cf_dout = commit_t'(cf_dout_w);
    VX_fifo_queue #(
        .DATAW  ($bits(commit_t)),
        .DEPTH  (4),
        .LUTRAM (1)
    ) commit_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (cf_push),
        .pop      (cf_pop),
        .data_in  ($bits(commit_t)'(cf_din)),
        .data_out (cf_dout_w),
        .empty    (cf_empty),
        `UNUSED_PIN (alm_empty),
        .full     (cf_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    // ═══════════════════════ window store ═════════════════════════════
    localparam WS_ADDRW = SLOT_W + RTU_WS_WORD_BITS;
    localparam ROW_BITS = NUM_LANES * 32;

    wire                 ws_wr, ws_rd;
    wire [WS_ADDRW-1:0]  ws_waddr, ws_raddr;
    wire [NUM_LANES-1:0] ws_wren;
    wire [ROW_BITS-1:0]  ws_wdata, ws_rdata;
    VX_dp_ram #(
        .DATAW    (ROW_BITS),
        .SIZE     (1 << WS_ADDRW),
        .WRENW    (NUM_LANES),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) window_store (
        .clk   (clk),
        .reset (reset),
        .read  (ws_rd),
        .write (ws_wr),
        .wren  (ws_wren),
        .waddr (ws_waddr),
        .wdata (ws_wdata),
        .raddr (ws_raddr),
        .rdata (ws_rdata)
    );
    assign win_rd_data = ws_rdata;

    // commit engine sequencing (one row per granted cycle)
    reg  [2:0] ce_step;
    wire       ce_active = ~cf_empty;
    wire [SLOT_W-1:0]    ce_slot = SLOT_W'(32'(cf_dout.ctx) / NUM_LANES);
    wire [NUM_LANES-1:0] ce_lane = NUM_LANES'(1) << (32'(cf_dout.ctx) % NUM_LANES);

    reg [RTU_WS_WORD_BITS-1:0] ce_word;
    reg [31:0]                 ce_data;
    reg                        ce_last;
    always @(*) begin
        // per-kind (row, field) walk; CK_YLDP skips inst/custom — an IS
        // candidate leaves those rows holding whatever was last staged
        logic [RTU_WS_WORD_BITS-1:0] base;
        base    = (cf_dout.kind == CK_HIT) ? RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE)
                                           : RTU_WS_WORD_BITS'(RTU_WS_YLD_BASE);
        ce_word = base + RTU_WS_WORD_BITS'(32'(ce_step));
        ce_data = 32'd0;
        ce_last = 1'b0;
        case (ce_step)
            3'd0: ce_data = cf_dout.t;
            3'd1: ce_data = cf_dout.u;
            3'd2: ce_data = cf_dout.v;
            3'd3: ce_data = cf_dout.prim;
            3'd4: begin
                if (cf_dout.kind == CK_YLDP) begin
                    ce_word = base + RTU_WS_WORD_BITS'(RTU_WS_F_GEOM);
                    ce_data = cf_dout.geom;
                end else begin
                    ce_data = cf_dout.inst;
                end
            end
            3'd5: begin
                if (cf_dout.kind == CK_YLDP) begin
                    ce_word = RTU_WS_WORD_BITS'(RTU_WS_YLD_SBT);
                    ce_data = 32'(cf_dout.sbt);
                    ce_last = 1'b1;
                end else begin
                    ce_word = base + RTU_WS_WORD_BITS'(RTU_WS_F_GEOM);
                    ce_data = cf_dout.geom;
                end
            end
            3'd6: begin
                ce_word = base + RTU_WS_WORD_BITS'(RTU_WS_F_CUST);
                ce_data = cf_dout.cust;
                ce_last = (cf_dout.kind == CK_HIT);
            end
            default: begin
                ce_word = RTU_WS_WORD_BITS'(RTU_WS_YLD_SBT);
                ce_data = 32'(cf_dout.sbt);
                ce_last = 1'b1;
            end
        endcase
    end

    // ═══════════════════════ barrier walker ═══════════════════════════
    // Runs the two whole-slot record operations one row at a time:
    //   FIN — stage CHS (hit->yld row copy) and MISS (zeroed attributes)
    //   RES — commit accepted candidates (yld->hit row copy, attr merge)
    localparam [3:0] BW_IDLE  = 4'd0,
                     BW_RD    = 4'd1,  // request the source row
                     BW_CAP   = 4'd2,  // capture it
                     BW_RD2   = 4'd3,  // request the CONT-t row (RES field 0)
                     BW_CAP2  = 4'd4,  // capture it
                     BW_WR    = 4'd5,  // write the destination row
                     BW_ZMISS = 4'd6,  // FIN: zero inst/geom/cust for MISS lanes
                     BW_ATTRR = 4'd7,  // RES: request the CONT hitAttribute
                     BW_ATTRC = 4'd8,  // capture it
                     BW_ATTRW = 4'd9;  // merge it into the bound attribute

    reg [3:0]           bw_state;
    reg                 bw_is_res;
    reg                 bw_job_done;
    reg [SLOT_W-1:0]    bw_slot;
    reg [2:0]           bw_field;
    reg [1:0]           bw_zidx;
    reg [NUM_LANES-1:0] bw_copy_mask;   // CHS lanes (FIN) / accepted lanes (RES)
    reg [NUM_LANES-1:0] bw_miss_mask;
    reg [ROW_BITS-1:0]  bw_data;
    reg [ROW_BITS-1:0]  bw_data2;

    wire bw_idle = (bw_state == BW_IDLE);
    wire ce_idle = cf_empty;

    // per-slot completion / finalise conditions (combinational on hot bits)
    wire [NUM_CTX-1:0] ctx_active = mask_q & ~done_q;
    wire [NUM_SLOTS-1:0] all_done;
    wire [NUM_SLOTS-1:0] yld_any;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_slot_status
        assign all_done[s] = ~(| ctx_active[s*NUM_LANES +: NUM_LANES]);
        assign yld_any[s]  = | yld_q[s*NUM_LANES +: NUM_LANES];
    end

    // CHS / MISS staging masks for the first slot awaiting finalise. The
    // commit queue must have drained: the staged rows it still holds belong
    // to this slot's own candidates.
    reg [SLOT_W-1:0] fin_slot;
    reg              fin_req;
    always @(*) begin
        fin_req  = 1'b0;
        fin_slot = '0;
        for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
            if (!fin_req && running[s] && all_done[s] && !finalised[s] && ce_idle) begin
                fin_req  = 1'b1;
                fin_slot = SLOT_W'(s);
            end
        end
    end
    wire [31:0] fin_flags = 32'(slot_flags[fin_slot]);
    wire fin_chs_en  = ((fin_flags & 32'(`VX_RT_FLAG_ENABLE_CHS)) != 0)
                    && ((fin_flags & 32'(`VX_RT_FLAG_SKIP_CLOSEST_HIT)) == 0);
    wire fin_miss_en = ((fin_flags & 32'(`VX_RT_FLAG_ENABLE_MISS)) != 0);
    reg [NUM_LANES-1:0] fin_chs_mask, fin_miss_mask;
    always @(*) begin
        for (integer j = 0; j < NUM_LANES; j = j + 1) begin
            fin_chs_mask[j]  = fin_req && fin_chs_en
                            && mask_q[32'(fin_slot)*NUM_LANES + j]
                            && !yld_q[32'(fin_slot)*NUM_LANES + j]
                            && hit_q[32'(fin_slot)*NUM_LANES + j];
            fin_miss_mask[j] = fin_req && fin_miss_en
                            && mask_q[32'(fin_slot)*NUM_LANES + j]
                            && !yld_q[32'(fin_slot)*NUM_LANES + j]
                            && !hit_q[32'(fin_slot)*NUM_LANES + j];
        end
    end
    wire fin_trivial = fin_req && (fin_chs_mask == '0) && (fin_miss_mask == '0);
    wire fin_start   = fin_req && !fin_trivial && bw_idle;

    wire [SLOT_W-1:0] res_slot_enc;
    wire              res_req;
    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) res_slot_pe (
        .data_in    (pend_resume),
        `UNUSED_PIN (onehot_out),
        .index_out  (res_slot_enc),
        .valid_out  (res_req)
    );
    wire res_start = res_req && !fin_start && bw_idle;

    reg [NUM_LANES-1:0] res_acc_mask;
    always @(*) begin
        for (integer j = 0; j < NUM_LANES; j = j + 1) begin
            res_acc_mask[j] = yld_q[32'(res_slot_enc)*NUM_LANES + j]
                && ((act_q[32'(res_slot_enc)*NUM_LANES + j] == RTU_CB_ACTION_BITS'(`VX_RT_CB_ACCEPT))
                 || (act_q[32'(res_slot_enc)*NUM_LANES + j] == RTU_CB_ACTION_BITS'(`VX_RT_CB_TERMINATE)));
        end
    end

    // walker window-port requests
    reg                 bw_rd_req, bw_wr_req;
    reg [WS_ADDRW-1:0]  bw_rd_addr, bw_wr_addr;
    reg [NUM_LANES-1:0] bw_wr_mask;
    reg [ROW_BITS-1:0]  bw_wr_data;

    wire [RTU_WS_WORD_BITS-1:0] bw_src_base = bw_is_res
        ? RTU_WS_WORD_BITS'(RTU_WS_YLD_BASE) : RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE);
    wire [RTU_WS_WORD_BITS-1:0] bw_dst_base = bw_is_res
        ? RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE) : RTU_WS_WORD_BITS'(RTU_WS_YLD_BASE);

    always @(*) begin
        bw_rd_req  = 1'b0;
        bw_wr_req  = 1'b0;
        bw_rd_addr = '0;
        bw_wr_addr = '0;
        bw_wr_mask = '0;
        bw_wr_data = '0;
        case (bw_state)
        BW_RD: begin
            bw_rd_req  = 1'b1;
            bw_rd_addr = {bw_slot, bw_src_base + RTU_WS_WORD_BITS'(32'(bw_field))};
        end
        BW_RD2: begin
            bw_rd_req  = 1'b1;
            bw_rd_addr = {bw_slot, RTU_WS_WORD_BITS'(RTU_WS_CONT_T)};
        end
        BW_WR: begin
            bw_wr_req  = 1'b1;
            bw_wr_addr = {bw_slot, bw_dst_base + RTU_WS_WORD_BITS'(32'(bw_field))};
            bw_wr_mask = bw_copy_mask;
            for (integer j = 0; j < NUM_LANES; j = j + 1) begin
                // a PROC accept commits the intersection shader's own t
                if (bw_is_res && (bw_field == 3'd0)
                 && (cbtype_q[32'(bw_slot)*NUM_LANES + j] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))) begin
                    bw_wr_data[j*32 +: 32] = bw_data2[j*32 +: 32];
                end else begin
                    bw_wr_data[j*32 +: 32] = bw_data[j*32 +: 32];
                end
            end
        end
        BW_ZMISS: begin
            bw_wr_req  = 1'b1;
            bw_wr_addr = {bw_slot, RTU_WS_WORD_BITS'(RTU_WS_YLD_BASE)
                          + ((bw_zidx == 2'd0) ? RTU_WS_WORD_BITS'(RTU_WS_F_INST)
                           : (bw_zidx == 2'd1) ? RTU_WS_WORD_BITS'(RTU_WS_F_GEOM)
                                               : RTU_WS_WORD_BITS'(RTU_WS_F_CUST))};
            bw_wr_mask = bw_miss_mask;
            bw_wr_data = '0;
        end
        BW_ATTRR: begin
            bw_rd_req  = 1'b1;
            bw_rd_addr = {bw_slot, RTU_WS_WORD_BITS'(RTU_WS_CONT_ATTR)};
        end
        BW_ATTRW: begin
            bw_wr_req  = 1'b1;
            bw_wr_addr = {bw_slot, RTU_WS_WORD_BITS'(RTU_WS_RES_ATTR)};
            bw_wr_mask = bw_copy_mask;
            bw_wr_data = bw_data;
        end
        default:;
        endcase
    end

    // ── window-store port arbitration ─────────────────────────────────
    // The core's accesses are always accepted (the CONTINUE beats ride the
    // never-stalling req channel); the engines yield and retry.
    wire ce_wr_gnt = !win_wr_valid && ce_active;
    wire bw_wr_gnt = !win_wr_valid && !ce_active && bw_wr_req;
    wire bw_rd_gnt = !win_rd_valid && bw_rd_req;

    assign ws_wr    = win_wr_valid || ce_wr_gnt || bw_wr_gnt;
    assign ws_waddr = win_wr_valid ? win_wr_addr
                    : ce_wr_gnt    ? {ce_slot, ce_word}
                                   : bw_wr_addr;
    assign ws_wren  = win_wr_valid ? win_wr_wren
                    : ce_wr_gnt    ? ce_lane
                                   : bw_wr_mask;
    assign ws_wdata = win_wr_valid ? win_wr_data
                    : ce_wr_gnt    ? {NUM_LANES{ce_data}}
                                   : bw_wr_data;
    assign ws_rd    = win_rd_valid || bw_rd_gnt;
    assign ws_raddr = win_rd_valid ? win_rd_addr : bw_rd_addr;

    assign cf_pop = ce_wr_gnt && ce_last;

    always_ff @(posedge clk) begin
        if (reset) begin
            ce_step <= '0;
        end else if (ce_wr_gnt) begin
            ce_step <= ce_last ? 3'd0 : (ce_step + 3'd1);
        end
    end

    // ═══════════════════════ EXEC: the context FSM ════════════════════
    // Effective word: a fresh (just-launched) context ignores the stale store
    // row and starts from the init template.
    ctx_state_t word_x;
    always @(*) begin
        word_x = word_q;
        if (fresh_q) begin
            word_x            = '0;
            word_x.cstate     = FLAT ? CS_HDR_REQ : CS_SETUP;
            word_x.best_t     = ray_q.t_max;
            word_x.yld_t      = ray_q.t_max;
        end
    end

    // EXEC outcome (combinational)
    ctx_state_t word_n;
    reg         wake_self;
    reg         exec_done;
    reg         exec_hit_set;
    reg         exec_yld_set;
    reg         exec_yld_clr;
    reg [RTU_CB_TYPE_BITS-1:0] exec_cbtype;
    reg         mem_issue;
    reg [LB-1:0] mem_fslot;
    reg         box_feed_r, box_raw_r, tri_feed_r, xform_feed_r;
    reg         recip_issue_r, recip_obj_r;
    reg         coll_alloc_r;
    reg         coll_free_r;
    reg         cf_push_r;
    commit_t    cf_din_r;
    reg         sp_inc, sp_dec;
    reg         stk_wr_r;
    reg [31:0]  stk_wdata_r;

    wire mem_fire = x_valid && mem_issue && mem_req_ready;

    wire [RTU_CHILD_BITS-1:0] last_child = node.n_children - RTU_CHILD_BITS'(1);

    always @(*) begin
        word_n        = word_x;
        wake_self     = 1'b0;
        exec_done     = 1'b0;
        exec_hit_set  = 1'b0;
        exec_yld_set  = 1'b0;
        exec_yld_clr  = 1'b0;
        exec_cbtype   = '0;
        mem_issue     = 1'b0;
        mem_fslot     = '0;
        box_feed_r    = 1'b0;
        box_raw_r     = 1'b0;
        tri_feed_r    = 1'b0;
        xform_feed_r  = 1'b0;
        recip_issue_r = 1'b0;
        recip_obj_r   = 1'b0;
        coll_alloc_r  = 1'b0;
        coll_free_r   = 1'b0;
        cf_push_r     = 1'b0;
        cf_din_r      = '0;
        sp_inc        = 1'b0;
        sp_dec        = 1'b0;
        stk_wr_r      = 1'b0;
        stk_wdata_r   = '0;

        cf_din_r.ctx = sel_q;

        case (word_x.cstate)
        CS_SETUP: begin
            recip_issue_r = 1'b1;
            if (word_x.setup_axis == 2'd2) begin
                word_n.cstate = CS_SETUP_WT;
            end else begin
                word_n.setup_axis = word_x.setup_axis + 2'd1;
                wake_self = 1'b1;
            end
        end
        CS_SETUP_WT: begin
            word_n.inv_d  = recip_q;
            word_n.cstate = CS_HDR_REQ;
            wake_self     = 1'b1;
        end
        CS_HDR_REQ: begin
            mem_issue = 1'b1;
            if (mem_fire) begin
                word_n.cstate = CS_HDR_WAIT;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_HDR_WAIT: begin
            if (FLAT) begin
                if (FLAT_TLAS != 0) begin
                    // flat TLAS scene: header word0 = instance count
                    if (hdr_count == 32'd0) begin
                        word_n.cstate = CS_DONE;
                        exec_done     = 1'b1;
                    end else begin
                        word_n.inst_cnt  = hdr_count;
                        word_n.inst_idx  = '0;
                        word_n.inst_base = 32'(RTU_SCENE_HDR_BYTES);
                        word_n.cur_off   = 32'(RTU_SCENE_HDR_BYTES);
                        word_n.cstate    = CS_INST_REQ;
                        wake_self        = 1'b1;
                    end
                end else begin
                    // flat scene: header word0 = triangle count
                    if (hdr_count == 32'd0 || skip_tris) begin
                        word_n.cstate = CS_DONE;
                        exec_done     = 1'b1;
                    end else begin
                        word_n.tri_n     = hdr_count;
                        word_n.tri_i     = '0;
                        word_n.prim_base = '0;
                        word_n.geom_r    = '0;
                        word_n.cur_off   = 32'(RTU_SCENE_HDR_BYTES);
                        word_n.cstate    = CS_LTRI_REQ0;
                        wake_self        = 1'b1;
                    end
                end
            end else begin
                word_n.cur_off  = f_aligned[RTU_SCENE_OFF_ROOT*8 +: 32];
                word_n.root_off = f_aligned[RTU_SCENE_OFF_ROOT*8 +: 32];
                word_n.cstate   = CS_REQ0;
                wake_self       = 1'b1;
            end
        end
        CS_REQ0: begin
            mem_issue = 1'b1;
            if (mem_fire) begin
                word_n.cstate = CS_RSP0;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_RSP0: begin
            if (node_kind == RTU_KIND_INTERNAL) begin
                word_n.f_total = node_lines;
                if (node_lines == LB'(1)) begin
                    word_n.cstate = CS_DISPATCH;
                end else begin
                    word_n.f_idx  = LB'(1);
                    word_n.cstate = CS_REQN;
                end
                wake_self = 1'b1;
            end else if ((node_kind == RTU_KIND_LEAF_TRI)
                      || (node_kind == RTU_KIND_LEAF_PROC)
                      || (node_kind == RTU_KIND_LEAF_INST)) begin
                word_n.f_total = leaf_lines;
                if (leaf_lines == LB'(1)) begin
                    word_n.cstate = CS_DISPATCH;
                end else begin
                    word_n.f_idx  = LB'(1);
                    word_n.cstate = CS_REQN;
                end
                wake_self = 1'b1;
            end else begin
                word_n.cstate = CS_POP;
                wake_self     = 1'b1;
            end
        end
        CS_REQN: begin
            mem_issue = 1'b1;
            mem_fslot = word_x.f_idx;
            if (mem_fire) begin
                word_n.cstate = CS_RSPN;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_RSPN: begin
            if ((word_x.f_idx + LB'(1)) == word_x.f_total) begin
                word_n.cstate = CS_DISPATCH;
            end else begin
                word_n.f_idx  = word_x.f_idx + LB'(1);
                word_n.cstate = CS_REQN;
            end
            wake_self = 1'b1;
        end
        CS_DISPATCH: begin
            if (node_kind == RTU_KIND_INTERNAL && node.n_children != '0) begin
                if (coll_alloc_ok) begin
                    coll_alloc_r    = 1'b1;
                    word_n.coll_id  = coll_alloc_idx;
                    word_n.feed_idx = '0;
                    word_n.cstate   = CS_FEED;
                end
                wake_self = 1'b1;   // retries while no collector entry is free
            end else if (node_kind == RTU_KIND_LEAF_TRI) begin
                word_n.prim_base = leaf_prim;
                word_n.geom_r    = leaf_geom;
                word_n.tri_n     = 32'(leaf_count);
                word_n.tri_i     = '0;
                if (leaf_count == 8'd0) begin
                    word_n.cstate = CS_POP;
                end else begin
                    word_n.cur_off = word_x.cur_off + 32'(RTU_LEAF_HDR_BYTES);
                    word_n.cstate  = CS_LTRI_REQ0;
                end
                wake_self = 1'b1;
            end else if (node_kind == RTU_KIND_LEAF_PROC) begin
                if (coll_alloc_ok) begin
                    coll_alloc_r     = 1'b1;
                    box_raw_r        = 1'b1;   // the raw AABB is fed this pass
                    word_n.coll_id   = coll_alloc_idx;
                    word_n.prim_base = leaf_prim;
                    word_n.geom_r    = leaf_geom;
                    word_n.proc_sbt  = RTU_CB_SBT_BITS'((leaf_flags >> RTU_TRI_SBT_IDX_SHIFT)
                                                        & RTU_TRI_SBT_IDX_MASK);
                    word_n.cstate    = CS_PROC_WAIT;
                end else begin
                    wake_self = 1'b1;
                end
            end else if (node_kind == RTU_KIND_LEAF_INST && leaf_count != 8'd0) begin
                word_n.inst_cnt   = {24'd0, leaf_count};
                word_n.inst_idx   = '0;
                word_n.inst_base  = word_x.cur_off + 32'(RTU_LEAF_HDR_BYTES);
                word_n.blas_floor = sp_q;
                word_n.cur_off    = word_x.cur_off + 32'(RTU_LEAF_HDR_BYTES);
                word_n.cstate     = CS_INST_REQ;
                wake_self         = 1'b1;
            end else begin
                word_n.cstate = CS_POP;
                wake_self     = 1'b1;
            end
        end
        CS_FEED: begin
            box_feed_r = 1'b1;
            if (word_x.feed_idx == last_child) begin
                word_n.cstate = CS_WAIT;
            end else begin
                word_n.feed_idx = word_x.feed_idx + RTU_CHILD_BITS'(1);
                wake_self = 1'b1;
            end
        end
        CS_WAIT: begin
            // woken by the collector: this node's ordering is complete
            word_n.push_ptr = (coll_cnt_q == RTU_CHILD_BITS'(0))
                            ? RTU_CHILD_BITS'(0)
                            : (coll_cnt_q - RTU_CHILD_BITS'(1));
            word_n.cstate = CS_PUSH;
            wake_self     = 1'b1;
        end
        CS_PUSH: begin
            if (word_x.push_ptr != RTU_CHILD_BITS'(0)) begin
                if (sp_q != RTU_STACK_BITS'(RTU_STACK_DEPTH)) begin
                    stk_wr_r    = 1'b1;
                    stk_wdata_r = coll_ordoff[word_x.coll_id][word_x.push_ptr[IDXW-1:0]]
                                & RTU_CHILD_OFF_MASK;
                    sp_inc      = 1'b1;
                end else if (word_x.in_blas) begin
                    word_n.ovf_o = 1'b1;
                end else begin
                    word_n.ovf_w = 1'b1;
                end
                word_n.push_ptr = word_x.push_ptr - RTU_CHILD_BITS'(1);
                wake_self = 1'b1;
            end else if (coll_cnt_q != RTU_CHILD_BITS'(0)) begin
                word_n.cur_off = coll_ordoff[word_x.coll_id][0] & RTU_CHILD_OFF_MASK;
                coll_free_r    = 1'b1;
                word_n.cstate  = CS_REQ0;
                wake_self      = 1'b1;
            end else begin
                coll_free_r   = 1'b1;
                word_n.cstate = CS_POP;
                wake_self     = 1'b1;
            end
        end
        CS_PROC_WAIT: begin
            // woken by the collector: the raw AABB result landed
            if (coll_hit_q
             && (coll_t0_q < word_x.best_t)
             && (!yld_q[sel_q] || (coll_t0_q < word_x.yld_t))) begin
                cf_din_r.kind = CK_YLDP;
                cf_din_r.t    = coll_t0_q;
                cf_din_r.prim = word_x.prim_base;
                cf_din_r.geom = word_x.geom_r;
                cf_din_r.sbt  = word_x.proc_sbt;
                if (!cf_full) begin
                    cf_push_r     = 1'b1;
                    exec_yld_set  = 1'b1;
                    exec_cbtype   = RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC);
                    word_n.yld_t  = coll_t0_q;
                    coll_free_r   = 1'b1;
                    word_n.cstate = CS_POP;
                end
                wake_self = 1'b1;   // retries while the commit queue is full
            end else begin
                coll_free_r   = 1'b1;
                word_n.cstate = CS_POP;
                wake_self     = 1'b1;
            end
        end
        CS_LTRI_REQ0: begin
            mem_issue = 1'b1;
            if (mem_fire) begin
                word_n.f_total = tri_rec_lines;
                word_n.cstate  = CS_LTRI_RSP0;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_LTRI_RSP0: begin
            if (word_x.f_total == LB'(1)) begin
                tri_feed_r       = 1'b1;   // vertices fed straight off the image
                word_n.tri_flags = ltri_flags;
                word_n.cstate    = CS_TRI_WAIT;
            end else begin
                word_n.f_idx  = LB'(1);
                word_n.cstate = CS_LTRI_REQN;
                wake_self     = 1'b1;
            end
        end
        CS_LTRI_REQN: begin
            mem_issue = 1'b1;
            mem_fslot = word_x.f_idx;
            if (mem_fire) begin
                word_n.cstate = CS_LTRI_RSPN;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_LTRI_RSPN: begin
            if ((word_x.f_idx + LB'(1)) == word_x.f_total) begin
                tri_feed_r       = 1'b1;
                word_n.tri_flags = ltri_flags;
                word_n.cstate    = CS_TRI_WAIT;
            end else begin
                word_n.f_idx  = word_x.f_idx + LB'(1);
                word_n.cstate = CS_LTRI_REQN;
                wake_self     = 1'b1;
            end
        end
        CS_TRI_WAIT: begin
            // woken by the tri PE result (held in its result RAM, so a retry
            // on a full commit queue re-reads the same result)
            if (tri_committable && tri_opaque) begin
                cf_din_r.kind = CK_HIT;
                cf_din_r.t    = trit_q;
                cf_din_r.u    = triu_q;
                cf_din_r.v    = triv_q;
                cf_din_r.prim = word_x.prim_base + word_x.tri_i;
                cf_din_r.inst = word_x.in_blas ? word_x.inst_id   : 32'd0;
                cf_din_r.cust = word_x.in_blas ? word_x.inst_cust : 32'd0;
                cf_din_r.geom = word_x.geom_r;
                if (cf_full) begin
                    wake_self = 1'b1;
                end else begin
                    cf_push_r     = 1'b1;
                    exec_hit_set  = 1'b1;
                    word_n.best_t = trit_q;
                    // a closer opaque hit occludes a farther candidate
                    if (yld_q[sel_q] && (word_x.yld_t >= trit_q)) begin
                        exec_yld_clr = 1'b1;
                    end
                    if (term_first) begin
                        word_n.cstate = CS_DONE;
                        exec_done     = 1'b1;
                    end else if ((word_x.tri_i + 32'd1) < word_x.tri_n) begin
                        word_n.tri_i   = word_x.tri_i + 32'd1;
                        word_n.cur_off = word_x.cur_off + 32'(RTU_TRI_STRIDE);
                        word_n.cstate  = CS_LTRI_REQ0;
                        wake_self      = 1'b1;
                    end else begin
                        word_n.cstate = CS_POP;
                        wake_self     = 1'b1;
                    end
                end
            end else if (tri_committable
                      && (!yld_q[sel_q] || (trit_q < word_x.yld_t))) begin
                cf_din_r.kind = CK_YLDA;
                cf_din_r.t    = trit_q;
                cf_din_r.u    = triu_q;
                cf_din_r.v    = triv_q;
                cf_din_r.prim = word_x.prim_base + word_x.tri_i;
                cf_din_r.inst = word_x.in_blas ? word_x.inst_id   : 32'd0;
                cf_din_r.cust = word_x.in_blas ? word_x.inst_cust : 32'd0;
                cf_din_r.geom = word_x.geom_r;
                cf_din_r.sbt  = cls_sbt;
                if (cf_full) begin
                    wake_self = 1'b1;
                end else begin
                    cf_push_r    = 1'b1;
                    exec_yld_set = 1'b1;
                    exec_cbtype  = cls_cbtype;
                    word_n.yld_t = trit_q;
                    if ((word_x.tri_i + 32'd1) < word_x.tri_n) begin
                        word_n.tri_i   = word_x.tri_i + 32'd1;
                        word_n.cur_off = word_x.cur_off + 32'(RTU_TRI_STRIDE);
                        word_n.cstate  = CS_LTRI_REQ0;
                    end else begin
                        word_n.cstate = CS_POP;
                    end
                    wake_self = 1'b1;
                end
            end else begin
                if ((word_x.tri_i + 32'd1) < word_x.tri_n) begin
                    word_n.tri_i   = word_x.tri_i + 32'd1;
                    word_n.cur_off = word_x.cur_off + 32'(RTU_TRI_STRIDE);
                    word_n.cstate  = CS_LTRI_REQ0;
                end else begin
                    word_n.cstate = CS_POP;
                end
                wake_self = 1'b1;
            end
        end
        CS_POP: begin
            if (FLAT) begin
                if (word_x.in_blas) begin
                    word_n.cstate = CS_INST_NEXT;
                    wake_self     = 1'b1;
                end else begin
                    word_n.cstate = CS_DONE;
                    exec_done     = 1'b1;
                end
            end else if (word_x.in_blas && (sp_q == word_x.blas_floor)) begin
                if (word_x.ovf_o && (word_x.rst_o != RST_CNTW'(RTU_RESTART_CAP))) begin
                    // an object-level subtree was dropped: re-descend the BLAS
                    // root pruning by the tightened best_t
                    word_n.ovf_o   = 1'b0;
                    word_n.rst_o   = word_x.rst_o + RST_CNTW'(1);
                    word_n.cur_off = word_x.blas_root;
                    word_n.cstate  = CS_REQ0;
                end else begin
                    word_n.cstate = CS_INST_NEXT;
                end
                wake_self = 1'b1;
            end else if (sp_q == '0) begin
                if (word_x.ovf_w && (word_x.rst_w != RST_CNTW'(RTU_RESTART_CAP))) begin
                    word_n.ovf_w   = 1'b0;
                    word_n.rst_w   = word_x.rst_w + RST_CNTW'(1);
                    word_n.cur_off = word_x.root_off;
                    word_n.cstate  = CS_REQ0;
                    wake_self      = 1'b1;
                end else begin
                    word_n.cstate = CS_DONE;
                    exec_done     = 1'b1;
                end
            end else begin
                word_n.cur_off = stacktop_q;
                sp_dec         = 1'b1;
                word_n.cstate  = CS_REQ0;
                wake_self      = 1'b1;
            end
        end
        CS_INST_REQ: begin
            mem_issue = 1'b1;
            if (mem_fire) begin
                word_n.f_total = inst_lines;
                word_n.cstate  = CS_INST_RSP0;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_INST_RSP0: begin
            if (word_x.f_total == LB'(1)) begin
                word_n.cstate = CS_INST_RSPN;
            end else begin
                word_n.f_idx  = LB'(1);
                word_n.cstate = CS_INST_REQN;
            end
            wake_self = 1'b1;
        end
        CS_INST_REQN: begin
            mem_issue = 1'b1;
            mem_fslot = word_x.f_idx;
            if (mem_fire) begin
                word_n.cstate = CS_INST_RSPN;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_INST_RSPN: begin
            if ((word_x.f_total != LB'(1))
             && ((word_x.f_idx + LB'(1)) != word_x.f_total)) begin
                word_n.f_idx  = word_x.f_idx + LB'(1);
                word_n.cstate = CS_INST_REQN;
                wake_self     = 1'b1;
            end else if (inst_culled) begin
                word_n.cstate = CS_INST_NEXT;
                wake_self     = 1'b1;
            end else begin
                xform_feed_r      = 1'b1;   // record fields fed straight off the image
                word_n.blas_root  = inst_blas;
                word_n.inst_id    = FLAT ? word_x.inst_idx : inst_id_w;
                word_n.inst_cust  = inst_custom;
                word_n.inst_flags = inst_fl_w;
                word_n.cstate     = CS_XFORM_WT;
            end
        end
        CS_XFORM_WT: begin
            // woken by the xform result: the object ray landed
            word_n.obj_o = xfo_q;
            word_n.obj_d = xfd_q;
            if (FLAT) begin
                // the flat scan runs no box tests: no object reciprocal
                word_n.in_blas = 1'b1;
                word_n.cur_off = word_x.blas_root;
                word_n.cstate  = CS_BHDR_REQ;
            end else begin
                word_n.setup_axis = 2'd0;
                word_n.cstate     = CS_OBJ_SETUP;
            end
            wake_self = 1'b1;
        end
        CS_OBJ_SETUP: begin
            recip_issue_r = 1'b1;
            recip_obj_r   = 1'b1;
            if (word_x.setup_axis == 2'd2) begin
                word_n.cstate = CS_OBJ_SETUP_WT;
            end else begin
                word_n.setup_axis = word_x.setup_axis + 2'd1;
                wake_self = 1'b1;
            end
        end
        CS_OBJ_SETUP_WT: begin
            word_n.obj_inv_d = recip_q;
            word_n.in_blas   = 1'b1;
            word_n.cur_off   = word_x.blas_root;
            word_n.rst_o     = '0;
            word_n.ovf_o     = 1'b0;
            word_n.cstate    = CS_REQ0;
            wake_self        = 1'b1;
        end
        CS_INST_NEXT: begin
            word_n.in_blas = 1'b0;
            if ((FLAT_TLAS != 0) && yld_q[sel_q]) begin
                // the flat instance loop stops on a staged candidate
                word_n.cstate = CS_DONE;
                exec_done     = 1'b1;
            end else if ((word_x.inst_idx + 32'd1) == word_x.inst_cnt) begin
                if (FLAT) begin
                    word_n.cstate = CS_DONE;
                    exec_done     = 1'b1;
                end else begin
                    word_n.cstate = CS_POP;
                    wake_self     = 1'b1;
                end
            end else begin
                word_n.inst_idx = word_x.inst_idx + 32'd1;
                word_n.cur_off  = word_x.inst_base
                                + ((word_x.inst_idx + 32'd1) * 32'(RTU_INST_STRIDE));
                word_n.cstate   = CS_INST_REQ;
                wake_self       = 1'b1;
            end
        end
        CS_BHDR_REQ: begin
            mem_issue = 1'b1;
            if (mem_fire) begin
                word_n.cstate = CS_BHDR_WAIT;
            end else begin
                wake_self = 1'b1;
            end
        end
        CS_BHDR_WAIT: begin
            // flat TLAS: BLAS header word0 = triangle count (object space)
            if (hdr_count == 32'd0 || skip_tris) begin
                word_n.cstate = CS_INST_NEXT;
            end else begin
                word_n.tri_n     = hdr_count;
                word_n.tri_i     = '0;
                word_n.prim_base = '0;
                word_n.geom_r    = '0;
                word_n.cur_off   = word_x.cur_off + 32'(RTU_SCENE_HDR_BYTES);
                word_n.cstate    = CS_LTRI_REQ0;
            end
            wake_self = 1'b1;
        end
        default:;   // CS_DONE
        endcase
    end

    // EXEC-driven strobes (qualified by stage validity)
    assign box_feed      = x_valid && box_feed_r;
    assign box_feed_raw  = x_valid && box_raw_r;
    assign tri_feed      = x_valid && tri_feed_r;
    assign xform_feed    = x_valid && xform_feed_r;
    assign recip_issue   = x_valid && recip_issue_r;
    assign recip_obj     = recip_obj_r;
    assign recip_axis    = word_x.setup_axis;
    assign cf_push       = x_valid && cf_push_r;
    assign cf_din        = cf_din_r;
    assign cs_wr         = x_valid;
    assign cs_wdata      = word_n;
    assign stk_wr        = x_valid && stk_wr_r;
    assign stk_wdata     = stk_wdata_r;
    assign mem_req_valid = x_valid && mem_issue;
    assign mem_req_addr  = structaddr_q + (ADDRW'(mem_fslot) << RTU_LINE_SEL_BITS);
    assign mem_req_tag   = sel_q;

    // ═══════════════════════ hot-state update ═════════════════════════
    integer k;
    always_ff @(posedge clk) begin
        if (reset) begin
            rdy_set     <= '0;
            fresh_set   <= '0;
            done_q      <= '0;
            mask_q      <= '0;
            hit_q       <= '0;
            yld_q       <= '0;
            attr_q      <= '0;
            running     <= '0;
            finalised   <= '0;
            done_r      <= '0;
            pend_resume <= '0;
        end else begin
            done_r <= '0;

            // SELECT consumes the wake bit
            if (grant_valid) begin
                rdy_set[grant_idx] <= 1'b0;
            end

            // slot launch: arm the slot before its rays land
            if (slot_start_valid) begin
                running[slot_start_slot]    <= 1'b1;
                finalised[slot_start_slot]  <= 1'b0;
                slot_flags[slot_start_slot] <= slot_start_flags;
                slot_cull[slot_start_slot]  <= slot_start_cull;
                slot_scene[slot_start_slot] <= slot_start_scene;
                for (k = 0; k < NUM_LANES; k = k + 1) begin
                    mask_q[32'(slot_start_slot)*NUM_LANES + k] <= slot_start_mask[k];
                    done_q[32'(slot_start_slot)*NUM_LANES + k] <= ~slot_start_mask[k];
                    hit_q[32'(slot_start_slot)*NUM_LANES + k]  <= 1'b0;
                    yld_q[32'(slot_start_slot)*NUM_LANES + k]  <= 1'b0;
                    attr_q[32'(slot_start_slot)*NUM_LANES + k] <= 1'b0;
                end
            end

            // a lane's ray landed: that context launches now
            if (ray_wr_valid) begin
                rdy_set[ray_wr_ctx]   <= 1'b1;
                fresh_set[ray_wr_ctx] <= 1'b1;
            end

            // event wake-ups (each targets a parked context)
            if (mem_rsp_valid) begin
                rdy_set[mem_rsp_tag] <= 1'b1;
            end
            if (tri_valid_out) begin
                rdy_set[tri_tag_out] <= 1'b1;
            end
            if (xform_valid_out) begin
                rdy_set[xform_tag_out] <= 1'b1;
            end
            if (recip_valid_out && recip_last_out) begin
                rdy_set[recip_tag_out] <= 1'b1;
            end
            // a box collection completing wakes its context
            if (box_valid_out
             && (coll_proc[box_coll] || (coll_cnt[box_coll] == coll_last[box_coll]))) begin
                rdy_set[coll_ctx[box_coll]] <= 1'b1;
            end

            // EXEC outcomes
            if (x_valid) begin
                fresh_set[sel_q] <= 1'b0;
                if (wake_self) begin
                    rdy_set[sel_q] <= 1'b1;
                end
                if (exec_done) begin
                    done_q[sel_q] <= 1'b1;
                end
                if (exec_hit_set) begin
                    hit_q[sel_q] <= 1'b1;
                end
                if (exec_yld_set) begin
                    yld_q[sel_q]    <= 1'b1;
                    cbtype_q[sel_q] <= exec_cbtype;
                end
                if (exec_yld_clr) begin
                    yld_q[sel_q] <= 1'b0;
                end
                if (fresh_q) begin
                    sp_q_arr[sel_q] <= '0;
                end else if (sp_inc) begin
                    sp_q_arr[sel_q] <= sp_q + RTU_STACK_BITS'(1);
                end else if (sp_dec) begin
                    sp_q_arr[sel_q] <= sp_q - RTU_STACK_BITS'(1);
                end
                if (mem_fire) begin
                    f_slot_q[sel_q] <= mem_fslot;
                end
            end

            // resume: capture the actions, queue the walker job
            for (k = 0; k < NUM_SLOTS; k = k + 1) begin
                if (resume[k]) begin
                    pend_resume[k] <= 1'b1;
                end
            end

            // trivial finalise (no CHS/MISS staging)
            if (fin_trivial) begin
                finalised[fin_slot] <= 1'b1;
            end
            if (res_start) begin
                pend_resume[res_slot_enc] <= 1'b0;
            end

            // barrier-walker completion commits the whole job's flag updates
            if (bw_job_done) begin
                if (bw_is_res) begin
                    for (k = 0; k < NUM_LANES; k = k + 1) begin
                        if (bw_copy_mask[k]) begin
                            hit_q[32'(bw_slot)*NUM_LANES + k]  <= 1'b1;
                            attr_q[32'(bw_slot)*NUM_LANES + k] <= 1'b1;
                        end
                        yld_q[32'(bw_slot)*NUM_LANES + k] <= 1'b0;
                    end
                end else begin
                    for (k = 0; k < NUM_LANES; k = k + 1) begin
                        if (bw_copy_mask[k]) begin
                            yld_q[32'(bw_slot)*NUM_LANES + k]    <= 1'b1;
                            cbtype_q[32'(bw_slot)*NUM_LANES + k] <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_CHS);
                        end
                        if (bw_miss_mask[k]) begin
                            yld_q[32'(bw_slot)*NUM_LANES + k]    <= 1'b1;
                            cbtype_q[32'(bw_slot)*NUM_LANES + k] <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_MISS);
                        end
                    end
                    finalised[bw_slot] <= 1'b1;
                end
            end

            // slot completion: all contexts retired, nothing left to yield
            for (k = 0; k < NUM_SLOTS; k = k + 1) begin
                if (running[k] && all_done[k] && finalised[k] && !yld_any[k]
                 && ce_idle && bw_idle && !pend_resume[k]) begin
                    running[k] <= 1'b0;
                    done_r[k]  <= 1'b1;
                end
            end
        end
    end

    // actions are captured at the resume pulse
    always_ff @(posedge clk) begin
        for (integer s2 = 0; s2 < NUM_SLOTS; s2 = s2 + 1) begin
            if (resume[s2]) begin
                for (integer j = 0; j < NUM_LANES; j = j + 1) begin
                    act_q[s2*NUM_LANES + j] <= action[s2*NUM_LANES + j];
                end
            end
        end
    end

    // ── collector update ──────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (reset) begin
            coll_busy <= '0;
        end else begin
            if (x_valid && coll_alloc_r) begin
                coll_busy[coll_alloc_idx]   <= 1'b1;
                coll_proc[coll_alloc_idx]   <= box_raw_r;
                coll_ctx[coll_alloc_idx]    <= sel_q;
                coll_cnt[coll_alloc_idx]    <= '0;
                coll_last[coll_alloc_idx]   <= box_raw_r ? RTU_CHILD_BITS'(0) : last_child;
                coll_ordcnt[coll_alloc_idx] <= '0;
            end
            if (x_valid && coll_free_r) begin
                coll_busy[word_x.coll_id] <= 1'b0;
            end
            if (box_valid_out) begin
                if (coll_proc[box_coll]) begin
                    coll_prochit[box_coll] <= box_hit;
                    coll_ordt[box_coll][0] <= box_t_near;
                end else begin
                    coll_cnt[box_coll] <= coll_cnt[box_coll] + RTU_CHILD_BITS'(1);
                    if (coll_pushable) begin
                        for (integer oc = 0; oc < RTU_BVH_WIDTH; oc = oc + 1) begin
                            if (RTU_CHILD_BITS'(oc) == ins_pos) begin
                                coll_ordoff[box_coll][oc[IDXW-1:0]] <= box_childoff;
                                coll_ordt[box_coll][oc[IDXW-1:0]]   <= box_t_near;
                            end else if (RTU_CHILD_BITS'(oc) > ins_pos) begin
                                coll_ordoff[box_coll][oc[IDXW-1:0]] <= coll_ordoff[box_coll][oc[IDXW-1:0]-1];
                                coll_ordt[box_coll][oc[IDXW-1:0]]   <= coll_ordt[box_coll][oc[IDXW-1:0]-1];
                            end
                        end
                        coll_ordcnt[box_coll] <= coll_ordcnt[box_coll] + RTU_CHILD_BITS'(1);
                    end
                end
            end
        end
    end

    // ── barrier walker FSM ────────────────────────────────────────────
    always_ff @(posedge clk) begin
        bw_job_done <= 1'b0;
        if (reset) begin
            bw_state <= BW_IDLE;
        end else begin
            case (bw_state)
            BW_IDLE: begin
                if (fin_start) begin
                    bw_is_res    <= 1'b0;
                    bw_slot      <= fin_slot;
                    bw_copy_mask <= fin_chs_mask;
                    bw_miss_mask <= fin_miss_mask;
                    bw_field     <= '0;
                    bw_zidx      <= '0;
                    bw_state     <= (fin_chs_mask != '0) ? BW_RD : BW_ZMISS;
                end else if (res_start) begin
                    bw_is_res    <= 1'b1;
                    bw_slot      <= res_slot_enc;
                    bw_copy_mask <= res_acc_mask;
                    bw_miss_mask <= '0;
                    bw_field     <= '0;
                    if (res_acc_mask == '0) begin
                        bw_job_done <= 1'b1;   // nothing accepted: just clear
                    end else begin
                        bw_state <= BW_RD;
                    end
                end
            end
            BW_RD: begin
                if (bw_rd_gnt) begin
                    bw_state <= BW_CAP;
                end
            end
            BW_CAP: begin
                bw_data  <= ws_rdata;
                bw_state <= (bw_is_res && (bw_field == 3'd0)) ? BW_RD2 : BW_WR;
            end
            BW_RD2: begin
                if (bw_rd_gnt) begin
                    bw_state <= BW_CAP2;
                end
            end
            BW_CAP2: begin
                bw_data2 <= ws_rdata;
                bw_state <= BW_WR;
            end
            BW_WR: begin
                if (bw_wr_gnt) begin
                    if (bw_field == 3'd6) begin
                        if (bw_is_res) begin
                            bw_state <= BW_ATTRR;
                        end else if (bw_miss_mask != '0) begin
                            bw_zidx  <= '0;
                            bw_state <= BW_ZMISS;
                        end else begin
                            bw_job_done <= 1'b1;
                            bw_state    <= BW_IDLE;
                        end
                    end else begin
                        bw_field <= bw_field + 3'd1;
                        bw_state <= BW_RD;
                    end
                end
            end
            BW_ZMISS: begin
                if (bw_wr_gnt) begin
                    if (bw_zidx == 2'd2) begin
                        bw_job_done <= 1'b1;
                        bw_state    <= BW_IDLE;
                    end else begin
                        bw_zidx <= bw_zidx + 2'd1;
                    end
                end
            end
            BW_ATTRR: begin
                if (bw_rd_gnt) begin
                    bw_state <= BW_ATTRC;
                end
            end
            BW_ATTRC: begin
                bw_data  <= ws_rdata;
                bw_state <= BW_ATTRW;
            end
            BW_ATTRW: begin
                if (bw_wr_gnt) begin
                    bw_job_done <= 1'b1;
                    bw_state    <= BW_IDLE;
                end
            end
            default: begin
                bw_state <= BW_IDLE;
            end
            endcase
        end
    end

    // ── outputs ───────────────────────────────────────────────────────
    assign busy = running;
    assign done = done_r;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_yield
        assign yield[s] = running[s] && all_done[s] && finalised[s] && yld_any[s]
                       && ce_idle && bw_idle && !pend_resume[s];
    end
    assign hit_bits = hit_q;
    assign yld_bits = yld_q;
    assign cb_types = cbtype_q;
    assign attr_vld = attr_q;

    `UNUSED_VAR ({f_aligned, s1_fresh, ray_q, ins_pos})

`ifdef DBG_TRACE_RTU
    always_ff @(posedge clk) begin
        if (x_valid && (word_x.cstate == CS_DISPATCH)) begin
            `TRACE(2, ("%t: %s rtu-node: ctx=%0d, addr=0x%0h, kind=%0d\n",
                $time, INSTANCE_ID, sel_q, structaddr_q, node_kind))
        end
        if (tri_valid_out) begin
            `TRACE(2, ("%t: %s rtu-tri: ctx=%0d, hit=%0d, t=0x%0h\n",
                $time, INSTANCE_ID, tri_tag_out, tri_hit, tri_t))
        end
        if (| done_r) begin
            `TRACE(1, ("%t: %s rtu-done: slots=%b\n", $time, INSTANCE_ID, done_r))
        end
    end
`endif

endmodule
