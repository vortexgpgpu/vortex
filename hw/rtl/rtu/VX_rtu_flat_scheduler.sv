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

// VX_rtu_flat_scheduler — flat triangle-list traversal control (the
// RTU_BVH_WIDTH==0 build). Holds one ray context per lane and time-multiplexes
// a single shared ray-triangle datapath across them: it linearly scans all
// triangles of a flat scene, keeping the closest opaque hit within
// [t_min, best_t). No BVH nodes, stack, box PE, or ray-setup reciprocal — the
// Möller-Trumbore tri PE consumes the ray directly. Mirrors the SimX FlatWalker;
// it presents the same scheduler↔core interface as VX_rtu_scheduler so
// VX_rtu_core can instantiate either by configured width.
//
// Like the BVH scheduler it runs a SELECT/EXEC micro-step pipeline and parks a
// context across the two long-latency operations (a cache-line fetch and a
// tri-PE test), tagging each by context id so responses route back. The flat
// triangle record (40 B, v0@0/v1@12/v2@24/flags@36, no leaf header) is fetched
// and byte-aligned exactly like a BVH leaf, since records straddle cache lines.

`include "VX_define.vh"

module VX_rtu_flat_scheduler import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_CTX   = 4,
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8,
    parameter CTX_TAG_W = `LOG2UP(NUM_CTX)
) (
    input  wire        clk,
    input  wire        reset,

    input  wire                       start,
    input  wire [NUM_CTX-1:0]         mask,
    input  rtu_ray_t [NUM_CTX-1:0]    rays,
    output wire                       busy,
    output wire                       done,

    output wire [NUM_CTX-1:0]         res_hit,
    output wire [NUM_CTX-1:0][31:0]   res_t,
    output wire [NUM_CTX-1:0][31:0]   res_u,
    output wire [NUM_CTX-1:0][31:0]   res_v,
    output wire [NUM_CTX-1:0][31:0]   res_prim,
    output wire [NUM_CTX-1:0][31:0]   res_geom,
    output wire [NUM_CTX-1:0][31:0]   res_inst,

    // Phase 2 callback yield barrier. After the walk completes, if any lane
    // staged a non-opaque candidate the scheduler holds here, asserts yield
    // with the per-lane candidate (res_* present the candidate attrs while
    // yield is high) and waits for the core to deliver per-lane actions on
    // resume. ACCEPT/TERMINATE commit the candidate; IGNORE keeps the opaque
    // hit. Single-yield-per-lane (proposal §minimum) — no re-walk.
    output wire                                       yield,
    output wire [NUM_CTX-1:0]                         yield_mask,
    output wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]   yield_cbtype,
    output wire [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]    yield_sbt,
    input  wire                                       resume,
    input  wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] action,
    input  wire [NUM_CTX-1:0][31:0]                   action_hit_t,

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
    localparam BUF_BITS = RTU_FLAT_LINES * LINE_BITS;
    localparam LB       = RTU_FLAT_LINES_BITS;

    // per-context FSM states
    localparam [4:0] CS_DONE     = 5'd0,   // retired (also idle lanes)
                     CS_HDR_REQ  = 5'd1,   // issue scene-header fetch
                     CS_HDR_WAIT = 5'd2,   // park: header line (triangle/instance count)
                     CS_REQ0     = 5'd3,   // issue triangle line 0
                     CS_RSP0     = 5'd4,   // park: line 0
                     CS_REQN     = 5'd5,   // issue triangle line N
                     CS_RSPN     = 5'd6,   // park: line N
                     CS_TRI_FEED = 5'd7,   // stream triangle to tri PE
                     CS_TRI_WAIT = 5'd8,   // park: tri result
                     CS_NEXT     = 5'd9,   // advance to next triangle / terminate
                     // TLAS-only states (VX_CFG_RTU_TLAS_ENABLE): instance loop
                     // over inline BLAS triangle lists, with a world→object xform.
                     CS_INST_REQ = 5'd10,  // issue instance-record line 0
                     CS_INST_RSP0= 5'd11,  // park: instance line 0
                     CS_INST_REQN= 5'd12,  // issue instance-record line N
                     CS_INST_RSPN= 5'd13,  // park: instance line N -> cull / xform
                     CS_XFORM    = 5'd14,  // feed the world ray + xform to VX_rtu_xform
                     CS_XFORM_WT = 5'd15,  // park: object ray
                     CS_BLAS_REQ = 5'd16,  // issue BLAS header line
                     CS_BLAS_RSP = 5'd17,  // park: BLAS header (triangle_count)
                     CS_INST_NEXT= 5'd18;  // advance to next instance / terminate

    // ── per-context state ─────────────────────────────────────────────
    reg [NUM_CTX-1:0][4:0]                cstate;
    rtu_ray_t [NUM_CTX-1:0]               ray_r;
    reg [NUM_CTX-1:0][31:0]               best_t;
    reg [NUM_CTX-1:0]                     hit_r;
    reg [NUM_CTX-1:0][31:0]               hit_t_r, hit_u_r, hit_v_r, hit_prim_r;
    reg [NUM_CTX-1:0][31:0]               hit_inst_r;   // committed hit's instance id (TLAS)
    reg [NUM_CTX-1:0][31:0]               tri_idx, tri_count, cur_off;
    reg [NUM_CTX-1:0][BUF_BITS-1:0]       f_buf;
    reg [NUM_CTX-1:0][LB-1:0]             f_idx, f_total, f_slot;
    reg [NUM_CTX-1:0]                     line_ready, tri_ready;
    reg [NUM_CTX-1:0]                     tri_hit_p, tri_back_p;
    reg [NUM_CTX-1:0][31:0]               tri_t_p, tri_u_p, tri_v_p, tri_prim_p;
    reg [NUM_CTX-1:0][31:0]               tri_flags_p;   // latched at TRI_FEED

    // Phase 2 per-context yield candidate (closest non-opaque hit so far).
    reg [NUM_CTX-1:0]                     yld_pending;
    reg [NUM_CTX-1:0][31:0]               yld_t, yld_u, yld_v, yld_prim;
    reg [NUM_CTX-1:0][31:0]               yld_inst;   // candidate hit's instance id (TLAS)
    reg [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0] yld_cbtype;
    reg [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]  yld_sbt;
    reg [NUM_CTX-1:0]                     mask_r;     // active-lane mask
    reg                                  finalised;  // one-shot end-of-walk finalise

`ifdef VX_CFG_RTU_TLAS_ENABLE
    // ── per-context TLAS state: the instance loop wrapping the BLAS scan ──
    reg [NUM_CTX-1:0][31:0]               inst_count, inst_idx, blas_off;
    reg [NUM_CTX-1:0][11:0][31:0]         inst_xform;   // latched 3x4 affine
    reg [NUM_CTX-1:0][2:0][31:0]          obj_o, obj_d; // object-space ray
    reg [NUM_CTX-1:0]                     xform_ready;  // async xform result landed
`endif

    reg                   running, done_r;
    reg [CTX_TAG_W-1:0]   cc;

    // ── micro-step pipeline: SELECT then EXEC ─────────────────────────
    reg                   phase;
    localparam PH_SELECT = 1'b0, PH_EXEC = 1'b1;
    reg [CTX_TAG_W-1:0]   sel_q;
    rtu_ray_t             ray_q;
    reg [BUF_BITS-1:0]    fbuf_q;
    reg [31:0]            curoff_q, bestt_q, triidx_q, tricount_q;
    reg [4:0]             cstate_q;
    reg [LB-1:0]          fidx_q, ftotal_q;
`ifdef VX_CFG_RTU_TLAS_ENABLE
    reg [31:0]            instidx_q, instcount_q, blasoff_q;
    reg [11:0][31:0]      xform_q;
    reg [2:0][31:0]       objo_q, objd_q;
`endif

    // ── runnable predicate ────────────────────────────────────────────
    wire [NUM_CTX-1:0] runnable;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_runnable
        reg r;
        always @(*) begin
            case (cstate[i])
                CS_DONE:     r = 1'b0;
                CS_HDR_WAIT,
                CS_RSP0,
                CS_RSPN:     r = line_ready[i];
                CS_TRI_WAIT: r = tri_ready[i];
`ifdef VX_CFG_RTU_TLAS_ENABLE
                CS_INST_RSP0,
                CS_INST_RSPN,
                CS_BLAS_RSP: r = line_ready[i];
                CS_XFORM_WT: r = xform_ready[i];
`endif
                default:     r = 1'b1;
            endcase
        end
        assign runnable[i] = r;
    end

    reg [CTX_TAG_W-1:0] sel;
    reg                 sel_valid;
    always @(*) begin
        sel       = cc;
        sel_valid = 1'b0;
        for (integer off = NUM_CTX-1; off >= 0; off = off - 1) begin
            integer cand;
            cand = (32'(cc) + off) % NUM_CTX;
            if (runnable[cand]) begin
                sel       = CTX_TAG_W'(cand);
                sel_valid = 1'b1;
            end
        end
    end
    wire exec = (phase == PH_EXEC);

    // ── combinational decode of the EXEC snapshot ─────────────────────
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] struct_addr = ray_q.scene_base + curoff_q;
    wire [RTU_LINE_SEL_BITS-1:0]      f_off   = struct_addr[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0]      f_shift = {f_off, 3'b000};
    wire [BUF_BITS-1:0]               f_aligned = fbuf_q >> f_shift;

    wire [2:0][31:0] tri_v0, tri_v1, tri_v2;
    for (genvar a = 0; a < 3; ++a) begin : g_tri_v
        assign tri_v0[a] = f_aligned[(RTU_FLAT_OFF_V0 + 4*a)*8 +: 32];
        assign tri_v1[a] = f_aligned[(RTU_FLAT_OFF_V1 + 4*a)*8 +: 32];
        assign tri_v2[a] = f_aligned[(RTU_FLAT_OFF_V2 + 4*a)*8 +: 32];
    end
    wire [31:0] tri_flags = f_aligned[RTU_FLAT_OFF_FLAGS*8 +: 32];

    wire [31:0] f_off32 = 32'(f_off);
    wire [LB-1:0] tri_lines =
        LB'(((f_off32 + RTU_FLAT_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    // header line 0: word0 = triangle_count (TRI_LIST) or instance_count (TLAS).
    wire [31:0] hdr_count = f_aligned[31:0];

    // The BLAS triangle scan runs in object space when TLAS is enabled (the
    // walker always treats the scene as a TLAS in that build); the world ray
    // otherwise. The object ray is the per-instance VX_rtu_xform output.
`ifdef VX_CFG_RTU_TLAS_ENABLE
    // ── instance-record decode (64 B; matches the shared host/SimX layout) ──
    wire [31:0]       inst_blas = f_aligned[RTU_INST_OFF_BLAS*8      +: 32];
    wire [31:0]       inst_cull = f_aligned[RTU_INST_OFF_CULL_FLAT*8 +: 32];
    wire [11:0][31:0] inst_xform_w;
    for (genvar k = 0; k < 12; ++k) begin : g_inst_xform
        assign inst_xform_w[k] = f_aligned[(RTU_INST_OFF_XFORM + 4*k)*8 +: 32];
    end
    wire [LB-1:0] inst_lines =
        LB'(((f_off32 + RTU_INST_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    // (instance_mask & ray.cull_mask & 0xff) == 0 -> skip the whole instance.
    wire inst_culled = ((inst_cull & ray_q.cull_mask & 32'hff) == 32'd0);
    wire [2:0][31:0] tri_ro = objo_q;
    wire [2:0][31:0] tri_rd = objd_q;
`else
    wire [2:0][31:0] tri_ro = ray_q.origin;
    wire [2:0][31:0] tri_rd = ray_q.dir;
`endif

    // ── tri PE: tagged by context id so results route back ────────────
    wire        tri_valid_in = exec && (cstate_q == CS_TRI_FEED);
    wire        tri_valid_out, tri_hit, tri_back;
    wire [CTX_TAG_W-1:0] tri_tag_out;
    wire [31:0] tri_t, tri_u, tri_v;
    VX_rtu_tri_pe #(.TAG_WIDTH (CTX_TAG_W)) tri_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (tri_valid_in),
        .tag_in (sel_q),
        .origin (tri_ro), .dir (tri_rd),
        .v0 (tri_v0), .v1 (tri_v1), .v2 (tri_v2),
        .t_min (ray_q.t_min), .t_max (bestt_q),
        .valid_out (tri_valid_out), .tag_out (tri_tag_out), .hit (tri_hit),
        .t (tri_t), .u (tri_u), .v (tri_v), .back_facing (tri_back)
    );

`ifdef VX_CFG_RTU_TLAS_ENABLE
    // ── world→object ray xform PE: tagged by context id ───────────────
    wire        xform_valid_in = exec && (cstate_q == CS_XFORM);
    wire        xform_valid_out;
    wire [CTX_TAG_W-1:0] xform_tag_out;
    wire [2:0][31:0] xform_obj_o, xform_obj_d;
    VX_rtu_xform #(.TAG_WIDTH (CTX_TAG_W)) xform_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (xform_valid_in),
        .tag_in (sel_q),
        .xform (xform_q), .ro (ray_q.origin), .rd (ray_q.dir),
        .valid_out (xform_valid_out), .tag_out (xform_tag_out),
        .obj_ro (xform_obj_o), .obj_rd (xform_obj_d)
    );
`endif

    // ── memory request (single shared port, tagged by context) ────────
    wire fetch_issue = (cstate_q == CS_HDR_REQ)
                    || (cstate_q == CS_REQ0)
                    || (cstate_q == CS_REQN)
`ifdef VX_CFG_RTU_TLAS_ENABLE
                    || (cstate_q == CS_INST_REQ)
                    || (cstate_q == CS_INST_REQN)
                    || (cstate_q == CS_BLAS_REQ)
`endif
                    ;
    assign mem_req_valid = exec && fetch_issue;
    assign mem_req_tag   = sel_q;
    // line-0 fetches (REQ0 / INST_REQ / BLAS_REQ) address the structure base
    // (struct_addr = scene_base + cur_off); line-N fetches add fidx*line.
    wire line0_req = (cstate_q == CS_REQ0)
`ifdef VX_CFG_RTU_TLAS_ENABLE
                  || (cstate_q == CS_INST_REQ) || (cstate_q == CS_BLAS_REQ)
`endif
                  ;
    assign mem_req_addr  = (cstate_q == CS_HDR_REQ) ? ray_q.scene_base
                         : line0_req                ? struct_addr
                         : (struct_addr + (`VX_CFG_MEM_ADDR_WIDTH'(fidx_q) << RTU_LINE_SEL_BITS));
    assign mem_rsp_ready = 1'b1;
    wire mem_req_fire = mem_req_valid && mem_req_ready;

    // classify_tri_hit (rtu_classifier): face culling, effective-opacity
    // override, opacity-class culling, terminate-on-first-hit. All keyed on
    // the EXEC-snapshot ray flags (ray_q) and the latched tri flags.
    wire cull_back  = (ray_q.flags & `VX_RT_FLAG_CULL_BACK_FACING)  != 0;
    wire cull_front = (ray_q.flags & `VX_RT_FLAG_CULL_FRONT_FACING) != 0;
    wire skip_tris  = (ray_q.flags & `VX_RT_FLAG_SKIP_TRIANGLES)    != 0;
    wire ray_opaque   = (ray_q.flags & `VX_RT_FLAG_OPAQUE)                 != 0;
    wire ray_noopaque = (ray_q.flags & `VX_RT_FLAG_NO_OPAQUE)              != 0;
    wire cull_opaque  = (ray_q.flags & `VX_RT_FLAG_CULL_OPAQUE)            != 0;
    wire cull_noopq   = (ray_q.flags & `VX_RT_FLAG_CULL_NO_OPAQUE)         != 0;
    wire term_first   = (ray_q.flags & `VX_RT_FLAG_TERMINATE_ON_FIRST_HIT) != 0;

    // Effective opacity of the latched tri being committed at CS_TRI_WAIT.
    wire [31:0] cls_flags = tri_flags_p[sel_q];
    wire tri_opaque = ray_opaque   ? 1'b1
                    : ray_noopaque ? 1'b0
                    : ((cls_flags & RTU_TRI_FLAG_OPAQUE) != 0);
    wire cls_cull   = (tri_opaque && cull_opaque) || (!tri_opaque && cull_noopq);
    wire [RTU_CB_TYPE_BITS-1:0] cls_cbtype =
        (cls_flags & RTU_TRI_FLAG_PROC) ? RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC)
                                        : RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_ANYHIT);
    wire [RTU_CB_SBT_BITS-1:0]  cls_sbt =
        RTU_CB_SBT_BITS'((cls_flags >> RTU_TRI_SBT_IDX_SHIFT) & RTU_TRI_SBT_IDX_MASK);

    // A geometric hit that survives face + opacity-class culling, closer than
    // the best committed opaque hit.
    wire tri_pass = tri_hit_p[sel_q]
                  && !(tri_back_p[sel_q] && cull_back)
                  && !(!tri_back_p[sel_q] && cull_front)
                  && !cls_cull;
    wire tri_committable = tri_pass && (tri_t_p[sel_q] < bestt_q);

    wire [NUM_CTX-1:0] ctx_done;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_ctx_done
        assign ctx_done[i] = (cstate[i] == CS_DONE);
    end
    wire all_done = &ctx_done;

    integer k;
    always @(posedge clk) begin
        if (reset) begin
            running   <= 1'b0;
            done_r    <= 1'b0;
            cc        <= '0;
            phase     <= PH_SELECT;
            finalised <= 1'b0;
            for (k = 0; k < NUM_CTX; k = k + 1) begin
                cstate[k]      <= CS_DONE;
                line_ready[k]  <= 1'b0;
                tri_ready[k]   <= 1'b0;
                yld_pending[k] <= 1'b0;
`ifdef VX_CFG_RTU_TLAS_ENABLE
                xform_ready[k] <= 1'b0;
`endif
            end
        end else begin
            done_r <= 1'b0;

            if (!running && start) begin
                running   <= 1'b1;
                cc        <= '0;
                phase     <= PH_SELECT;
                mask_r    <= mask;
                finalised <= 1'b0;
                for (k = 0; k < NUM_CTX; k = k + 1) begin
                    ray_r[k]      <= rays[k];
                    best_t[k]     <= rays[k].t_max;
                    cur_off[k]    <= '0;
                    tri_idx[k]    <= '0;
                    tri_count[k]  <= '0;
                    line_ready[k] <= 1'b0;
                    tri_ready[k]  <= 1'b0;
                    hit_r[k]      <= 1'b0;
                    hit_t_r[k]    <= rays[k].t_max;
                    hit_u_r[k]    <= '0;
                    hit_v_r[k]    <= '0;
                    hit_prim_r[k] <= '0;
                    hit_inst_r[k] <= '0;
                    yld_inst[k]   <= '0;
                    yld_pending[k]<= 1'b0;
                    yld_t[k]      <= rays[k].t_max;
                    yld_u[k]      <= '0;
                    yld_v[k]      <= '0;
                    yld_prim[k]   <= '0;
                    yld_cbtype[k] <= '0;
                    yld_sbt[k]    <= '0;
`ifdef VX_CFG_RTU_TLAS_ENABLE
                    inst_count[k] <= '0;
                    inst_idx[k]   <= '0;
                    blas_off[k]   <= '0;
                    xform_ready[k]<= 1'b0;
`endif
                    cstate[k]     <= mask[k] ? CS_HDR_REQ : CS_DONE;
                end
            end

            // async line-fetch response → route to its context
            if (mem_rsp_valid) begin
                f_buf[mem_rsp_tag][f_slot[mem_rsp_tag] * LINE_BITS +: LINE_BITS] <= mem_rsp_data;
                line_ready[mem_rsp_tag] <= 1'b1;
            end

            // async tri-PE result → route to its context
            if (tri_valid_out) begin
                tri_ready[tri_tag_out]  <= 1'b1;
                tri_hit_p[tri_tag_out]  <= tri_hit;
                tri_back_p[tri_tag_out] <= tri_back;
                tri_t_p[tri_tag_out]    <= tri_t;
                tri_u_p[tri_tag_out]    <= tri_u;
                tri_v_p[tri_tag_out]    <= tri_v;
            end

`ifdef VX_CFG_RTU_TLAS_ENABLE
            // async xform-PE result → object ray routes back to its context
            if (xform_valid_out) begin
                obj_o[xform_tag_out]       <= xform_obj_o;
                obj_d[xform_tag_out]       <= xform_obj_d;
                xform_ready[xform_tag_out] <= 1'b1;
            end
`endif

            if (running) begin
                if (phase == PH_SELECT) begin
                    if (sel_valid) begin
                        sel_q      <= sel;
                        cc         <= sel;
                        ray_q      <= ray_r[sel];
                        fbuf_q     <= f_buf[sel];
                        curoff_q   <= cur_off[sel];
                        bestt_q    <= best_t[sel];
                        cstate_q   <= cstate[sel];
                        fidx_q     <= f_idx[sel];
                        ftotal_q   <= f_total[sel];
                        triidx_q   <= tri_idx[sel];
                        tricount_q <= tri_count[sel];
`ifdef VX_CFG_RTU_TLAS_ENABLE
                        instidx_q   <= inst_idx[sel];
                        instcount_q <= inst_count[sel];
                        blasoff_q   <= blas_off[sel];
                        xform_q     <= inst_xform[sel];
                        objo_q      <= obj_o[sel];
                        objd_q      <= obj_d[sel];
`endif
                        phase      <= PH_EXEC;
                    end
                end else begin
                    phase <= PH_SELECT;
                    case (cstate_q)
                    CS_HDR_REQ: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_HDR_WAIT;
                        end
                    end
                    CS_HDR_WAIT: begin
                        if (line_ready[sel_q]) begin
`ifdef VX_CFG_RTU_TLAS_ENABLE
                            // TLAS header word0 = instance_count; iterate
                            // instances, each over its own inline BLAS.
                            if (hdr_count == 32'd0) begin
                                cstate[sel_q] <= CS_DONE;
                            end else begin
                                inst_count[sel_q] <= hdr_count;
                                inst_idx[sel_q]   <= '0;
                                cur_off[sel_q]    <= 32'(RTU_SCENE_HDR_BYTES);
                                cstate[sel_q]     <= CS_INST_REQ;
                            end
`else
                            if (hdr_count == 32'd0 || skip_tris) begin
                                cstate[sel_q] <= CS_DONE;
                            end else begin
                                tri_count[sel_q] <= hdr_count;
                                tri_idx[sel_q]   <= '0;
                                cur_off[sel_q]   <= 32'(RTU_SCENE_HDR_BYTES);
                                cstate[sel_q]    <= CS_REQ0;
                            end
`endif
                        end
                    end
                    CS_REQ0: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            f_total[sel_q]    <= tri_lines;
                            cstate[sel_q]     <= CS_RSP0;
                        end
                    end
                    CS_RSP0: begin
                        if (line_ready[sel_q]) begin
                            if (ftotal_q == LB'(1)) begin
                                cstate[sel_q] <= CS_TRI_FEED;
                            end else begin
                                f_idx[sel_q]  <= LB'(1);
                                cstate[sel_q] <= CS_REQN;
                            end
                        end
                    end
                    CS_REQN: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= fidx_q;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_RSPN;
                        end
                    end
                    CS_RSPN: begin
                        if (line_ready[sel_q]) begin
                            if ((fidx_q + LB'(1)) == ftotal_q) begin
                                cstate[sel_q] <= CS_TRI_FEED;
                            end else begin
                                f_idx[sel_q]  <= fidx_q + LB'(1);
                                cstate[sel_q] <= CS_REQN;
                            end
                        end
                    end
                    CS_TRI_FEED: begin
                        tri_prim_p[sel_q]  <= triidx_q;  // flat prim_id = triangle index
                        tri_flags_p[sel_q] <= tri_flags;  // latch flags for CS_TRI_WAIT classify
                        cstate[sel_q]      <= CS_TRI_WAIT;
                    end
                    CS_TRI_WAIT: begin
                        if (tri_ready[sel_q]) begin
                            tri_ready[sel_q] <= 1'b0;
                            if (tri_committable) begin
                                if (tri_opaque) begin
                                    // commit closest opaque hit.
                                    best_t[sel_q]     <= tri_t_p[sel_q];
                                    hit_r[sel_q]      <= 1'b1;
                                    hit_t_r[sel_q]    <= tri_t_p[sel_q];
                                    hit_u_r[sel_q]    <= tri_u_p[sel_q];
                                    hit_v_r[sel_q]    <= tri_v_p[sel_q];
                                    hit_prim_r[sel_q] <= tri_prim_p[sel_q];
`ifdef VX_CFG_RTU_TLAS_ENABLE
                                    hit_inst_r[sel_q] <= instidx_q;  // TLAS instance index
`endif
                                    // a closer opaque hit occludes a farther candidate.
                                    if (yld_pending[sel_q] && (yld_t[sel_q] >= tri_t_p[sel_q]))
                                        yld_pending[sel_q] <= 1'b0;
                                end else begin
                                    // stage closest non-opaque candidate (AHS/IS yield).
                                    if (!yld_pending[sel_q] || (tri_t_p[sel_q] < yld_t[sel_q])) begin
                                        yld_pending[sel_q] <= 1'b1;
                                        yld_t[sel_q]       <= tri_t_p[sel_q];
                                        yld_u[sel_q]       <= tri_u_p[sel_q];
                                        yld_v[sel_q]       <= tri_v_p[sel_q];
                                        yld_prim[sel_q]    <= tri_prim_p[sel_q];
                                        yld_cbtype[sel_q]  <= cls_cbtype;
                                        yld_sbt[sel_q]     <= cls_sbt;
`ifdef VX_CFG_RTU_TLAS_ENABLE
                                        yld_inst[sel_q]    <= instidx_q;
`endif
                                    end
                                end
                            end
                            // opaque TERMINATE_ON_FIRST_HIT stops this lane's scan.
                            cstate[sel_q] <= (tri_committable && tri_opaque && term_first)
                                           ? CS_DONE : CS_NEXT;
                        end
                    end
                    CS_NEXT: begin
                        if ((triidx_q + 32'd1) == tricount_q) begin
                            // BLAS exhausted: next instance (TLAS) or retire.
`ifdef VX_CFG_RTU_TLAS_ENABLE
                            cstate[sel_q] <= CS_INST_NEXT;
`else
                            cstate[sel_q] <= CS_DONE;
`endif
                        end else begin
                            tri_idx[sel_q] <= triidx_q + 32'd1;
                            cur_off[sel_q] <= curoff_q + 32'(RTU_TRI_STRIDE);
                            cstate[sel_q]  <= CS_REQ0;
                        end
                    end
`ifdef VX_CFG_RTU_TLAS_ENABLE
                    // ── instance-record fetch (64 B, may straddle two lines) ──
                    CS_INST_REQ: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            f_total[sel_q]    <= inst_lines;
                            cstate[sel_q]     <= CS_INST_RSP0;
                        end
                    end
                    CS_INST_RSP0: begin
                        if (line_ready[sel_q]) begin
                            if (ftotal_q == LB'(1)) begin
                                cstate[sel_q] <= CS_INST_RSPN;
                            end else begin
                                f_idx[sel_q]  <= LB'(1);
                                cstate[sel_q] <= CS_INST_REQN;
                            end
                        end
                    end
                    CS_INST_REQN: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= fidx_q;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_INST_RSPN;
                        end
                    end
                    CS_INST_RSPN: begin
                        // multi-line record assembled; decode it. The byte-align
                        // shift is valid once the last needed line has landed.
                        if (line_ready[sel_q]) begin
                            if ((ftotal_q != LB'(1)) && ((fidx_q + LB'(1)) != ftotal_q)) begin
                                f_idx[sel_q]  <= fidx_q + LB'(1);
                                cstate[sel_q] <= CS_INST_REQN;
                            end else if (inst_culled) begin
                                // §8.8 cull gate: skip transform + BLAS scan.
                                cstate[sel_q] <= CS_INST_NEXT;
                            end else begin
                                inst_xform[sel_q] <= inst_xform_w;
                                blas_off[sel_q]   <= inst_blas;
                                xform_ready[sel_q]<= 1'b0;
                                cstate[sel_q]     <= CS_XFORM;
                            end
                        end
                    end
                    CS_XFORM: begin
                        // the world ray + xform were fed to VX_rtu_xform this
                        // EXEC cycle (xform_valid_in); await the object ray.
                        cstate[sel_q] <= CS_XFORM_WT;
                    end
                    CS_XFORM_WT: begin
                        if (xform_ready[sel_q]) begin
                            // object ray latched async; fetch the BLAS header.
                            cur_off[sel_q] <= blasoff_q;
                            cstate[sel_q]  <= CS_BLAS_REQ;
                        end
                    end
                    CS_BLAS_REQ: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_BLAS_RSP;
                        end
                    end
                    CS_BLAS_RSP: begin
                        if (line_ready[sel_q]) begin
                            // BLAS header word0 = triangle_count (object space).
                            if (hdr_count == 32'd0 || skip_tris) begin
                                cstate[sel_q] <= CS_INST_NEXT;
                            end else begin
                                tri_count[sel_q] <= hdr_count;
                                tri_idx[sel_q]   <= '0;
                                cur_off[sel_q]   <= blasoff_q + 32'(RTU_SCENE_HDR_BYTES);
                                cstate[sel_q]    <= CS_REQ0;
                            end
                        end
                    end
                    CS_INST_NEXT: begin
                        // Advance to the next instance. A staged non-opaque
                        // candidate stops the instance loop (single-yield, like
                        // SimX FlatWalker's `!yield_pending`).
                        if (yld_pending[sel_q] || ((instidx_q + 32'd1) == instcount_q)) begin
                            cstate[sel_q] <= CS_DONE;
                        end else begin
                            inst_idx[sel_q] <= instidx_q + 32'd1;
                            cur_off[sel_q]  <= 32'(RTU_SCENE_HDR_BYTES)
                                             + ((instidx_q + 32'd1) * 32'(RTU_INST_STRIDE));
                            cstate[sel_q]   <= CS_INST_REQ;
                        end
                    end
`endif
                    default:;
                    endcase
                end
            end

            // ── post-walk yield barrier ───────────────────────────────
            // Walk complete. If any lane staged a non-opaque candidate, hold
            // here and present a CB_YIELD (yield output) until the core
            // returns per-lane actions on `resume`; ACCEPT/TERMINATE commit
            // the candidate, IGNORE keeps the opaque hit. Once no candidate
            // remains pending the slot retires (done).
            if (running && all_done) begin
                if (!finalised) begin
                    // finalise_lane: a lane with no candidate yield still fires
                    // CHS (committed opaque hit + ENABLE_CHS, not SKIP_CLOSEST)
                    // or MISS (no hit + ENABLE_MISS). The dispatcher exits these
                    // with cb_ret(DONE) — no hit mutation on resume.
                    for (k = 0; k < NUM_CTX; k = k + 1) begin
                        if (mask_r[k] && !yld_pending[k]) begin
                            if (hit_r[k] && ((ray_r[k].flags & `VX_RT_FLAG_ENABLE_CHS) != 0)
                                         && ((ray_r[k].flags & `VX_RT_FLAG_SKIP_CLOSEST_HIT) == 0)) begin
                                yld_pending[k] <= 1'b1;
                                yld_cbtype[k]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_CHS);
                                yld_t[k]       <= hit_t_r[k];
                                yld_u[k]       <= hit_u_r[k];
                                yld_v[k]       <= hit_v_r[k];
                                yld_prim[k]    <= hit_prim_r[k];
                            end else if (!hit_r[k] && ((ray_r[k].flags & `VX_RT_FLAG_ENABLE_MISS) != 0)) begin
                                yld_pending[k] <= 1'b1;
                                yld_cbtype[k]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_MISS);
                            end
                        end
                    end
                    finalised <= 1'b1;
                end else if (|yld_pending) begin
                    if (resume) begin
                        for (k = 0; k < NUM_CTX; k = k + 1) begin
                            if (yld_pending[k]) begin
                                if ((action[k] == RTU_CB_ACTION_BITS'(`VX_RT_CB_ACCEPT))
                                 || (action[k] == RTU_CB_ACTION_BITS'(`VX_RT_CB_TERMINATE))) begin
                                    hit_r[k]      <= 1'b1;
                                    // A procedural (IS) accept commits the IS-computed
                                    // t; a geometric (AHS) accept keeps the candidate t.
                                    hit_t_r[k]    <= (yld_cbtype[k] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))
                                                   ? action_hit_t[k] : yld_t[k];
                                    hit_u_r[k]    <= yld_u[k];
                                    hit_v_r[k]    <= yld_v[k];
                                    hit_prim_r[k] <= yld_prim[k];
                                end
                                yld_pending[k] <= 1'b0;
                            end
                        end
                        // next cycle (|yld_pending == 0) retires the slot.
                    end
                end else begin
                    running <= 1'b0;
                    done_r  <= 1'b1;
                end
            end
        end
    end

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (tri_valid_out) begin
            `TRACE(2, ("%t: %s rtu-flat-tri: ctx=%0d, hit=%0d, t=0x%0h\n",
                $time, INSTANCE_ID, tri_tag_out, tri_hit, tri_t))
        end
        if (done_r) begin
            `TRACE(1, ("%t: %s rtu-flat-done\n", $time, INSTANCE_ID))
        end
    end
`endif

    // While at the yield barrier, present the candidate attrs (the CB_YIELD
    // payload) on res_* for the yielding lanes; otherwise the committed hit.
    assign yield        = running && all_done && finalised && (|yld_pending);
    assign yield_mask   = yld_pending;
    assign yield_cbtype = yld_cbtype;
    assign yield_sbt    = yld_sbt;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_res
        wire cand_i = yield && yld_pending[i];
        assign res_hit[i]  = hit_r[i];
        assign res_t[i]    = cand_i ? yld_t[i]    : hit_t_r[i];
        assign res_u[i]    = cand_i ? yld_u[i]    : hit_u_r[i];
        assign res_v[i]    = cand_i ? yld_v[i]    : hit_v_r[i];
        assign res_prim[i] = cand_i ? yld_prim[i] : hit_prim_r[i];
        assign res_geom[i] = 32'd0;   // flat scenes report geometry 0
        assign res_inst[i] = cand_i ? yld_inst[i] : hit_inst_r[i];
    end

    assign busy = running;
    assign done = done_r;

endmodule
