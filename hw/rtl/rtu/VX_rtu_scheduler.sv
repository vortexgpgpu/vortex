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

// VX_rtu_scheduler — context-pool BVH traversal control. Holds one ray context
// per lane (origin/dir/inv_d, short stack, best_t, hit record, traversal state)
// and time-multiplexes a single shared datapath across them: one box PE, one
// tri PE, one ray-setup reciprocal, one node decoder. A round-robin micro-
// engine runs one context at a time and, on the two long-latency operations
// (a cache line fetch and a ray-triangle test), parks the context and switches
// to another runnable one — hiding memory and tri-PE latency across rays. Line
// fetches and tri tests carry the context id as a tag so responses route back
// to their context; the box test is short and stays exclusive to the running
// context.
//
// Per context: set up the ray, read the scene header for the root, then depth-
// first walk the short stack. Each popped structure is fetched and byte-aligned
// (nodes/leaves are packed at arbitrary offsets and may straddle cache lines).
// An internal node streams its children through the box PE, pushing those whose
// AABB the ray enters within [t_min, best_t). A triangle leaf streams its
// vertices through the tri PE; a hit closer than best_t shrinks best_t and
// latches the closest-hit record.

`include "VX_define.vh"

module VX_rtu_scheduler import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_CTX   = 4,
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8,
    parameter CTX_TAG_W = `LOG2UP(NUM_CTX)    // derived: context-id tag width
) (
    input  wire        clk,
    input  wire        reset,

    // warp launch: one ray per active lane
    input  wire                       start,
    input  wire [NUM_CTX-1:0]         mask,
    input  rtu_ray_t [NUM_CTX-1:0]    rays,
    output wire                       busy,
    output wire                       done,

    // per-lane closest-hit results
    output wire [NUM_CTX-1:0]         res_hit,
    output wire [NUM_CTX-1:0][31:0]   res_t,
    output wire [NUM_CTX-1:0][31:0]   res_u,
    output wire [NUM_CTX-1:0][31:0]   res_v,
    output wire [NUM_CTX-1:0][31:0]   res_prim,
    output wire [NUM_CTX-1:0][31:0]   res_geom,

    // node/leaf fetch (to VX_rtu_mem, tagged by context id)
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
    localparam SETUP_LAT  = RTU_FDIV_LAT;
    localparam SETUP_CW   = `CLOG2(SETUP_LAT + 1);
    localparam BUF_BITS   = RTU_NODE_LINES * LINE_BITS;
    localparam IDXW       = `CLOG2(RTU_BVH_WIDTH);

    // per-context FSM states
    localparam [3:0] CS_DONE     = 4'd0,   // retired (also idle lanes)
                     CS_SETUP    = 4'd1,   // computing inv_d = 1/dir
                     CS_HDR_REQ  = 4'd2,   // issue scene-header fetch
                     CS_HDR_WAIT = 4'd3,   // park: header line
                     CS_REQ0     = 4'd4,   // issue structure line 0
                     CS_RSP0     = 4'd5,   // park: line 0
                     CS_REQN     = 4'd6,   // issue structure line N
                     CS_RSPN     = 4'd7,   // park: line N
                     CS_DISPATCH = 4'd8,   // internal vs leaf decode
                     CS_FEED     = 4'd9,   // stream children to box PE
                     CS_WAIT     = 4'd10,  // collect box results
                     CS_PUSH     = 4'd11,  // push hit children
                     CS_TRI_FEED = 4'd12,  // stream triangle to tri PE
                     CS_TRI_WAIT = 4'd13,  // park: tri result
                     CS_POP      = 4'd14;  // pop next node / terminate

    // ── per-context state ─────────────────────────────────────────────
    reg [NUM_CTX-1:0][3:0]                       cstate;
    rtu_ray_t [NUM_CTX-1:0]                       ray_r;
    reg [NUM_CTX-1:0][2:0][31:0]                  inv_d_r;
    reg [NUM_CTX-1:0][31:0]                       best_t;
    reg [NUM_CTX-1:0]                             hit_r;
    reg [NUM_CTX-1:0][31:0]                       hit_t_r, hit_u_r, hit_v_r, hit_prim_r, hit_geom_r;
    reg [NUM_CTX-1:0][RTU_STACK_DEPTH-1:0][31:0]  stack;
    reg [NUM_CTX-1:0][RTU_STACK_BITS-1:0]         sp;
    reg [NUM_CTX-1:0][31:0]                       cur_off;
    reg [NUM_CTX-1:0][BUF_BITS-1:0]               f_buf;
    reg [NUM_CTX-1:0][RTU_LINES_BITS-1:0]         f_idx, f_total, f_slot;
    reg [NUM_CTX-1:0][RTU_CHILD_BITS-1:0]         feed_idx, coll_idx, push_idx;
    reg [NUM_CTX-1:0][RTU_BVH_WIDTH-1:0]          child_hit;
    reg [NUM_CTX-1:0][31:0]                       leaf_geom_r, leaf_prim_r;
    reg [NUM_CTX-1:0][SETUP_CW-1:0]               setup_ctr;
    reg [NUM_CTX-1:0]                             line_ready, tri_ready;
    reg [NUM_CTX-1:0]                             tri_hit_p;
    reg [NUM_CTX-1:0][31:0]                       tri_t_p, tri_u_p, tri_v_p;

    reg                       running;
    reg                       done_r;
    reg [CTX_TAG_W-1:0]       cc;          // round-robin start pointer

    // ── runnable predicate per context ────────────────────────────────
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
                default:     r = 1'b1;
            endcase
        end
        assign runnable[i] = r;
    end

    // ── selected (acting) context: prefer cc, else round-robin ────────
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
    wire act = running && sel_valid;   // the selected context advances this cycle

    // ── combinational decode for the selected context ─────────────────
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] struct_addr = ray_r[sel].scene_base + cur_off[sel];
    wire [RTU_LINE_SEL_BITS-1:0]      f_off   = struct_addr[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0]      f_shift = {f_off, 3'b000};

    wire [BUF_BITS-1:0] f_aligned = f_buf[sel] >> f_shift;
    wire [RTU_NODE_IMG_BITS-1:0] node_img = f_aligned[RTU_NODE_IMG_BITS-1:0];

    wire [7:0]  node_kind;
    rtu_node_t  node;
    VX_rtu_node_decode #(.IMG_BITS (RTU_NODE_IMG_BITS)) decode (
        .line (node_img), .kind (node_kind), .node (node)
    );

    wire [2:0][31:0] leaf_v0, leaf_v1, leaf_v2;
    for (genvar a = 0; a < 3; ++a) begin : g_tri_v
        assign leaf_v0[a] = f_aligned[(RTU_TRI_OFF_V0 + 4*a)*8 +: 32];
        assign leaf_v1[a] = f_aligned[(RTU_TRI_OFF_V1 + 4*a)*8 +: 32];
        assign leaf_v2[a] = f_aligned[(RTU_TRI_OFF_V2 + 4*a)*8 +: 32];
    end
    wire [31:0] leaf_geom = f_aligned[RTU_LEAF_OFF_GEOM*8 +: 32];
    wire [31:0] leaf_prim = f_aligned[RTU_LEAF_OFF_PRIM*8 +: 32];

    wire [31:0] f_off32 = 32'(f_off);
    wire [RTU_LINES_BITS-1:0] node_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_NODE_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [RTU_LINES_BITS-1:0] leaf_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_LEAF_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);

    wire [IDXW-1:0] feed_ci = feed_idx[sel][IDXW-1:0];
    wire [IDXW-1:0] coll_ci = coll_idx[sel][IDXW-1:0];
    wire [IDXW-1:0] push_ci = push_idx[sel][IDXW-1:0];
    wire [RTU_CHILD_BITS-1:0] last_child = node.n_children - RTU_CHILD_BITS'(1);

    // ── ray setup datapath (driven by the selected context). inv_d = 1/dir;
    // the box PE subtracts the ray origin itself, so there is no origin*inv_d
    // precompute (which would lose precision on axis-aligned rays where inv_d
    // is infinite) ─────────────────────────────────────────────────────
    wire [2:0][31:0] inv_d_w;
    for (genvar a = 0; a < 3; ++a) begin : g_setup
        VX_fdivsqrt_unit #(.LATENCY (RTU_FDIV_LAT)) recip (
            .clk (clk), .reset (reset), .enable (1'b1), .mask (1'b1),
            .fmt ('0), .frm (INST_FRM_RNE),
            .dataa (32'h3F800000 /*1.0*/), .datab (ray_r[sel].dir[a]), .is_sqrt (1'b0),
            .result (inv_d_w[a]), `UNUSED_PIN (fflags)
        );
    end

    // ── box PE: one child per cycle while the selected context feeds ──
    wire        box_valid_in = act && (cstate[sel] == CS_FEED);
    wire        box_valid_out, box_hit;
    wire [31:0] box_t_near;
    `UNUSED_VAR (box_t_near)
    VX_rtu_box_pe box_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (box_valid_in),
        .origin (node.origin), .exp (node.exp),
        .qmin (node.qmin[feed_ci]), .qmax (node.qmax[feed_ci]),
        .ro (ray_r[sel].origin), .inv_d (inv_d_r[sel]),
        .t_min (ray_r[sel].t_min), .t_max (best_t[sel]),
        .valid_out (box_valid_out), .hit (box_hit), .t_near (box_t_near)
    );
    wire coll_pushable = box_hit && (node.child_off[coll_ci] != 32'd0);

    // ── tri PE: tagged by context id so results route back ────────────
    wire        tri_valid_in = act && (cstate[sel] == CS_TRI_FEED);
    wire        tri_valid_out, tri_hit, tri_back;
    wire [CTX_TAG_W-1:0] tri_tag_out;
    wire [31:0] tri_t, tri_u, tri_v;
    `UNUSED_VAR (tri_back)
    VX_rtu_tri_pe #(.TAG_WIDTH (CTX_TAG_W)) tri_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (tri_valid_in),
        .tag_in (sel),
        .origin (ray_r[sel].origin), .dir (ray_r[sel].dir),
        .v0 (leaf_v0), .v1 (leaf_v1), .v2 (leaf_v2),
        .t_min (ray_r[sel].t_min), .t_max (best_t[sel]),
        .valid_out (tri_valid_out), .tag_out (tri_tag_out), .hit (tri_hit),
        .t (tri_t), .u (tri_u), .v (tri_v), .back_facing (tri_back)
    );

    // ── memory request (single shared port, tagged by context) ────────
    wire fetch_issue = (cstate[sel] == CS_HDR_REQ)
                    || (cstate[sel] == CS_REQ0)
                    || (cstate[sel] == CS_REQN);
    assign mem_req_valid = act && fetch_issue;
    assign mem_req_tag   = sel;
    assign mem_req_addr  = (cstate[sel] == CS_HDR_REQ) ? ray_r[sel].scene_base
                         : (cstate[sel] == CS_REQ0)    ? struct_addr
                         : (struct_addr + (`VX_CFG_MEM_ADDR_WIDTH'(f_idx[sel]) << RTU_LINE_SEL_BITS));
    assign mem_rsp_ready = 1'b1;
    wire mem_req_fire = mem_req_valid && mem_req_ready;

    wire [NUM_CTX-1:0] ctx_done;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_ctx_done
        assign ctx_done[i] = (cstate[i] == CS_DONE);
    end
    wire all_done = &ctx_done;

    integer k;
    always @(posedge clk) begin
        if (reset) begin
            running <= 1'b0;
            done_r  <= 1'b0;
            cc      <= '0;
            for (k = 0; k < NUM_CTX; k = k + 1) begin
                cstate[k]     <= CS_DONE;
                line_ready[k] <= 1'b0;
                tri_ready[k]  <= 1'b0;
            end
        end else begin
            done_r <= 1'b0;

            // launch: seed one context per active lane
            if (!running && start) begin
                running <= 1'b1;
                cc      <= '0;
                for (k = 0; k < NUM_CTX; k = k + 1) begin
                    ray_r[k]      <= rays[k];
                    best_t[k]     <= rays[k].t_max;
                    sp[k]         <= '0;
                    cur_off[k]    <= '0;
                    setup_ctr[k]  <= '0;
                    line_ready[k] <= 1'b0;
                    tri_ready[k]  <= 1'b0;
                    hit_r[k]      <= 1'b0;
                    hit_t_r[k]    <= rays[k].t_max;
                    hit_u_r[k]    <= '0;
                    hit_v_r[k]    <= '0;
                    hit_prim_r[k] <= '0;
                    hit_geom_r[k] <= '0;
                    cstate[k]     <= mask[k] ? CS_SETUP : CS_DONE;
                end
            end

            // async line-fetch response → route to its context
            if (mem_rsp_valid) begin
                f_buf[mem_rsp_tag][f_slot[mem_rsp_tag] * LINE_BITS +: LINE_BITS] <= mem_rsp_data;
                line_ready[mem_rsp_tag] <= 1'b1;
            end

            // async tri-PE result → route to its context
            if (tri_valid_out) begin
                tri_ready[tri_tag_out] <= 1'b1;
                tri_hit_p[tri_tag_out] <= tri_hit;
                tri_t_p[tri_tag_out]   <= tri_t;
                tri_u_p[tri_tag_out]   <= tri_u;
                tri_v_p[tri_tag_out]   <= tri_v;
            end

            // advance the selected context
            if (act) begin
                cc <= sel;
                case (cstate[sel])
                CS_SETUP: begin
                    if (setup_ctr[sel] != SETUP_CW'(SETUP_LAT)) begin
                        setup_ctr[sel] <= setup_ctr[sel] + SETUP_CW'(1);
                    end else begin
                        inv_d_r[sel]   <= inv_d_w;
                        cstate[sel]    <= CS_HDR_REQ;
                    end
                end
                CS_HDR_REQ: begin
                    if (mem_req_fire) begin
                        f_slot[sel]     <= '0;
                        line_ready[sel] <= 1'b0;
                        cstate[sel]     <= CS_HDR_WAIT;
                    end
                end
                CS_HDR_WAIT: begin
                    if (line_ready[sel]) begin
                        cur_off[sel] <= f_aligned[RTU_SCENE_OFF_ROOT*8 +: 32];
                        cstate[sel]  <= CS_REQ0;
                    end
                end
                CS_REQ0: begin
                    if (mem_req_fire) begin
                        f_slot[sel]     <= '0;
                        line_ready[sel] <= 1'b0;
                        cstate[sel]     <= CS_RSP0;
                    end
                end
                CS_RSP0: begin
                    if (line_ready[sel]) begin
                        if (node_kind == RTU_KIND_INTERNAL) begin
                            f_total[sel] <= node_lines;
                            if (node_lines == RTU_LINES_BITS'(1)) begin
                                cstate[sel] <= CS_DISPATCH;
                            end else begin
                                f_idx[sel]  <= RTU_LINES_BITS'(1);
                                cstate[sel] <= CS_REQN;
                            end
                        end else if (node_kind == RTU_KIND_LEAF_TRI) begin
                            f_total[sel] <= leaf_lines;
                            if (leaf_lines == RTU_LINES_BITS'(1)) begin
                                cstate[sel] <= CS_DISPATCH;
                            end else begin
                                f_idx[sel]  <= RTU_LINES_BITS'(1);
                                cstate[sel] <= CS_REQN;
                            end
                        end else begin
                            cstate[sel] <= CS_POP;
                        end
                    end
                end
                CS_REQN: begin
                    if (mem_req_fire) begin
                        f_slot[sel]     <= f_idx[sel];
                        line_ready[sel] <= 1'b0;
                        cstate[sel]     <= CS_RSPN;
                    end
                end
                CS_RSPN: begin
                    if (line_ready[sel]) begin
                        if ((f_idx[sel] + RTU_LINES_BITS'(1)) == f_total[sel]) begin
                            cstate[sel] <= CS_DISPATCH;
                        end else begin
                            f_idx[sel]  <= f_idx[sel] + RTU_LINES_BITS'(1);
                            cstate[sel] <= CS_REQN;
                        end
                    end
                end
                CS_DISPATCH: begin
                    if (node_kind == RTU_KIND_INTERNAL && node.n_children != '0) begin
                        feed_idx[sel]  <= '0;
                        coll_idx[sel]  <= '0;
                        child_hit[sel] <= '0;
                        cstate[sel]    <= CS_FEED;
                    end else if (node_kind == RTU_KIND_LEAF_TRI) begin
                        leaf_geom_r[sel] <= leaf_geom;
                        leaf_prim_r[sel] <= leaf_prim;
                        cstate[sel]      <= CS_TRI_FEED;
                    end else begin
                        cstate[sel] <= CS_POP;
                    end
                end
                CS_FEED: begin
                    if (feed_idx[sel] == last_child) begin
                        cstate[sel] <= CS_WAIT;
                    end
                    feed_idx[sel] <= feed_idx[sel] + RTU_CHILD_BITS'(1);
                end
                CS_WAIT: begin
                    if (box_valid_out) begin
                        child_hit[sel][coll_ci] <= coll_pushable;
                        if (coll_idx[sel] == last_child) begin
                            push_idx[sel] <= '0;
                            cstate[sel]   <= CS_PUSH;
                        end
                        coll_idx[sel] <= coll_idx[sel] + RTU_CHILD_BITS'(1);
                    end
                end
                CS_PUSH: begin
                    if (child_hit[sel][push_ci] && (sp[sel] != RTU_STACK_BITS'(RTU_STACK_DEPTH))) begin
                        stack[sel][sp[sel]] <= node.child_off[push_ci] & RTU_CHILD_OFF_MASK;
                        sp[sel]             <= sp[sel] + RTU_STACK_BITS'(1);
                    end
                    if (push_idx[sel] == last_child) begin
                        cstate[sel] <= CS_POP;
                    end
                    push_idx[sel] <= push_idx[sel] + RTU_CHILD_BITS'(1);
                end
                CS_TRI_FEED: begin
                    cstate[sel] <= CS_TRI_WAIT;
                end
                CS_TRI_WAIT: begin
                    if (tri_ready[sel]) begin
                        tri_ready[sel] <= 1'b0;
                        if (tri_hit_p[sel]) begin
                            best_t[sel]     <= tri_t_p[sel];
                            hit_r[sel]      <= 1'b1;
                            hit_t_r[sel]    <= tri_t_p[sel];
                            hit_u_r[sel]    <= tri_u_p[sel];
                            hit_v_r[sel]    <= tri_v_p[sel];
                            hit_prim_r[sel] <= leaf_prim_r[sel];
                            hit_geom_r[sel] <= leaf_geom_r[sel];
                        end
                        cstate[sel] <= CS_POP;
                    end
                end
                CS_POP: begin
                    if (sp[sel] == '0) begin
                        cstate[sel] <= CS_DONE;
                    end else begin
                        cur_off[sel] <= stack[sel][sp[sel] - RTU_STACK_BITS'(1)];
                        sp[sel]      <= sp[sel] - RTU_STACK_BITS'(1);
                        cstate[sel]  <= CS_REQ0;
                    end
                end
                default:;
                endcase
            end

            // all contexts retired → pulse done and idle
            if (running && all_done) begin
                running <= 1'b0;
                done_r  <= 1'b1;
            end
        end
    end

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (act && (cstate[sel] == CS_DISPATCH)) begin
            `TRACE(2, ("%t: %s rtu-node: ctx=%0d, off=%0d, kind=%0d, children=%0d\n",
                $time, INSTANCE_ID, sel, cur_off[sel], node_kind, node.n_children))
        end
        if (tri_valid_out) begin
            `TRACE(2, ("%t: %s rtu-tri: ctx=%0d, hit=%0d, t=0x%0h\n",
                $time, INSTANCE_ID, tri_tag_out, tri_hit, tri_t))
        end
        if (done_r) begin
            `TRACE(1, ("%t: %s rtu-done\n", $time, INSTANCE_ID))
        end
    end
`endif

    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_res
        assign res_hit[i]  = hit_r[i];
        assign res_t[i]    = hit_t_r[i];
        assign res_u[i]    = hit_u_r[i];
        assign res_v[i]    = hit_v_r[i];
        assign res_prim[i] = hit_prim_r[i];
        assign res_geom[i] = hit_geom_r[i];
    end

    assign busy = running;
    assign done = done_r;

endmodule
