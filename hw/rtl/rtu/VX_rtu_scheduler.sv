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

// VX_rtu_scheduler — single-context BVH traversal control. Sets up the ray
// (inv_d = 1/dir, ro_invd = origin*inv_d), reads the scene header for the root,
// then depth-first walks the short stack. Each popped structure is fetched and
// byte-aligned: nodes and leaves are packed at arbitrary scene offsets and can
// straddle several aligned cache lines, so the engine reads the spanning lines,
// assembles them, and shifts the structure's word0 to bit 0 before decoding.
// An internal node streams its children through the box PE, pushing those whose
// AABB the ray enters within [t_min, best_t). A triangle leaf streams its
// vertices through the tri PE; a hit closer than best_t shrinks best_t (which
// prunes later box tests) and latches the closest-hit record.

`include "VX_define.vh"

module VX_rtu_scheduler import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8
) (
    input  wire        clk,
    input  wire        reset,

    // trace launch
    input  wire        start,
    input  rtu_ray_t   ray,
    output wire        busy,

    // terminal result
    output wire        done,
    output wire        result_hit,
    output wire [31:0] result_t,
    output wire [31:0] result_u,
    output wire [31:0] result_v,
    output wire [31:0] result_prim,
    output wire [31:0] result_geom,
    output wire [31:0] nodes_visited,

    // node/leaf fetch (to VX_rtu_mem, single outstanding)
    output wire                              mem_req_valid,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] mem_req_addr,
    input  wire                              mem_req_ready,
    input  wire                              mem_rsp_valid,
    input  wire [LINE_BITS-1:0]              mem_rsp_data,
    output wire                              mem_rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam SETUP_LAT = `VX_CFG_LATENCY_FDIV + `VX_CFG_LATENCY_FMA;
    localparam SETUP_CW  = `CLOG2(SETUP_LAT + 1);
    localparam BUF_BITS  = RTU_NODE_LINES * LINE_BITS;

    localparam [3:0] ST_IDLE    = 4'd0,
                     ST_SETUP   = 4'd1,
                     ST_HDR_REQ = 4'd2,
                     ST_HDR_RSP = 4'd3,
                     ST_POP     = 4'd4,
                     ST_REQ0    = 4'd5,
                     ST_RSP0    = 4'd6,
                     ST_REQN    = 4'd7,
                     ST_RSPN    = 4'd8,
                     ST_DISPATCH= 4'd9,
                     ST_FEED    = 4'd10,
                     ST_WAIT    = 4'd11,
                     ST_PUSH    = 4'd12,
                     ST_TRI_FEED= 4'd13,
                     ST_TRI_WAIT= 4'd14,
                     ST_DONE    = 4'd15;

    reg [3:0] state;

    rtu_ray_t          ray_r;
    reg [2:0][31:0]    inv_d_r, ro_invd_r;
    reg [31:0]         best_t;
    reg [31:0]         node_cnt;
    reg [SETUP_CW-1:0] setup_ctr;

    reg [RTU_STACK_DEPTH-1:0][31:0] stack;
    reg [RTU_STACK_BITS-1:0]        sp;
    reg [31:0]                      cur_off;

    // multi-line structure buffer + byte-aligned image
    reg [BUF_BITS-1:0]              f_buf;
    reg [RTU_LINES_BITS-1:0]        f_idx, f_total;

    reg [RTU_CHILD_BITS-1:0]        feed_idx, coll_idx, push_idx;
    reg [RTU_BVH_WIDTH-1:0]         child_hit;

    // closest-hit record
    reg                hit_r;
    reg [31:0]         hit_t_r, hit_u_r, hit_v_r, hit_prim_r, hit_geom_r;
    reg [31:0]         leaf_geom_r, leaf_prim_r;

    // child counters span 0..n_children; array indices need only clog2(WIDTH)
    localparam IDXW = `CLOG2(RTU_BVH_WIDTH);
    wire [IDXW-1:0] feed_ci = feed_idx[IDXW-1:0];
    wire [IDXW-1:0] coll_ci = coll_idx[IDXW-1:0];
    wire [IDXW-1:0] push_ci = push_idx[IDXW-1:0];

    // byte offset of the current structure within its first aligned line
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] struct_addr = ray_r.scene_base + cur_off;
    wire [RTU_LINE_SEL_BITS-1:0]      f_off   = struct_addr[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0]      f_shift = {f_off, 3'b000};   // f_off * 8

    // assembled structure aligned so word0 sits at bit 0
    wire [BUF_BITS-1:0] f_aligned = f_buf >> f_shift;
    wire [RTU_NODE_IMG_BITS-1:0] node_img = f_aligned[RTU_NODE_IMG_BITS-1:0];

    wire [7:0]  node_kind;
    rtu_node_t  node;
    VX_rtu_node_decode #(.IMG_BITS (RTU_NODE_IMG_BITS)) decode (
        .line (node_img), .kind (node_kind), .node (node)
    );

    // triangle vertices from the aligned leaf image
    wire [2:0][31:0] leaf_v0, leaf_v1, leaf_v2;
    for (genvar a = 0; a < 3; ++a) begin : g_tri_v
        assign leaf_v0[a] = f_aligned[(RTU_TRI_OFF_V0 + 4*a)*8 +: 32];
        assign leaf_v1[a] = f_aligned[(RTU_TRI_OFF_V1 + 4*a)*8 +: 32];
        assign leaf_v2[a] = f_aligned[(RTU_TRI_OFF_V2 + 4*a)*8 +: 32];
    end
    wire [31:0] leaf_geom = f_aligned[RTU_LEAF_OFF_GEOM*8 +: 32];
    wire [31:0] leaf_prim = f_aligned[RTU_LEAF_OFF_PRIM*8 +: 32];

    // kind of the just-fetched structure (from line0 at its byte offset)
    wire [7:0] line0_kind = mem_rsp_data[f_shift +: 8];

    // lines a node / leaf span at the current byte offset
    wire [31:0] f_off32 = 32'(f_off);
    wire [RTU_LINES_BITS-1:0] node_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_NODE_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [RTU_LINES_BITS-1:0] leaf_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_LEAF_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);

    // ── ray setup datapath: inv_d = 1/dir, ro_invd = origin*inv_d ─────
    wire [2:0][31:0] inv_d_w, ro_invd_w;
    for (genvar a = 0; a < 3; ++a) begin : g_setup
        VX_fdivsqrt_unit #(.LATENCY (`VX_CFG_LATENCY_FDIV)) recip (
            .clk (clk), .reset (reset), .enable (1'b1), .mask (1'b1),
            .fmt ('0), .frm (INST_FRM_RNE),
            .dataa (32'h3F800000 /*1.0*/), .datab (ray_r.dir[a]), .is_sqrt (1'b0),
            .result (inv_d_w[a]), `UNUSED_PIN (fflags)
        );
        VX_fma_unit #(.LATENCY (`VX_CFG_LATENCY_FMA)) mul (
            .clk (clk), .reset (reset), .enable (1'b1), .mask (1'b1),
            .op_type (INST_FPU_MADD), .fmt (2'b00), .frm (INST_FRM_RNE),
            .dataa (ray_r.origin[a]), .datab (inv_d_w[a]), .datac (32'h0),
            .result (ro_invd_w[a]), `UNUSED_PIN (fflags)
        );
    end

    // ── box PE: one child per cycle while in ST_FEED ──────────────────
    wire             box_valid_in = (state == ST_FEED);
    wire             box_valid_out, box_hit;
    wire [31:0]      box_t_near;
    `UNUSED_VAR (box_t_near)

    VX_rtu_box_pe box_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (box_valid_in),
        .origin (node.origin), .exp (node.exp),
        .qmin (node.qmin[feed_ci]), .qmax (node.qmax[feed_ci]),
        .inv_d (inv_d_r), .ro_invd (ro_invd_r), .t_min (ray_r.t_min), .t_max (best_t),
        .valid_out (box_valid_out), .hit (box_hit), .t_near (box_t_near)
    );

    wire coll_pushable = box_hit && (node.child_off[coll_ci] != 32'd0);
    wire [RTU_CHILD_BITS-1:0] last_child = node.n_children - RTU_CHILD_BITS'(1);

    // ── tri PE: stream the leaf triangle once in ST_TRI_FEED ──────────
    wire        tri_valid_in = (state == ST_TRI_FEED);
    wire        tri_valid_out, tri_hit, tri_back;
    wire [31:0] tri_t, tri_u, tri_v;
    `UNUSED_VAR (tri_back)

    VX_rtu_tri_pe tri_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (tri_valid_in),
        .origin (ray_r.origin), .dir (ray_r.dir),
        .v0 (leaf_v0), .v1 (leaf_v1), .v2 (leaf_v2),
        .t_min (ray_r.t_min), .t_max (best_t),
        .valid_out (tri_valid_out), .hit (tri_hit),
        .t (tri_t), .u (tri_u), .v (tri_v), .back_facing (tri_back)
    );

    // ── memory request (combinational; single outstanding) ────────────
    wire fetching = (state == ST_HDR_REQ) || (state == ST_REQ0) || (state == ST_REQN);
    assign mem_req_valid = fetching;
    // line0 of a structure starts at its base; subsequent lines step by one
    // cache line. ST_REQN only ever runs with f_idx >= 1.
    assign mem_req_addr  = (state == ST_HDR_REQ) ? ray_r.scene_base
                         : (state == ST_REQ0)    ? (ray_r.scene_base + cur_off)
                         : (ray_r.scene_base + cur_off
                            + (`VX_CFG_MEM_ADDR_WIDTH'(f_idx) << RTU_LINE_SEL_BITS));
    assign mem_rsp_ready = 1'b1;
    wire mem_req_fire = mem_req_valid && mem_req_ready;

    always @(posedge clk) begin
        if (reset) begin
            state    <= ST_IDLE;
            sp       <= '0;
            node_cnt <= '0;
        end else begin
            case (state)
            ST_IDLE: begin
                if (start) begin
                    ray_r      <= ray;
                    best_t     <= ray.t_max;
                    node_cnt   <= '0;
                    sp         <= '0;
                    setup_ctr  <= '0;
                    hit_r      <= 1'b0;
                    hit_t_r    <= ray.t_max;
                    hit_u_r    <= '0;
                    hit_v_r    <= '0;
                    hit_prim_r <= '0;
                    hit_geom_r <= '0;
                    state      <= ST_SETUP;
                end
            end
            ST_SETUP: begin
                if (setup_ctr != SETUP_CW'(SETUP_LAT)) begin
                    setup_ctr <= setup_ctr + SETUP_CW'(1);
                end else begin
                    inv_d_r   <= inv_d_w;
                    ro_invd_r <= ro_invd_w;
                    state     <= ST_HDR_REQ;
                end
            end
            ST_HDR_REQ: begin
                if (mem_req_fire) begin
                    state <= ST_HDR_RSP;
                end
            end
            ST_HDR_RSP: begin
                if (mem_rsp_valid) begin
                    cur_off <= mem_rsp_data[RTU_SCENE_OFF_ROOT*8 +: 32];
                    state   <= ST_REQ0;
                end
            end
            ST_POP: begin
                if (sp == '0) begin
                    state <= ST_DONE;
                end else begin
                    cur_off <= stack[sp - RTU_STACK_BITS'(1)];
                    sp      <= sp - RTU_STACK_BITS'(1);
                    state   <= ST_REQ0;
                end
            end
            ST_REQ0: begin
                if (mem_req_fire) begin
                    state <= ST_RSP0;
                end
            end
            ST_RSP0: begin
                if (mem_rsp_valid) begin
                    f_buf[0 +: LINE_BITS] <= mem_rsp_data;
                    node_cnt <= node_cnt + 32'd1;
                    if (line0_kind == RTU_KIND_INTERNAL) begin
                        f_total <= node_lines;
                        if (node_lines == RTU_LINES_BITS'(1)) begin
                            state <= ST_DISPATCH;
                        end else begin
                            f_idx <= RTU_LINES_BITS'(1);
                            state <= ST_REQN;
                        end
                    end else if (line0_kind == RTU_KIND_LEAF_TRI) begin
                        f_total <= leaf_lines;
                        if (leaf_lines == RTU_LINES_BITS'(1)) begin
                            state <= ST_DISPATCH;
                        end else begin
                            f_idx <= RTU_LINES_BITS'(1);
                            state <= ST_REQN;
                        end
                    end else begin
                        state <= ST_POP;   // unsupported leaf kind: skip
                    end
                end
            end
            ST_REQN: begin
                if (mem_req_fire) begin
                    state <= ST_RSPN;
                end
            end
            ST_RSPN: begin
                if (mem_rsp_valid) begin
                    f_buf[f_idx * LINE_BITS +: LINE_BITS] <= mem_rsp_data;
                    if ((f_idx + RTU_LINES_BITS'(1)) == f_total) begin
                        state <= ST_DISPATCH;
                    end else begin
                        f_idx <= f_idx + RTU_LINES_BITS'(1);
                        state <= ST_REQN;
                    end
                end
            end
            ST_DISPATCH: begin
                if (node_kind == RTU_KIND_INTERNAL && node.n_children != '0) begin
                    feed_idx  <= '0;
                    coll_idx  <= '0;
                    child_hit <= '0;
                    state     <= ST_FEED;
                end else if (node_kind == RTU_KIND_LEAF_TRI) begin
                    leaf_geom_r <= leaf_geom;
                    leaf_prim_r <= leaf_prim;
                    state       <= ST_TRI_FEED;
                end else begin
                    state <= ST_POP;   // empty internal node or unsupported kind
                end
            end
            ST_FEED: begin
                if (feed_idx == last_child) begin
                    state <= ST_WAIT;
                end
                feed_idx <= feed_idx + RTU_CHILD_BITS'(1);
            end
            ST_WAIT: begin
                if (box_valid_out) begin
                    child_hit[coll_ci] <= coll_pushable;
                    if (coll_idx == last_child) begin
                        push_idx <= '0;
                        state    <= ST_PUSH;
                    end
                    coll_idx <= coll_idx + RTU_CHILD_BITS'(1);
                end
            end
            ST_PUSH: begin
                if (child_hit[push_ci] && (sp != RTU_STACK_BITS'(RTU_STACK_DEPTH))) begin
                    stack[sp] <= node.child_off[push_ci] & RTU_CHILD_OFF_MASK;
                    sp        <= sp + RTU_STACK_BITS'(1);
                end
                if (push_idx == last_child) begin
                    state <= ST_POP;
                end
                push_idx <= push_idx + RTU_CHILD_BITS'(1);
            end
            ST_TRI_FEED: begin
                state <= ST_TRI_WAIT;
            end
            ST_TRI_WAIT: begin
                if (tri_valid_out) begin
                    if (tri_hit) begin
                        best_t     <= tri_t;
                        hit_r      <= 1'b1;
                        hit_t_r    <= tri_t;
                        hit_u_r    <= tri_u;
                        hit_v_r    <= tri_v;
                        hit_prim_r <= leaf_prim_r;
                        hit_geom_r <= leaf_geom_r;
                    end
                    state <= ST_POP;
                end
            end
            ST_DONE: begin
                state <= ST_IDLE;
            end
            default:;
            endcase
        end
    end

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (state == ST_DISPATCH) begin
            `TRACE(2, ("%t: %s rtu-node: off=%0d, kind=%0d, children=%0d\n",
                $time, INSTANCE_ID, cur_off, node_kind, node.n_children))
        end
        if ((state == ST_TRI_WAIT) && tri_valid_out) begin
            `TRACE(2, ("%t: %s rtu-tri: hit=%0d, t=0x%0h, u=0x%0h, v=0x%0h\n",
                $time, INSTANCE_ID, tri_hit, tri_t, tri_u, tri_v))
        end
        if (state == ST_DONE) begin
            `TRACE(1, ("%t: %s rtu-done: hit=%0d, t=0x%0h, geom=0x%0h, nodes=%0d\n",
                $time, INSTANCE_ID, hit_r, hit_t_r, hit_geom_r, node_cnt))
        end
    end
`endif

    assign busy          = (state != ST_IDLE);
    assign done          = (state == ST_DONE);
    assign result_hit    = hit_r;
    assign result_t      = hit_t_r;
    assign result_u      = hit_u_r;
    assign result_v      = hit_v_r;
    assign result_prim   = hit_prim_r;
    assign result_geom   = hit_geom_r;
    assign nodes_visited = node_cnt;

endmodule
