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

// VX_rtu_scheduler — single-context BVH traversal control. Mirrors the SimX
// walk_bvh4_subtree: set up the ray (inv_d = 1/dir, ro_invd = origin*inv_d),
// read the scene header for the root, then depth-first walk the short stack —
// fetch a node, decode it, and for an internal node stream its children
// through the box PE, pushing those whose AABB the ray enters within
// [t_min, best_t). Leaves are fetched and counted but not intersected here
// (Phase 1 reports a miss; ray-triangle commit lands with the tri PE). Child
// push order does not affect which nodes are visited while best_t is fixed,
// so no near/far sort is done yet.

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

    localparam [3:0] ST_IDLE    = 4'd0,
                     ST_SETUP   = 4'd1,
                     ST_HDR_REQ = 4'd2,
                     ST_HDR_RSP = 4'd3,
                     ST_POP     = 4'd4,
                     ST_FETCH   = 4'd5,
                     ST_DECODE  = 4'd6,
                     ST_DECODE2 = 4'd7,
                     ST_FEED    = 4'd8,
                     ST_WAIT    = 4'd9,
                     ST_PUSH    = 4'd10,
                     ST_DONE    = 4'd11;

    reg [3:0] state;

    rtu_ray_t          ray_r;
    reg [2:0][31:0]    inv_d_r, ro_invd_r;
    reg [31:0]         best_t;
    reg [31:0]         node_cnt;
    reg [SETUP_CW-1:0] setup_ctr;

    reg [RTU_STACK_DEPTH-1:0][31:0] stack;
    reg [RTU_STACK_BITS-1:0]        sp;
    reg [31:0]                      cur_off;

    reg [LINE_BITS-1:0]             node_line;
    reg [RTU_CHILD_BITS-1:0]        feed_idx, coll_idx, push_idx;
    reg [RTU_BVH_WIDTH-1:0]         child_hit;

    // child counters span 0..n_children; array indices need only clog2(WIDTH)
    localparam IDXW = `CLOG2(RTU_BVH_WIDTH);
    wire [IDXW-1:0] feed_ci = feed_idx[IDXW-1:0];
    wire [IDXW-1:0] coll_ci = coll_idx[IDXW-1:0];
    wire [IDXW-1:0] push_ci = push_idx[IDXW-1:0];

    wire [7:0]  node_kind;
    rtu_node_t  node;
    VX_rtu_node_decode decode (.line (node_line), .kind (node_kind), .node (node));

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

    // ── memory request (combinational; single outstanding) ────────────
    assign mem_req_valid = (state == ST_HDR_REQ) || (state == ST_FETCH);
    assign mem_req_addr  = (state == ST_HDR_REQ) ? ray_r.scene_base
                                                 : (ray_r.scene_base + cur_off);
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
                    ray_r     <= ray;
                    best_t    <= ray.t_max;
                    node_cnt  <= '0;
                    sp        <= '0;
                    setup_ctr <= '0;
                    state     <= ST_SETUP;
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
                    state   <= ST_FETCH;
                end
            end
            ST_POP: begin
                if (sp == '0) begin
                    state <= ST_DONE;
                end else begin
                    cur_off <= stack[sp - RTU_STACK_BITS'(1)];
                    sp      <= sp - RTU_STACK_BITS'(1);
                    state   <= ST_FETCH;
                end
            end
            ST_FETCH: begin
                if (mem_req_fire) begin
                    state <= ST_DECODE;
                end
            end
            ST_DECODE: begin
                if (mem_rsp_valid) begin
                    node_line <= mem_rsp_data;
                    node_cnt  <= node_cnt + 32'd1;
                    state     <= ST_DECODE2;
                end
            end
            ST_DECODE2: begin
                if (node_kind == RTU_KIND_INTERNAL && node.n_children != '0) begin
                    feed_idx  <= '0;
                    coll_idx  <= '0;
                    child_hit <= '0;
                    state     <= ST_FEED;
                end else begin
                    state <= ST_POP;   // leaf (counted, all-miss) or empty
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
            ST_DONE: begin
                state <= ST_IDLE;
            end
            default:;
            endcase
        end
    end

    assign busy          = (state != ST_IDLE);
    assign done          = (state == ST_DONE);
    assign result_hit    = 1'b0;            // Phase 1: traversal-only, always miss
    assign result_t      = ray_r.t_max;
    assign nodes_visited = node_cnt;

endmodule
