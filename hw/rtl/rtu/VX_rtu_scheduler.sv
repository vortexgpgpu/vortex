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
// tri PE, one ray-setup reciprocal, one node decoder.
//
// Each traversal micro-step runs as two pipeline phases so the per-context
// selection and the wide datapath fan-out sit in different clock cycles:
//   SELECT : pick a runnable context and snapshot its working set (ray, inv_d,
//            fetched line, stack/counters, best_t) into the stage registers.
//   EXEC   : decode the snapshot, drive the box/tri PEs and the memory port,
//            and advance the context FSM, writing results back to the context.
// The selection mux therefore feeds registers rather than the decoder and PE
// inputs directly, keeping each cycle's logic short.
//
// On the two long-latency operations (a cache line fetch and a ray-triangle
// test) the context parks and another runnable one is picked, hiding memory and
// tri-PE latency across rays. Line fetches and tri tests carry the context id as
// a tag so responses route back to their context; box results stream back to the
// running context, which stays selected for the span of one node's children.
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
    output wire [NUM_CTX-1:0][31:0]   res_inst,

    // Phase 2 callback yield barrier (see VX_rtu_flat_scheduler). The BVH
    // walker is opaque-only for now, so it never yields — these are tied off
    // and resume/action are unused until BVH AHS lands.
    output wire                                       yield,
    output wire [NUM_CTX-1:0]                         yield_mask,
    output wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]   yield_cbtype,
    output wire [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]    yield_sbt,
    input  wire                                       resume,
    input  wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] action,
    input  wire [NUM_CTX-1:0][31:0]                   action_hit_t,

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
    localparam [4:0] CS_DONE      = 5'd0,   // retired (also idle lanes)
                     CS_SETUP     = 5'd1,   // computing inv_d = 1/dir
                     CS_HDR_REQ   = 5'd2,   // issue scene-header fetch
                     CS_HDR_WAIT  = 5'd3,   // park: header line
                     CS_REQ0      = 5'd4,   // issue structure line 0
                     CS_RSP0      = 5'd5,   // park: line 0
                     CS_REQN      = 5'd6,   // issue structure line N
                     CS_RSPN      = 5'd7,   // park: line N
                     CS_DISPATCH  = 5'd8,   // internal vs leaf decode
                     CS_FEED      = 5'd9,   // stream children to box PE
                     CS_WAIT      = 5'd10,  // collect box results
                     CS_PUSH      = 5'd11,  // push hit children
                     CS_TRI_FEED  = 5'd12,  // stream triangle to tri PE
                     CS_TRI_WAIT  = 5'd13,  // park: tri result
                     CS_POP       = 5'd14,  // pop next node / terminate
                     CS_PROC_FEED = 5'd15,  // feed procedural-leaf AABB (raw box)
                     CS_PROC_WAIT = 5'd16,  // park: proc box result -> IS yield
                     // Instancing states: a LEAF_INST node iterates instances,
                     // each descending into its inline BLAS subtree under the
                     // object-space ray on the same short-stack.
                     CS_INST_REQ  = 5'd17,  // issue instance-record line 0
                     CS_INST_RSP0 = 5'd18,  // park: instance line 0
                     CS_INST_REQN = 5'd19,  // issue instance-record line N
                     CS_INST_RSPN = 5'd20,  // park: instance line N -> cull / xform
                     CS_XFORM     = 5'd21,  // feed world ray + xform to VX_rtu_xform
                     CS_XFORM_WT  = 5'd22,  // park: object ray
                     CS_OBJ_SETUP = 5'd23,  // object inv_d = 1/obj_dir (recip)
                     CS_INST_NEXT = 5'd24;  // advance to next instance / resume TLAS

    // ── per-context state ─────────────────────────────────────────────
    reg [NUM_CTX-1:0][4:0]                       cstate;
    rtu_ray_t [NUM_CTX-1:0]                       ray_r;
    reg [NUM_CTX-1:0][2:0][31:0]                  inv_d_r;
    reg [NUM_CTX-1:0][31:0]                       best_t;
    reg [NUM_CTX-1:0]                             hit_r;
    reg [NUM_CTX-1:0][31:0]                       hit_t_r, hit_u_r, hit_v_r, hit_prim_r, hit_geom_r;
    reg [NUM_CTX-1:0][31:0]                       hit_inst_r;   // committed hit's instance id (TLAS)
    reg [NUM_CTX-1:0][RTU_STACK_DEPTH-1:0][31:0]  stack;
    reg [NUM_CTX-1:0][RTU_STACK_BITS-1:0]         sp;
    reg [NUM_CTX-1:0][31:0]                       cur_off;
    reg [NUM_CTX-1:0][BUF_BITS-1:0]               f_buf;
    reg [NUM_CTX-1:0][RTU_LINES_BITS-1:0]         f_idx, f_total, f_slot;
    reg [NUM_CTX-1:0][RTU_CHILD_BITS-1:0]         feed_idx, coll_idx, push_idx;
    reg [NUM_CTX-1:0][RTU_BVH_WIDTH-1:0]          child_hit;
    reg [NUM_CTX-1:0]                             box_done;
    reg [NUM_CTX-1:0][31:0]                       leaf_geom_r, leaf_prim_r;
    reg [NUM_CTX-1:0][SETUP_CW-1:0]               setup_ctr;
    reg [NUM_CTX-1:0]                             line_ready, tri_ready;
    reg [NUM_CTX-1:0]                             tri_hit_p;
    reg [NUM_CTX-1:0][31:0]                       tri_t_p, tri_u_p, tri_v_p;
    // procedural-leaf box result (routed off the child-hit collection)
    reg [NUM_CTX-1:0]                             proc_ready, proc_hit_p;
    reg [NUM_CTX-1:0][31:0]                       proc_t_p;
    reg [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]        proc_sbt_p;
    // Phase 2 per-context IS/AHS yield candidate + finalise bookkeeping.
    reg [NUM_CTX-1:0]                             yld_pending;
    reg [NUM_CTX-1:0][31:0]                       yld_t, yld_u, yld_v, yld_prim;
    reg [NUM_CTX-1:0][31:0]                       yld_inst;   // candidate hit's instance id (TLAS)
    reg [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]       yld_cbtype;
    reg [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]        yld_sbt;
    reg [NUM_CTX-1:0]                             mask_r;
    reg                                          finalised;

    // ── per-context TLAS state: the LEAF_INST instance loop + BLAS descent ──
    reg [NUM_CTX-1:0][31:0]                       inst_count, inst_idx;
    reg [NUM_CTX-1:0][31:0]                       inst_base, blas_root, inst_id_r;
    reg [NUM_CTX-1:0][11:0][31:0]                 inst_xform;     // latched 3x4 affine
    reg [NUM_CTX-1:0][2:0][31:0]                  obj_o, obj_d;   // object-space ray
    reg [NUM_CTX-1:0][2:0][31:0]                  obj_inv_d_r;    // 1/obj_dir
    reg [NUM_CTX-1:0][RTU_STACK_BITS-1:0]         blas_floor;     // sp at instance loop
    reg [NUM_CTX-1:0]                             in_blas;        // object ray active
    reg [NUM_CTX-1:0]                             xform_ready;

    reg                       running;
    reg                       done_r;
    reg [CTX_TAG_W-1:0]       cc;          // round-robin start pointer

    // ── micro-step pipeline: SELECT (phase 0) latches a snapshot, EXEC
    //    (phase 1) runs the datapath + FSM from it ──────────────────────
    reg                       phase;       // 0 = SELECT, 1 = EXEC
    localparam PH_SELECT = 1'b0, PH_EXEC = 1'b1;

    reg [CTX_TAG_W-1:0]       sel_q;       // context being executed
    rtu_ray_t                 ray_q;
    reg [2:0][31:0]           invd_q;
    reg [BUF_BITS-1:0]        fbuf_q;
    // Precomputed absolute structure address (scene_base + cur_off) latched in
    // SELECT, so the EXEC critical cone (byte-align shift -> node decode ->
    // state) starts after the add instead of through it. Same register/adder
    // count as latching the raw offset — a pure phase move, no latency/area cost.
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] structaddr_q;
    reg [31:0]                bestt_q;
    reg [4:0]                 cstate_q;
    reg [SETUP_CW-1:0]        setupctr_q;
    reg [RTU_LINES_BITS-1:0]  fidx_q, ftotal_q;
    reg [RTU_CHILD_BITS-1:0]  feed_q, push_q;
    reg [RTU_STACK_BITS-1:0]  sp_q;
    reg [31:0]                stacktop_q;
    reg [RTU_BVH_WIDTH-1:0]   childhit_q;
    reg [31:0]                instidx_q, instcount_q, instbase_q, blasroot_q, instid_q;
    reg [11:0][31:0]          xform_q;
    reg [2:0][31:0]           objo_q, objd_q, objinvd_q;
    reg [RTU_STACK_BITS-1:0]  blasfloor_q;
    reg                       inblas_q;

    // ── runnable predicate per context ────────────────────────────────
    wire [NUM_CTX-1:0] runnable;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_runnable
        reg r;
        always @(*) begin
            case (cstate[i])
                CS_DONE:     r = 1'b0;
                CS_HDR_WAIT,
                CS_RSP0,
                CS_RSPN:      r = line_ready[i];
                CS_TRI_WAIT:  r = tri_ready[i];
                CS_INST_RSP0,
                CS_INST_RSPN: r = line_ready[i];
                CS_XFORM_WT:  r = xform_ready[i];
                // CS_PROC_WAIT busy-waits (default=1) like CS_WAIT, so the
                // context stays selected and its box result routes back.
                default:      r = 1'b1;
            endcase
        end
        assign runnable[i] = r;
    end

    // ── selected context for the next EXEC: prefer cc, else round-robin ─
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
    wire exec = (phase == PH_EXEC);   // the snapshot context advances this cycle

    // ── combinational decode of the EXEC snapshot ─────────────────────
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] struct_addr = structaddr_q;
    wire [RTU_LINE_SEL_BITS-1:0]      f_off   = struct_addr[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0]      f_shift = {f_off, 3'b000};

    wire [BUF_BITS-1:0] f_aligned = fbuf_q >> f_shift;
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
    wire [31:0] leaf_geom  = f_aligned[RTU_LEAF_OFF_GEOM*8 +: 32];
    wire [31:0] leaf_prim  = f_aligned[RTU_LEAF_OFF_PRIM*8 +: 32];
    wire [31:0] leaf_flags = f_aligned[RTU_LEAF_OFF_FLAGS*8 +: 32];

    wire [31:0] f_off32 = 32'(f_off);
    wire [RTU_LINES_BITS-1:0] node_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_NODE_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [RTU_LINES_BITS-1:0] leaf_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_LEAF_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);

    wire [IDXW-1:0] feed_ci = feed_q[IDXW-1:0];
    wire [IDXW-1:0] coll_ci = coll_idx[sel_q][IDXW-1:0];   // live: box results stream async
    wire [IDXW-1:0] push_ci = push_q[IDXW-1:0];
    wire [RTU_CHILD_BITS-1:0] last_child = node.n_children - RTU_CHILD_BITS'(1);

    // LEAF_INST count: leaf-header word0 bits 8..15 (kind|count<<8).
    wire [31:0] leaf_inst_count = {24'd0, f_aligned[15:8]};
    // ── instance-record decode (64 B BVH instance: matches VxBvhInstance) ──
    wire [31:0]       inst_blas = f_aligned[RTU_INST_OFF_BLAS*8     +: 32];
    wire [31:0]       inst_id   = f_aligned[RTU_INST_OFF_ID_BVH*8   +: 32];
    wire [31:0]       inst_cull = f_aligned[RTU_INST_OFF_CULL_BVH*8 +: 32];
    wire [11:0][31:0] inst_xform_w;
    for (genvar k2 = 0; k2 < 12; ++k2) begin : g_inst_xform
        assign inst_xform_w[k2] = f_aligned[(RTU_INST_OFF_XFORM + 4*k2)*8 +: 32];
    end
    wire [RTU_LINES_BITS-1:0] inst_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_INST_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire inst_culled = ((inst_cull & ray_q.cull_mask & 32'hff) == 32'd0);
    // BLAS traversal runs the object-space ray; the world ray otherwise.
    wire obj_setup = (cstate_q == CS_OBJ_SETUP);
    wire [2:0][31:0] walk_ro    = inblas_q ? objo_q    : ray_q.origin;
    wire [2:0][31:0] walk_rd    = inblas_q ? objd_q    : ray_q.dir;
    wire [2:0][31:0] walk_inv_d = inblas_q ? objinvd_q : invd_q;

    // ── ray setup datapath (driven by the EXEC snapshot ray). inv_d = 1/dir;
    // the box PE subtracts the ray origin itself, so there is no origin*inv_d
    // precompute (which would lose precision on axis-aligned rays where inv_d
    // is infinite). The snapshot ray is stable for the span of CS_SETUP, so the
    // fixed-latency reciprocal sees a steady input ────────────────────────
    // The shared reciprocal computes 1/dir for the world ray (CS_SETUP) or the
    // current instance's object ray (CS_OBJ_SETUP); the input is muxed so the
    // same units feed both. The snapshot dir is stable across the setup span.
    wire [2:0][31:0] inv_d_w;
    for (genvar a = 0; a < 3; ++a) begin : g_setup
        wire [31:0] recip_din = obj_setup ? objd_q[a] : ray_q.dir[a];
        VX_fdivsqrt_unit #(.LATENCY (RTU_FDIV_LAT)) recip (
            .clk (clk), .reset (reset), .enable (1'b1), .mask (1'b1),
            .fmt ('0), .frm (INST_FRM_RNE),
            .dataa (32'h3F800000 /*1.0*/), .datab (recip_din), .is_sqrt (1'b0),
            .result (inv_d_w[a]), `UNUSED_PIN (fflags)
        );
    end

    // ── box PE: one child per EXEC cycle while the snapshot context feeds ──
    wire        box_valid_in = exec && ((cstate_q == CS_FEED) || (cstate_q == CS_PROC_FEED));
    wire        box_valid_out, box_hit;
    wire [31:0] box_t_near;
    // Procedural-leaf raw AABB (float min/max == leaf_v0/leaf_v1) fed in raw
    // mode; internal-node child boxes stay quantized (raw=0).
    wire             box_raw    = (cstate_q == CS_PROC_FEED);
    wire [2:0][31:0] box_rawmin = leaf_v0;
    wire [2:0][31:0] box_rawmax = leaf_v1;
    VX_rtu_box_pe box_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (box_valid_in),
        .origin (node.origin), .exp (node.exp),
        .qmin (node.qmin[feed_ci]), .qmax (node.qmax[feed_ci]),
        .raw (box_raw), .raw_min (box_rawmin), .raw_max (box_rawmax),
        .ro (walk_ro), .inv_d (walk_inv_d),
        .t_min (ray_q.t_min), .t_max (bestt_q),
        .valid_out (box_valid_out), .hit (box_hit), .t_near (box_t_near)
    );
    wire coll_pushable = box_hit && (node.child_off[coll_ci] != 32'd0);

    // ── tri PE: tagged by context id so results route back ────────────
    wire        tri_valid_in = exec && (cstate_q == CS_TRI_FEED);
    wire        tri_valid_out, tri_hit, tri_back;
    wire [CTX_TAG_W-1:0] tri_tag_out;
    wire [31:0] tri_t, tri_u, tri_v;
    `UNUSED_VAR (tri_back)
    VX_rtu_tri_pe #(.TAG_WIDTH (CTX_TAG_W)) tri_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (tri_valid_in),
        .tag_in (sel_q),
        .origin (walk_ro), .dir (walk_rd),
        .v0 (leaf_v0), .v1 (leaf_v1), .v2 (leaf_v2),
        .t_min (ray_q.t_min), .t_max (bestt_q),
        .valid_out (tri_valid_out), .tag_out (tri_tag_out), .hit (tri_hit),
        .t (tri_t), .u (tri_u), .v (tri_v), .back_facing (tri_back)
    );

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

    // ── memory request (single shared port, tagged by context) ────────
    wire fetch_issue = (cstate_q == CS_HDR_REQ)
                    || (cstate_q == CS_REQ0)
                    || (cstate_q == CS_REQN)
                    || (cstate_q == CS_INST_REQ)
                    || (cstate_q == CS_INST_REQN)
                    ;
    assign mem_req_valid = exec && fetch_issue;
    assign mem_req_tag   = sel_q;
    wire line0_req = (cstate_q == CS_REQ0)
                  || (cstate_q == CS_INST_REQ)
                  ;
    assign mem_req_addr  = (cstate_q == CS_HDR_REQ) ? ray_q.scene_base
                         : line0_req                ? struct_addr
                         : (struct_addr + (`VX_CFG_MEM_ADDR_WIDTH'(fidx_q) << RTU_LINE_SEL_BITS));
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
            running  <= 1'b0;
            done_r   <= 1'b0;
            cc       <= '0;
            phase    <= PH_SELECT;
            for (k = 0; k < NUM_CTX; k = k + 1) begin
                cstate[k]      <= CS_DONE;
                line_ready[k]  <= 1'b0;
                tri_ready[k]   <= 1'b0;
                box_done[k]    <= 1'b0;
                proc_ready[k]  <= 1'b0;
                yld_pending[k] <= 1'b0;
                xform_ready[k] <= 1'b0;
                in_blas[k]     <= 1'b0;
            end
            finalised <= 1'b0;
        end else begin
            done_r <= 1'b0;

            // launch: seed one context per active lane
            if (!running && start) begin
                running   <= 1'b1;
                cc        <= '0;
                phase     <= PH_SELECT;
                mask_r    <= mask;
                finalised <= 1'b0;
                for (k = 0; k < NUM_CTX; k = k + 1) begin
                    ray_r[k]      <= rays[k];
                    best_t[k]     <= rays[k].t_max;
                    sp[k]         <= '0;
                    cur_off[k]    <= '0;
                    setup_ctr[k]  <= '0;
                    line_ready[k] <= 1'b0;
                    tri_ready[k]  <= 1'b0;
                    box_done[k]   <= 1'b0;
                    proc_ready[k] <= 1'b0;
                    hit_r[k]      <= 1'b0;
                    hit_t_r[k]    <= rays[k].t_max;
                    hit_u_r[k]    <= '0;
                    hit_v_r[k]    <= '0;
                    hit_prim_r[k] <= '0;
                    hit_geom_r[k] <= '0;
                    hit_inst_r[k] <= '0;
                    yld_inst[k]   <= '0;
                    yld_pending[k]<= 1'b0;
                    yld_t[k]      <= rays[k].t_max;
                    inst_count[k] <= '0;
                    inst_idx[k]   <= '0;
                    in_blas[k]    <= 1'b0;
                    xform_ready[k]<= 1'b0;
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

            // async xform-PE result → object ray routes back to its context
            if (xform_valid_out) begin
                obj_o[xform_tag_out]       <= xform_obj_o;
                obj_d[xform_tag_out]       <= xform_obj_d;
                xform_ready[xform_tag_out] <= 1'b1;
            end

            // box results stream back to the running context (exclusive to the
            // context selected across its node's CS_FEED/CS_WAIT span). Collected
            // every cycle so a result that lands on a SELECT phase is not missed.
            if (box_valid_out) begin
                if (cstate[sel_q] == CS_PROC_WAIT) begin
                    // procedural-leaf raw box test result -> IS yield candidate
                    proc_ready[sel_q] <= 1'b1;
                    proc_hit_p[sel_q] <= box_hit;
                    proc_t_p[sel_q]   <= box_t_near;
                end else begin
                    child_hit[sel_q][coll_ci] <= coll_pushable;
                    coll_idx[sel_q]           <= coll_idx[sel_q] + RTU_CHILD_BITS'(1);
                    if (coll_idx[sel_q] == last_child) begin
                        box_done[sel_q] <= 1'b1;
                    end
                end
            end

            // ── micro-step pipeline ───────────────────────────────────
            if (running) begin
                if (phase == PH_SELECT) begin
                    // snapshot the selected context's working set for EXEC
                    if (sel_valid) begin
                        sel_q      <= sel;
                        cc         <= sel;
                        ray_q      <= ray_r[sel];
                        invd_q     <= inv_d_r[sel];
                        fbuf_q     <= f_buf[sel];
                        structaddr_q <= ray_r[sel].scene_base + cur_off[sel];
                        bestt_q    <= best_t[sel];
                        cstate_q   <= cstate[sel];
                        setupctr_q <= setup_ctr[sel];
                        fidx_q     <= f_idx[sel];
                        ftotal_q   <= f_total[sel];
                        feed_q     <= feed_idx[sel];
                        push_q     <= push_idx[sel];
                        sp_q       <= sp[sel];
                        stacktop_q <= stack[sel][sp[sel] - RTU_STACK_BITS'(1)];
                        childhit_q <= child_hit[sel];
                        instidx_q   <= inst_idx[sel];
                        instcount_q <= inst_count[sel];
                        instbase_q  <= inst_base[sel];
                        blasroot_q  <= blas_root[sel];
                        instid_q    <= inst_id_r[sel];
                        xform_q     <= inst_xform[sel];
                        objo_q      <= obj_o[sel];
                        objd_q      <= obj_d[sel];
                        objinvd_q   <= obj_inv_d_r[sel];
                        blasfloor_q <= blas_floor[sel];
                        inblas_q    <= in_blas[sel];
                        phase      <= PH_EXEC;
                    end
                end else begin
                    // EXEC: advance the snapshot context, write back results
                    phase <= PH_SELECT;
                    case (cstate_q)
                    CS_SETUP: begin
                        if (setupctr_q != SETUP_CW'(SETUP_LAT)) begin
                            setup_ctr[sel_q] <= setupctr_q + SETUP_CW'(1);
                        end else begin
                            inv_d_r[sel_q]   <= inv_d_w;
                            cstate[sel_q]    <= CS_HDR_REQ;
                        end
                    end
                    CS_HDR_REQ: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_HDR_WAIT;
                        end
                    end
                    CS_HDR_WAIT: begin
                        if (line_ready[sel_q]) begin
                            cur_off[sel_q] <= f_aligned[RTU_SCENE_OFF_ROOT*8 +: 32];
                            cstate[sel_q]  <= CS_REQ0;
                        end
                    end
                    CS_REQ0: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_RSP0;
                        end
                    end
                    CS_RSP0: begin
                        if (line_ready[sel_q]) begin
                            if (node_kind == RTU_KIND_INTERNAL) begin
                                f_total[sel_q] <= node_lines;
                                if (node_lines == RTU_LINES_BITS'(1)) begin
                                    cstate[sel_q] <= CS_DISPATCH;
                                end else begin
                                    f_idx[sel_q]  <= RTU_LINES_BITS'(1);
                                    cstate[sel_q] <= CS_REQN;
                                end
                            end else if ((node_kind == RTU_KIND_LEAF_TRI)
                                      || (node_kind == RTU_KIND_LEAF_PROC)
                                      || (node_kind == RTU_KIND_LEAF_INST)) begin
                                // LEAF_INST shares the leaf-line fetch: only its
                                // 16 B header (kind|count) is decoded at DISPATCH;
                                // the instance records are fetched in CS_INST_REQ.
                                f_total[sel_q] <= leaf_lines;
                                if (leaf_lines == RTU_LINES_BITS'(1)) begin
                                    cstate[sel_q] <= CS_DISPATCH;
                                end else begin
                                    f_idx[sel_q]  <= RTU_LINES_BITS'(1);
                                    cstate[sel_q] <= CS_REQN;
                                end
                            end else begin
                                cstate[sel_q] <= CS_POP;
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
                            if ((fidx_q + RTU_LINES_BITS'(1)) == ftotal_q) begin
                                cstate[sel_q] <= CS_DISPATCH;
                            end else begin
                                f_idx[sel_q]  <= fidx_q + RTU_LINES_BITS'(1);
                                cstate[sel_q] <= CS_REQN;
                            end
                        end
                    end
                    CS_DISPATCH: begin
                        if (node_kind == RTU_KIND_INTERNAL && node.n_children != '0) begin
                            feed_idx[sel_q]  <= '0;
                            coll_idx[sel_q]  <= '0;
                            child_hit[sel_q] <= '0;
                            box_done[sel_q]  <= 1'b0;
                            cstate[sel_q]    <= CS_FEED;
                        end else if (node_kind == RTU_KIND_LEAF_TRI) begin
                            leaf_geom_r[sel_q] <= leaf_geom;
                            leaf_prim_r[sel_q] <= leaf_prim;
                            cstate[sel_q]      <= CS_TRI_FEED;
                        end else if (node_kind == RTU_KIND_LEAF_PROC) begin
                            leaf_geom_r[sel_q] <= leaf_geom;
                            leaf_prim_r[sel_q] <= leaf_prim;
                            proc_sbt_p[sel_q]  <= RTU_CB_SBT_BITS'((leaf_flags >> RTU_TRI_SBT_IDX_SHIFT) & RTU_TRI_SBT_IDX_MASK);
                            proc_ready[sel_q]  <= 1'b0;
                            cstate[sel_q]      <= CS_PROC_FEED;
                        end else begin
                            if (node_kind == RTU_KIND_LEAF_INST && leaf_inst_count != 32'd0) begin
                                // TLAS leaf: iterate instances, each descending
                                // into its BLAS subtree under the object ray on
                                // this same short stack (floor = current sp).
                                inst_count[sel_q] <= leaf_inst_count;
                                inst_idx[sel_q]   <= '0;
                                inst_base[sel_q]  <= cur_off[sel_q] + 32'(RTU_LEAF_HDR_BYTES);
                                blas_floor[sel_q] <= sp_q;
                                cur_off[sel_q]    <= cur_off[sel_q] + 32'(RTU_LEAF_HDR_BYTES);
                                cstate[sel_q]     <= CS_INST_REQ;
                            end else
                            cstate[sel_q] <= CS_POP;
                        end
                    end
                    CS_PROC_FEED: begin
                        // the raw AABB box was fed this EXEC cycle; await result.
                        cstate[sel_q] <= CS_PROC_WAIT;
                    end
                    CS_PROC_WAIT: begin
                        if (proc_ready[sel_q]) begin
                            proc_ready[sel_q] <= 1'b0;
                            // procedural primitive is non-opaque: stage an IS yield
                            // for the AABB-entry candidate (its t is a lower bound,
                            // overridden by the IS via cb_hit_t on accept).
                            if (proc_hit_p[sel_q] && (proc_t_p[sel_q] < bestt_q)
                                && (!yld_pending[sel_q] || (proc_t_p[sel_q] < yld_t[sel_q]))) begin
                                yld_pending[sel_q] <= 1'b1;
                                yld_t[sel_q]       <= proc_t_p[sel_q];
                                yld_u[sel_q]       <= '0;
                                yld_v[sel_q]       <= '0;
                                yld_prim[sel_q]    <= leaf_prim_r[sel_q];
                                yld_cbtype[sel_q]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC);
                                yld_sbt[sel_q]     <= proc_sbt_p[sel_q];
                            end
                            cstate[sel_q] <= CS_POP;
                        end
                    end
                    CS_FEED: begin
                        if (feed_q == last_child) begin
                            cstate[sel_q] <= CS_WAIT;
                        end
                        feed_idx[sel_q] <= feed_q + RTU_CHILD_BITS'(1);
                    end
                    CS_WAIT: begin
                        if (box_done[sel_q]) begin
                            box_done[sel_q] <= 1'b0;
                            push_idx[sel_q] <= '0;
                            cstate[sel_q]   <= CS_PUSH;
                        end
                    end
                    CS_PUSH: begin
                        if (childhit_q[push_ci] && (sp_q != RTU_STACK_BITS'(RTU_STACK_DEPTH))) begin
                            stack[sel_q][sp_q] <= node.child_off[push_ci] & RTU_CHILD_OFF_MASK;
                            sp[sel_q]          <= sp_q + RTU_STACK_BITS'(1);
                        end
                        if (push_q == last_child) begin
                            cstate[sel_q] <= CS_POP;
                        end
                        push_idx[sel_q] <= push_q + RTU_CHILD_BITS'(1);
                    end
                    CS_TRI_FEED: begin
                        cstate[sel_q] <= CS_TRI_WAIT;
                    end
                    CS_TRI_WAIT: begin
                        if (tri_ready[sel_q]) begin
                            tri_ready[sel_q] <= 1'b0;
                            if (tri_hit_p[sel_q]) begin
                                best_t[sel_q]     <= tri_t_p[sel_q];
                                hit_r[sel_q]      <= 1'b1;
                                hit_t_r[sel_q]    <= tri_t_p[sel_q];
                                hit_u_r[sel_q]    <= tri_u_p[sel_q];
                                hit_v_r[sel_q]    <= tri_v_p[sel_q];
                                hit_prim_r[sel_q] <= leaf_prim_r[sel_q];
                                hit_geom_r[sel_q] <= leaf_geom_r[sel_q];
                                // a BLAS hit carries its instance's id; a
                                // top-level (non-instanced) tri reports 0.
                                hit_inst_r[sel_q] <= inblas_q ? instid_q : 32'd0;
                            end
                            cstate[sel_q] <= CS_POP;
                        end
                    end
                    CS_POP: begin
                        if (inblas_q && (sp_q == blasfloor_q)) begin
                            // BLAS subtree drained back to the instance-loop
                            // floor: resume the instance loop in world space.
                            cstate[sel_q] <= CS_INST_NEXT;
                        end else
                        if (sp_q == '0) begin
                            cstate[sel_q] <= CS_DONE;
                        end else begin
                            cur_off[sel_q] <= stacktop_q;
                            sp[sel_q]      <= sp_q - RTU_STACK_BITS'(1);
                            cstate[sel_q]  <= CS_REQ0;
                        end
                    end
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
                            if (ftotal_q == RTU_LINES_BITS'(1)) begin
                                cstate[sel_q] <= CS_INST_RSPN;
                            end else begin
                                f_idx[sel_q]  <= RTU_LINES_BITS'(1);
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
                        if (line_ready[sel_q]) begin
                            if ((ftotal_q != RTU_LINES_BITS'(1))
                             && ((fidx_q + RTU_LINES_BITS'(1)) != ftotal_q)) begin
                                f_idx[sel_q]  <= fidx_q + RTU_LINES_BITS'(1);
                                cstate[sel_q] <= CS_INST_REQN;
                            end else if (inst_culled) begin
                                // §8.8 cull gate: skip transform + BLAS descent.
                                cstate[sel_q] <= CS_INST_NEXT;
                            end else begin
                                inst_xform[sel_q] <= inst_xform_w;
                                blas_root[sel_q]  <= inst_blas;
                                inst_id_r[sel_q]  <= inst_id;
                                xform_ready[sel_q]<= 1'b0;
                                cstate[sel_q]     <= CS_XFORM;
                            end
                        end
                    end
                    CS_XFORM: begin
                        // world ray + xform fed to VX_rtu_xform this EXEC cycle.
                        cstate[sel_q] <= CS_XFORM_WT;
                    end
                    CS_XFORM_WT: begin
                        if (xform_ready[sel_q]) begin
                            // object ray latched; compute its inv_d next.
                            setup_ctr[sel_q] <= '0;
                            cstate[sel_q]    <= CS_OBJ_SETUP;
                        end
                    end
                    CS_OBJ_SETUP: begin
                        if (setupctr_q != SETUP_CW'(SETUP_LAT)) begin
                            setup_ctr[sel_q] <= setupctr_q + SETUP_CW'(1);
                        end else begin
                            obj_inv_d_r[sel_q] <= inv_d_w;
                            // enter the BLAS subtree under the object ray.
                            in_blas[sel_q]     <= 1'b1;
                            cur_off[sel_q]     <= blasroot_q;
                            cstate[sel_q]      <= CS_REQ0;
                        end
                    end
                    CS_INST_NEXT: begin
                        // BLAS done: back to world space; advance the instance.
                        in_blas[sel_q] <= 1'b0;
                        if ((instidx_q + 32'd1) == instcount_q) begin
                            // all instances visited: resume the TLAS walk by
                            // popping the (world-space) stack from the floor.
                            cstate[sel_q] <= CS_POP;
                        end else begin
                            inst_idx[sel_q] <= instidx_q + 32'd1;
                            cur_off[sel_q]  <= instbase_q
                                             + ((instidx_q + 32'd1) * 32'(RTU_INST_STRIDE));
                            cstate[sel_q]   <= CS_INST_REQ;
                        end
                    end
                    default:;
                    endcase
                end
            end

            // ── post-walk callback yield barrier (mirrors the flat walker) ──
            if (running && all_done) begin
                if (!finalised) begin
                    // finalise_lane: CHS (committed hit + ENABLE_CHS) or MISS
                    // (no hit + ENABLE_MISS) for lanes without a candidate yield.
                    for (k = 0; k < NUM_CTX; k = k + 1) begin
                        if (mask_r[k] && !yld_pending[k]) begin
                            if (hit_r[k] && ((ray_r[k].flags & `VX_RT_FLAG_ENABLE_CHS) != 0)
                                         && ((ray_r[k].flags & `VX_RT_FLAG_SKIP_CLOSEST_HIT) == 0)) begin
                                yld_pending[k] <= 1'b1;
                                yld_cbtype[k]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_CHS);
                                yld_t[k] <= hit_t_r[k]; yld_u[k] <= hit_u_r[k];
                                yld_v[k] <= hit_v_r[k]; yld_prim[k] <= hit_prim_r[k];
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
                                    // PROC accept commits the IS-computed t.
                                    hit_t_r[k]    <= (yld_cbtype[k] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))
                                                   ? action_hit_t[k] : yld_t[k];
                                    hit_u_r[k]    <= yld_u[k];
                                    hit_v_r[k]    <= yld_v[k];
                                    hit_prim_r[k] <= yld_prim[k];
                                end
                                yld_pending[k] <= 1'b0;
                            end
                        end
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
        if (exec && (cstate_q == CS_DISPATCH)) begin
            `TRACE(2, ("%t: %s rtu-node: ctx=%0d, off=%0d, kind=%0d, children=%0d\n",
                $time, INSTANCE_ID, sel_q, structaddr_q, node_kind, node.n_children))
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

    // While at the yield barrier, present the candidate attrs (CB_YIELD payload)
    // on res_* for the yielding lanes; otherwise the committed hit.
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
        assign res_geom[i] = hit_geom_r[i];
        assign res_inst[i] = cand_i ? yld_inst[i] : hit_inst_r[i];
    end

    assign busy = running;
    assign done = done_r;

endmodule
