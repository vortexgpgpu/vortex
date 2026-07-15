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

// VX_rtu_bvh_scheduler — context-pool BVH traversal control. Holds one ray context
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

module VX_rtu_bvh_scheduler import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_SLOTS = 1,
    parameter NUM_CTX   = 4,
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8,
    parameter CTX_TAG_W = `LOG2UP(NUM_CTX)    // derived: context-id tag width
) (
    input  wire        clk,
    input  wire        reset,

    // Slot launch: one ray per active lane of the slot's warp. A SLOT OWNS its
    // contexts — slot s owns contexts [s*CTX_PER_SLOT, (s+1)*CTX_PER_SLOT) — so
    // several warps traverse CONCURRENTLY and the shared front end below switches
    // between them whenever one parks on memory. That switch is the whole point:
    // it is what hides the fetch latency a single warp would sit through.
    input  wire [NUM_SLOTS-1:0]       start,
    input  wire [NUM_CTX-1:0]         mask,
    input  rtu_ray_t [NUM_CTX-1:0]    rays,
    output wire [NUM_SLOTS-1:0]       busy,
    output wire [NUM_SLOTS-1:0]       done,

    // per-lane closest-hit results
    output wire [NUM_CTX-1:0]         res_hit,
    output wire [NUM_CTX-1:0][31:0]   res_t,
    output wire [NUM_CTX-1:0][31:0]   res_u,
    output wire [NUM_CTX-1:0][31:0]   res_v,
    output wire [NUM_CTX-1:0][31:0]   res_prim,
    output wire [NUM_CTX-1:0][31:0]   res_geom,
    output wire [NUM_CTX-1:0][31:0]   res_inst,
    output wire [NUM_CTX-1:0][31:0]   res_custom,

    // Callback yield barrier (see VX_rtu_flat_scheduler). The BVH
    // walker yields per-triangle any-hit (non-opaque tri), procedural-leaf
    // intersection, and post-walk CHS/MISS callbacks through this interface.
    output wire [NUM_SLOTS-1:0]                       yield,
    output wire [NUM_CTX-1:0]                         yield_mask,
    output wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]   yield_cbtype,
    output wire [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]    yield_sbt,
    input  wire [NUM_SLOTS-1:0]                       resume,
    input  wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] action,
    input  wire [NUM_CTX-1:0][31:0]                   action_hit_t,

    // node/leaf fetch (to the RTCache port in VX_rtu_core, tagged by context id)
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

    // A slot owns a fixed, contiguous run of contexts. Static ownership is what
    // keeps this free: a context's slot is just the high bits of its index, so
    // nothing here has to carry a slot id around.
    localparam CTX_PER_SLOT = NUM_CTX / NUM_SLOTS;
    `STATIC_ASSERT(((CTX_PER_SLOT * NUM_SLOTS) == NUM_CTX),
        ("RTU_NUM_CTX must be a whole multiple of RTU_NUM_SLOTS"))

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
                     CS_INST_NEXT = 5'd24,  // advance to next instance / resume TLAS
                     // Fat-leaf triangle loop: a LEAF_TRI packs `count`
                     // triangles; each is fetched as a 40 B record at the leaf's
                     // triangle stride and streamed through the tri PE.
                     CS_LTRI_REQ0 = 5'd25,  // issue leaf-triangle record line 0
                     CS_LTRI_RSP0 = 5'd26,  // park: record line 0
                     CS_LTRI_REQN = 5'd27,  // issue record line N
                     CS_LTRI_RSPN = 5'd28;  // park: record line N -> tri PE

    // ── per-context state ─────────────────────────────────────────────
    reg [NUM_CTX-1:0][4:0]                       cstate;
    // The ray is NOT copied here. `rays` is driven from the core's per-slot ray
    // registers and is stable for the whole traversal — the core may only overwrite a
    // slot's rays once that slot's contexts have all retired (it preloads under a
    // TERMINAL record write, never under a candidate). Copying it would be a second
    // ~350 b per context of flops holding what the core already holds.
    reg [NUM_CTX-1:0][2:0][31:0]                  inv_d_r;
    reg [NUM_CTX-1:0][31:0]                       best_t;
    reg [NUM_CTX-1:0]                             hit_r;
    reg [NUM_CTX-1:0][31:0]                       hit_t_r, hit_u_r, hit_v_r, hit_prim_r, hit_geom_r;
    reg [NUM_CTX-1:0][31:0]                       hit_inst_r;   // committed hit's instance id (TLAS)
    reg [NUM_CTX-1:0][31:0]                       hit_custom_r; // committed hit's custom index (TLAS)
    reg [NUM_CTX-1:0][RTU_STACK_BITS-1:0]         sp;
    reg [NUM_CTX-1:0][31:0]                       cur_off;
    // f_buf (per-context node image) is held in g_fbuf_ram (context-id RAM; below).
    reg [NUM_CTX-1:0][RTU_LINES_BITS-1:0]         f_idx, f_total, f_slot;
    reg [NUM_CTX-1:0][RTU_CHILD_BITS-1:0]         feed_idx, coll_idx;
    // Incremental nearest-first ordering: as box results stream back one child per
    // cycle, each hit child is insertion-sorted (ascending by t_near) into
    // ord_off/ord_t. ord_off[0] is the nearest (descended directly); entries
    // [1..ord_cnt-1] are the siblings, pushed farthest-first. Building the order
    // during collection makes the CS_PUSH stack-write a registered index -> RAM,
    // instead of a combinational WIDTH-wide t-compare scan feeding the stack RAM.
    reg [NUM_CTX-1:0][RTU_BVH_WIDTH-1:0][31:0]     ord_off;   // hit children, t-ascending
    // Children per node, latched at CS_DISPATCH. The collector counts results
    // against this instead of reading node_r, which a parked context no longer owns.
    reg [NUM_CTX-1:0][RTU_CHILD_BITS-1:0]         coll_last;
    reg [NUM_CTX-1:0][RTU_BVH_WIDTH-1:0][31:0]     ord_t;     // their t_near keys
    reg [NUM_CTX-1:0][RTU_CHILD_BITS-1:0]         ord_cnt;    // number of hit children
    reg [NUM_CTX-1:0][RTU_CHILD_BITS-1:0]         push_ptr;   // CS_PUSH cursor (farthest->1)
    reg [NUM_CTX-1:0]                             box_done;
    reg [NUM_CTX-1:0][31:0]                       leaf_geom_r, leaf_prim_r;
    reg [NUM_CTX-1:0][2:0][31:0]                  leaf_v0_r, leaf_v1_r, leaf_v2_r;
    // Fat-leaf: per-leaf triangle count + current index. prim reported is
    // leaf_prim_r (prim_base) + leaf_tidx.
    reg [NUM_CTX-1:0][7:0]                        leaf_tcnt, leaf_tidx;
    reg [NUM_CTX-1:0][SETUP_CW-1:0]               setup_ctr;
    reg [NUM_CTX-1:0][1:0]                        setup_axis;   // 1/dir axis being reciprocated
    reg [NUM_CTX-1:0]                             line_ready, tri_ready;
    reg [NUM_CTX-1:0]                             tri_hit_p, tri_back_p;
    reg [NUM_CTX-1:0][31:0]                       tri_t_p, tri_u_p, tri_v_p;
    reg [NUM_CTX-1:0][31:0]                       tri_flags_p;  // per-triangle flags, latched at leaf decode
    // procedural-leaf box result (routed off the child-hit collection)
    reg [NUM_CTX-1:0]                             proc_ready, proc_hit_p;
    reg [NUM_CTX-1:0][31:0]                       proc_t_p;
    reg [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]        proc_sbt_p;
    // Per-context IS/AHS yield candidate + finalise bookkeeping.
    reg [NUM_CTX-1:0]                             yld_pending;
    reg [NUM_CTX-1:0][31:0]                       yld_t, yld_u, yld_v, yld_prim;
    reg [NUM_CTX-1:0][31:0]                       yld_inst;   // candidate hit's instance id (TLAS)
    reg [NUM_CTX-1:0][31:0]                       yld_custom; // candidate hit's custom index (TLAS)
    reg [NUM_CTX-1:0][31:0]                       yld_geom;   // candidate hit's gl_GeometryIndexEXT
    reg [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0]       yld_cbtype;
    reg [NUM_CTX-1:0][RTU_CB_SBT_BITS-1:0]        yld_sbt;
    reg [NUM_CTX-1:0]                             mask_r;
    reg [NUM_SLOTS-1:0]                          finalised;

    // ── per-context TLAS state: the LEAF_INST instance loop + BLAS descent ──
    reg [NUM_CTX-1:0][31:0]                       inst_count, inst_idx;
    reg [NUM_CTX-1:0][31:0]                       inst_base, blas_root, inst_id_r, inst_custom_r;
    reg [NUM_CTX-1:0][7:0]                         inst_flags_r; // latched VkGeometryInstanceFlagBits

    // ── short-stack overflow restart ──────────────────────────────
    // The short stack is RTU_STACK_DEPTH deep; a deeper tree overflows and a
    // child push is dropped. Rather than silently losing that subtree, the
    // walker records the drop and, when the current subtree drains, re-descends
    // it from its root pruning by the committed best_t — a bounded "re-descend
    // from root" backstop. Each re-descent bumps a (capped) budget so traversal
    // always terminates.
    //
    // The drop is recorded at its OWN level — world (TLAS) vs object
    // (BLAS) — so a world-level drop is not silently cleared by a BLAS-floor
    // restart. Each level's marker is re-descended from its own root (scene
    // root for world, blas_root for object). The restart budget is split
    // per level and the object budget is RESET on every BLAS entry, so a deep
    // multi-instance ray doesn't exhaust one global budget in its first
    // instance and then drop later subtrees. Descending nearest-first
    // (CS_PUSH) tightens best_t early so a bounded number of re-descents converge
    // on the true closest hit.
    localparam RTU_RESTART_CAP = 8;
    localparam RST_CNTW        = `CLOG2(RTU_RESTART_CAP + 1);
    reg [NUM_CTX-1:0][31:0]                       root_off_r;   // scene root (restart target)
    reg [NUM_CTX-1:0]                             ovf_world_r;  // world(TLAS)-level push dropped
    reg [NUM_CTX-1:0]                             ovf_obj_r;    // object(BLAS)-level push dropped
    reg [NUM_CTX-1:0][RST_CNTW-1:0]               rst_world;    // world re-descents taken (capped)
    reg [NUM_CTX-1:0][RST_CNTW-1:0]               rst_obj;      // per-BLAS re-descents taken (capped)
    // inst_xform (latched 3x4 affine) is held in xform_ram (context-id RAM; below).
    reg [NUM_CTX-1:0][2:0][31:0]                  obj_o, obj_d;   // object-space ray
    reg [NUM_CTX-1:0][2:0][31:0]                  obj_inv_d_r;    // 1/obj_dir
    reg [NUM_CTX-1:0][RTU_STACK_BITS-1:0]         blas_floor;     // sp at instance loop
    reg [NUM_CTX-1:0]                             in_blas;        // object ray active
    reg [NUM_CTX-1:0]                             xform_ready;

    // Per SLOT, not per machine: several slots traverse at once.
    reg [NUM_SLOTS-1:0]       running;
    reg [NUM_SLOTS-1:0]       done_r;
    wire                      running_any = (| running);
    reg [CTX_TAG_W-1:0]       cc;          // round-robin start pointer

    // ── micro-step pipeline: SELECT latches the narrow snapshot and issues the
    //    f_buf RAM read; ALIGN registers the RAM node image into a fabric flop;
    //    EXEC runs the byte-align shift + decode + FSM from it. The ALIGN flop
    //    keeps the BlockRAM read output off the f_aligned barrel-shift cone, so
    //    only the states that read that image need it (see needs_img); the rest
    //    run from the snapshot alone and skip straight to EXEC. ─────────────────
    reg [1:0]                 phase;
    localparam [1:0] PH_SELECT = 2'd0, PH_ALIGN = 2'd1, PH_EXEC = 2'd2;
    reg [BUF_BITS-1:0]        fbuf_q;      // ALIGN-registered node image (off-BRAM)

    reg [CTX_TAG_W-1:0]       sel_q;       // context being executed
    rtu_ray_t                 ray_q;
    `UNUSED_VAR (ray_q.t_max)
    reg [2:0][31:0]           invd_q;
    // The node image is read from g_fbuf_ram (issued in SELECT), registered into
    // fbuf_q in ALIGN, and decoded in EXEC. The instance transform (xform_rd) is
    // the direct xform_ram output, consumed by the xform PE in EXEC.
    // Precomputed absolute structure address (scene_base + cur_off) latched in
    // SELECT, so the EXEC critical cone (byte-align shift -> node decode ->
    // state) starts after the add instead of through it. Same register/adder
    // count as latching the raw offset — a pure phase move, no latency/area cost.
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] structaddr_q;
    reg [31:0]                bestt_q;
    reg [4:0]                 cstate_q;
    reg [SETUP_CW-1:0]        setupctr_q;
    reg [1:0]                 setupaxis_q;
    reg [RTU_LINES_BITS-1:0]  fidx_q, ftotal_q;
    reg [RTU_CHILD_BITS-1:0]  feed_q;
    reg [RTU_STACK_BITS-1:0]  sp_q;
    reg [31:0]                stacktop_q;
    reg [RTU_BVH_WIDTH-1:0][31:0] ordoff_q;   // snapshot of ord_off
    reg [RTU_CHILD_BITS-1:0]  ordcnt_q;            // snapshot of ord_cnt
    reg [RTU_CHILD_BITS-1:0]  pushptr_q;           // snapshot of push_ptr
    reg [RTU_CHILD_BITS-1:0]  ins_pos;             // insertion index (collection scratch)
    reg [2:0][31:0]           leafv0_q, leafv1_q, leafv2_q;
    reg [7:0]                 leaftidx_q, leaftcnt_q;
    reg [31:0]                instidx_q, instcount_q, instbase_q, blasroot_q, instid_q, custid_q;
    reg [7:0]                 instflags_q;
    reg [31:0]                rootoff_q;
    reg                       ovfw_q, ovfo_q;
    reg [RST_CNTW-1:0]        rstw_q, rsto_q;
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
                CS_RSPN,
                CS_LTRI_RSP0,
                CS_LTRI_RSPN: r = line_ready[i];
                CS_TRI_WAIT:  r = tri_ready[i];
                // Box results carry their context id, so a context awaiting them
                // parks rather than holding the machine for the PE drain.
                CS_WAIT:      r = box_done[i];
                CS_PROC_WAIT: r = proc_ready[i];
                CS_INST_RSP0,
                CS_INST_RSPN: r = line_ready[i];
                CS_XFORM_WT:  r = xform_ready[i];
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
            logic [CTX_TAG_W-1:0] cand;
            cand = CTX_TAG_W'((32'(cc) + off) % NUM_CTX);
            if (runnable[cand]) begin
                sel       = cand;
                sel_valid = 1'b1;
            end
        end
    end
    wire exec = (phase == PH_EXEC);   // the snapshot context advances this cycle

    // States whose EXEC reads the f_buf node image. Must list every f_aligned
    // consumer: an omission reads a stale image, not a stalled one.
    function automatic logic needs_img (input logic [4:0] s);
        case (s)
            CS_HDR_WAIT,
            CS_RSP0,
            CS_DISPATCH,
            CS_LTRI_RSP0,
            CS_LTRI_RSPN,
            CS_INST_RSPN: needs_img = 1'b1;
            default:      needs_img = 1'b0;
        endcase
    endfunction
    wire sel_needs_img = needs_img(cstate[sel]);

    // ── combinational decode of the EXEC snapshot ─────────────────────
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] struct_addr = structaddr_q;
    wire [RTU_LINE_SEL_BITS-1:0]      f_off   = struct_addr[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0]      f_shift = {f_off, 3'b000};

    wire [BUF_BITS-1:0] f_aligned = fbuf_q >> f_shift;
    wire [RTU_NODE_IMG_BITS-1:0] node_img = f_aligned[RTU_NODE_IMG_BITS-1:0];
    `UNUSED_VAR (f_aligned[BUF_BITS-1:RTU_NODE_IMG_BITS])

    wire [7:0]  node_kind;
    rtu_node_t  node;
    VX_rtu_node_decode #(
        .IMG_BITS (RTU_NODE_IMG_BITS)
    ) decode (
        .line (node_img),
        .kind (node_kind),
        .node (node)
    );

    // Decode the internal node ONCE (latched at CS_DISPATCH into node_r) and drive
    // the per-cycle box-feed / collection / push paths from the register instead of
    // re-running the fbuf byte-align barrel-shift + decode every cycle. Collection/
    // feed/push are atomic to the selected (pinned) context, so a single current-
    // node register suffices (same discipline as the box-collect scratch). This
    // keeps the barrel-shift+decode cone off the CS_WAIT/CS_PUSH paths, which would
    // otherwise run structaddr -> decode -> child_off in a single cycle.
    rtu_node_t  node_r;

    // Leaf header + procedural-AABB corners (LEAF_PROC feeds leaf_v0/leaf_v1 as
    // the raw min/max box). Triangle-leaf vertices are decoded per record in the
    // fat-leaf loop (ltri_*), so no fixed single-triangle decode here.
    wire [2:0][31:0] leaf_v0, leaf_v1;
    for (genvar a = 0; a < 3; ++a) begin : g_tri_v
        assign leaf_v0[a] = f_aligned[(RTU_TRI_OFF_V0 + 4*a)*8 +: 32];
        assign leaf_v1[a] = f_aligned[(RTU_TRI_OFF_V1 + 4*a)*8 +: 32];
    end
    wire [31:0] leaf_geom  = f_aligned[RTU_LEAF_OFF_GEOM*8 +: 32];
    wire [31:0] leaf_prim  = f_aligned[RTU_LEAF_OFF_PRIM*8 +: 32];
    wire [31:0] leaf_flags = f_aligned[RTU_LEAF_OFF_FLAGS*8 +: 32];
    // Fat-leaf: leaf triangle count (kind|count<<8, low byte of count).
    wire [7:0]  leaf_tri_count = f_aligned[15:8];
    // A leaf-triangle record is fetched standalone at its own byte offset, so
    // its vertices sit at the record-relative flat offsets (v0@0, v1@12, v2@24,
    // flags@36) — same layout the flat walker decodes.
    wire [2:0][31:0] ltri_v0, ltri_v1, ltri_v2;
    for (genvar a2 = 0; a2 < 3; ++a2) begin : g_ltri_v
        assign ltri_v0[a2] = f_aligned[(RTU_FLAT_OFF_V0 + 4*a2)*8 +: 32];
        assign ltri_v1[a2] = f_aligned[(RTU_FLAT_OFF_V1 + 4*a2)*8 +: 32];
        assign ltri_v2[a2] = f_aligned[(RTU_FLAT_OFF_V2 + 4*a2)*8 +: 32];
    end
    wire [31:0] ltri_flags = f_aligned[RTU_FLAT_OFF_FLAGS*8 +: 32];

    wire [31:0] f_off32 = 32'(f_off);
    wire [RTU_LINES_BITS-1:0] node_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_NODE_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire [RTU_LINES_BITS-1:0] leaf_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_LEAF_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    // Fat-leaf: a standalone 40 B triangle record straddles <= 2 lines.
    wire [RTU_LINES_BITS-1:0] tri_rec_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_TRI_STRIDE - 1) >> RTU_LINE_SEL_BITS) + 1);

    wire [IDXW-1:0] feed_ci = feed_q[IDXW-1:0];
    wire [RTU_CHILD_BITS-1:0] last_child = node_r.n_children - RTU_CHILD_BITS'(1);

    // ── Nearest-first child ordering (precomputed during collection) ──────
    // The order is built incrementally as box results stream back (see the
    // collection block): hit children are insertion-sorted ascending by t_near
    // into ord_off. Descending the NEAREST (ord_off[0]) directly and pushing the
    // siblings farthest-first tightens best_t early (so the bounded restart
    // converges on the true closest hit and far subtrees are pruned before they
    // overflow). t_near is a non-negative float, so an unsigned magnitude compare
    // orders it correctly (same convention as the tri/box t-compares). CS_PUSH
    // walks push_ptr from the farthest sibling (ord_cnt-1) down to 1; the
    // stack-write child index is therefore a registered value, not the output of
    // a combinational t-compare scan feeding the stack RAM.
    wire push_active = (pushptr_q != RTU_CHILD_BITS'(0));       // siblings remain to push
    wire [31:0] push_child_off = ordoff_q[pushptr_q[IDXW-1:0]]; // sibling being pushed
    wire near_found_q = (ordcnt_q != RTU_CHILD_BITS'(0));       // any hit child this node

    // Short stacks held in a 1R1W RAM (one read in SELECT, one push in EXEC) keyed
    // by {context, depth} instead of a per-context flip-flop array + wide mux.
    localparam STK_IDXW = `CLOG2(RTU_STACK_DEPTH);
    localparam STK_SIZE = NUM_CTX << STK_IDXW;
    wire                stk_wr    = running_any && exec && (cstate_q == CS_PUSH)
                                 && push_active && (sp_q != RTU_STACK_BITS'(RTU_STACK_DEPTH));
    wire [STK_IDXW-1:0] stk_ridx  = STK_IDXW'(sp[sel] - RTU_STACK_BITS'(1));
    wire [31:0]         stk_rdata;
    VX_dp_ram #(
        .DATAW    (32),
        .SIZE     (STK_SIZE),
        .LUTRAM   (1),
        .OUT_REG  (0),
        .RDW_MODE ("W")
    ) node_stack_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (stk_wr),
        .wren  (1'b1),
        .waddr ({sel_q, sp_q[STK_IDXW-1:0]}),
        .wdata (push_child_off & RTU_CHILD_OFF_MASK),
        .raddr ({sel, stk_ridx}),
        .rdata (stk_rdata)
    );

    // LEAF_INST count: leaf-header word0 bits 8..15 (kind|count<<8).
    wire [31:0] leaf_inst_count = {24'd0, f_aligned[15:8]};
    // ── instance-record decode (64 B BVH instance) ──
    wire [31:0]       inst_blas = f_aligned[RTU_INST_OFF_BLAS*8     +: 32];
    wire [31:0]       inst_id   = f_aligned[RTU_INST_OFF_ID_BVH*8   +: 32];
    wire [31:0]       inst_custom = f_aligned[RTU_INST_OFF_CUSTOM*8 +: 32];
    wire [31:0]       inst_cull = f_aligned[RTU_INST_OFF_CULL_BVH*8 +: 32];
    // VkGeometryInstanceFlagBits packed into cull_mask bits 15..8.
    wire [7:0]        inst_flags = inst_cull[RTU_INST_FLAGS_SHIFT +: 8];
    wire [11:0][31:0] inst_xform_w;
    for (genvar k2 = 0; k2 < 12; ++k2) begin : g_inst_xform
        assign inst_xform_w[k2] = f_aligned[(RTU_INST_OFF_XFORM + 4*k2)*8 +: 32];
    end
    wire [RTU_LINES_BITS-1:0] inst_lines =
        RTU_LINES_BITS'(((f_off32 + RTU_INST_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    wire inst_culled = ((inst_cull & ray_q.cull_mask & 32'hff) == 32'd0);

    // ── Per-triangle any-hit classification (mirrors VX_rtu_flat_scheduler) ──
    // Evaluated at CS_TRI_WAIT on the latched tri flags / back-facing of the
    // selected context. A geometric triangle hit is committed when opaque, or
    // yielded as an any-hit callback when non-opaque; ray flags override the
    // opacity and cull front/back/opaque-class candidates.
    wire cull_back    = (ray_q.flags & 32'(`VX_RT_FLAG_CULL_BACK_FACING))     != 0;
    wire cull_front   = (ray_q.flags & 32'(`VX_RT_FLAG_CULL_FRONT_FACING))    != 0;
    wire skip_tris    = (ray_q.flags & 32'(`VX_RT_FLAG_SKIP_TRIANGLES))       != 0;
    wire ray_opaque   = (ray_q.flags & 32'(`VX_RT_FLAG_OPAQUE))               != 0;
    wire ray_noopaque = (ray_q.flags & 32'(`VX_RT_FLAG_NO_OPAQUE))            != 0;
    wire cull_opaque  = (ray_q.flags & 32'(`VX_RT_FLAG_CULL_OPAQUE))          != 0;
    wire cull_noopq   = (ray_q.flags & 32'(`VX_RT_FLAG_CULL_NO_OPAQUE))       != 0;
    wire term_first   = (ray_q.flags & 32'(`VX_RT_FLAG_TERMINATE_ON_FIRST_HIT)) != 0;
    // Per-instance flags (VkGeometryInstanceFlagBits) of the enclosing BLAS
    // instance — 0 for a top-level (non-instanced) triangle. FLIP inverts the
    // winding, CULL_DIS disables face culling, FORCE_{,NO_}OPAQUE override the
    // geometry opacity (ray flags still win).
    wire [7:0] cur_iflags = inblas_q ? instflags_q : 8'd0;
    wire inst_flip    = (cur_iflags & RTU_INST_FLAG_TRI_FLIP)     != 0;
    wire inst_culldis = (cur_iflags & RTU_INST_FLAG_TRI_CULL_DIS) != 0;
    wire inst_fopq    = (cur_iflags & RTU_INST_FLAG_FORCE_OPAQUE) != 0;
    wire inst_fnopq   = (cur_iflags & RTU_INST_FLAG_FORCE_NO_OPQ) != 0;
    wire eff_back     = tri_back_p[sel_q] ^ inst_flip;
    wire [31:0] cls_flags = tri_flags_p[sel_q];
    wire tri_opaque = ray_opaque   ? 1'b1
                    : ray_noopaque ? 1'b0
                    : inst_fopq    ? 1'b1
                    : inst_fnopq   ? 1'b0
                    : ((cls_flags & RTU_TRI_FLAG_OPAQUE) != 0);
    wire cls_cull = (tri_opaque && cull_opaque) || (!tri_opaque && cull_noopq);
    wire [RTU_CB_SBT_BITS-1:0] cls_sbt =
        RTU_CB_SBT_BITS'((cls_flags >> RTU_TRI_SBT_IDX_SHIFT) & RTU_TRI_SBT_IDX_MASK);
    // A geometric hit that survives skip / face / opacity-class culling and is
    // closer than the best committed opaque hit.
    wire tri_pass = tri_hit_p[sel_q]
                  && !skip_tris
                  && (inst_culldis || !(eff_back  && cull_back))
                  && (inst_culldis || !(!eff_back && cull_front))
                  && !cls_cull;
    // Compare against the per-context best-t snapshot (also fed to the box/tri
    // PEs) rather than the live array read, to keep the NUM_CTX mux off the
    // t-compare cone and match the flat walker / proc-IS paths.
    wire tri_committable = tri_pass && (tri_t_p[sel_q] < bestt_q);

    // BLAS traversal runs the object-space ray; the world ray otherwise.
    wire obj_setup = (cstate_q == CS_OBJ_SETUP);
    wire [2:0][31:0] walk_ro    = inblas_q ? objo_q    : ray_q.origin;
    wire [2:0][31:0] walk_rd    = inblas_q ? objd_q    : ray_q.dir;
    wire [2:0][31:0] walk_inv_d = inblas_q ? objinvd_q : invd_q;

    // ── per-context working set in RAM (node image + instance transform) ──
    // f_buf and inst_xform are held in context-id-addressed RAMs: the read is
    // issued in SELECT (raddr = sel) and its registered output lands the next
    // cycle (the f_buf image is restaged through the ALIGN flop fbuf_q before the
    // EXEC decode; see the phase machine). The entries are wide (>= 16b), so
    // VX_dp_ram maps them to BlockRAM, keeping the per-context working set on BRAM
    // (flat in fabric as NUM_CTX grows) rather than a flip-flop file + NUM_CTX:1 mux.
    wire ram_rd_en = running_any && (phase == PH_SELECT) && sel_valid;

    // f_buf: RTU_NODE_LINES fetched lines per context, each line slot its own 1R1W
    // RAM — full-line write on the matching mem response, full-line read in SELECT;
    // the slots concatenate (low slot = low bits) into the byte-aligned node image.
    wire [RTU_LINES_BITS-1:0] fbuf_wslot = f_slot[mem_rsp_tag];
    wire [BUF_BITS-1:0]       fbuf;
    for (genvar s = 0; s < RTU_NODE_LINES; ++s) begin : g_fbuf_ram
        wire [LINE_BITS-1:0] line_rd;
        VX_dp_ram #(
            .DATAW    (LINE_BITS),
            .SIZE     (NUM_CTX),
            .OUT_REG  (1),
            .RDW_MODE ("R")
        ) fbuf_ram (
            .clk   (clk),
            .reset (reset),
            .read  (ram_rd_en),
            .write (mem_rsp_valid && (fbuf_wslot == RTU_LINES_BITS'(s))),
            .wren  (1'b1),
            .waddr (mem_rsp_tag),
            .wdata (mem_rsp_data),
            .raddr (sel),
            .rdata (line_rd)
        );
        assign fbuf[s*LINE_BITS +: LINE_BITS] = line_rd;
    end

    // inst_xform: the latched 3x4 affine of the active TLAS instance. Full-word
    // write when CS_INST_RSPN accepts the (unculled) instance, full-word read in
    // SELECT. The write gate mirrors the CS_INST_RSPN accept branch in the FSM.
    wire inst_last_line = !((ftotal_q != RTU_LINES_BITS'(1))
                         && ((fidx_q + RTU_LINES_BITS'(1)) != ftotal_q));
    wire xform_wr = running_any && exec && (cstate_q == CS_INST_RSPN)
                 && line_ready[sel_q] && inst_last_line && !inst_culled;
    wire [11:0][31:0] xform_rd;
    VX_dp_ram #(
        .DATAW    (12*32),
        .SIZE     (NUM_CTX),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) xform_ram (
        .clk   (clk),
        .reset (reset),
        .read  (ram_rd_en),
        .write (xform_wr),
        .wren  (1'b1),
        .waddr (sel_q),
        .wdata (inst_xform_w),
        .raddr (sel),
        .rdata (xform_rd)
    );

    // ── ray setup datapath (driven by the EXEC snapshot ray). inv_d = 1/dir;
    // the box PE subtracts the ray origin itself, so there is no origin*inv_d
    // precompute (which would lose precision on axis-aligned rays where inv_d
    // is infinite). The snapshot ray is stable for the span of CS_SETUP, so the
    // fixed-latency reciprocal sees a steady input ────────────────────────
    // The shared reciprocal computes 1/dir for the world ray (CS_SETUP) or the
    // current instance's object ray (CS_OBJ_SETUP); the input is muxed so the
    // same units feed both. The snapshot dir is stable across the setup span.
    // One shared reciprocal, time-multiplexed over the 3 axes (the context stays
    // selected for the whole setup span, so the divider is fed one axis at a time
    // for its full latency). Trades 2 dividers for 2 extra setup passes.
    wire [31:0] inv_d_w;
    wire [31:0] recip_din = obj_setup ? objd_q[setupaxis_q] : ray_q.dir[setupaxis_q];
    // Reciprocal backend: LUT_NR (default) or the BRAM-seed + DSP Newton-Raphson
    // VX_rtu_recip when VX_CFG_RTU_RECIP_DSP_SEED is set (opt-in; adds DSP/BRAM,
    // saves LUT). Both honour the same fixed SETUP_LAT span.
    VX_rtu_recip #(
        .LATENCY  (RTU_FDIV_LAT),
        .DSP_SEED (`VX_CFG_RTU_RECIP_DSP_SEED)
    ) recip (
        .clk    (clk),
        .reset  (reset),
        .enable (1'b1),
        .mask   (1'b1),
        .x      (recip_din),
        .result (inv_d_w)
    );

    // ── box PE: one child per EXEC cycle while the snapshot context feeds ──
    wire        box_valid_in = exec && ((cstate_q == CS_FEED) || (cstate_q == CS_PROC_FEED));
    wire        box_valid_out, box_hit;
    wire [31:0] box_t_near;
    // Procedural-leaf raw AABB (float min/max == leaf_v0/leaf_v1) fed in raw
    // mode; internal-node child boxes stay quantized (raw=0).
    wire             box_raw    = (cstate_q == CS_PROC_FEED);
    wire [2:0][31:0] box_rawmin = leafv0_q;
    wire [2:0][31:0] box_rawmax = leafv1_q;
    // Tagged by context id, exactly as the tri and xform PEs are. Without a tag
    // the results could only be collected into whichever context happened to be
    // selected, so a context feeding the box PE had to stay selected for the
    // whole pipeline drain -- a busy-wait that burned the shared front end while
    // doing nothing, and starved every other context of it.
    wire [CTX_TAG_W+32-1:0] box_tag_out;
    VX_rtu_box_pe #(
        .TAG_WIDTH (CTX_TAG_W + 32)
    ) box_pe (
        .clk       (clk),
        .reset     (reset),
        .enable    (1'b1),
        .valid_in  (box_valid_in),
        .tag_in    ({sel_q, node_r.child_off[feed_ci]}),
        .origin    (node_r.origin),
        .exp       (node_r.exp),
        .qmin      (node_r.qmin[feed_ci]),
        .qmax      (node_r.qmax[feed_ci]),
        .raw       (box_raw),
        .raw_min   (box_rawmin),
        .raw_max   (box_rawmax),
        .ro        (walk_ro),
        .inv_d     (walk_inv_d),
        .t_min     (ray_q.t_min),
        .t_max     (bestt_q),
        .valid_out (box_valid_out),
        .tag_out   (box_tag_out),
        .hit       (box_hit),
        .t_near    (box_t_near)
    );
    // The result's own context and the child offset it was fed with. Everything
    // the collector needs rides the tag, so it reads no per-node state at all.
    wire [CTX_TAG_W-1:0] box_ctx      = box_tag_out[CTX_TAG_W+32-1 : 32];
    wire [31:0]          box_childoff = box_tag_out[31:0];
    wire coll_pushable = box_hit && (box_childoff != 32'd0);

    // ── tri PE: tagged by context id so results route back ────────────
    wire        tri_valid_in = exec && (cstate_q == CS_TRI_FEED);
    wire        tri_valid_out, tri_hit, tri_back;
    wire [CTX_TAG_W-1:0] tri_tag_out;
    wire [31:0] tri_t, tri_u, tri_v;
    VX_rtu_tri_pe #(
        .TAG_WIDTH (CTX_TAG_W)
    ) tri_pe (
        .clk         (clk),
        .reset       (reset),
        .enable      (1'b1),
        .valid_in    (tri_valid_in),
        .tag_in      (sel_q),
        .origin      (walk_ro),
        .dir         (walk_rd),
        .v0          (leafv0_q),
        .v1          (leafv1_q),
        .v2          (leafv2_q),
        .t_min       (ray_q.t_min),
        .t_max       (bestt_q),
        .valid_out   (tri_valid_out),
        .tag_out     (tri_tag_out),
        .hit         (tri_hit),
        .t           (tri_t),
        .u           (tri_u),
        .v           (tri_v),
        .back_facing (tri_back)
    );

    // ── world→object ray xform PE: tagged by context id ───────────────
    wire        xform_valid_in = exec && (cstate_q == CS_XFORM);
    wire        xform_valid_out;
    wire [CTX_TAG_W-1:0] xform_tag_out;
    wire [2:0][31:0] xform_obj_o, xform_obj_d;
    VX_rtu_xform #(
        .TAG_WIDTH (CTX_TAG_W)
    ) xform_pe (
        .clk       (clk),
        .reset     (reset),
        .enable    (1'b1),
        .valid_in  (xform_valid_in),
        .tag_in    (sel_q),
        .xform     (xform_rd),
        .ro        (ray_q.origin),
        .rd        (ray_q.dir),
        .valid_out (xform_valid_out),
        .tag_out   (xform_tag_out),
        .obj_ro    (xform_obj_o),
        .obj_rd    (xform_obj_d)
    );

    // ── memory request (single shared port, tagged by context) ────────
    wire fetch_issue = (cstate_q == CS_HDR_REQ)
                    || (cstate_q == CS_REQ0)
                    || (cstate_q == CS_REQN)
                    || (cstate_q == CS_INST_REQ)
                    || (cstate_q == CS_INST_REQN)
                    || (cstate_q == CS_LTRI_REQ0)
                    || (cstate_q == CS_LTRI_REQN)
                    ;
    assign mem_req_valid = exec && fetch_issue;
    assign mem_req_tag   = sel_q;
    wire line0_req = (cstate_q == CS_REQ0)
                  || (cstate_q == CS_INST_REQ)
                  || (cstate_q == CS_LTRI_REQ0)
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

    // The completion barrier is PER SLOT (Vulkan-Sim's Warp Status: a warp is done
    // when no thread of it is still traversing). A slot whose contexts have all
    // retired finalises and frees while its neighbours keep walking.
    wire [NUM_SLOTS-1:0] all_done;
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_all_done
        assign all_done[s] = &ctx_done[s * CTX_PER_SLOT +: CTX_PER_SLOT];
    end

    integer k;

    // Insertion index for the t-ascending child ordering: # of collected
    // entries with t <= the incoming child's t, consumed the same cycle by
    // the ordered-shift below.
    always @(*) begin
        ins_pos = RTU_CHILD_BITS'(0);
        for (integer oc = 0; oc < RTU_BVH_WIDTH; oc = oc + 1) begin
            if ((RTU_CHILD_BITS'(oc) < ord_cnt[box_ctx])
                && (ord_t[box_ctx][oc] <= box_t_near)) begin
                ins_pos = ins_pos + RTU_CHILD_BITS'(1);
            end
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            running  <= '0;
            done_r   <= '0;
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
            finalised <= '0;
        end else begin
            done_r <= '0;

            // Launch: seed one context per active lane OF THE STARTING SLOT. Slots
            // start independently, so a warp's ray enters the machine while other
            // warps are mid-walk — nothing is quiesced, and there is no bubble
            // between one trace terminating and the next beginning.
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
              if (!running[s] && start[s]) begin
                running[s]   <= 1'b1;
                finalised[s] <= 1'b0;
                // Only re-home the front end if the machine was IDLE. `phase` is
                // SHARED by every slot's contexts, so a slot starting while another
                // is mid-micro-step must not touch it — doing so aborts the EXEC of
                // whatever context is in flight.
                if (!running_any) begin
                    phase <= PH_SELECT;
                end
                for (integer j = 0; j < CTX_PER_SLOT; j = j + 1) begin
                    k = s * CTX_PER_SLOT + j;
                    mask_r[k] <= mask[k];
                    best_t[k]     <= rays[k].t_max;
                    sp[k]         <= '0;
                    cur_off[k]    <= '0;
                    setup_ctr[k]  <= '0;
                    setup_axis[k] <= '0;
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
                    hit_custom_r[k] <= '0;
                    yld_inst[k]   <= '0;
                    yld_custom[k] <= '0;
                    yld_geom[k]   <= '0;
                    yld_pending[k]<= 1'b0;
                    yld_t[k]      <= rays[k].t_max;
                    inst_count[k] <= '0;
                    inst_idx[k]   <= '0;
                    inst_id_r[k]     <= '0;
                    inst_custom_r[k] <= '0;
                    inst_flags_r[k]  <= '0;
                    leaf_tcnt[k]  <= '0;
                    leaf_tidx[k]  <= '0;
                    in_blas[k]    <= 1'b0;
                    xform_ready[k]<= 1'b0;
                    root_off_r[k] <= '0;
                    ovf_world_r[k]<= 1'b0;
                    ovf_obj_r[k]  <= 1'b0;
                    rst_world[k]  <= '0;
                    rst_obj[k]    <= '0;
                    cstate[k]     <= mask[k] ? CS_SETUP : CS_DONE;
                end
              end
            end

            // async line-fetch response → route to its context. The line data is
            // captured by g_fbuf_ram through its combinational write port (keyed on
            // mem_rsp_tag / f_slot); here we only flag the context runnable.
            if (mem_rsp_valid) begin
                line_ready[mem_rsp_tag] <= 1'b1;
            end

            // async tri-PE result → route to its context
            if (tri_valid_out) begin
                tri_ready[tri_tag_out] <= 1'b1;
                tri_hit_p[tri_tag_out] <= tri_hit;
                tri_back_p[tri_tag_out]<= tri_back;
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
                if (cstate[box_ctx] == CS_PROC_WAIT) begin
                    // procedural-leaf raw box test result -> IS yield candidate
                    proc_ready[box_ctx] <= 1'b1;
                    proc_hit_p[box_ctx] <= box_hit;
                    proc_t_p[box_ctx]   <= box_t_near;
                end else begin
                    coll_idx[box_ctx] <= coll_idx[box_ctx] + RTU_CHILD_BITS'(1);
                    // insert this child into the running t-ascending order if it hit
                    if (coll_pushable) begin
                        // shift entries at/above ins_pos up one; drop the new child in.
                        // (nonblocking RHS reads the pre-shift values -> a clean shift.)
                        for (integer oc = 0; oc < RTU_BVH_WIDTH; oc = oc + 1) begin
                            if (RTU_CHILD_BITS'(oc) == ins_pos) begin
                                ord_off[box_ctx][oc] <= box_childoff;
                                ord_t[box_ctx][oc]   <= box_t_near;
                            end else if (RTU_CHILD_BITS'(oc) > ins_pos) begin
                                ord_off[box_ctx][oc] <= ord_off[box_ctx][oc-1];
                                ord_t[box_ctx][oc]   <= ord_t[box_ctx][oc-1];
                            end
                        end
                        ord_cnt[box_ctx] <= ord_cnt[box_ctx] + RTU_CHILD_BITS'(1);
                    end
                    if (coll_idx[box_ctx] == coll_last[box_ctx]) begin
                        box_done[box_ctx] <= 1'b1;
                    end
                end
            end

            // ── micro-step pipeline ───────────────────────────────────
            // ONE front end, shared by every slot's contexts. `sel` scans the
            // runnable set across ALL of them, so a context parked on a memory
            // response is simply passed over and another slot's context executes in
            // its place. That is the GTO switch, and it costs nothing here: the
            // selector was always scanning contexts — it just never had contexts
            // from a second warp to find.
            if (running_any) begin
                if (phase == PH_SELECT) begin
                    // snapshot the selected context's working set for EXEC
                    if (sel_valid) begin
                        sel_q      <= sel;
                        cc         <= sel;
                        ray_q      <= rays[sel];
                        invd_q     <= inv_d_r[sel];
                        structaddr_q <= rays[sel].scene_base + `VX_CFG_MEM_ADDR_WIDTH'(cur_off[sel]);
                        bestt_q    <= best_t[sel];
                        cstate_q   <= cstate[sel];
                        setupctr_q <= setup_ctr[sel];
                        setupaxis_q <= setup_axis[sel];
                        fidx_q     <= f_idx[sel];
                        ftotal_q   <= f_total[sel];
                        feed_q     <= feed_idx[sel];
                        sp_q       <= sp[sel];
                        stacktop_q <= stk_rdata;
                        ordoff_q   <= ord_off[sel];
                        ordcnt_q   <= ord_cnt[sel];
                        pushptr_q  <= push_ptr[sel];
                        leafv0_q    <= leaf_v0_r[sel];
                        leafv1_q    <= leaf_v1_r[sel];
                        leafv2_q    <= leaf_v2_r[sel];
                        leaftidx_q  <= leaf_tidx[sel];
                        leaftcnt_q  <= leaf_tcnt[sel];
                        instidx_q   <= inst_idx[sel];
                        instcount_q <= inst_count[sel];
                        instbase_q  <= inst_base[sel];
                        blasroot_q  <= blas_root[sel];
                        instid_q    <= inst_id_r[sel];
                        custid_q    <= inst_custom_r[sel];
                        instflags_q <= inst_flags_r[sel];
                        rootoff_q   <= root_off_r[sel];
                        ovfw_q      <= ovf_world_r[sel];
                        ovfo_q      <= ovf_obj_r[sel];
                        rstw_q      <= rst_world[sel];
                        rsto_q      <= rst_obj[sel];
                        objo_q      <= obj_o[sel];
                        objd_q      <= obj_d[sel];
                        objinvd_q   <= obj_inv_d_r[sel];
                        blasfloor_q <= blas_floor[sel];
                        inblas_q    <= in_blas[sel];
                        phase      <= sel_needs_img ? PH_ALIGN : PH_EXEC;
                    end
                end else if (phase == PH_ALIGN) begin
                    // ALIGN: register the BlockRAM node image into a fabric flop so
                    // the EXEC byte-align shift starts from a fast FF rather than the
                    // slower BlockRAM read output (the f_buf-BRAM critical path).
                    fbuf_q <= fbuf;
                    phase  <= PH_EXEC;
                end else begin
                    // EXEC: advance the snapshot context, write back results
                    phase <= PH_SELECT;
                    case (cstate_q)
                    CS_SETUP: begin
                        if (setupctr_q != SETUP_CW'(SETUP_LAT)) begin
                            setup_ctr[sel_q] <= setupctr_q + SETUP_CW'(1);
                        end else begin
                            inv_d_r[sel_q][setupaxis_q] <= inv_d_w;
                            setup_ctr[sel_q]            <= '0;
                            if (setupaxis_q == 2'd2) begin
                                setup_axis[sel_q] <= 2'd0;
                                cstate[sel_q]     <= CS_HDR_REQ;
                            end else begin
                                setup_axis[sel_q] <= setupaxis_q + 2'd1;
                            end
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
                            cur_off[sel_q]    <= f_aligned[RTU_SCENE_OFF_ROOT*8 +: 32];
                            root_off_r[sel_q] <= f_aligned[RTU_SCENE_OFF_ROOT*8 +: 32];
                            cstate[sel_q]     <= CS_REQ0;
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
                        // Latch the fully-assembled decoded node once, for the span
                        // in which this context owns the machine: CS_DISPATCH ->
                        // CS_FEED. It parks at CS_WAIT, and from there on neither
                        // the collector nor CS_PUSH reads node_r -- the child offset
                        // rides the box-PE tag and the ordering array holds offsets.
                        node_r <= node;
                        if (node_kind == RTU_KIND_INTERNAL && node.n_children != '0) begin
                            feed_idx[sel_q]  <= '0;
                            coll_idx[sel_q]  <= '0;
                            ord_cnt[sel_q]   <= '0;
                            box_done[sel_q]  <= 1'b0;
                            // This node's child count, kept per context: the
                            // collector counts results against it once the context
                            // has parked and no longer owns node_r.
                            coll_last[sel_q] <= node.n_children - RTU_CHILD_BITS'(1);
                            cstate[sel_q]    <= CS_FEED;
                        end else if (node_kind == RTU_KIND_LEAF_TRI) begin
                            // Fat-leaf: a LEAF_TRI packs `count` triangles.
                            // Latch the leaf header (geometry + prim_base) and set
                            // up the per-triangle re-fetch loop from the first
                            // record, iterating all `count` tris.
                            leaf_geom_r[sel_q] <= leaf_geom;
                            leaf_prim_r[sel_q] <= leaf_prim;
                            leaf_tcnt[sel_q]   <= leaf_tri_count;
                            leaf_tidx[sel_q]   <= 8'd0;
                            if (leaf_tri_count == 8'd0) begin
                                cstate[sel_q]  <= CS_POP;
                            end else begin
                                cur_off[sel_q] <= cur_off[sel_q] + 32'(RTU_LEAF_HDR_BYTES);
                                cstate[sel_q]  <= CS_LTRI_REQ0;
                            end
                        end else if (node_kind == RTU_KIND_LEAF_PROC) begin
                            leaf_geom_r[sel_q] <= leaf_geom;
                            leaf_prim_r[sel_q] <= leaf_prim;
                            leaf_v0_r[sel_q]   <= leaf_v0;
                            leaf_v1_r[sel_q]   <= leaf_v1;
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
                                yld_geom[sel_q]    <= leaf_geom_r[sel_q];
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
                            // cursor starts at the farthest sibling; 0 when <=1 hit
                            // child (nothing to push, descend the nearest directly).
                            push_ptr[sel_q] <= (ord_cnt[sel_q] == RTU_CHILD_BITS'(0))
                                             ? RTU_CHILD_BITS'(0)
                                             : (ord_cnt[sel_q] - RTU_CHILD_BITS'(1));
                            cstate[sel_q]   <= CS_PUSH;
                        end
                    end
                    CS_PUSH: begin
                        // Walk push_ptr farthest->1, pushing one sibling per cycle
                        // (node_stack_ram writes ord_off[push_ptr] via stk_wr); when the
                        // siblings are exhausted, descend the NEAREST hit child
                        // (ord_off[0]) directly as the current node (it never rides
                        // the short stack, so an overflow can't drop the nearest path).
                        if (push_active) begin
                            if (sp_q != RTU_STACK_BITS'(RTU_STACK_DEPTH)) begin
                                sp[sel_q] <= sp_q + RTU_STACK_BITS'(1);
                            end else begin
                                // short stack full: this sibling subtree is dropped —
                                // flag a level-scoped overflow so CS_POP re-descends.
                                if (inblas_q) begin
                                    ovf_obj_r[sel_q] <= 1'b1;
                                end else begin
                                    ovf_world_r[sel_q] <= 1'b1;
                                end
                            end
                            push_ptr[sel_q] <= pushptr_q - RTU_CHILD_BITS'(1);
                        end else if (near_found_q) begin
                            cur_off[sel_q] <= ordoff_q[0] & RTU_CHILD_OFF_MASK;
                            cstate[sel_q]  <= CS_REQ0;
                        end else begin
                            // no child's AABB was entered: nothing to descend.
                            cstate[sel_q] <= CS_POP;
                        end
                    end
                    CS_TRI_FEED: begin
                        cstate[sel_q] <= CS_TRI_WAIT;
                    end
                    CS_TRI_WAIT: begin
                        if (tri_ready[sel_q]) begin
                            tri_ready[sel_q] <= 1'b0;
                            if (tri_committable) begin
                                if (tri_opaque) begin
                                    // commit the closest opaque hit.
                                    best_t[sel_q]     <= tri_t_p[sel_q];
                                    hit_r[sel_q]      <= 1'b1;
                                    hit_t_r[sel_q]    <= tri_t_p[sel_q];
                                    hit_u_r[sel_q]    <= tri_u_p[sel_q];
                                    hit_v_r[sel_q]    <= tri_v_p[sel_q];
                                    hit_prim_r[sel_q] <= leaf_prim_r[sel_q] + 32'(leaftidx_q);
                                    hit_geom_r[sel_q] <= leaf_geom_r[sel_q];
                                    // a BLAS hit carries its instance's id; a
                                    // top-level (non-instanced) tri reports 0.
                                    hit_inst_r[sel_q] <= inblas_q ? instid_q : 32'd0;
                                    hit_custom_r[sel_q] <= inblas_q ? custid_q : 32'd0;
                                    // a closer opaque hit occludes a farther candidate.
                                    if (yld_pending[sel_q] && (yld_t[sel_q] >= tri_t_p[sel_q])) begin
                                        yld_pending[sel_q] <= 1'b0;
                                    end
                                end else begin
                                    // non-opaque: stage the closest any-hit candidate.
                                    if (!yld_pending[sel_q] || (tri_t_p[sel_q] < yld_t[sel_q])) begin
                                        yld_pending[sel_q] <= 1'b1;
                                        yld_t[sel_q]       <= tri_t_p[sel_q];
                                        yld_u[sel_q]       <= tri_u_p[sel_q];
                                        yld_v[sel_q]       <= tri_v_p[sel_q];
                                        yld_prim[sel_q]    <= leaf_prim_r[sel_q] + 32'(leaftidx_q);
                                        yld_geom[sel_q]    <= leaf_geom_r[sel_q];
                                        yld_inst[sel_q]    <= inblas_q ? instid_q : 32'd0;
                                        yld_custom[sel_q]  <= inblas_q ? custid_q : 32'd0;
                                        yld_cbtype[sel_q]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_ANYHIT);
                                        yld_sbt[sel_q]     <= cls_sbt;
                                    end
                                end
                            end
                            // opaque TERMINATE_ON_FIRST_HIT stops this lane's walk;
                            // otherwise advance the fat-leaf triangle loop:
                            // fetch the next record if any remain, else pop.
                            if (tri_committable && tri_opaque && term_first) begin
                                cstate[sel_q] <= CS_DONE;
                            end else if ((leaftidx_q + 8'd1) < leaftcnt_q) begin
                                leaf_tidx[sel_q] <= leaftidx_q + 8'd1;
                                cur_off[sel_q]   <= cur_off[sel_q] + 32'(RTU_TRI_STRIDE);
                                cstate[sel_q]    <= CS_LTRI_REQ0;
                            end else begin
                                cstate[sel_q] <= CS_POP;
                            end
                        end
                    end
                    // ── fat-leaf triangle record fetch (40 B, <=2 lines) ──
                    CS_LTRI_REQ0: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            f_total[sel_q]    <= tri_rec_lines;
                            cstate[sel_q]     <= CS_LTRI_RSP0;
                        end
                    end
                    CS_LTRI_RSP0: begin
                        if (line_ready[sel_q]) begin
                            if (ftotal_q == RTU_LINES_BITS'(1)) begin
                                tri_flags_p[sel_q] <= ltri_flags;
                                leaf_v0_r[sel_q]   <= ltri_v0;
                                leaf_v1_r[sel_q]   <= ltri_v1;
                                leaf_v2_r[sel_q]   <= ltri_v2;
                                cstate[sel_q]      <= CS_TRI_FEED;
                            end else begin
                                f_idx[sel_q]  <= RTU_LINES_BITS'(1);
                                cstate[sel_q] <= CS_LTRI_REQN;
                            end
                        end
                    end
                    CS_LTRI_REQN: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= fidx_q;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_LTRI_RSPN;
                        end
                    end
                    CS_LTRI_RSPN: begin
                        if (line_ready[sel_q]) begin
                            if ((fidx_q + RTU_LINES_BITS'(1)) == ftotal_q) begin
                                tri_flags_p[sel_q] <= ltri_flags;
                                leaf_v0_r[sel_q]   <= ltri_v0;
                                leaf_v1_r[sel_q]   <= ltri_v1;
                                leaf_v2_r[sel_q]   <= ltri_v2;
                                cstate[sel_q]      <= CS_TRI_FEED;
                            end else begin
                                f_idx[sel_q]  <= fidx_q + RTU_LINES_BITS'(1);
                                cstate[sel_q] <= CS_LTRI_REQN;
                            end
                        end
                    end
                    CS_POP: begin
                        if (inblas_q && (sp_q == blasfloor_q)) begin
                            if (ovfo_q && (rsto_q != RST_CNTW'(RTU_RESTART_CAP))) begin
                                // a BLAS(object)-level subtree was dropped;
                                // re-descend the BLAS root pruning by the tightened
                                // best_t, charging the per-BLAS restart budget.
                                ovf_obj_r[sel_q] <= 1'b0;
                                rst_obj[sel_q]   <= rsto_q + RST_CNTW'(1);
                                cur_off[sel_q]   <= blasroot_q;
                                cstate[sel_q]    <= CS_REQ0;   // sp stays at floor
                            end else begin
                                // BLAS subtree drained back to the instance-loop
                                // floor: resume the instance loop in world space.
                                // A pending WORLD overflow is left intact for the
                                // top-level restart.
                                cstate[sel_q] <= CS_INST_NEXT;
                            end
                        end else
                        if (sp_q == '0) begin
                            if (ovfw_q && (rstw_q != RST_CNTW'(RTU_RESTART_CAP))) begin
                                // a WORLD(TLAS)-level subtree was dropped;
                                // re-descend from scene root pruning by best_t,
                                // charging the world restart budget.
                                ovf_world_r[sel_q] <= 1'b0;
                                rst_world[sel_q]   <= rstw_q + RST_CNTW'(1);
                                cur_off[sel_q]     <= rootoff_q;
                                cstate[sel_q]      <= CS_REQ0;   // sp stays 0
                            end else begin
                                cstate[sel_q] <= CS_DONE;
                            end
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
                                // cull gate: skip transform + BLAS descent.
                                cstate[sel_q] <= CS_INST_NEXT;
                            end else begin
                                // inst_xform is captured by xform_ram (xform_wr) this cycle.
                                blas_root[sel_q]  <= inst_blas;
                                inst_id_r[sel_q]  <= inst_id;
                                inst_custom_r[sel_q] <= inst_custom;
                                inst_flags_r[sel_q]  <= inst_flags;
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
                            setup_ctr[sel_q]  <= '0;
                            setup_axis[sel_q] <= 2'd0;
                            cstate[sel_q]     <= CS_OBJ_SETUP;
                        end
                    end
                    CS_OBJ_SETUP: begin
                        if (setupctr_q != SETUP_CW'(SETUP_LAT)) begin
                            setup_ctr[sel_q] <= setupctr_q + SETUP_CW'(1);
                        end else begin
                            obj_inv_d_r[sel_q][setupaxis_q] <= inv_d_w;
                            setup_ctr[sel_q]                <= '0;
                            if (setupaxis_q == 2'd2) begin
                                setup_axis[sel_q] <= 2'd0;
                                // enter the BLAS subtree under the object ray.
                                in_blas[sel_q]    <= 1'b1;
                                cur_off[sel_q]    <= blasroot_q;
                                // fresh per-BLAS restart budget + overflow
                                // marker so a deep multi-instance ray gets a full
                                // budget in every instance (not one global pool).
                                rst_obj[sel_q]    <= '0;
                                ovf_obj_r[sel_q]  <= 1'b0;
                                cstate[sel_q]     <= CS_REQ0;
                            end else begin
                                setup_axis[sel_q] <= setupaxis_q + 2'd1;
                            end
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

            // ── post-walk callback yield barrier, PER SLOT ─────────────
            // Each slot barriers on its OWN contexts, so one warp's callback round
            // trip through the shader does not hold another warp's walk.
            for (integer s = 0; s < NUM_SLOTS; s = s + 1) begin
              if (running[s] && all_done[s]) begin
                if (!finalised[s]) begin
                    // Finalise: CHS (committed hit + ENABLE_CHS) or MISS
                    // (no hit + ENABLE_MISS) for lanes without a candidate yield.
                    for (integer j = 0; j < CTX_PER_SLOT; j = j + 1) begin
                        k = s * CTX_PER_SLOT + j;
                        if (mask_r[k] && !yld_pending[k]) begin
                            if (hit_r[k] && ((rays[k].flags & 32'(`VX_RT_FLAG_ENABLE_CHS)) != 0)
                                         && ((rays[k].flags & 32'(`VX_RT_FLAG_SKIP_CLOSEST_HIT)) == 0)) begin
                                yld_pending[k] <= 1'b1;
                                yld_cbtype[k]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_CHS);
                                yld_t[k] <= hit_t_r[k]; yld_u[k] <= hit_u_r[k];
                                yld_v[k] <= hit_v_r[k]; yld_prim[k] <= hit_prim_r[k];
                                // Stage the committed hit's instance/geometry attributes
                                // so the CHS reads the right gl_Instance*/gl_GeometryIndex
                                // and a CHS accept re-commits them unchanged.
                                yld_inst[k] <= hit_inst_r[k]; yld_custom[k] <= hit_custom_r[k];
                                yld_geom[k] <= hit_geom_r[k];
                            end else if (!hit_r[k] && ((rays[k].flags & 32'(`VX_RT_FLAG_ENABLE_MISS)) != 0)) begin
                                yld_pending[k] <= 1'b1;
                                yld_cbtype[k]  <= RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_MISS);
                                // A miss carries no instance/geometry.
                                yld_inst[k] <= '0; yld_custom[k] <= '0;
                                yld_geom[k] <= '0;
                            end
                        end
                    end
                    finalised[s] <= 1'b1;
                end else if (|yld_pending[s * CTX_PER_SLOT +: CTX_PER_SLOT]) begin
                    if (resume[s]) begin
                        for (integer j = 0; j < CTX_PER_SLOT; j = j + 1) begin
                            k = s * CTX_PER_SLOT + j;
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
                                    // Commit the accepted candidate's instance
                                    // attributes so the post-wait / CHS read
                                    // reports the accepted instance. The CHS
                                    // finalise below stages yld_inst/yld_custom
                                    // from the committed hit, so a CHS accept
                                    // writes them back unchanged.
                                    hit_inst_r[k]   <= yld_inst[k];
                                    hit_custom_r[k] <= yld_custom[k];
                                    // accepted candidate's geometry becomes committed.
                                    hit_geom_r[k]   <= yld_geom[k];
                                end
                                yld_pending[k] <= 1'b0;
                            end
                        end
                    end
                end else begin
                    running[s] <= 1'b0;
                    done_r[s]  <= 1'b1;
                end
              end
            end
        end
    end

`ifdef DBG_TRACE_RTU
    always_ff @(posedge clk) begin
        if (exec && (cstate_q == CS_DISPATCH)) begin
            `TRACE(2, ("%t: %s rtu-node: ctx=%0d, off=%0d, kind=%0d, children=%0d\n",
                $time, INSTANCE_ID, sel_q, structaddr_q, node_kind, node.n_children))
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

    // While at the yield barrier, present the candidate attrs (CB_YIELD payload)
    // on res_* for the yielding lanes; otherwise the committed hit.
    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_yield
        assign yield[s] = running[s] && all_done[s] && finalised[s]
                       && (| yld_pending[s * CTX_PER_SLOT +: CTX_PER_SLOT]);
    end
    assign yield_mask   = yld_pending;
    assign yield_cbtype = yld_cbtype;
    assign yield_sbt    = yld_sbt;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_res
        wire cand_i = yield[i / CTX_PER_SLOT] && yld_pending[i];
        assign res_hit[i]  = hit_r[i];
        assign res_t[i]    = cand_i ? yld_t[i]    : hit_t_r[i];
        assign res_u[i]    = cand_i ? yld_u[i]    : hit_u_r[i];
        assign res_v[i]    = cand_i ? yld_v[i]    : hit_v_r[i];
        assign res_prim[i] = cand_i ? yld_prim[i] : hit_prim_r[i];
        assign res_geom[i] = cand_i ? yld_geom[i] : hit_geom_r[i];
        assign res_inst[i] = cand_i ? yld_inst[i] : hit_inst_r[i];
        assign res_custom[i] = cand_i ? yld_custom[i] : hit_custom_r[i];
    end

    assign busy = running;
    assign done = done_r;

endmodule
