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

`include "VX_define.vh"

// Dispatches warps for incoming CTAs from the KMU. Warps launch on NUM_LANES
// lanes in parallel: lane l owns the warps with wid % NUM_LANES == l, and each
// lane launches one of its free warps per cycle until the CTA is fully launched.
module VX_cta_dispatch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input wire                      clk,
    input wire                      reset,

    // from KMU
    VX_kmu_bus_if.slave             kmu_bus_if,

    // from scheduler
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    input wire                      warp_done,
    input wire [NW_WIDTH-1:0]       warp_done_wid,

    // to scheduler (one warp per lane per cycle)
    output wire [NUM_LANES-1:0]     cta_fire,
    output wire [NUM_LANES-1:0][NW_WIDTH-1:0] cta_wid,
    output wire [PC_BITS-1:0]       cta_PC,
    output wire [NUM_LANES-1:0][`VX_CFG_NUM_THREADS-1:0] cta_tmask,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param,
    output wire [NUM_LANES-1:0]     cta_init,

    // CTA state for the CTA-CSR stores (see VX_csr_cta): the CTA context,
    // and each lane's warp records
    output cta_bcast_t              cta_bcast,
    output cta_lane_bcast_t [NUM_LANES-1:0] cta_lane_bcast,

    // register-file layout of the running kernel, to the operand collectors
    output gpr_layout_t             gpr_layout,

    output wire                     busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT ((`VX_CFG_NUM_WARPS % NUM_LANES) == 0, ("invalid parameter"))
    localparam NUM_CTA_SLOTS= `VX_CFG_NUM_WARPS;
    localparam LANE_WARPS   = `VX_CFG_NUM_WARPS / NUM_LANES;
    localparam LANE_WIDTH   = `LOG2UP(NUM_LANES);
    localparam LCNT_WIDTH   = `CLOG2(NUM_LANES+1);
    localparam CS_BITS      = NW_WIDTH; // UP(NW_BITS): at least 1 to avoid zero-width when NUM_WARPS=1
    localparam LMEM_SIZE    = `VX_CFG_LMEM_SIZE;

    // -------------------------------------------------------------------------
    // CTA table — fixed-stride slots (no in-order reclaim).
    // Per-slot remaining-warp counts live in rem_warps_ram; occupancy is mirrored
    // in slot_valid_r to keep it off the table read path. Slots are allocated
    // round-robin and freed immediately (out of order) when a CTA's last warp
    // retires.
    // -------------------------------------------------------------------------

    wire                    rem_warps_read;
    wire                    rem_warps_write;
    wire [CS_BITS-1:0]      rem_warps_waddr;
    wire [NW_WIDTH:0]       rem_warps_wdata;
    wire [CS_BITS-1:0]      rem_warps_raddr;
    wire [NW_WIDTH:0]       rem_warps_rdata;

    VX_dp_ram #(
        .DATAW (NW_WIDTH+1),
        .SIZE  (NUM_CTA_SLOTS),
        .RDW_MODE ("R"),
        .OUT_REG (1)
    ) rem_warps_ram (
        .clk   (clk),
        .reset (reset),
        .wren  (1'b1),
        .read  (rem_warps_read),
        .write (rem_warps_write),
        .waddr (rem_warps_waddr),
        .wdata (rem_warps_wdata),
        .raddr (rem_warps_raddr),
        .rdata (rem_warps_rdata)
    );

    reg [NUM_CTA_SLOTS-1:0] slot_valid_r;          // per-slot occupancy mirror
    reg [CS_BITS-1:0]       tail_r;                // round-robin slot pointer

    // Fixed-stride LMEM partition: resident CTA in slot i gets LMEM base
    // i * stride, where stride = aligned_lmem_size (uniform within a kernel).
    // The product is formed from the slot and stride latched at accept, a cycle
    // later; its first reader, the context write of the first launched warp,
    // comes at least two cycles after accept.
    reg [`VX_CFG_LMEM_LOG_SIZE:0]   cur_stride_r;
    reg [`VX_CFG_LMEM_LOG_SIZE-1:0] cur_lmem_base_r;

    // Reverse lookup: warp-ID → CTA slot index. A flop array indexed by wid,
    // updated on each warp dispatch. Combinational read on warp_done lets the
    // retirement path skip the registered raddr that a DP-RAM would require.
    reg [`VX_CFG_NUM_WARPS-1:0][CS_BITS-1:0] cta_slot_per_warp_r;
    wire [CS_BITS-1:0] done_slot = cta_slot_per_warp_r[warp_done_wid];

    // Registered retirement signals. The pipeline holds two stages: warp_done_r
    // captures the retirement event, then warp_done_r_dly aligns with the
    // rem_warps_ram rdata (OUT_REG=1) for cta_done evaluation.
    reg                 warp_done_r;
    reg                 warp_done_r_dly;
    reg [CS_BITS-1:0]   done_slot_r;
    reg [CS_BITS-1:0]   done_slot_r_dly;

    // Per-lane startup record: the stub derives each lane's stack from its hart id,
    // so a slot relaunched with lanes it has not run before must run it again.
    reg [7:0]           cur_ctx_id_r;
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_THREADS-1:0] warp_init_lanes_r;

    // -------------------------------------------------------------------------
    // FSM + per-dispatch registers
    // -------------------------------------------------------------------------
    typedef enum logic { IDLE = 0, DISPATCH = 1 } state_t;
    state_t state;

    reg [PC_BITS-1:0]               warp_PC;
    reg [PC_BITS-1:0]               entry_r;
    reg [2:0][31:0]                 block_idx_r;
    reg [2:0][CTA_TID_WIDTH:0]      block_dim_r;
    reg [1:0][CTA_TID_WIDTH-1:0]    block_last_r;
    reg [2:0][31:0]                 grid_dim_r;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] param_r;
    reg [CTA_TID_WIDTH:0]           block_size_r;
    reg [2:0][CTA_TID_WIDTH-1:0]    warp_step_r;
    reg [NW_WIDTH:0]                cluster_size_r;
    reg [NW_WIDTH-1:0]              cta_rank_r;
    reg [CS_BITS-1:0]               cur_slot_r;

    // Warps launched last cycle, one per lane; `order` is the warp's position
    // among them, so its CTA rank is cta_rank_r + order.
    reg [NUM_LANES-1:0]                     warp_fire_r;
    reg [NUM_LANES-1:0][NW_WIDTH-1:0]       warp_id_r;
    reg [NUM_LANES-1:0][`VX_CFG_NUM_THREADS-1:0] warp_tmask_r;
    reg [NUM_LANES-1:0][LANE_WIDTH-1:0]     warp_order_r;
    reg [NUM_LANES-1:0]                     warp_skip_init_r;
    reg [LCNT_WIDTH-1:0]                    nfired_r;
    wire                                    fire_any_r = (| warp_fire_r);

    // Thread-index window: thread_idx_r[k] is the base thread index of the warp
    // ranked cta_rank_r + k. Launching n warps shifts the window by n; the entries
    // shifted in advance by group_step_r (NUM_LANES warps), so no two advances
    // chain within a cycle.
    reg [NUM_LANES-1:0][2:0][CTA_TID_WIDTH-1:0] thread_idx_r;
    reg [2:0][CTA_TID_WIDTH-1:0]            group_step_r;

    // -------------------------------------------------------------------------
    // Free-warp selection: one candidate per lane
    // -------------------------------------------------------------------------
    reg  [`VX_CFG_NUM_WARPS-1:0] dispatched_warps;
    reg  [`VX_CFG_NUM_WARPS-1:0] warp_fits_r;
    wire [`VX_CFG_NUM_WARPS-1:0] free_warps = ~(active_warps | dispatched_warps) & warp_fits_r;
    wire [NUM_LANES-1:0]               lane_ready;
    wire [NUM_LANES-1:0][NW_WIDTH-1:0] lane_wid;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lane_select
        wire [LANE_WARPS-1:0] lane_free;
        for (genvar j = 0; j < LANE_WARPS; ++j) begin : g_free
            assign lane_free[j] = free_warps[j * NUM_LANES + l];
        end
        wire [`LOG2UP(LANE_WARPS)-1:0] lane_idx;
        VX_priority_encoder #(
            .N       (LANE_WARPS),
            .REVERSE (0)
        ) priority_enc (
            .data_in   (lane_free),
            `UNUSED_PIN(onehot_out),
            .index_out (lane_idx),
            .valid_out (lane_ready[l])
        );
        assign lane_wid[l] = NW_WIDTH'(lane_idx) * NW_WIDTH'(NUM_LANES) + NW_WIDTH'(l);
    end

    wire kmu_bus_if_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    // A launch is one beat. The routing sidebands (kind/eop/dest) are consumed
    // by the fan-out arbiters; the endpoint discriminates via kmu_req.kind.
    kmu_req_t kmu_req;
    assign kmu_req = kmu_req_t'(kmu_bus_if.data);
    `UNUSED_VAR (kmu_bus_if.kind)
    `UNUSED_VAR (kmu_bus_if.dest)
    `UNUSED_VAR (kmu_bus_if.eop)

    // A fragment is not a CTA: it carries only args.fragment, and the compute-descriptor
    // fields (grid geometry, cluster, block_size) read as stamp bits for a fragment. So the
    // consumer supplies a fragment's constants from `kind` -- grid [1,1,1], block [0,0,0], a single
    // warp of count*FRAG_QUAD_LANES active lanes, admitted to a single slot (cluster 1).
    // When RASTER is off, `kind` is always COMPUTE and these are the plain compute reads.
    wire kmu_is_frag = (kmu_req.kind == KMU_KIND_FRAGMENT);
    wire [2:0][31:0] kmu_grid_dim  = kmu_is_frag ? {3{32'd1}} : kmu_req.args.compute.grid_dim;
    wire [2:0][31:0] kmu_block_idx = kmu_is_frag ? '0         : kmu_req.args.compute.block_idx;

`ifdef VX_CFG_EXT_RASTER_ENABLE
    wire [CTA_TID_WIDTH:0] eff_block_size = kmu_is_frag
        ? (CTA_TID_WIDTH+1)'(kmu_req.args.fragment.count) * (CTA_TID_WIDTH+1)'(FRAG_QUAD_LANES)
        : kmu_req.args.compute.block_size;
    wire [NW_WIDTH:0]      eff_cluster_size = kmu_is_frag ? (NW_WIDTH+1)'(1) : kmu_req.args.compute.cluster_size;
    wire                   eff_is_first     = kmu_is_frag ? 1'b1              : kmu_req.args.compute.is_first_of_cluster;
`else
    wire [CTA_TID_WIDTH:0] eff_block_size   = kmu_req.args.compute.block_size;
    wire [NW_WIDTH:0]      eff_cluster_size = kmu_req.args.compute.cluster_size;
    wire                   eff_is_first     = kmu_req.args.compute.is_first_of_cluster;
`endif

    // Fragment-launch signals, defaulted so their consumers stay unconditional (the
    // RASTER-only logic that drives them lives in one place, below). `frag_inflight`
    // holds the header off while an earlier fragment warp is still in the thread-index
    // pipeline; `is_frag_warp`/`frag_stamps_sel` pick the lane-record view being
    // written. All benign (0) in a build with no rasterizer.
    wire frag_inflight;
    wire [`VX_CFG_NUM_THREADS-1:0][FRAG_LANE_BITS-1:0] frag_stamps_sel;

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // The wave's stamps, taken from the launch header. The lane record is an overlay
    // (see cta_warp_t), so the write side must also know WHICH view to land: a
    // fragment launch lands a stamp slice, a compute launch its expanded thread
    // index. Both are held with the header and are stable until the next one.
    reg [`VX_CFG_NUM_THREADS-1:0][FRAG_LANE_BITS-1:0] frag_stamps_r;
    reg is_frag_r;

    always @(posedge clk) begin
        if ((state == IDLE) && kmu_bus_if_fire) begin
            frag_stamps_r <= kmu_req.args.fragment.stamps;   // the fragment's stamps
            is_frag_r     <= kmu_is_frag;
        end
    end
`endif

    // -------------------------------------------------------------------------
    // Power-of-two NUM_THREADS arithmetic — all combinational, zero adders
    // -------------------------------------------------------------------------
    wire [NW_WIDTH:0]      kmu_num_warps;
    wire [CTA_TID_WIDTH:0] block_size_next;
    wire [`VX_CFG_NUM_THREADS-1:0] partial_tmask;
    wire                   part_warp_r;

    if (NT_BITS > 0) begin : g_nt_nonzero
        // Ceiling division block_size / NUM_THREADS: upper bits + OR of lower bits.
        // From KMU data at accept time (used to initialise table + cta_size output)
        assign kmu_num_warps = (NW_WIDTH+1)'(eff_block_size[CTA_TID_WIDTH:NT_BITS]) + (NW_WIDTH+1)'(|eff_block_size[NT_BITS-1:0]);
        // Shared block_size decrement: low NT_BITS bits unchanged; upper bits drop by
        // the warps launched.
        assign block_size_next = {block_size_r[CTA_TID_WIDTH:NT_BITS] - (CTA_TID_WIDTH+1-NT_BITS)'(nfired_r), block_size_r[NT_BITS-1:0]};
        // Partial-warp mask: (1 << count) - 1 where count = block_size_r[NT_BITS-1:0],
        // which launches never change.
        assign partial_tmask = (`VX_CFG_NUM_THREADS'(1) << block_size_r[NT_BITS-1:0]) - `VX_CFG_NUM_THREADS'(1);
        assign part_warp_r = (| block_size_r[NT_BITS-1:0]);
    end else begin : g_nt_zero
        // NT_BITS=0: NUM_THREADS=1, each warp has exactly 1 thread, no partial warps.
        assign kmu_num_warps = (NW_WIDTH+1)'(eff_block_size);
        assign block_size_next = (CTA_TID_WIDTH+1)'(block_size_r - (CTA_TID_WIDTH+1)'(nfired_r));
        assign partial_tmask = `VX_CFG_NUM_THREADS'(0);
        assign part_warp_r = 1'b0;
    end

    // -------------------------------------------------------------------------
    // Register-file occupancy. Warps interleave across the GPR_NC operand
    // collectors, so warp w takes register set w / GPR_NC of its collector and
    // fits iff (w / GPR_NC + 1) * R <= GPR_ROWS, tested against a constant per w.
    // A CTA's own warps are always granted, as a CTA that cannot launch all of
    // them would never finish. Latched at accept: it holds while the CTA's warps
    // launch, whatever the bus carries then.
    // -------------------------------------------------------------------------
    localparam GPR_NC   = `VX_CFG_ISSUE_WIDTH * `VX_CFG_NUM_OPCS;
    localparam GPR_ROWS = `VX_CFG_GPR_ROWS;
    `STATIC_ASSERT ((GPR_ROWS >= NUM_REGS) && ((GPR_ROWS % 4) == 0), ("invalid VX_CFG_GPR_ROWS"))
    wire       kreq_all_regs = (kmu_req.num_regs == '0);
    wire [8:0] kreq_rx = kreq_all_regs ? 9'(RV_REGS) : 9'(kmu_req.num_regs[7:0]);
    wire [8:0] kreq_r  = kreq_all_regs ? 9'(NUM_REGS) : (9'(kmu_req.num_regs[7:0]) + 9'(kmu_req.num_regs[15:8]));
    wire [`VX_CFG_NUM_WARPS-1:0] warp_fits;
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_warp_fits
        assign warp_fits[w] = (32'(kreq_r) <= 32'(GPR_ROWS / (w / GPR_NC + 1)))
                           || ((NW_WIDTH+1)'(w) < kmu_num_warps);
    end

    gpr_layout_t gpr_layout_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            warp_fits_r          <= '1;
            gpr_layout_r.size4   <= GPR_SET_W'(NUM_REGS / 4);
            gpr_layout_r.fp_off4 <= GPR_SET_W'(RV_REGS / 4);
        end else if ((state == IDLE) && kmu_bus_if_fire) begin
            warp_fits_r          <= warp_fits;
            gpr_layout_r.size4   <= GPR_SET_W'(kreq_r >> 2);
            gpr_layout_r.fp_off4 <= GPR_SET_W'(kreq_rx >> 2);
        end
    end
    assign gpr_layout = gpr_layout_r;

    // Warps still to launch, after those latched last cycle, compared with each
    // launch position k as thermometers: block_size_r retires the warps latched
    // last cycle a cycle late, so excluding them shifts the thermometers of its
    // own counts by nfired_r. full_gt[k]: more than k full warps are left;
    // rem_gt[k]: more than k warps are left.
    localparam FULL_W = CTA_TID_WIDTH + 1 - NT_BITS;
    wire [FULL_W-1:0]        full_warps_r = block_size_r[CTA_TID_WIDTH:NT_BITS];
    wire [2*NUM_LANES-1:0]   full_r_gt, rem_r_gt;
    for (genvar j = 0; j < 2*NUM_LANES; ++j) begin : g_rem_therm
        // a block holds at most 2^FULL_W-1 full warps
        if (j < (1 << FULL_W) - 1) begin : g_full_cmp
            assign full_r_gt[j] = (full_warps_r > FULL_W'(j));
        end else begin : g_full_never
            assign full_r_gt[j] = 1'b0;
        end
        if (j == 0) begin : g_first
            assign rem_r_gt[j] = full_r_gt[j] | part_warp_r;
        end else begin : g_rest
            assign rem_r_gt[j] = full_r_gt[j] | (part_warp_r & full_r_gt[j-1]);
        end
    end
    wire [LCNT_WIDTH-1:0] nfired_cur = fire_any_r ? nfired_r : '0;
    wire [NUM_LANES-1:0]  full_gt = NUM_LANES'(full_r_gt >> nfired_cur);
    wire [NUM_LANES-1:0]  rem_gt  = NUM_LANES'(rem_r_gt >> nfired_cur);

    // Launch selection: lanes with a free warp, taken in round-robin order from
    // lane_rr_r, up to the warps left. A lane's order is its position among the
    // selected lanes: the ready lanes ahead of it in the rotation, so a ready lane
    // is selected while fewer than the warps left are ahead of it.
    `STATIC_ASSERT (`IS_POW2(NUM_LANES), ("invalid parameter"))
    reg  [LANE_WIDTH-1:0] lane_rr_r;
    wire                  launch_en;
    wire [NUM_LANES-1:0]                 lane_sel;
    wire [NUM_LANES-1:0][LANE_WIDTH-1:0] lane_order;
    wire [LCNT_WIDTH-1:0]                lane_cnt;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lane_sel
        wire [NUM_LANES-1:0] ahead;
        for (genvar m = 0; m < NUM_LANES; ++m) begin : g_ahead
            assign ahead[m] = (LANE_WIDTH'(m) - lane_rr_r) < (LANE_WIDTH'(l) - lane_rr_r);
        end
        wire [NUM_LANES-1:0]  ready_ahead = lane_ready & ahead;
        wire [LCNT_WIDTH-1:0] order;
        VX_popcount #(
            .N (NUM_LANES)
        ) order_count (
            .data_in  (ready_ahead),
            .data_out (order)
        );
        assign lane_order[l] = LANE_WIDTH'(order);
        assign lane_sel[l]   = launch_en && lane_ready[l] && 1'(rem_gt >> order);
    end
    VX_popcount #(
        .N (NUM_LANES)
    ) lane_count (
        .data_in  (lane_sel),
        .data_out (lane_cnt)
    );

    // A warp is full unless it is the CTA's last and the block size is not a
    // multiple of the warp width.
    wire [NUM_LANES-1:0][`VX_CFG_NUM_THREADS-1:0] lane_tmask;
    wire [NUM_LANES-1:0] lane_full;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lane_tmask
        assign lane_full[l] = 1'(full_gt >> lane_order[l]);
        assign lane_tmask[l] = lane_full[l] ? {`VX_CFG_NUM_THREADS{1'b1}} : partial_tmask;
    end

    // Does slot w still owe the startup for any lane a full launch activates? A
    // partial last warp re-runs the check on its own mask. Both answers are
    // looked up per lane, and the lane's full/partial decision -- the deepest
    // input -- only selects between them.
    wire [`VX_CFG_NUM_WARPS-1:0] slot_owes_full, slot_owes_part;
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_needs_init
        assign slot_owes_full[w] = ~(&warp_init_lanes_r[w]);
        assign slot_owes_part[w] = |(partial_tmask & ~warp_init_lanes_r[w]);
    end
    wire [NUM_LANES-1:0] lane_needs_init;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lane_needs_init
        assign lane_needs_init[l] = lane_full[l] ? slot_owes_full[lane_wid[l]] : slot_owes_part[lane_wid[l]];
    end

    // One thread-index advance by `step`, with a single wrap per dimension: every
    // step is normalized below the block dimensions. Y and Z are formed for both
    // carries in parallel, so the lower dimension's wrap only selects.
    function automatic logic [2:0][CTA_TID_WIDTH-1:0] tid_advance (
        input logic [2:0][CTA_TID_WIDTH-1:0] base,
        input logic [2:0][CTA_TID_WIDTH-1:0] step,
        input logic [CTA_TID_WIDTH:0]        bdim_x,
        input logic [CTA_TID_WIDTH:0]        bdim_y
    );
        logic [CTA_TID_WIDTH:0] nx, ny0, ny1;
        logic wx, wy0, wy1;
        logic [CTA_TID_WIDTH-1:0] y0, y1, z0, z1;
        nx  = {1'b0, base[0]} + {1'b0, step[0]};
        wx  = (nx >= bdim_x);
        ny0 = {1'b0, base[1]} + {1'b0, step[1]};
        ny1 = {1'b0, base[1]} + {1'b0, step[1]} + (CTA_TID_WIDTH+1)'(1);
        wy0 = (ny0 >= bdim_y);
        wy1 = (ny1 >= bdim_y);
        y0  = wy0 ? CTA_TID_WIDTH'(ny0 - bdim_y) : CTA_TID_WIDTH'(ny0);
        y1  = wy1 ? CTA_TID_WIDTH'(ny1 - bdim_y) : CTA_TID_WIDTH'(ny1);
        z0  = base[2] + step[2];
        z1  = base[2] + step[2] + CTA_TID_WIDTH'(1);
        tid_advance[0] = wx ? CTA_TID_WIDTH'(nx - bdim_x) : CTA_TID_WIDTH'(nx);
        tid_advance[1] = wx ? y1 : y0;
        tid_advance[2] = (wx ? wy1 : wy0) ? z1 : z0;
    endfunction

    // The window after the warps latched last cycle launch: entries shift down by
    // nfired_r, and the ones shifted in are the entries they replace advanced by
    // one group step.
    wire [NUM_LANES-1:0][2:0][CTA_TID_WIDTH-1:0] thread_idx_grp;
    wire [NUM_LANES-1:0][2:0][CTA_TID_WIDTH-1:0] thread_idx_adv;
    for (genvar k = 0; k < NUM_LANES; ++k) begin : g_tid_window
        assign thread_idx_grp[k] = tid_advance(thread_idx_r[k], group_step_r, block_dim_r[0], block_dim_r[1]);
        reg [2:0][CTA_TID_WIDTH-1:0] shifted;
        always @(*) begin
            shifted = thread_idx_r[k];
            for (integer n = 1; n <= NUM_LANES; ++n) begin
                if (int'(nfired_r) == n) begin
                    shifted = ((k + n) < NUM_LANES) ? thread_idx_r[(k + n) % NUM_LANES]
                                                    : thread_idx_grp[(k + n) % NUM_LANES];
                end
            end
        end
        assign thread_idx_adv[k] = shifted;
    end

    // Kernel constants of the window: the base thread indices of ranks 0..NUM_LANES-1
    // and the NUM_LANES-warp step, built by stepping warp_step_r once per cycle
    // whenever a launch's block shape differs from the last one's.
    reg [NUM_LANES-1:0][2:0][CTA_TID_WIDTH-1:0] thread_idx_init_r;
    reg [2:0][CTA_TID_WIDTH-1:0] shape_step_r;
    reg [2:0][CTA_TID_WIDTH:0]   shape_dim_r;
    reg [LCNT_WIDTH-1:0]         init_cnt_r;
    reg [2:0][CTA_TID_WIDTH-1:0] init_acc_r;
    wire                         init_busy = (init_cnt_r != '0);
    wire [2:0][CTA_TID_WIDTH-1:0] init_next = tid_advance(init_acc_r, warp_step_r, block_dim_r[0], block_dim_r[1]);
    wire shape_new = (NUM_LANES > 1) && !kmu_is_frag
                  && ((kmu_req.args.compute.warp_step != shape_step_r)
                   || (kmu_req.args.compute.block_dim != shape_dim_r));

    // The window: a launch header reloads it (cleared for a new shape), the
    // window build writes one entry per cycle, and launches shift it. No warp
    // launches while the window is built, so the three writes are exclusive and
    // the shift, the deepest input, takes the last select.
    `RUNTIME_ASSERT(reset || ~(init_busy && fire_any_r), ("%t: %s window shifted while being built", $time, INSTANCE_ID))
    always @(posedge clk) begin
        if (~reset) begin
            for (integer k = 0; k < NUM_LANES; ++k) begin
                if ((state == DISPATCH) && fire_any_r) begin
                    thread_idx_r[k] <= thread_idx_adv[k];
                end else if ((state == IDLE) && kmu_bus_if_fire) begin
                    thread_idx_r[k] <= shape_new ? '0 : thread_idx_init_r[k];
                end else if ((state == DISPATCH) && init_busy && (int'(init_cnt_r) == k)) begin
                    thread_idx_r[k] <= init_next;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Retirement decode — pipeline:
    //   T0: warp_done arrives; done_slot = cta_slot_per_warp_r[warp_done_wid].
    //   T1: warp_done_r/done_slot_r latched; rem_warps_ram read issued.
    //   T2: rem_warps_rdata available; cta_done evaluated using
    //       warp_done_r_dly + done_slot_r_dly. Two-cycle write forwarding
    //       handles back-to-back retirements to the same slot:
    //         _r  covers 1-cycle gap (write set last cycle, still asserted)
    //         _rr covers 2-cycle gap (write asserted but not yet visible to a
    //                                  read sampled at the same RAM cycle —
    //                                  RDW_MODE="R" + OUT_REG=1 gives the read
    //                                  the pre-write value)
    // -------------------------------------------------------------------------
    reg rem_warps_write_r;
    reg [CS_BITS-1:0] rem_warps_waddr_r;
    reg [NW_WIDTH:0]  rem_warps_wdata_r;
    reg rem_warps_write_rr;
    reg [CS_BITS-1:0] rem_warps_waddr_rr;
    reg [NW_WIDTH:0]  rem_warps_wdata_rr;

    wire [NW_WIDTH:0] rem_warps_rdata_fwd =
        (rem_warps_write_r  && (rem_warps_waddr_r  == done_slot_r_dly)) ? rem_warps_wdata_r  :
        (rem_warps_write_rr && (rem_warps_waddr_rr == done_slot_r_dly)) ? rem_warps_wdata_rr :
        rem_warps_rdata;
    wire cta_done = warp_done_r_dly
                 && slot_valid_r[done_slot_r_dly]
                 && (rem_warps_rdata_fwd == (NW_WIDTH+1)'(1));

    // -------------------------------------------------------------------------
    // Admission control — fixed-stride LMEM slots
    // -------------------------------------------------------------------------
    //
    // Within one kernel every CTA has the same aligned LMEM footprint, so LMEM
    // is partitioned into equal slots of pitch `stride = aligned_lmem_size`.
    // Resident CTA in slot i owns LMEM bytes [i*stride, (i+1)*stride). The
    // occupancy bound `usable = min(NUM_WARPS, floor(LMEM/stride))` is a
    // kernel constant and is registered so the per-CTA admission/ready path sees
    // only a free-slot test, never the divide/encode.
    //
    // Cluster co-residency: the first CTA of a cluster (is_first_of_cluster)
    // reserves K = cluster_size CONSECUTIVE usable slots, pre-wrapping the slot
    // window to 0 if it would overrun the usable range. Members 2..K then take the
    // following slots, so member r lands at issuer_base + r*stride — exactly what
    // DXA multicast resolves. Slots free immediately (out of order) on retire.

    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE;

    // Per-CTA LMEM footprint, block-aligned to MEM_BLOCK_SIZE by the KMU (DXA
    // multicast resolves receiver addresses as issuer_addr + r*smem_stride;
    // a non-aligned stride would target the wrong block).
    wire [LMEM_LOG:0] stride = kmu_req.aligned_lmem_size;
    wire is_first_of_cluster = eff_is_first;

    // Occupancy bound as a thermometer, registered (off the ready path):
    // usable_therm_r[m-1] is set iff m slots fit, i.e. m*stride <= LMEM_SIZE,
    // tested as stride <= floor(LMEM_SIZE/m), a constant per m, so there is
    // neither a multiplier nor a divider. All slots fit when stride == 0, and
    // slot 0 always does.
    localparam PROD_W = LMEM_LOG + NW_WIDTH + 2;
    reg [NUM_CTA_SLOTS-1:0] usable_therm_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            usable_therm_r <= '1;
        end else begin
            for (integer m = 1; m <= NUM_CTA_SLOTS; m = m + 1) begin
                // Granting one slot to a CTA that exceeds capacity keeps the
                // launch making forward progress, at the cost of letting it
                // write past its slot.
                usable_therm_r[m-1] <= (m == 1) || (stride == 0)
                                    || (PROD_W'(stride) <= PROD_W'(LMEM_SIZE / m));
            end
        end
    end

    // `data` is a launch descriptor only while `valid` is high: the launch buffers
    // reset the valid bit alone, so an idle beat carries arbitrary bits.
    `RUNTIME_ASSERT(~kmu_bus_if.valid || (stride == 0) || (PROD_W'(stride) <= PROD_W'(LMEM_SIZE)),
        ("%t: %s CTA shared-memory footprint %0d exceeds the %0d-byte capacity; the CTA overruns its slot", $time, INSTANCE_ID, stride, LMEM_SIZE))

    // The slot base is either the round-robin tail or slot 0: the tail is
    // normalized to 0 when a kernel transition shrank the usable range below it,
    // and a cluster window that would overrun the usable range pre-wraps to 0.
    // Every test is a bit of the registered maps shifted by the tail:
    // tail_room[j] is set iff slot tail+j is usable.
    wire [NUM_CTA_SLOTS:0]   usable_room = {1'b0, usable_therm_r};
    wire [NUM_CTA_SLOTS:0]   tail_room = usable_room >> tail_r;
    wire [NUM_CTA_SLOTS-1:0] tail_busy = slot_valid_r >> tail_r;
    if (NUM_CTA_SLOTS > 1) begin : g_room_top
        `UNUSED_VAR (tail_room[NUM_CTA_SLOTS])
    end

    // Window of K = max(cluster_size, 1) slots. The fit test may take K unclamped:
    // a K above the usable count never fits past slot 0, and at tail 0 both bases
    // agree. The window mask is clamped to the usable count, as a cluster larger
    // than co-residency degenerates to a clamp.
    wire [NUM_CTA_SLOTS-1:0] cluster_therm;
    for (genvar j = 0; j < NUM_CTA_SLOTS; ++j) begin : g_cluster_therm
        assign cluster_therm[j] = (j == 0) || (eff_cluster_size > (NW_WIDTH+1)'(j));
    end
    wire [NUM_CTA_SLOTS-1:0] window_ones = cluster_therm & usable_therm_r;
    wire window_fits_tail = ((cluster_therm & ~tail_room[NUM_CTA_SLOTS-1:0]) == '0);
    wire window_free_tail = ((tail_busy & window_ones) == '0);
    wire window_free_zero = ((slot_valid_r & window_ones) == '0);

    wire tail_usable = tail_room[0];
    wire base_zero   = is_first_of_cluster ? ~window_fits_tail : ~tail_usable;
    wire [CS_BITS-1:0] base_slot = base_zero ? CS_BITS'(0) : tail_r;

    // Round-robin advance: next slot wraps to 0 at the usable count.
    wire [CS_BITS-1:0] tail_inc = tail_r + CS_BITS'(1);
    wire [CS_BITS-1:0] next_tail = base_zero ? CS_BITS'(usable_room[1])
                                 : (tail_room[1] ? tail_inc : CS_BITS'(0));

    // Admission: a standalone CTA / following cluster member needs its single
    // slot free; a first-of-cluster needs all K window slots free. The window
    // [base_slot, base_slot+K) never overruns the usable range, so the mask is
    // built at NUM_CTA_SLOTS width directly.
    wire cluster_window_free = window_fits_tail ? window_free_tail : window_free_zero;
    wire single_slot_free = tail_usable ? ~tail_busy[0] : ~slot_valid_r[0];
    wire admit_ok = is_first_of_cluster ? cluster_window_free : single_slot_free;
    // A fragment launch's stamps land in the warp record TID_STAGES cycles after the
    // warp fires (the thread-index ripple), but frag_stamps_r is a single bank. So a
    // new launch must not overwrite it while an earlier fragment warp is still in that
    // pipeline, or the earlier warp's record is written from the later launch's stamps
    // (frag_inflight, below). The window is TID_STAGES deep, which grows with the warp
    // width -- invisible at 4 threads, and a fragment-dropping race by 16. Holding the
    // header off costs a launch bubble; pipelining the stamps instead would cost
    // NUM_THREADS x LANE_LAUNCH_BITS per stage -- the whole record replicated.
    assign kmu_bus_if.ready = (state == IDLE) && admit_ok && !rem_warps_write_r
                           && !frag_inflight;

    // -------------------------------------------------------------------------
    // BRAM access
    // -------------------------------------------------------------------------

    // rem_warps_ram access — retirement exclusively owns the read port.
    // cta_size_r is captured at accept time (kmu_num_warps) to avoid contention.
    assign rem_warps_read  = warp_done_r;
    assign rem_warps_raddr = done_slot_r;

    assign rem_warps_write = (kmu_bus_if_fire && state == IDLE) || rem_warps_write_r;
    assign rem_warps_waddr = (kmu_bus_if_fire && state == IDLE) ? base_slot : rem_warps_waddr_r;
    assign rem_warps_wdata = (kmu_bus_if_fire && state == IDLE) ? (NW_WIDTH+1)'(kmu_num_warps) : rem_warps_wdata_r;

    // -------------------------------------------------------------------------
    // Sequential
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state           <= IDLE;
            warp_fire_r     <= '0;
            warp_id_r       <= '0;
            warp_tmask_r    <= '0;
            warp_order_r    <= '0;
            nfired_r        <= '0;
            lane_rr_r       <= '0;
            init_cnt_r      <= '0;
            shape_step_r    <= '0;
            shape_dim_r     <= '0;
            thread_idx_init_r <= '0;
            group_step_r    <= '0;
            init_acc_r      <= '0;
            cur_ctx_id_r    <= '0;
            warp_init_lanes_r <= '0;
            warp_skip_init_r <= '0;
            tail_r          <= '0;
            cur_stride_r    <= '0;
            slot_valid_r    <= '0;
            dispatched_warps<= '0;
            warp_done_r     <= 0;
            warp_done_r_dly <= 0;
            done_slot_r     <= '0;
            done_slot_r_dly <= '0;
            cur_slot_r      <= '0;
            cluster_size_r <= (NW_WIDTH+1)'(1);  // default: no grouping
            rem_warps_waddr_r <= '0;
            rem_warps_wdata_r <= '0;
            rem_warps_write_r <= 0;
            rem_warps_waddr_rr <= '0;
            rem_warps_wdata_rr <= '0;
            rem_warps_write_rr <= 0;
            cta_slot_per_warp_r <= '0;

        end else begin

            // ---- Register retirement signals (1-stage pipeline aligned with
            //      rem_warps_ram OUT_REG=1 latency) ---------------------------
            warp_done_r     <= warp_done;
            warp_done_r_dly <= warp_done_r;
            if (warp_done) done_slot_r <= done_slot;
            done_slot_r_dly <= done_slot_r;

            // ---- wid → cta-slot map: latch on warp dispatch ----------------
            // (Used internally by the retirement path to recover the slot
            // index of a finishing warp; no longer exposed externally.)
            for (integer l = 0; l < NUM_LANES; ++l) begin
                if (lane_sel[l]) begin
                    cta_slot_per_warp_r[lane_wid[l]] <= cur_slot_r;
                end
            end

            // ---- Warp retirement -------------------------------------------
            // Snapshot _r into _rr so 2-cycle forwarding can cover the
            // window where the rem_warps_ram read started concurrently with
            // a pending write (RDW_MODE="R" returns the pre-write value).
            rem_warps_write_rr <= rem_warps_write_r;
            rem_warps_waddr_rr <= rem_warps_waddr_r;
            rem_warps_wdata_rr <= rem_warps_wdata_r;
            if (warp_done_r_dly && slot_valid_r[done_slot_r_dly]) begin
                rem_warps_waddr_r <= done_slot_r_dly;
                rem_warps_wdata_r <= rem_warps_rdata_fwd - 1;
                rem_warps_write_r <= 1;
                if (cta_done) begin
                    slot_valid_r[done_slot_r_dly] <= 1'b0;
                end
            end else begin
                rem_warps_write_r <= 0;
            end

            // ---- FSM -------------------------------------------------------
            case (state)
                IDLE: begin
                    if (kmu_bus_if_fire) begin
                        if (kmu_req.ctx_id != cur_ctx_id_r) begin
                            cur_ctx_id_r <= kmu_req.ctx_id;
                            warp_init_lanes_r <= '0;
                        end
                        warp_PC      <= kmu_req.PC;
                        entry_r      <= kmu_req.entry;
                        block_idx_r  <= kmu_block_idx;
                        block_dim_r  <= kmu_req.args.compute.block_dim;
                        for (integer d = 0; d < 2; ++d) begin
                            block_last_r[d] <= CTA_TID_WIDTH'(kmu_req.args.compute.block_dim[d] - (CTA_TID_WIDTH+1)'(1));
                        end
                        grid_dim_r   <= kmu_grid_dim;
                        param_r      <= kmu_req.param;
                        block_size_r <= eff_block_size;
                        warp_step_r  <= kmu_req.args.compute.warp_step;
                        cluster_size_r <= eff_cluster_size;
                        cta_rank_r   <= '0;
                        group_step_r <= (NUM_LANES == 1) ? kmu_req.args.compute.warp_step : group_step_r;
                        // A new block shape rebuilds the window constants before launching.
                        if (shape_new) begin
                            shape_step_r <= kmu_req.args.compute.warp_step;
                            shape_dim_r  <= kmu_req.args.compute.block_dim;
                            init_cnt_r   <= LCNT_WIDTH'(1);
                            init_acc_r   <= '0;
                        end

                        // Fixed-stride placement: this CTA lands in slot base_slot
                        // (the round-robin tail, pre-wrapped for a cluster window)
                        // at LMEM base base_slot * stride. The round-robin pointer
                        // advances to next_tail (wraps at the usable count). Slots
                        // free out of order on retirement, so a cluster's window is
                        // checked free up front (admit_ok) rather than reclaimed
                        // in order.
                        cur_stride_r <= stride;
                        slot_valid_r[base_slot] <= 1'b1;
                        tail_r     <= (`VX_CFG_NUM_WARPS > 1) ? next_tail : CS_BITS'(0);
                        cur_slot_r <= base_slot;
                        dispatched_warps <= '0;
                        state <= DISPATCH;
                    end
                end

                DISPATCH: begin
                    for (integer l = 0; l < NUM_LANES; ++l) begin
                        warp_fire_r[l]  <= lane_sel[l];
                        if (lane_sel[l]) begin
                            warp_id_r[l]    <= lane_wid[l];
                            warp_tmask_r[l] <= lane_tmask[l];
                            warp_order_r[l] <= lane_order[l];
                            warp_skip_init_r[l] <= ~lane_needs_init[l];
                            dispatched_warps[lane_wid[l]] <= 1'b1;
                        end
                    end
                    nfired_r  <= lane_cnt;
                    lane_rr_r <= (NUM_LANES > 1) ? (lane_rr_r + LANE_WIDTH'(lane_cnt)) : '0;

                    if (fire_any_r) begin
                        cta_rank_r   <= cta_rank_r + NW_WIDTH'(nfired_r);
                        block_size_r <= block_size_next;
                        for (integer l = 0; l < NUM_LANES; ++l) begin
                            if (warp_fire_r[l]) begin
                                warp_init_lanes_r[warp_id_r[l]] <= warp_init_lanes_r[warp_id_r[l]] | warp_tmask_r[l];
                            end
                        end
                        if (~rem_gt[0]) begin
                            state <= IDLE;
                        end
                    end

                    // window constants: step warp_step_r once per cycle
                    if (init_busy) begin
                        if (int'(init_cnt_r) < NUM_LANES) begin
                            for (integer k = 0; k < NUM_LANES; ++k) begin
                                if (int'(init_cnt_r) == k) begin
                                    thread_idx_init_r[k] <= init_next;
                                end
                            end
                            init_cnt_r <= init_cnt_r + LCNT_WIDTH'(1);
                        end else begin
                            group_step_r <= init_next;
                            init_cnt_r   <= '0;
                        end
                        init_acc_r <= init_next;
                    end
                end
                default:;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            cur_lmem_base_r <= '0;
        end else begin
            cur_lmem_base_r <= (LMEM_LOG)'((LMEM_LOG+NW_WIDTH+1)'(cur_slot_r) * (LMEM_LOG+NW_WIDTH+1)'(cur_stride_r));
        end
    end

    // -------------------------------------------------------------------------
    // Outputs — all driven combinationally from registered state
    // -------------------------------------------------------------------------
    // Launch-time CTA CSRs assembled from FSM state; feed the context tables
    // and cta_param.
    cta_csrs_t cta_csrs;

    assign launch_en = (state == DISPATCH) && !init_busy;

    assign cta_fire  = warp_fire_r;
    assign cta_wid   = warp_id_r;
    assign cta_PC    = warp_PC;
    assign cta_tmask = warp_tmask_r;
    assign cta_param = cta_csrs.param;
    assign cta_init  = ~warp_skip_init_r;

    reg [NW_WIDTH:0] cta_size_r;
    always @(posedge clk) begin
        if (reset) begin
            cta_size_r <= '0;
        end else if (kmu_bus_if_fire) begin
            cta_size_r <= (NW_WIDTH+1)'(kmu_num_warps);
        end
    end

    assign cta_csrs.cta_id     = cur_slot_r;
    assign cta_csrs.cta_rank   = cta_rank_r;
    assign cta_csrs.cta_size   = cta_size_r;
    assign cta_csrs.block_idx  = block_idx_r;
    assign cta_csrs.block_dim  = block_dim_r;
    assign cta_csrs.grid_dim   = grid_dim_r;
    assign cta_csrs.entry      = entry_r;
    assign cta_csrs.param      = param_r;
    assign cta_csrs.lmem_addr  = `VX_CFG_MEM_ADDR_WIDTH'(`VX_MEM_LMEM_BASE_ADDR)
                               | `VX_CFG_MEM_ADDR_WIDTH'(cur_lmem_base_r);
    assign cta_csrs.cluster_size = 32'(cluster_size_r);

    // The accept cycle (kmu_bus_if_fire) is covered by the issuer of the launch,
    // the socket, so busy has no path from this unit's inputs. Without it, a
    // registered busy aggregation would leave a 1-cycle device-busy gap on the
    // final CTA, which the host's edge-sensitive idle-wait latches as premature
    // completion.
    assign busy = (state == DISPATCH);

    // -------------------------------------------------------------------------
    // CTA context storage + per-thread coordinate expansion
    // -------------------------------------------------------------------------
    // Per-CTA context and per-warp residue, broadcast at launch to the CTA-CSR
    // stores, which resolve CTA CSRs in one cycle with no divider.

    // Per-CTA context write port, shared by every lane.
    wire                  cta_ctx_write;
    wire [NCTA_WIDTH-1:0] cta_ctx_waddr;
    cta_ctx_t             cta_ctx_wdata;

    // Per-lane CTA thread coordinates, expanded divide-free from the warp base
    // (lane 0 = base; each later lane steps +1 along X with a single wrap into Y
    // then Z) and stored in cta_warp_ram so CTA_THREAD_ID reads cost one cycle
    // with no divider. The expansion is a serial ripple, so it is pipelined
    // TID_STEP lanes/cycle to meet timing; each launch lane has its own pipeline.
    // cta_warp_ram is read many cycles after a warp launches (fetch/decode/issue
    // latency >> TID_STAGES), so the added write latency is hidden.
    // one step per stage once the ripple across many threads is the long path
    localparam TID_STEP   = (`VX_CFG_NUM_THREADS > 8) ? 1 : 2;
    localparam TID_STAGES = (`VX_CFG_NUM_THREADS - 1 + TID_STEP - 1) / TID_STEP;

    function automatic logic [2:0][CTA_TID_WIDTH-1:0] tid_next (
        input logic [2:0][CTA_TID_WIDTH-1:0] prev,
        input logic [CTA_TID_WIDTH-1:0]      last_x,
        input logic [CTA_TID_WIDTH-1:0]      last_y
    );
        // prev is a reduced coordinate (prev[0] <= last_x, prev[1] <= last_y:
        // the warp bases come from tid_advance), so a +1 step wraps exactly
        // when it leaves the last index, and wraps to 0. The wraps are equality
        // tests on prev itself, beside the increments rather than after them.
        logic wx, wy;
        wx = (prev[0] == last_x);
        wy = wx && (prev[1] == last_y);
        tid_next[0] = wx ? '0 : (prev[0] + CTA_TID_WIDTH'(1));
        tid_next[1] = wy ? '0 : (wx ? (prev[1] + CTA_TID_WIDTH'(1)) : prev[1]);
        tid_next[2] = wy ? (prev[2] + CTA_TID_WIDTH'(1)) : prev[2];
    endfunction

    typedef logic [`VX_CFG_NUM_THREADS-1:0][2:0][CTA_TID_WIDTH-1:0] cta_tid_arr_t;

    wire [NUM_LANES-1:0] lane_frag_inflight;

    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lanes
        cta_tid_arr_t                          tidp_tid [TID_STAGES+1];
        wire [TID_STAGES:0]                    tidp_valid;
        wire [TID_STAGES:0][NW_WIDTH-1:0]      tidp_wid;
        wire [TID_STAGES:0][NW_WIDTH-1:0]      tidp_rank;
        // The last index of each dimension, block_dim - 1: block_dim spans
        // 1..NUM_WARPS*NUM_THREADS, so its last index is a thread index.
        wire [TID_STAGES:0][CTA_TID_WIDTH-1:0] tidp_last_x;
        wire [TID_STAGES:0][CTA_TID_WIDTH-1:0] tidp_last_y;

        // stage 0: combinational inputs; seed lane 0 with the warp base.
        cta_tid_arr_t tidp_tid0;
        always @(*) begin
            tidp_tid0 = '0;
            tidp_tid0[0] = thread_idx_r[warp_order_r[l]];
        end
        assign tidp_tid[0]   = tidp_tid0;
        assign tidp_valid[0] = warp_fire_r[l];
        assign tidp_wid[0]   = warp_id_r[l];
        assign tidp_rank[0]  = cta_csrs.cta_rank + NW_WIDTH'(warp_order_r[l]);
        assign tidp_last_x[0] = block_last_r[0];
        assign tidp_last_y[0] = block_last_r[1];

        for (genvar s = 1; s <= TID_STAGES; ++s) begin : g_tid_pipe
            cta_tid_arr_t nxt;
            always @(*) begin
                nxt = tidp_tid[s-1];
                for (integer k = 0; k < TID_STEP; k = k + 1) begin
                    if (((s-1)*TID_STEP + k + 1) < `VX_CFG_NUM_THREADS) begin
                        nxt[(s-1)*TID_STEP + k + 1] = tid_next(nxt[(s-1)*TID_STEP + k], tidp_last_x[s-1], tidp_last_y[s-1]);
                    end
                end
            end
            reg                    v_r;
            reg [NW_WIDTH-1:0]     wid_r, rank_r;
            reg [CTA_TID_WIDTH-1:0] last_x_r, last_y_r;
            cta_tid_arr_t          tid_r;
            always @(posedge clk) begin
                if (reset) begin
                    v_r <= 1'b0;
                end else begin
                    v_r <= tidp_valid[s-1];
                end
                tid_r  <= nxt;
                wid_r  <= tidp_wid[s-1];
                rank_r <= tidp_rank[s-1];
                last_x_r <= tidp_last_x[s-1];
                last_y_r <= tidp_last_y[s-1];
            end
            assign tidp_tid[s]   = tid_r;
            assign tidp_valid[s] = v_r;
            assign tidp_wid[s]   = wid_r;
            assign tidp_rank[s]  = rank_r;
            assign tidp_last_x[s] = last_x_r;
            assign tidp_last_y[s] = last_y_r;
        end

        wire is_frag_warp;
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        // Which in-flight warps came from a fragment launch. One bit per stage rather
        // than the stamps themselves: the write side needs to know WHICH view of the
        // lane record to land, and the header side needs to know whether frag_stamps_r
        // is still owed to someone. Shaped exactly like tidp_valid -- stage 0
        // combinational, the rest registered -- so the bit and the warp it describes
        // stay in step.
        wire [TID_STAGES:0] tidp_frag;
        assign tidp_frag[0] = warp_fire_r[l] && is_frag_r;
        for (genvar s = 1; s <= TID_STAGES; ++s) begin : g_frag_pipe
            reg f_r;
            always @(posedge clk) begin
                if (reset) begin
                    f_r <= 1'b0;
                end else begin
                    f_r <= tidp_frag[s-1];
                end
            end
            assign tidp_frag[s] = f_r;
        end
        assign lane_frag_inflight[l] = |tidp_frag;
        assign is_frag_warp = tidp_frag[TID_STAGES];
    `else
        assign lane_frag_inflight[l] = 1'b0;
        assign is_frag_warp = 1'b0;
        // its only consumer (the lane-launch view mux) is RASTER-only
        `UNUSED_VAR (is_frag_warp)
    `endif

        // last_x/last_y are consumed only by stages still rippling; the final
        // stage's forwarded copies are unused.
        `UNUSED_VAR (tidp_last_x[TID_STAGES])
        `UNUSED_VAR (tidp_last_y[TID_STAGES])

        cta_warp_t cta_warp_wdata;
        assign cta_warp_wdata.cta_rank = tidp_rank[TID_STAGES];

        // The lane record is an overlay: a fragment launch lands its stamp slice, and a
        // compute launch lands the expanded thread index. Which of the two applies is a
        // property of the WARP being written, not of whatever launch the header stage
        // happens to be holding now, so it rides the pipeline with it.
        for (genvar i = 0; i < `VX_CFG_NUM_THREADS; ++i) begin : g_lane_launch
            cta_lane_t lane_compute;
            assign lane_compute = LANE_LAUNCH_BITS'(tidp_tid[TID_STAGES][i]);
        `ifdef VX_CFG_EXT_RASTER_ENABLE
            cta_lane_t lane_fragment;
            assign lane_fragment = LANE_LAUNCH_BITS'(frag_stamps_sel[i]);
            assign cta_warp_wdata.lane_launch[i] = is_frag_warp ? lane_fragment : lane_compute;
        `else
            assign cta_warp_wdata.lane_launch[i] = lane_compute;
        `endif
        end

        assign cta_lane_bcast[l].warp_write = tidp_valid[TID_STAGES];
        assign cta_lane_bcast[l].warp_waddr = tidp_wid[TID_STAGES];
        assign cta_lane_bcast[l].warp_wdata = cta_warp_wdata;
        assign cta_lane_bcast[l].id_write   = warp_fire_r[l];
        assign cta_lane_bcast[l].id_wid     = warp_id_r[l];
        assign cta_lane_bcast[l].id_cta     = cta_csrs.cta_id;
    end

`ifdef VX_CFG_EXT_RASTER_ENABLE
    assign frag_inflight   = |lane_frag_inflight;
    assign frag_stamps_sel = frag_stamps_r;
`else
    assign frag_inflight   = 1'b0;
    assign frag_stamps_sel = '0;
    `UNUSED_VAR (lane_frag_inflight)
    `UNUSED_VAR (frag_stamps_sel)
`endif

    assign cta_ctx_write = fire_any_r;
    assign cta_ctx_waddr = cta_csrs.cta_id;
    assign cta_ctx_wdata.cta_size  = cta_csrs.cta_size;
    assign cta_ctx_wdata.block_idx = cta_csrs.block_idx;
    assign cta_ctx_wdata.block_dim = cta_csrs.block_dim;
    assign cta_ctx_wdata.grid_dim  = cta_csrs.grid_dim;
    assign cta_ctx_wdata.param     = cta_csrs.param;
    assign cta_ctx_wdata.lmem_addr = cta_csrs.lmem_addr;
    assign cta_ctx_wdata.cluster_size = cta_csrs.cluster_size;
    assign cta_ctx_wdata.entry     = cta_csrs.entry;

    assign cta_bcast.ctx_write  = cta_ctx_write;
    assign cta_bcast.ctx_waddr  = cta_ctx_waddr;
    assign cta_bcast.ctx_wdata  = cta_ctx_wdata;

`ifdef DBG_TRACE_PIPELINE
    // Pipeline warp_done_wid alongside the retirement chain for trace logging.
    reg [NW_WIDTH-1:0] warp_done_wid_r, warp_done_wid_r_dly;
    always @(posedge clk) begin
        if (reset) begin
            warp_done_wid_r     <= '0;
            warp_done_wid_r_dly <= '0;
        end else begin
            if (warp_done) warp_done_wid_r <= warp_done_wid;
            warp_done_wid_r_dly <= warp_done_wid_r;
        end
    end

    always @(posedge clk) begin
        // CTA accepted from KMU. cta_id is the dispatcher slot (= VX_CSR_CTA_ID
        // value seen by the kernel).
        if (kmu_bus_if_fire) begin
            `TRACE(1, ("%t: %s kmu-accept: cta_id=%0d, PC=0x%0h, param=0x%0h, stride=%0d, num_warps=%0d, usable_slots=%0d\n",
                $time, INSTANCE_ID, base_slot, to_fullPC(kmu_req.PC),
                kmu_req.param,
                stride, kmu_num_warps, $countones(usable_therm_r)))
        end
        // Warps dispatched to the schedulers
        for (integer l = 0; l < NUM_LANES; ++l) begin
            if (warp_fire_r[l]) begin
                `TRACE(1, ("%t: %s dispatch: wid=%0d, cta_id=%0d, rank=%0d, PC=0x%0h, tmask=%b, param=0x%0h, lmem_addr=0x%0h, init=%b\n",
                    $time, INSTANCE_ID, warp_id_r[l], cur_slot_r, cta_rank_r + NW_WIDTH'(warp_order_r[l]), to_fullPC(warp_PC),
                    warp_tmask_r[l], param_r,
                    (`VX_CFG_MEM_ADDR_WIDTH'(`VX_MEM_LMEM_BASE_ADDR) | `VX_CFG_MEM_ADDR_WIDTH'(cur_lmem_base_r)),
                    ~warp_skip_init_r[l]))
            end
        end
        // Warp retirement / CTA done
        if (warp_done_r_dly && slot_valid_r[done_slot_r_dly]) begin
            `TRACE(1, ("%t: %s warp-done: wid=%0d, cta_id=%0d, rem_warps=%0d, cta_done=%b\n",
                $time, INSTANCE_ID, warp_done_wid_r_dly, done_slot_r_dly,
                rem_warps_rdata - (NW_WIDTH+1)'(1), cta_done))
        end
        // Admission gate status when KMU presents a CTA but is stalled
        if (kmu_bus_if.valid && !kmu_bus_if.ready && state == IDLE) begin
            `TRACE(4, ("%t: %s stall: admit_ok=%b, base_slot=%0d, stride=%0d, usable_slots=%0d\n",
                $time, INSTANCE_ID, admit_ok, base_slot, stride, $countones(usable_therm_r)))
        end
    end
`endif

endmodule
