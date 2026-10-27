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

    output wire                     busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT ((`VX_CFG_NUM_WARPS % NUM_LANES) == 0, ("invalid parameter"))
    localparam NUM_CTA_SLOTS= `VX_CFG_NUM_WARPS;
    localparam LANE_WARPS   = `VX_CFG_NUM_WARPS / NUM_LANES;
    localparam LANE_WIDTH   = `LOG2UP(NUM_LANES);
    localparam LCNT_WIDTH   = `CLOG2(NUM_LANES+1);
    localparam CS_BITS      = NW_WIDTH; // UP(NW_BITS): at least 1 to avoid zero-width when NUM_WARPS=1
    localparam LMEM_SIZE    = (1 << `VX_CFG_LMEM_LOG_SIZE);

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
    // cur_lmem_base_r is latched at accept and held stable through DISPATCH.
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
    wire [`VX_CFG_NUM_WARPS-1:0] free_warps = ~(active_warps | dispatched_warps);
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
    wire [CTA_TID_WIDTH:0] block_size_cur;
    wire [`VX_CFG_NUM_THREADS-1:0] partial_tmask;

    // block_size_r retires the warps latched last cycle a cycle late, so the count
    // left for the warps being latched now excludes them.
    assign block_size_cur = fire_any_r ? block_size_next : block_size_r;

    if (NT_BITS > 0) begin : g_nt_nonzero
        // Ceiling division block_size / NUM_THREADS: upper bits + OR of lower bits.
        // From KMU data at accept time (used to initialise table + cta_size output)
        assign kmu_num_warps = (NW_WIDTH+1)'(eff_block_size[CTA_TID_WIDTH:NT_BITS]) + (NW_WIDTH+1)'(|eff_block_size[NT_BITS-1:0]);
        // Shared block_size decrement: low NT_BITS bits unchanged; upper bits drop by
        // the warps launched.
        assign block_size_next = {block_size_r[CTA_TID_WIDTH:NT_BITS] - (CTA_TID_WIDTH+1-NT_BITS)'(nfired_r), block_size_r[NT_BITS-1:0]};
        // Partial-warp mask: (1 << count) - 1 where count = block_size_cur[NT_BITS-1:0]
        assign partial_tmask = (`VX_CFG_NUM_THREADS'(1) << block_size_cur[NT_BITS-1:0]) - `VX_CFG_NUM_THREADS'(1);
    end else begin : g_nt_zero
        // NT_BITS=0: NUM_THREADS=1, each warp has exactly 1 thread, no partial warps.
        assign kmu_num_warps = (NW_WIDTH+1)'(eff_block_size);
        assign block_size_next = (CTA_TID_WIDTH+1)'(block_size_r - (CTA_TID_WIDTH+1)'(nfired_r));
        assign partial_tmask = `VX_CFG_NUM_THREADS'(0);
    end

    // Warps still to launch, after those latched last cycle.
    wire [CTA_TID_WIDTH-NT_BITS:0] full_warps_cur = block_size_cur[CTA_TID_WIDTH:NT_BITS];
    wire [NW_WIDTH:0] rem_warps_cur;
    if (NT_BITS > 0) begin : g_rem_nt
        assign rem_warps_cur = (NW_WIDTH+1)'(full_warps_cur) + (NW_WIDTH+1)'(|block_size_cur[NT_BITS-1:0]);
    end else begin : g_rem_nt1
        assign rem_warps_cur = (NW_WIDTH+1)'(full_warps_cur);
    end

    // Launch selection: lanes with a free warp, taken in round-robin order from
    // lane_rr_r, up to the warps left. order[l] is the lane's position among
    // the selected lanes.
    reg [LANE_WIDTH-1:0] lane_rr_r;
    wire                 launch_en;
    reg [NUM_LANES-1:0]                 lane_sel;
    reg [NUM_LANES-1:0][LANE_WIDTH-1:0] lane_order;
    reg [LCNT_WIDTH-1:0]                lane_cnt;
    always @(*) begin
        lane_sel   = '0;
        lane_order = '0;
        lane_cnt   = '0;
        for (integer o = 0; o < NUM_LANES; ++o) begin
            if (launch_en && lane_ready[(int'(lane_rr_r) + o) % NUM_LANES]
             && ((NW_WIDTH+1)'(lane_cnt) < rem_warps_cur)) begin
                lane_sel[(int'(lane_rr_r) + o) % NUM_LANES]   = 1'b1;
                lane_order[(int'(lane_rr_r) + o) % NUM_LANES] = LANE_WIDTH'(lane_cnt);
                lane_cnt = lane_cnt + LCNT_WIDTH'(1);
            end
        end
    end

    // A warp is full unless it is the CTA's last and the block size is not a
    // multiple of the warp width.
    wire [NUM_LANES-1:0][`VX_CFG_NUM_THREADS-1:0] lane_tmask;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lane_tmask
        wire is_full_warp = ((CTA_TID_WIDTH+1-NT_BITS)'(lane_order[l]) < full_warps_cur);
        assign lane_tmask[l] = is_full_warp ? {`VX_CFG_NUM_THREADS{1'b1}} : partial_tmask;
    end

    // Does slot w still owe the startup for any lane a full launch activates? A
    // partial last warp re-runs the check on its own mask.
    wire [`VX_CFG_NUM_WARPS-1:0] slot_needs_init;
    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_needs_init
        localparam l = w % NUM_LANES;
        assign slot_needs_init[w] = |(lane_tmask[l] & ~warp_init_lanes_r[w]);
    end

    // One thread-index advance by `step`, with a single wrap per dimension: every
    // step is normalized below the block dimensions.
    function automatic logic [2:0][CTA_TID_WIDTH-1:0] tid_advance (
        input logic [2:0][CTA_TID_WIDTH-1:0] base,
        input logic [2:0][CTA_TID_WIDTH-1:0] step,
        input logic [CTA_TID_WIDTH:0]        bdim_x,
        input logic [CTA_TID_WIDTH:0]        bdim_y
    );
        logic [CTA_TID_WIDTH:0] nx, ny;
        logic wx, wy;
        nx = {1'b0, base[0]} + {1'b0, step[0]};
        wx = (nx >= bdim_x);
        ny = ({1'b0, base[1]} + {1'b0, step[1]}) + (CTA_TID_WIDTH+1)'(wx);
        wy = (ny >= bdim_y);
        tid_advance[0] = wx ? CTA_TID_WIDTH'(nx - bdim_x) : CTA_TID_WIDTH'(nx);
        tid_advance[1] = wy ? CTA_TID_WIDTH'(ny - bdim_y) : CTA_TID_WIDTH'(ny);
        tid_advance[2] = base[2] + step[2] + CTA_TID_WIDTH'(wy);
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
    // occupancy bound `usable_slots_r = min(NUM_WARPS, floor(LMEM/stride))` is a
    // kernel constant and is registered so the per-CTA admission/ready path sees
    // only a free-slot test, never the divide/encode.
    //
    // Cluster co-residency: the first CTA of a cluster (is_first_of_cluster)
    // reserves K = cluster_size CONSECUTIVE usable slots, pre-wrapping the slot
    // window to 0 if it would overrun usable_slots_r. Members 2..K then take the
    // following slots, so member r lands at issuer_base + r*stride — exactly what
    // DXA multicast resolves. Slots free immediately (out of order) on retire.

    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE;

    // Per-CTA LMEM footprint, block-aligned to MEM_BLOCK_SIZE by the KMU (DXA
    // multicast resolves receiver addresses as issuer_addr + r*smem_stride;
    // a non-aligned stride would target the wrong block).
    wire [LMEM_LOG:0] stride = kmu_req.aligned_lmem_size;
    wire is_first_of_cluster = eff_is_first;

    reg [NW_WIDTH:0] usable_slots_r;
    // Cluster member count K, capped at the slot count (a cluster larger than
    // co-residency degenerates to a clamp).
    wire [NW_WIDTH:0] cluster_k_raw = eff_cluster_size;
    wire [NW_WIDTH:0] cluster_k = (cluster_k_raw > usable_slots_r) ? usable_slots_r
                                : (cluster_k_raw == 0) ? (NW_WIDTH+1)'(1) : cluster_k_raw;

    // Occupancy bound, registered (off the ready path). usable = largest m in
    // [1, NUM_WARPS] with m*stride <= LMEM_SIZE; all slots when stride == 0.
    // The product m*stride is a constant-times-variable (shift/add), so the
    // NUM_WARPS-wide comparator tree has no divider.
    localparam PROD_W = LMEM_LOG + NW_WIDTH + 2;
    always_ff @(posedge clk) begin
        if (reset) begin
            usable_slots_r <= (NW_WIDTH+1)'(NUM_CTA_SLOTS);
        end else begin
            if (stride == 0) begin
                usable_slots_r <= (NW_WIDTH+1)'(NUM_CTA_SLOTS);
            end else begin
                // Granting one slot to a CTA that exceeds capacity keeps the
                // launch making forward progress, at the cost of letting it
                // write past its slot.
                usable_slots_r <= (NW_WIDTH+1)'(1);
                for (integer m = 1; m <= NUM_CTA_SLOTS; m = m + 1) begin
                    if (PROD_W'(m) * PROD_W'(stride) <= PROD_W'(LMEM_SIZE)) begin
                        usable_slots_r <= (NW_WIDTH+1)'(m);
                    end
                end
            end
        end
    end

    // `data` is a launch descriptor only while `valid` is high: the launch buffers
    // reset the valid bit alone, so an idle beat carries arbitrary bits.
    `RUNTIME_ASSERT(~kmu_bus_if.valid || (stride == 0) || (PROD_W'(stride) <= PROD_W'(LMEM_SIZE)),
        ("%t: %s CTA shared-memory footprint %0d exceeds the %0d-byte capacity; the CTA overruns its slot", $time, INSTANCE_ID, stride, LMEM_SIZE))

    // Normalize the round-robin pointer to the usable range (covers a kernel
    // transition that shrank usable_slots_r), then apply cluster pre-wrap.
    // Compares are full (NW_WIDTH+2)-bit to admit usable_slots_r == NUM_CTA_SLOTS.
    wire [NW_WIDTH+1:0] tail_ext = {2'b0, tail_r};
    wire [NW_WIDTH+1:0] usable_ext = {1'b0, usable_slots_r};
    wire [CS_BITS-1:0] base_tail = (tail_ext >= usable_ext) ? CS_BITS'(0) : tail_r;
    wire cluster_prewrap = is_first_of_cluster
        && (({2'b0, base_tail} + {1'b0, cluster_k}) > usable_ext);
    wire [CS_BITS-1:0] base_slot = cluster_prewrap ? CS_BITS'(0) : base_tail;

    // Round-robin advance: next slot wraps to 0 at usable_slots_r.
    wire [NW_WIDTH+1:0] next_tail_raw = {2'b0, base_slot} + (NW_WIDTH+2)'(1);
    wire [CS_BITS-1:0] next_tail = (next_tail_raw >= usable_ext)
                                 ? CS_BITS'(0) : next_tail_raw[CS_BITS-1:0];

    // Slot LMEM base = base_slot * stride (small multiply, latched at accept;
    // off the ready path).
    wire [LMEM_LOG-1:0] base_lmem = (LMEM_LOG)'((LMEM_LOG+NW_WIDTH+1)'(base_slot) * (LMEM_LOG+NW_WIDTH+1)'(stride));

    // Admission: a standalone CTA / following cluster member needs its single
    // slot free; a first-of-cluster needs all K window slots free. The window
    // [base_slot, base_slot+K) never overruns NUM_CTA_SLOTS (pre-wrap ensures
    // base_slot+K <= usable_slots_r <= NUM_CTA_SLOTS), so the mask is built at
    // NUM_CTA_SLOTS width directly. K == NUM_CTA_SLOTS wraps to all-ones, which
    // is the intended full mask.
    wire [NUM_CTA_SLOTS-1:0] window_ones = (NUM_CTA_SLOTS'(1) << cluster_k) - NUM_CTA_SLOTS'(1);
    wire [NUM_CTA_SLOTS-1:0] cluster_window = window_ones << base_slot;
    wire cluster_window_free = ((slot_valid_r & cluster_window) == '0);
    wire admit_ok = is_first_of_cluster ? cluster_window_free : ~slot_valid_r[base_slot];
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
            cur_lmem_base_r <= '0;
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
                        grid_dim_r   <= kmu_grid_dim;
                        param_r      <= kmu_req.param;
                        block_size_r <= eff_block_size;
                        warp_step_r  <= kmu_req.args.compute.warp_step;
                        cluster_size_r <= eff_cluster_size;
                        cta_rank_r   <= '0;
                        thread_idx_r <= thread_idx_init_r;
                        group_step_r <= (NUM_LANES == 1) ? kmu_req.args.compute.warp_step : group_step_r;
                        // A new block shape rebuilds the window constants before launching.
                        if ((NUM_LANES > 1) && !kmu_is_frag
                         && ((kmu_req.args.compute.warp_step != shape_step_r)
                          || (kmu_req.args.compute.block_dim != shape_dim_r))) begin
                            shape_step_r <= kmu_req.args.compute.warp_step;
                            shape_dim_r  <= kmu_req.args.compute.block_dim;
                            init_cnt_r   <= LCNT_WIDTH'(1);
                            init_acc_r   <= '0;
                            thread_idx_r <= '0;
                        end

                        // Fixed-stride placement: this CTA lands in slot base_slot
                        // (the round-robin tail, pre-wrapped for a cluster window)
                        // at LMEM base base_slot * stride. The round-robin pointer
                        // advances to next_tail (wraps at usable_slots_r). Slots
                        // free out of order on retirement, so a cluster's window is
                        // checked free up front (admit_ok) rather than reclaimed
                        // in order.
                        cur_lmem_base_r <= base_lmem;
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
                            warp_skip_init_r[l] <= ~slot_needs_init[lane_wid[l]];
                            dispatched_warps[lane_wid[l]] <= 1'b1;
                        end
                    end
                    nfired_r  <= lane_cnt;
                    lane_rr_r <= LANE_WIDTH'((int'(lane_rr_r) + int'(lane_cnt)) % NUM_LANES);

                    if (fire_any_r) begin
                        cta_rank_r   <= cta_rank_r + NW_WIDTH'(nfired_r);
                        block_size_r <= block_size_next;
                        thread_idx_r <= thread_idx_adv;
                        for (integer l = 0; l < NUM_LANES; ++l) begin
                            if (warp_fire_r[l]) begin
                                warp_init_lanes_r[warp_id_r[l]] <= warp_init_lanes_r[warp_id_r[l]] | warp_tmask_r[l];
                            end
                        end
                        if (rem_warps_cur == '0) begin
                            state <= IDLE;
                        end
                    end

                    // window constants: step warp_step_r once per cycle
                    if (init_busy) begin
                        if (int'(init_cnt_r) < NUM_LANES) begin
                            thread_idx_r[init_cnt_r] <= tid_advance(init_acc_r, warp_step_r, block_dim_r[0], block_dim_r[1]);
                            thread_idx_init_r[init_cnt_r] <= tid_advance(init_acc_r, warp_step_r, block_dim_r[0], block_dim_r[1]);
                            init_cnt_r <= init_cnt_r + LCNT_WIDTH'(1);
                        end else begin
                            group_step_r <= tid_advance(init_acc_r, warp_step_r, block_dim_r[0], block_dim_r[1]);
                            init_cnt_r   <= '0;
                        end
                        init_acc_r <= tid_advance(init_acc_r, warp_step_r, block_dim_r[0], block_dim_r[1]);
                    end
                end
                default:;
            endcase
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

    // Busy from the accept cycle (kmu_bus_if_fire), not just while in DISPATCH:
    // the accept->DISPATCH transition is registered, so gating on state alone
    // leaves the accept cycle un-busy. With SOCKET_SIZE>1 the socket busy
    // aggregation is registered, so on the final CTA kmu_busy drops before the
    // buffered per-core busy rises, opening a 1-cycle device-busy gap the host's
    // edge-sensitive idle-wait latches as premature completion (cores>1 launch
    // failure). Covering the accept cycle closes the gap.
    assign busy = (state == DISPATCH) || kmu_bus_if_fire;

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
    localparam TID_STEP   = 2;
    localparam TID_STAGES = (`VX_CFG_NUM_THREADS - 1 + TID_STEP - 1) / TID_STEP;

    function automatic logic [2:0][CTA_TID_WIDTH-1:0] tid_next (
        input logic [2:0][CTA_TID_WIDTH-1:0] prev,
        input logic [CTA_TID_WIDTH:0]        bd_x,
        input logic [CTA_TID_WIDTH:0]        bd_y
    );
        logic [CTA_TID_WIDTH:0] nx, ny;
        logic wx, wy;
        nx = {1'b0, prev[0]} + (CTA_TID_WIDTH+1)'(1);
        wx = (nx >= bd_x);
        ny = {1'b0, prev[1]} + (CTA_TID_WIDTH+1)'(wx);
        wy = wx && (ny >= bd_y);
        tid_next[0] = wx ? CTA_TID_WIDTH'(nx - bd_x) : CTA_TID_WIDTH'(nx);
        tid_next[1] = wy ? CTA_TID_WIDTH'(ny - bd_y) : CTA_TID_WIDTH'(ny);
        tid_next[2] = prev[2] + CTA_TID_WIDTH'(wy);
    endfunction

    typedef logic [`VX_CFG_NUM_THREADS-1:0][2:0][CTA_TID_WIDTH-1:0] cta_tid_arr_t;

    wire [NUM_LANES-1:0] lane_frag_inflight;

    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lanes
        cta_tid_arr_t                          tidp_tid [TID_STAGES+1];
        wire [TID_STAGES:0]                    tidp_valid;
        wire [TID_STAGES:0][NW_WIDTH-1:0]      tidp_wid;
        wire [TID_STAGES:0][NW_WIDTH-1:0]      tidp_rank;
        // block_dim spans 1..NUM_WARPS*NUM_THREADS inclusive, one bit more than a
        // thread index -- truncating it to CTA_TID_WIDTH turns a full-width dimension
        // into 0 and makes every lane wrap.
        wire [TID_STAGES:0][CTA_TID_WIDTH:0] tidp_bdx;
        wire [TID_STAGES:0][CTA_TID_WIDTH:0] tidp_bdy;

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
        assign tidp_bdx[0]   = cta_csrs.block_dim[0];
        assign tidp_bdy[0]   = cta_csrs.block_dim[1];

        for (genvar s = 1; s <= TID_STAGES; ++s) begin : g_tid_pipe
            cta_tid_arr_t nxt;
            always @(*) begin
                nxt = tidp_tid[s-1];
                for (integer k = 0; k < TID_STEP; k = k + 1) begin
                    if (((s-1)*TID_STEP + k + 1) < `VX_CFG_NUM_THREADS) begin
                        nxt[(s-1)*TID_STEP + k + 1] = tid_next(nxt[(s-1)*TID_STEP + k], tidp_bdx[s-1], tidp_bdy[s-1]);
                    end
                end
            end
            reg                    v_r;
            reg [NW_WIDTH-1:0]     wid_r, rank_r;
            reg [CTA_TID_WIDTH:0]  bdx_r, bdy_r;
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
                bdx_r  <= tidp_bdx[s-1];
                bdy_r  <= tidp_bdy[s-1];
            end
            assign tidp_tid[s]   = tid_r;
            assign tidp_valid[s] = v_r;
            assign tidp_wid[s]   = wid_r;
            assign tidp_rank[s]  = rank_r;
            assign tidp_bdx[s]   = bdx_r;
            assign tidp_bdy[s]   = bdy_r;
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

        // block_dim is consumed only by stages still rippling; the final stage's
        // forwarded copy is unused.
        `UNUSED_VAR (tidp_bdx[TID_STAGES])
        `UNUSED_VAR (tidp_bdy[TID_STAGES])

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
                stride, kmu_num_warps, usable_slots_r))
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
                $time, INSTANCE_ID, admit_ok, base_slot, stride, usable_slots_r))
        end
    end
`endif

endmodule
