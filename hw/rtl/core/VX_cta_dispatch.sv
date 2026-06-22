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

// Dispatches warps for incoming CTAs from the KMU.
module VX_cta_dispatch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire                      clk,
    input wire                      reset,

    // from KMU
    VX_kmu_bus_if.slave             kmu_bus_if,

    // from scheduler
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    input wire                      warp_done,
    input wire [NW_WIDTH-1:0]       warp_done_wid,

    // to scheduler (one warp per cycle)
    output wire                     cta_fire,
    output wire [NW_WIDTH-1:0]      cta_wid,
    output wire [PC_BITS-1:0]       cta_PC,
    output wire [`VX_CFG_NUM_THREADS-1:0] cta_tmask,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param,
    output wire                     cta_init,

    // CTA-CSR read-back
    input wire [NW_WIDTH-1:0]       csr_rd_wid,
    input wire [NCTA_WIDTH-1:0]     csr_rd_cta_id,
    output cta_csrs_t               cta_rd_csrs,
    output wire [`VX_CFG_NUM_THREADS-1:0][2:0][CTA_TID_WIDTH-1:0] cta_rd_tid,

    // CTA id of the scheduled warp
    input wire [NW_WIDTH-1:0]       schedule_wid,
    output wire [NCTA_WIDTH-1:0]    schedule_cta_id,

    output wire                     busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam NUM_CTA_SLOTS= `VX_CFG_NUM_WARPS;
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

    // Kernel initialization tracking
    reg [7:0]           cur_ctx_id_r;
    reg [`VX_CFG_NUM_WARPS-1:0] warp_init_mask_r;
    reg                 warp_skip_init_r;

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
    reg                             warp_fire_r;
    reg [NW_WIDTH-1:0]              warp_id_r;
    reg [`VX_CFG_NUM_THREADS-1:0]   warp_tmask_r;
    reg [NW_WIDTH-1:0]              cta_rank_r;
    reg [2:0][CTA_TID_WIDTH-1:0]    thread_idx_r;
    reg [CS_BITS-1:0]               cur_slot_r;

    // -------------------------------------------------------------------------
    // Free-warp selection
    // -------------------------------------------------------------------------
    reg  [`VX_CFG_NUM_WARPS-1:0] dispatched_warps;
    wire [NW_WIDTH-1:0] warp_id_n;
    wire                warp_ready;
    VX_priority_encoder #(
        .N       (`VX_CFG_NUM_WARPS),
        .REVERSE (0)
    ) priority_enc (
        .data_in   (~(active_warps | dispatched_warps)),
        `UNUSED_PIN(onehot_out),
        .index_out (warp_id_n),
        .valid_out (warp_ready)
    );

    wire kmu_bus_if_fire = kmu_bus_if.valid && kmu_bus_if.ready;

    // -------------------------------------------------------------------------
    // Power-of-two NUM_THREADS arithmetic — all combinational, zero adders
    // -------------------------------------------------------------------------
    wire [NW_WIDTH:0]      cta_num_warps;
    wire [NW_WIDTH:0]      kmu_num_warps;
    wire [CTA_TID_WIDTH:0] block_size_next;
    wire [`VX_CFG_NUM_THREADS-1:0] partial_tmask;

    if (NT_BITS > 0) begin : g_nt_nonzero
        // Ceiling division block_size / NUM_THREADS: upper bits + OR of lower bits.
        assign cta_num_warps = (NW_WIDTH+1)'(block_size_r[CTA_TID_WIDTH:NT_BITS]) + (NW_WIDTH+1)'(|block_size_r[NT_BITS-1:0]);
        // From KMU data at accept time (used to initialise table + cta_size output)
        assign kmu_num_warps = (NW_WIDTH+1)'(kmu_bus_if.data.block_size[CTA_TID_WIDTH:NT_BITS]) + (NW_WIDTH+1)'(|kmu_bus_if.data.block_size[NT_BITS-1:0]);
        // Shared block_size decrement: low NT_BITS bits unchanged; upper bits decrement by 1.
        assign block_size_next = {block_size_r[CTA_TID_WIDTH:NT_BITS] - 1'b1, block_size_r[NT_BITS-1:0]};
        // Partial-warp mask: (1 << count) - 1 where count = block_size_r[NT_BITS-1:0]
        assign partial_tmask = (`VX_CFG_NUM_THREADS'(1) << block_size_r[NT_BITS-1:0]) - `VX_CFG_NUM_THREADS'(1);
    end else begin : g_nt_zero
        // NT_BITS=0: NUM_THREADS=1, each warp has exactly 1 thread, no partial warps.
        assign cta_num_warps = (NW_WIDTH+1)'(block_size_r);
        assign kmu_num_warps = (NW_WIDTH+1)'(kmu_bus_if.data.block_size);
        assign block_size_next = (CTA_TID_WIDTH+1)'(block_size_r - 1'b1);
        assign partial_tmask = `VX_CFG_NUM_THREADS'(0);
    end

    // Full-warp test: upper bits non-zero (no comparator)
    wire is_full_warp = |block_size_r[CTA_TID_WIDTH:NT_BITS];
    // Last-warp test: ceiling(remaining/NUM_THREADS) == 1.
    // Covers (upper==1, lower==0) full-last and (upper==0, lower!=0) partial-only cases.
    wire is_last_warp = (cta_num_warps == (NW_WIDTH+1)'(1));

    // 3D thread-index carry-propagation (combinational on registered state)
    wire [CTA_TID_WIDTH:0] next_x = {1'b0, thread_idx_r[0]} + {1'b0, warp_step_r[0]};
    wire                   wrap_x = (next_x >= {1'b0, block_dim_r[0][CTA_TID_WIDTH-1:0]});
    wire [CTA_TID_WIDTH:0] next_y = ({1'b0, thread_idx_r[1]} + {1'b0, warp_step_r[1]}) + (CTA_TID_WIDTH+1)'(wrap_x);
    wire                   wrap_y = (next_y >= {1'b0, block_dim_r[1][CTA_TID_WIDTH-1:0]});

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
    wire [LMEM_LOG:0] stride = kmu_bus_if.data.aligned_lmem_size;
    wire is_first_of_cluster = kmu_bus_if.data.is_first_of_cluster;

    // Cluster member count K, capped at the slot count (a cluster larger than
    // co-residency degenerates to a clamp — matches the SimX model).
    wire [NW_WIDTH:0] cluster_k_raw = kmu_bus_if.data.cluster_size;
    wire [NW_WIDTH:0] cluster_k = (cluster_k_raw > usable_slots_r) ? usable_slots_r
                                : (cluster_k_raw == 0) ? (NW_WIDTH+1)'(1) : cluster_k_raw;

    // Occupancy bound, registered (off the ready path). usable = largest m in
    // [1, NUM_WARPS] with m*stride <= LMEM_SIZE; all slots when stride == 0.
    // The product m*stride is a constant-times-variable (shift/add), so the
    // NUM_WARPS-wide comparator tree has no divider.
    localparam PROD_W = LMEM_LOG + NW_WIDTH + 2;
    reg [NW_WIDTH:0] usable_slots_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            usable_slots_r <= (NW_WIDTH+1)'(NUM_CTA_SLOTS);
        end else begin
            if (stride == 0) begin
                usable_slots_r <= (NW_WIDTH+1)'(NUM_CTA_SLOTS);
            end else begin
                usable_slots_r <= (NW_WIDTH+1)'(1);
                for (integer m = 1; m <= NUM_CTA_SLOTS; m = m + 1) begin
                    if (PROD_W'(m) * PROD_W'(stride) <= PROD_W'(LMEM_SIZE)) begin
                        usable_slots_r <= (NW_WIDTH+1)'(m);
                    end
                end
            end
        end
    end

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
    assign kmu_bus_if.ready = (state == IDLE) && admit_ok && !rem_warps_write_r;

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
            warp_fire_r     <= 0;
            warp_id_r       <= '0;
            warp_tmask_r    <= '0;
            cur_ctx_id_r    <= '0;
            warp_init_mask_r<= '0;
            warp_skip_init_r<= 0;
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
            if ((state == DISPATCH) && warp_ready) begin
                cta_slot_per_warp_r[warp_id_n] <= cur_slot_r;
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
                        if (kmu_bus_if.data.ctx_id != cur_ctx_id_r) begin
                            cur_ctx_id_r <= kmu_bus_if.data.ctx_id;
                            warp_init_mask_r  <= '0;
                        end
                        warp_PC      <= kmu_bus_if.data.PC;
                        entry_r      <= kmu_bus_if.data.entry;
                        block_idx_r  <= kmu_bus_if.data.block_idx;
                        block_dim_r  <= kmu_bus_if.data.block_dim;
                        grid_dim_r   <= kmu_bus_if.data.grid_dim;
                        param_r      <= kmu_bus_if.data.param;
                        block_size_r <= kmu_bus_if.data.block_size;
                        warp_step_r  <= kmu_bus_if.data.warp_step;
                        cluster_size_r <= kmu_bus_if.data.cluster_size;
                        cta_rank_r   <= '0;
                        thread_idx_r <= '0;

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
                    if (warp_ready) begin
                        warp_fire_r  <= 1;
                        warp_id_r    <= warp_id_n;
                        dispatched_warps[warp_id_n] <= 1'b1;
                        // Full warp: all ones.  Partial: (1<<count)-1, no subtrahend barrel shift.
                        warp_tmask_r <= is_full_warp ? {`VX_CFG_NUM_THREADS{1'b1}} : partial_tmask;
                        warp_skip_init_r <= warp_init_mask_r[warp_id_n];
                    end else begin
                        warp_fire_r <= 0;
                    end

                    if (warp_fire_r) begin
                        cta_rank_r <= cta_rank_r + NW_WIDTH'(1);
                        // Single shared adder result used for both decrement and last-warp test
                        block_size_r <= block_size_next;

                        warp_init_mask_r[warp_id_r] <= 1'b1;
                        thread_idx_r[0] <= wrap_x ? CTA_TID_WIDTH'(next_x - {1'b0, block_dim_r[0][CTA_TID_WIDTH-1:0]}) : CTA_TID_WIDTH'(next_x);
                        thread_idx_r[1] <= wrap_y ? CTA_TID_WIDTH'(next_y - {1'b0, block_dim_r[1][CTA_TID_WIDTH-1:0]}) : CTA_TID_WIDTH'(next_y);
                        thread_idx_r[2] <= thread_idx_r[2] + warp_step_r[2] + CTA_TID_WIDTH'(wrap_y);

                        if (is_last_warp) begin
                            warp_fire_r <= 0;
                            state       <= IDLE;
                        end
                    end
                end
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Outputs — all driven combinationally from registered state
    // -------------------------------------------------------------------------
    // Launch-time CTA CSRs assembled from FSM state; feed the context tables
    // and cta_param.
    cta_csrs_t cta_csrs;

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

    assign busy = (state == DISPATCH);

    // -------------------------------------------------------------------------
    // CTA context storage + per-thread coordinate expansion
    // -------------------------------------------------------------------------
    // Per-CTA context (cta_ctx_ram) and per-warp residue (cta_warp_ram), written
    // at launch and read back via csr_rd_wid / csr_rd_cta_id so CTA CSRs resolve
    // in one cycle with no divider. BRAM-backed to avoid per-warp full-record flops.

    // Per-CTA context.
    wire                  cta_ctx_write;
    wire [NCTA_WIDTH-1:0] cta_ctx_waddr;
    cta_ctx_t             cta_ctx_wdata;
    wire [NCTA_WIDTH-1:0] cta_ctx_raddr;
    cta_ctx_t             cta_ctx_rdata;

    VX_dp_ram #(
        .DATAW     ($bits(cta_ctx_t)),
        .SIZE      (NUM_CTA_MAX),
        .RDW_MODE  ("R"),
        .RADDR_REG (1)
    ) cta_ctx_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (cta_ctx_write),
        .wren  (1'b1),
        .waddr (cta_ctx_waddr),
        .wdata (cta_ctx_wdata),
        .raddr (cta_ctx_raddr),
        .rdata (cta_ctx_rdata)
    );

    // Per-warp residue.
    wire                cta_warp_write;
    wire [NW_WIDTH-1:0] cta_warp_waddr;
    cta_warp_t          cta_warp_wdata;
    wire [NW_WIDTH-1:0] cta_warp_raddr;
    cta_warp_t          cta_warp_rdata;

    VX_dp_ram #(
        .DATAW     ($bits(cta_warp_t)),
        .SIZE      (`VX_CFG_NUM_WARPS),
        .RDW_MODE  ("R"),
        .RADDR_REG (1)
    ) cta_warp_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (cta_warp_write),
        .wren  (1'b1),
        .waddr (cta_warp_waddr),
        .wdata (cta_warp_wdata),
        .raddr (cta_warp_raddr),
        .rdata (cta_warp_rdata)
    );

    // Per-lane CTA thread coordinates, expanded divide-free from the warp base
    // (lane 0 = base; each later lane steps +1 along X with a single wrap into Y
    // then Z) and stored in cta_warp_ram so CTA_THREAD_ID reads cost one cycle
    // with no divider. The expansion is a serial ripple, so it is pipelined
    // TID_STEP lanes/cycle to meet timing. CTA dispatch is infrequent and
    // cta_warp_ram is read many cycles after a warp launches (fetch/decode/issue
    // latency >> TID_STAGES), so the added write latency is hidden.
    wire [2:0][CTA_TID_WIDTH-1:0] cta_base_tid = thread_idx_r;

    localparam TID_STEP   = 2;
    localparam TID_STAGES = (`VX_CFG_NUM_THREADS - 1 + TID_STEP - 1) / TID_STEP;

    function automatic logic [2:0][CTA_TID_WIDTH-1:0] tid_next (
        input logic [2:0][CTA_TID_WIDTH-1:0] prev,
        input logic [CTA_TID_WIDTH-1:0]      bd_x,
        input logic [CTA_TID_WIDTH-1:0]      bd_y
    );
        logic [CTA_TID_WIDTH:0] nx, ny;
        logic wx, wy;
        nx = {1'b0, prev[0]} + (CTA_TID_WIDTH+1)'(1);
        wx = (nx >= {1'b0, bd_x});
        ny = {1'b0, prev[1]} + (CTA_TID_WIDTH+1)'(wx);
        wy = wx && (ny >= {1'b0, bd_y});
        tid_next[0] = wx ? CTA_TID_WIDTH'(nx - {1'b0, bd_x}) : CTA_TID_WIDTH'(nx);
        tid_next[1] = wy ? CTA_TID_WIDTH'(ny - {1'b0, bd_y}) : CTA_TID_WIDTH'(ny);
        tid_next[2] = prev[2] + CTA_TID_WIDTH'(wy);
    endfunction

    typedef logic [`VX_CFG_NUM_THREADS-1:0][2:0][CTA_TID_WIDTH-1:0] cta_tid_arr_t;

    cta_tid_arr_t                          tidp_tid [TID_STAGES+1];
    wire [TID_STAGES:0]                    tidp_valid;
    wire [TID_STAGES:0][NW_WIDTH-1:0]      tidp_wid;
    wire [TID_STAGES:0][NW_WIDTH-1:0]      tidp_rank;
    wire [TID_STAGES:0][CTA_TID_WIDTH-1:0] tidp_bdx;
    wire [TID_STAGES:0][CTA_TID_WIDTH-1:0] tidp_bdy;

    // stage 0: combinational inputs; seed lane 0 with the warp base.
    cta_tid_arr_t tidp_tid0;
    always @(*) begin
        tidp_tid0 = '0;
        tidp_tid0[0] = cta_base_tid;
    end
    assign tidp_tid[0]   = tidp_tid0;
    assign tidp_valid[0] = cta_fire;
    assign tidp_wid[0]   = cta_wid;
    assign tidp_rank[0]  = cta_csrs.cta_rank;
    assign tidp_bdx[0]   = cta_csrs.block_dim[0][CTA_TID_WIDTH-1:0];
    assign tidp_bdy[0]   = cta_csrs.block_dim[1][CTA_TID_WIDTH-1:0];

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
        reg [CTA_TID_WIDTH-1:0] bdx_r, bdy_r;
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

    // block_dim is consumed only by stages still rippling; the final stage's
    // forwarded copy is unused.
    `UNUSED_VAR (tidp_bdx[TID_STAGES])
    `UNUSED_VAR (tidp_bdy[TID_STAGES])

    assign cta_warp_write          = tidp_valid[TID_STAGES];
    assign cta_warp_waddr          = tidp_wid[TID_STAGES];
    assign cta_warp_wdata.cta_rank = tidp_rank[TID_STAGES];
    assign cta_warp_wdata.cta_tid  = tidp_tid[TID_STAGES];

    assign cta_ctx_write = cta_fire;
    assign cta_ctx_waddr = cta_csrs.cta_id;
    assign cta_ctx_wdata.cta_size  = cta_csrs.cta_size;
    assign cta_ctx_wdata.block_idx = cta_csrs.block_idx;
    assign cta_ctx_wdata.block_dim = cta_csrs.block_dim;
    assign cta_ctx_wdata.grid_dim  = cta_csrs.grid_dim;
    assign cta_ctx_wdata.param     = cta_csrs.param;
    assign cta_ctx_wdata.lmem_addr = cta_csrs.lmem_addr;
    assign cta_ctx_wdata.cluster_size = cta_csrs.cluster_size;
    assign cta_ctx_wdata.entry     = cta_csrs.entry;

    // rdata is valid one cycle after raddr; csr_unit holds its read address
    // stable for that cycle.
    assign cta_warp_raddr = csr_rd_wid;
    assign cta_ctx_raddr  = csr_rd_cta_id;

    assign cta_rd_csrs.cta_id     = csr_rd_cta_id;
    assign cta_rd_csrs.cta_rank   = cta_warp_rdata.cta_rank;
    assign cta_rd_tid             = cta_warp_rdata.cta_tid;
    assign cta_rd_csrs.cta_size   = cta_ctx_rdata.cta_size;
    assign cta_rd_csrs.block_idx  = cta_ctx_rdata.block_idx;
    assign cta_rd_csrs.block_dim  = cta_ctx_rdata.block_dim;
    assign cta_rd_csrs.grid_dim   = cta_ctx_rdata.grid_dim;
    assign cta_rd_csrs.param      = cta_ctx_rdata.param;
    assign cta_rd_csrs.lmem_addr  = cta_ctx_rdata.lmem_addr;
    assign cta_rd_csrs.cluster_size = cta_ctx_rdata.cluster_size;
    assign cta_rd_csrs.entry      = cta_ctx_rdata.entry;

    // Per-warp -> CTA-id map, latched on launch and indexed by schedule_wid.
    reg [`VX_CFG_NUM_WARPS-1:0][NCTA_WIDTH-1:0] cta_id_per_warp_r;
    always @(posedge clk) begin
        if (cta_fire) begin
            cta_id_per_warp_r[cta_wid] <= cta_csrs.cta_id;
        end
    end
    assign schedule_cta_id = cta_id_per_warp_r[schedule_wid];

    `UNUSED_VAR (kmu_bus_if.data.cta_id)

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
        // value seen by the kernel); kmu_cta_idx is the KMU's global grid-rank
        // counter for cross-CTA correlation.
        if (kmu_bus_if_fire) begin
            `TRACE(1, ("%t: %s kmu-accept: cta_id=%0d, PC=0x%0h, param=0x%0h, kmu_cta_idx=%0d, stride=%0d, num_warps=%0d, usable_slots=%0d\n",
                $time, INSTANCE_ID, base_slot, to_fullPC(kmu_bus_if.data.PC),
                kmu_bus_if.data.param, kmu_bus_if.data.cta_id,
                stride, kmu_num_warps, usable_slots_r))
        end
        // Warp dispatched to scheduler
        if (warp_fire_r) begin
            `TRACE(1, ("%t: %s dispatch: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, param=0x%0h, lmem_addr=0x%0h, init=%b\n",
                $time, INSTANCE_ID, warp_id_r, cur_slot_r, to_fullPC(warp_PC),
                warp_tmask_r, param_r,
                (`VX_CFG_MEM_ADDR_WIDTH'(`VX_MEM_LMEM_BASE_ADDR) | `VX_CFG_MEM_ADDR_WIDTH'(cur_lmem_base_r)),
                ~warp_skip_init_r))
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
