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

module VX_bar_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire                clk,
    input wire                reset,

    // local barrier interface
    input wire                req_valid,
    input wire [NW_WIDTH-1:0] req_wid,
    input barrier_t           req_data,
    input wire [BAR_ADDR_W-1:0] read_addr, // valid one cycle before req_valid
    output wire               read_phase,  // asynchronous phase bit return at read_addr

    // global barrier interface
    output VX_gbar_bus_if.master gbar_bus_if,

    // scheduler interface
    input wire [`NUM_WARPS-1:0] active_warps,
    output wire               unlock_valid, // unlock stalled warps
    output wire [`NUM_WARPS-1:0] unlock_mask // warps to unlock
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    //                    warp mask + warp count + event count
    localparam EVENT_WIDTH = 16;
    localparam BAR_STATEW = `NUM_WARPS + NW_WIDTH + EVENT_WIDTH;
    localparam USE_GBAR = (NUM_CORES > 1);

    logic [`NUM_WARPS-1:0] mask_r, mask_n;
    logic [NW_WIDTH-1:0]   count_r, count_n;
    logic [EVENT_WIDTH-1:0] events_r, events_n;
    logic                  phase_r, phase_n;

    logic                  unlock_valid_n;
    logic [`NUM_WARPS-1:0] unlock_mask_n;

    logic gbar_req_valid_r, gbar_req_valid_n;
    logic [NB_WIDTH-1:0] gbar_req_id_r, gbar_req_id_n;
    logic [NC_WIDTH-1:0] gbar_req_size_m1_r, gbar_req_size_m1_n;
    wire gbar_req_ready;
    wire gbar_rsp_valid;
    wire gbar_rsp_ready = ~req_valid;

    wire [`NUM_WARPS-1:0] wait_mask = ((`NUM_WARPS)'(1) << req_wid) | mask_r;
    wire [NW_WIDTH-1:0] next_count  = count_r + NW_WIDTH'(1);
    wire next_phase  = ~phase_r;

    // Forward current-cycle TX count update into barrier-arrive checks.
    // This prevents a read-before-write hazard: if tx_valid fires in the same
    // clock cycle as the last barrier arrive, the combinational check would see
    // the stale (pre-update) tx_count_r[write_addr] and incorrectly unlock
    // warps before the DXA transfer completes (causing wrong results).
    // Root cause: warp_ctl_if.valid is registered (+1 cycle via wctl_reg) while
    // tx_bar_if.valid (SETUP) is combinational, creating a 1-cycle overlap window.
    wire tx_fwd_same_bar = tx_valid && (tx_bar_addr == write_addr);
    wire [TX_COUNT_W-1:0] tx_count_fwd =
        tx_fwd_same_bar
            ? (tx_is_done ? tx_count_r[write_addr] - TX_COUNT_W'(1)
                          : tx_count_r[write_addr] + TX_COUNT_W'(1))
            : tx_count_r[write_addr];

    always @(*) begin
        mask_n  = mask_r;
        count_n = count_r;
        events_n = events_r;
        phase_n = phase_r;
        unlock_valid_n = 0;
        unlock_mask_n = 'x;
        gbar_req_valid_n = gbar_req_valid_r;
        gbar_req_id_n = gbar_req_id_r;
        gbar_req_size_m1_n = gbar_req_size_m1_r;

        // local barrier scheduling
        if (req_data_valid && ~req_data.is_global) begin
            if (req_data.is_arrive) begin
                // barrier arrival
                if (count_r == NW_WIDTH'(req_data.size_m1)) begin
                    // All warps arrived — check tx state
                    if (tx_count_fwd == '0) begin
                        // No pending DXA: immediate unlock (same as native)
                        count_n = '0;
                        mask_n  = '0;
                        unlock_valid_n = 1;
                        unlock_mask_n = req_data.is_async ? mask_r : wait_mask;
                        phase_n = next_phase;
                end else begin
                    events_n = events_r - EVENT_WIDTH'(1);
                end
                    end
                end else begin
                    count_n = next_count;
                    if (!req_data.is_async) begin
                        mask_n = wait_mask;
                    end
                end
            end else begin
                // barrier waiting
                if (req_data.phase != phase_r) begin
                    unlock_valid_n = 1;
                    unlock_mask_n = (`NUM_WARPS)'(1) << req_wid;
                end else begin
                    mask_n = wait_mask;
                end
            end
        end

    `ifdef GBAR_ENABLE
        // global barrier scheduling
        if (req_data_valid && req_data.is_global) begin
            if (req_data.is_arrive) begin
                if (wait_mask == active_warps) begin
                    mask_n  = '0;
                    gbar_req_valid_n = 1;
                    gbar_req_id_n = req_data.id;
                    gbar_req_size_m1_n = NC_WIDTH'(req_data.size_m1);
                end else begin
                    mask_n = wait_mask;
                end
            end
        end
        if (gbar_bus_if.rsp_valid && (gbar_bus_if.rsp_data.id == gbar_req_id_r)) begin
            unlock_valid_n = 1;
            unlock_mask_n = active_warps;
        end
        if (gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
            gbar_req_valid_n = 0;
        end
    `endif
    end

    // ── Shadow register loading + dp_ram state ───────────────────────────
    // Phase output (combinational, same as native)
    // Use store_wdata for hazard bypass — correct for both req and deferred writes
    wire phase_with_reset = store_valids[read_addr] ? store_rdata[0] : '0;
    wire phase_async = is_rdw_hazard ? store_wdata[0] : phase_with_reset;

    always @(posedge clk) begin
        if (reset) begin
            store_valids <= '0;
            mask_r  <= '0;
            count_r <= '0;
            phase_r <= '0;
        end else begin
            if (store_we) begin
                store_valids[store_waddr] <= 1'b1;
            end
            if (is_rdw_hazard) begin
                {mask_r, count_r, phase_r} <= store_wdata;
            end else begin
                {mask_r, count_r, phase_r} <= store_valids[read_addr] ? store_rdata : '0;
            end
        end
        write_addr <= read_addr;
    end

    // ── TX register array update ─────────────────────────────────────────
    // Detect deferred arrival: all warps arrived but tx still pending
    wire deferred_arrive = req_data_valid && ~req_data.is_global && req_data.is_arrive
        && (count_r == NW_WIDTH'(req_data.size_m1))
        && (tx_count_fwd != '0);

    // Detect bar.wait stall during deferred period — warp must be added to deferred mask
    wire wait_stall_deferred = req_data_valid && ~req_data.is_global && ~req_data.is_arrive
        && (req_data.phase == phase_r)  // phase matches → stall (barrier not yet advanced)
        && arrived_all_r[write_addr];   // in deferred period for this barrier

    always @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < BAR_COUNT; i++) begin
                tx_count_r[i]       <= '0;
                arrived_all_r[i]    <= 1'b0;
                deferred_mask_r[i]  <= '0;
                deferred_phase_r[i] <= 1'b0;
            end
        end else begin
            // TX count update (always accepted, no contention)
            if (tx_valid) begin
                if (tx_is_done)
                    tx_count_r[tx_bar_addr] <= tx_count_r[tx_bar_addr] - TX_COUNT_W'(1);
                else
                    tx_count_r[tx_bar_addr] <= tx_count_r[tx_bar_addr] + TX_COUNT_W'(1);
            end
            // Deferred arrival: save state for later unlock
            if (deferred_arrive) begin
                arrived_all_r[write_addr]   <= 1'b1;
                deferred_mask_r[write_addr] <= req_data.is_async ? mask_r : wait_mask;
                deferred_phase_r[write_addr] <= next_phase;
            end
            // Accumulate bar.wait warps that stall during the deferred period
            if (wait_stall_deferred) begin
                deferred_mask_r[write_addr] <= deferred_mask_r[write_addr]
                    | ((`NUM_WARPS)'(1) << req_wid);
            end
            // Deferred unlock: tx_count reached 0 with arrived_all
            if (tx_deferred_fire) begin
                arrived_all_r[tx_bar_addr]  <= 1'b0;
                deferred_mask_r[tx_bar_addr] <= '0;
            end
            // Immediate unlock: clear arrived_all if it was set
            if (req_data_valid && ~req_data.is_global && req_data.is_arrive
                && (count_r == NW_WIDTH'(req_data.size_m1))
                && (tx_count_fwd == '0)) begin
                arrived_all_r[write_addr] <= 1'b0;
            end
        end
    end

    // ── Deferred dp_ram write + deferred unlock ──────────────────────────
    always @(posedge clk) begin
        if (reset) begin
            deferred_dp_pending_r    <= 1'b0;
            deferred_unlock_valid_r  <= 1'b0;
            deferred_unlock_mask_r   <= '0;
        end else begin
            // Clear deferred dp_ram write when applied.
            // This comes FIRST so that tx_deferred_fire below takes priority
            // (last NBS wins): when both fire in the same cycle, the new deferred
            // write is not lost to the simultaneous apply clear.
            if (deferred_dp_apply) begin
                deferred_dp_pending_r <= 1'b0;
            end
            // Fire deferred unlock when tx_count transitions 1→0.
            // LAST in block so tx_deferred_fire always wins over the clear above.
            if (tx_deferred_fire) begin
                deferred_unlock_valid_r <= 1'b1;
                deferred_unlock_mask_r  <= deferred_mask_r[tx_bar_addr];
                // Queue deferred dp_ram write
                deferred_dp_pending_r <= 1'b1;
                deferred_dp_addr_r    <= tx_bar_addr;
                deferred_dp_mask_r    <= '0;
                deferred_dp_count_r   <= '0;
                deferred_dp_phase_r   <= deferred_phase_r[tx_bar_addr];
            end else begin
                deferred_unlock_valid_r <= 1'b0;
            end
        end
    end

    // ── Unlock output (immediate OR deferred) ────────────────────────────
    reg unlock_valid_r;
    reg [`NUM_WARPS-1:0] unlock_mask_r;

    always @(posedge clk) begin
        if (reset) begin
            unlock_valid_r <= 0;
        end else begin
            unlock_valid_r <= unlock_valid_n;
        end
        unlock_mask_r <= unlock_mask_n;
    end

    assign read_phase   = phase_async;
    assign unlock_valid = unlock_valid_r || deferred_unlock_valid_r;
    assign unlock_mask  = (unlock_valid_r ? unlock_mask_r : '0)
                        | (deferred_unlock_valid_r ? deferred_unlock_mask_r : '0);

`ifdef GBAR_ENABLE
    always @(posedge clk) begin
        if (reset) begin
            gbar_req_valid_r <= 0;
        end else begin
            gbar_req_valid_r <= gbar_req_valid_n;
        end
        gbar_req_size_m1_r <= gbar_req_size_m1_n;
        gbar_req_id_r <= gbar_req_id_n;
    end

    assign gbar_bus_if.req_valid        = gbar_req_valid_r;
    assign gbar_bus_if.req_data.id      = gbar_req_id_r;
    assign gbar_bus_if.req_data.size_m1 = gbar_req_size_m1_r;
    assign gbar_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID % `NUM_CORES);
`endif

    // ── Debug traces ─────────────────────────────────────────────────────
`ifdef DBG_TRACE_DXA_TX_BAR
    always @(posedge clk) begin
        if (~reset) begin
            if (req_data_valid && ~req_data.is_global) begin
                `TRACE(2, ("%t: %s req: bar=%0d wid=%0d arrive=%b async=%b phase=%b count=%0d size_m1=%0d tx_count=%0d\n",
                    $time, INSTANCE_ID, write_addr, req_wid, req_data.is_arrive, req_data.is_async,
                    req_data.phase, count_r, req_data.size_m1, tx_count_r[write_addr]))
            end
            if (tx_valid) begin
                `TRACE(2, ("%t: %s tx-evt: bar=%0d done=%b tx_count=%0d arrived_all=%0d\n",
                    $time, INSTANCE_ID, tx_bar_addr, tx_is_done, tx_count_r[tx_bar_addr], arrived_all_r[tx_bar_addr]))
            end
            if (tx_deferred_fire) begin
                `TRACE(1, ("%t: %s deferred-unlock: bar=%0d mask=%b\n",
                    $time, INSTANCE_ID, tx_bar_addr, deferred_mask_r[tx_bar_addr]))
            end
            if (unlock_valid_n) begin
                `TRACE(1, ("%t: %s unlock: mask=%b\n", $time, INSTANCE_ID, unlock_mask_n))
            end
        end
    end
`endif

    // ── Debug watchdog ───────────────────────────────────────────────────
    for (genvar bi = 0; bi < BAR_COUNT; ++bi) begin : g_tx_done_watchdog
        reg [31:0] state_stall_ctr_r;
        always @(posedge clk) begin
            if (reset) begin
                state_stall_ctr_r <= '0;
            end else begin
                if (arrived_all_r[bi] && (tx_count_r[bi] != '0)) begin
                    state_stall_ctr_r <= state_stall_ctr_r + 32'd1;
                end else begin
                    state_stall_ctr_r <= '0;
                end
            end
        end
        `RUNTIME_ASSERT(state_stall_ctr_r < STALL_TIMEOUT, (
            "*** %s barrier-tx stall: bar=%0d, tx_count=%0d, arrived_all=%0d",
            INSTANCE_ID, bi, tx_count_r[bi], arrived_all_r[bi]))
    end


`else  // !BAR_TX_ENABLE

    // ========================================================================
    // Native async barrier (no DXA transaction tracking).
    // Uses dp_ram for efficient multi-entry storage (single write port).
    // Base algorithm: arrive increments count, unlocks when count == size_m1.
    // Wait checks phase; if already advanced, unlock immediately.
    // ========================================================================

    `UNUSED_VAR ({tx_valid, tx_bar_addr, tx_is_done})
`ifndef GBAR_ENABLE
    `UNUSED_VAR ({req_data.id, active_warps})
`endif

    // Scheduler has no backpressure path for barrier requests.
    assign tx_ready = 1'b1;

    //                     warp mask + warp count_r + phase_r
    localparam BAR_DATAW = `NUM_WARPS + NW_WIDTH + 1;

    logic [`NUM_WARPS-1:0] mask_r, mask_n;
    logic [NW_WIDTH-1:0]   count_r, count_n;
    logic                  phase_r, phase_n;

    logic                  unlock_valid_n;
    logic [`NUM_WARPS-1:0] unlock_mask_n;

`ifdef GBAR_ENABLE
    logic gbar_req_valid_r, gbar_req_valid_n;
    logic [NB_WIDTH-1:0] gbar_req_id_r, gbar_req_id_n;
    logic [NC_WIDTH-1:0] gbar_req_size_m1_r, gbar_req_size_m1_n;
`endif

    wire req_data_valid = req_valid && req_data.valid;

    wire [`NUM_WARPS-1:0] wait_mask = ((`NUM_WARPS)'(1) << req_wid) | mask_r;
    wire [NW_WIDTH-1:0] next_count  = count_r + NW_WIDTH'(1);
    wire next_phase  = ~phase_r;

    always @(*) begin
        mask_n  = mask_r;
        count_n = count_r;
        phase_n = phase_r;
        unlock_valid_n = 0;
        unlock_mask_n = 'x;
    `ifdef GBAR_ENABLE
        gbar_req_valid_n = gbar_req_valid_r;
        gbar_req_id_n = gbar_req_id_r;
        gbar_req_size_m1_n = gbar_req_size_m1_r;
    `endif
        // local barrier scheduling
        if (req_data_valid && ~req_data.is_global) begin
            if (req_data.is_arrive) begin
                // barrier arrival
                if (count_r == NW_WIDTH'(req_data.size_m1)) begin
                    count_n = '0;
                    mask_n = '0;
                    unlock_valid_n = 1; // release waiting warps
                    unlock_mask_n = mask_r;
                    phase_n = next_phase; // advance phase
                end
            end else if (req_data.is_arrive) begin
                // barrier arrival
                if (count_r == NW_WIDTH'(req_data.size_m1)) begin
                    count_n = '0;
                    if (events_r == 0) begin
                        mask_n = '0;
                        unlock_valid_n = 1; // release waiting warps
                        unlock_mask_n = req_data.is_sync ? wait_mask : mask_r;
                        phase_n = next_phase; // advance phase
                    end
                end else begin
                    count_n = next_count;
                    if (req_data.is_sync) begin
                        mask_n = wait_mask;
                    end
                end
            end else begin
                // barrier waiting
                if (req_data.phase != phase_r) begin
                    unlock_valid_n = 1; // release warp
                    unlock_mask_n = (`NUM_WARPS)'(1) << req_wid;
                end else begin
                    // add warp to wait list
                    mask_n = wait_mask;
                end
            end
        end
        if (USE_GBAR) begin
            // global barrier scheduling
            if (req_valid && req_data.is_global) begin
                if (req_data.is_event) begin
                    // event tracking
                    if (req_data.phase) begin
                        events_n = events_r + EVENT_WIDTH'(1);
                    end else begin
                        events_n = events_r - EVENT_WIDTH'(1);
                    end
                    // unlock warps if decrementing event to 0 and all warps have arrived
                    if ((req_data.phase == 0) && (events_r == EVENT_WIDTH'(1)) && (wait_mask == active_warps)) begin
                        mask_n = '0;
                        gbar_req_valid_n = 1; // notify global barrier
                        gbar_req_id_n = req_data.id;
                        gbar_req_size_m1_n = NC_WIDTH'(count_r); // was saved in barrier_arrive
                    end
                end else if (req_data.is_arrive) begin
                    // barrier arrival
                    count_n = NW_WIDTH'(req_data.size_m1); // store participating number of cores
                    if (wait_mask == active_warps && events_r == 0) begin
                        mask_n = '0;
                        gbar_req_valid_n = 1; // notify global barrier
                        gbar_req_id_n = req_data.id;
                        gbar_req_size_m1_n = NC_WIDTH'(req_data.size_m1);
                    end else begin
                        mask_n = wait_mask;
                    end
                end else begin
                    // barrier waiting
                    if (req_data.phase != phase_r) begin
                        unlock_valid_n = 1; // release warp
                        unlock_mask_n = (`NUM_WARPS)'(1) << req_wid;
                    end else begin
                        // add warp to wait list
                        mask_n = wait_mask;
                    end
                end
            end

            // global barrier response handling
            if (gbar_rsp_valid && gbar_rsp_ready && (gbar_bus_if.rsp_data.id == gbar_req_id_r)) begin
                unlock_valid_n = 1; // release stalled warps
                unlock_mask_n = active_warps; // release all active warps
                phase_n = next_phase; // advance phase
            end

            // global barrier request handshake
            if (gbar_req_valid_r && gbar_req_ready) begin
                gbar_req_valid_n = 0;
            end
        end
    end

    // Barriers store
    wire [BAR_STATEW-1:0] store_state_rdata;
    wire                  store_phase_rdata;
    wire [BAR_ADDR_W-1:0] store_raddr = read_addr;
    reg [BAR_ADDR_W-1:0]  store_waddr;
    wire [BAR_STATEW-1:0] store_state_wdata = {mask_n, count_n, events_n};
    wire                  store_phase_wdata = phase_n;
    wire                  store_write = req_valid || gbar_rsp_valid;

    VX_dp_ram #(
        .DATAW    (BAR_STATEW),
        .SIZE     (1 << BAR_ADDR_BITS),
        .RDW_MODE ("W"),
        .OUT_REG  (1),
        .RADDR_REG(1)
    ) barrier_state_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (store_write),
        .wren  (1'b1),
        .raddr (store_raddr),
        .waddr (store_waddr),
        .wdata (store_state_wdata),
        .rdata (store_state_rdata)
    );

    VX_dp_ram #(
        .DATAW    (1),
        .SIZE     (1 << BAR_ADDR_BITS),
        .RDW_MODE ("W"),
        .RADDR_REG(1)
    ) barrier_phase_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (store_write),
        .wren  (1'b1),
        .raddr (store_raddr),
        .waddr (store_waddr),
        .wdata (store_phase_wdata),
        .rdata (store_phase_rdata)
    );

    // Store reset handling
    reg [(1 << BAR_ADDR_BITS)-1:0] store_valids;
    wire is_rdw_hazard = store_write && (store_waddr == store_raddr);

    wire store_phase_rdata_v = store_valids[store_raddr] ? store_phase_rdata : '0;

    always @(posedge clk) begin
        if (reset) begin
            store_valids <= '0;
            phase_r <= '0;
        end else begin
            if (store_write) begin
                store_valids[store_waddr] <= 1'b1;
            end
            phase_r <= store_write ? store_phase_wdata : store_phase_rdata_v;
        end
        store_waddr <= store_raddr;
    end

    assign {mask_r, count_r, events_r} = store_valids[store_waddr] ? store_state_rdata : '0;

    wire phase_async = is_rdw_hazard ? phase_n : store_phase_rdata_v;

    reg unlock_valid_r;
    reg [`NUM_WARPS-1:0] unlock_mask_r;

    always @(posedge clk) begin
        if (reset) begin
            unlock_valid_r <= 0;
        end else begin
            unlock_valid_r <= unlock_valid_n;
        end
        unlock_mask_r <= unlock_mask_n;
    end

    assign read_phase   = phase_async;
    assign unlock_valid = unlock_valid_r;
    assign unlock_mask  = unlock_mask_r;

    if (USE_GBAR) begin : g_gbar

        assign gbar_req_ready = gbar_bus_if.req_ready;
        assign gbar_rsp_valid = gbar_bus_if.rsp_valid;

        always @(posedge clk) begin
            if (reset) begin
                gbar_req_valid_r <= 0;
            end else begin
                gbar_req_valid_r <= gbar_req_valid_n;
            end
            gbar_req_size_m1_r <= gbar_req_size_m1_n;
            gbar_req_id_r <= gbar_req_id_n;
        end

        assign gbar_bus_if.req_valid        = gbar_req_valid_r;
        assign gbar_bus_if.req_data.id      = gbar_req_id_r;
        assign gbar_bus_if.req_data.size_m1 = gbar_req_size_m1_r;
        assign gbar_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID % NUM_CORES);
        assign gbar_bus_if.rsp_ready        = gbar_rsp_ready;

    end else begin : g_nogbar

        assign gbar_req_ready = 0;
        assign gbar_rsp_valid = 0;

        assign gbar_req_valid_r = 0;
        assign gbar_req_size_m1_r = 'x;
        assign gbar_req_id_r    = 'x;

        assign gbar_bus_if.req_valid = 0;
        assign gbar_bus_if.req_data  = 'x;
        assign gbar_bus_if.rsp_ready = 0;

        `UNUSED_VAR ({gbar_req_valid_n, gbar_req_size_m1_n, gbar_req_id_n})

    end

endmodule
