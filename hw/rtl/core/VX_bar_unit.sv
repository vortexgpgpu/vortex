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
    VX_gbar_bus_if.master gbar_bus_if,

    // scheduler interface
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    output wire               unlock_valid, // unlock stalled warps
    output wire [`VX_CFG_NUM_WARPS-1:0] unlock_mask // warps to unlock
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    //                    warp mask + warp count + event count
    localparam EVENT_WIDTH = `CLOG2(`VX_CFG_MAX_BAR_EVENTS + 1);
    localparam BAR_STATEW = `VX_CFG_NUM_WARPS + NW_WIDTH + EVENT_WIDTH;
    localparam GBAR_REQW = NB_WIDTH + NC_WIDTH;
    localparam USE_GBAR = (`VX_CFG_NUM_CORES > 1);

    logic [`VX_CFG_NUM_WARPS-1:0] mask_r, mask_n;
    logic [NW_WIDTH-1:0]   count_r, count_n;
    logic [EVENT_WIDTH-1:0] events_r, events_n;
    logic                  phase_r, phase_n;

    logic                  unlock_valid_n;
    logic [`VX_CFG_NUM_WARPS-1:0] unlock_mask_n;

    // A response carries only its id after the barrier-store read port has moved on.
    logic [`VX_CFG_NUM_BARRIERS-1:0] gbar_pending_r, gbar_pending_n;
    logic [`VX_CFG_NUM_BARRIERS-1:0] gbar_pending_phase_r, gbar_pending_phase_n;
    logic [`VX_CFG_NUM_BARRIERS-1:0][`VX_CFG_NUM_WARPS-1:0] gbar_waiters_r, gbar_waiters_n;

    logic gbar_enqueue;
    logic [NB_WIDTH-1:0] gbar_enqueue_id;
    logic [NC_WIDTH-1:0] gbar_enqueue_size_m1;

    wire gbar_rsp_ready = ~req_valid;
    wire gbar_rsp_fire = gbar_bus_if.rsp_valid && gbar_rsp_ready;
    wire gbar_rsp_apply = USE_GBAR && gbar_rsp_fire && gbar_pending_r[gbar_bus_if.rsp_data.id];

    wire [`VX_CFG_NUM_WARPS-1:0] wait_mask = ((`VX_CFG_NUM_WARPS)'(1) << req_wid) | mask_r;
    wire [NW_WIDTH-1:0] next_count  = count_r + NW_WIDTH'(1);
    wire next_phase  = ~phase_r;

    always @(*) begin
        mask_n  = mask_r;
        count_n = count_r;
        events_n = events_r;
        phase_n = phase_r;
        unlock_valid_n = 0;
        unlock_mask_n = 'x;
        gbar_pending_n = gbar_pending_r;
        gbar_pending_phase_n = gbar_pending_phase_r;
        gbar_waiters_n = gbar_waiters_r;
        gbar_enqueue = 0;
        gbar_enqueue_id = 'x;
        gbar_enqueue_size_m1 = 'x;

        // local barrier scheduling
        if (req_valid && ~req_data.is_global) begin
            if (req_data.is_event) begin
                // event tracking
                if (req_data.phase) begin
                    // attach/expect_tx: increment by (size_m1 + 1) so that
                    // a software expect_tx(N) adds N events, while ordinary
                    // single-event paths (size_m1=0) add 1.
                    events_n = events_r + EVENT_WIDTH'(req_data.size_m1) + EVENT_WIDTH'(1);
                end else begin
                    events_n = events_r - EVENT_WIDTH'(1);
                end
                // unlock warps if decrementing event to 0 and all all warps have arrived
                if ((req_data.phase == 0) && (events_r == EVENT_WIDTH'(1)) && (count_r == 0)) begin
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
                    end else if (req_data.is_sync) begin
                        // Add arriving warp to wait mask
                        mask_n = wait_mask;
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
                    unlock_mask_n = (`VX_CFG_NUM_WARPS)'(1) << req_wid;
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
                        // expect_tx(N): add (size_m1 + 1)
                        events_n = events_r + EVENT_WIDTH'(req_data.size_m1) + EVENT_WIDTH'(1);
                    end else begin
                        events_n = events_r - EVENT_WIDTH'(1);
                    end
                    // unlock warps if decrementing event to 0 and all warps have arrived
                    if ((req_data.phase == 0) && (events_r == EVENT_WIDTH'(1)) && (wait_mask == active_warps)) begin
                        mask_n = '0;
                        gbar_enqueue = 1; // notify global barrier
                        gbar_enqueue_id = req_data.id;
                        gbar_enqueue_size_m1 = NC_WIDTH'(count_r); // was saved in barrier_arrive
                    end
                end else if (req_data.is_arrive) begin
                    // barrier arrival
                    count_n = NW_WIDTH'(req_data.size_m1); // store participating number of cores
                    if (req_data.is_sync) begin
                        gbar_waiters_n[req_data.id][req_wid] = 1;
                    end
                    if (wait_mask == active_warps && events_r == 0) begin
                        mask_n = '0;
                        gbar_enqueue = 1; // notify global barrier
                        gbar_enqueue_id = req_data.id;
                        gbar_enqueue_size_m1 = NC_WIDTH'(req_data.size_m1);
                    end else begin
                        // Add arriving warp to arrival mask
                        mask_n = wait_mask;
                    end
                end else begin
                    // barrier waiting
                    if (req_data.phase != phase_r) begin
                        unlock_valid_n = 1; // release warp
                        unlock_mask_n = (`VX_CFG_NUM_WARPS)'(1) << req_wid;
                    end else begin
                        // add warp to wait list
                        gbar_waiters_n[req_data.id][req_wid] = 1;
                    end
                end
            end

            // global barrier response handling
            if (gbar_rsp_apply) begin
                unlock_valid_n = (gbar_waiters_r[gbar_bus_if.rsp_data.id] != '0);
                unlock_mask_n = gbar_waiters_r[gbar_bus_if.rsp_data.id];
                gbar_waiters_n[gbar_bus_if.rsp_data.id] = '0;
                gbar_pending_n[gbar_bus_if.rsp_data.id] = 0;
            end

            if (gbar_enqueue) begin
                gbar_pending_n[gbar_enqueue_id] = 1;
                gbar_pending_phase_n[gbar_enqueue_id] = phase_r;
            end
        end
    end

    // Barriers store
    wire [BAR_STATEW-1:0] store_state_rdata;
    wire                  store_phase_rdata;
    wire [BAR_ADDR_W-1:0] store_raddr = read_addr;
    reg [BAR_ADDR_W-1:0]  store_waddr;
    wire [BAR_STATEW-1:0] store_state_wdata = {mask_n, count_n, events_n};
    wire                  store_state_write = req_valid;
    wire [BAR_ADDR_W-1:0] store_phase_waddr = gbar_rsp_apply ? BAR_ADDR_W'(gbar_bus_if.rsp_data.id) : store_waddr;
    wire                  store_phase_wdata = gbar_rsp_apply ? ~gbar_pending_phase_r[gbar_bus_if.rsp_data.id] : phase_n;
    wire                  store_phase_write = req_valid || gbar_rsp_apply;

    VX_dp_ram #(
        .DATAW    (BAR_STATEW),
        .SIZE     (1 << BAR_ADDR_BITS),
        .RDW_MODE ("W"),
        .OUT_REG  (1)
    ) barrier_state_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (store_state_write),
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
        .write (store_phase_write),
        .wren  (1'b1),
        .raddr (store_raddr),
        .waddr (store_phase_waddr),
        .wdata (store_phase_wdata),
        .rdata (store_phase_rdata)
    );

    // Store reset handling
    reg [(1 << BAR_ADDR_BITS)-1:0] store_valids;
    wire is_phase_rdw_hazard = store_phase_write && (store_phase_waddr == store_raddr);

    wire store_phase_rdata_v = store_valids[store_raddr] ? store_phase_rdata : '0;

    always @(posedge clk) begin
        if (reset) begin
            store_valids <= '0;
            phase_r <= '0;
        end else begin
            if (store_state_write) begin
                store_valids[store_waddr] <= 1'b1;
            end
            phase_r <= is_phase_rdw_hazard ? store_phase_wdata : store_phase_rdata_v;
        end
        store_waddr <= store_raddr;
    end

    assign {mask_r, count_r, events_r} = store_valids[store_waddr] ? store_state_rdata : '0;

    wire phase_async = is_phase_rdw_hazard ? store_phase_wdata : store_phase_rdata_v;

    reg unlock_valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0] unlock_mask_r;

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

    always @(posedge clk) begin
        if (reset) begin
            gbar_pending_r <= '0;
            gbar_pending_phase_r <= '0;
            gbar_waiters_r <= '0;
        end else begin
            gbar_pending_r <= gbar_pending_n;
            gbar_pending_phase_r <= gbar_pending_phase_n;
            gbar_waiters_r <= gbar_waiters_n;
        end
    end

    if (USE_GBAR) begin : g_gbar

        wire [GBAR_REQW-1:0] req_queue_data;
        wire req_empty;
        wire req_pop = ~req_empty && gbar_bus_if.req_ready;

        VX_fifo_queue #(
            .DATAW  (GBAR_REQW),
            .DEPTH  (1 << NB_WIDTH),
            .LUTRAM (1)
        ) req_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (gbar_enqueue),
            .pop     (req_pop),
            .data_in ({gbar_enqueue_id, gbar_enqueue_size_m1}),
            .data_out(req_queue_data),
            .empty   (req_empty),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (full),
            `UNUSED_PIN (size)
        );

        assign gbar_bus_if.req_valid        = ~req_empty;
        assign {gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1} = req_queue_data;
        assign gbar_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID % `VX_CFG_NUM_CORES);
        assign gbar_bus_if.rsp_ready        = gbar_rsp_ready;

        `RUNTIME_ASSERT(~gbar_enqueue || ~gbar_pending_r[gbar_enqueue_id], ("%s duplicate global barrier generation: id=%0d", INSTANCE_ID, gbar_enqueue_id))
        `RUNTIME_ASSERT(~gbar_enqueue || (store_waddr == BAR_ADDR_W'(gbar_enqueue_id)), ("%s invalid global barrier slot: id=%0d, slot=%0d", INSTANCE_ID, gbar_enqueue_id, store_waddr))
    end else begin : g_nogbar

        assign gbar_bus_if.req_valid = 0;
        assign gbar_bus_if.req_data  = 'x;
        assign gbar_bus_if.rsp_ready = 0;

        `UNUSED_VAR (gbar_enqueue_size_m1)

    end

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (req_valid) begin
            `TRACE(2, ("%t: %s req: wid=%0d, bar_id=%0d, is_global=%b, is_event=%b, is_arrive=%b, is_sync=%b, phase=%b, size_m1=%0d\n",
                $time, INSTANCE_ID, req_wid, req_data.id, req_data.is_global, req_data.is_event, req_data.is_arrive, req_data.is_sync, req_data.phase, req_data.size_m1))
        end
        if (USE_GBAR && gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
            `TRACE(2, ("%t: %s global-req: bar_id=%0d, size_m1=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1))
        end
        if (USE_GBAR && gbar_bus_if.rsp_valid && gbar_rsp_ready) begin
            `TRACE(2, ("%t: %s global-rsp: bar_id=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.rsp_data.id))
        end
        if (unlock_valid_n) begin
            `TRACE(2, ("%t: %s unlock: mask=%b\n",
                $time, INSTANCE_ID, unlock_mask_n))
        end
    end
`endif

endmodule
