//!/bin/bash

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

`include "VX_raster_define.vh"

module VX_raster_core import VX_gpu_pkg::*; import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX    = 0,
    parameter NUM_INSTANCES   = 1,
    parameter NUM_SLICES      = 1, // number of slices
    parameter TILE_LOGSIZE    = 5, // tile log size
    parameter BLOCK_LOGSIZE   = 2, // block log size
    parameter MEM_FIFO_DEPTH  = 4, // memory queue size
    parameter QUAD_FIFO_DEPTH = 4, // quad queue size
    parameter OUTPUT_QUADS    = 4   // number of output quads
) (
    `SCOPE_IO_DECL

    // Clock
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_raster_perf_if.master perf_raster_if,
`endif

    // Memory interface (primitive/tile fetch through the rcache)
    VX_mem_bus_if.master    cache_bus_if [RCACHE_NUM_REQS],

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Early-Z committed-depth read port (through the cluster ocache, coherent
    // with the OM's write-through depth stores).
    VX_mem_bus_if.master    earlyz_cache_bus_if [OCACHE_NUM_REQS],
`endif

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,
    VX_raster_launch_if.slave launch_if,

    // Outputs — fragment launches onto the cluster's KMU launch stream. The
    // stamp rides inside the launch, so there is no separate fragment data bus
    // to every core any more.
    VX_kmu_bus_if.master    kmu_bus_if,

    // Status — high from the frame kick until the engine is fully drained.
    // Out-of-band drain signal (replaces the in-band `done` token); plumbed up
    // the gfx hierarchy into the device busy aggregation so the host's
    // launch-drain wait covers the in-flight frame.
    output wire             busy
);
    localparam EDGE_FUNC_LATENCY = `LATENCY_IMUL;
    localparam SLICES_BITS = `CLOG2(NUM_SLICES+1);

    // A primitive data contains (xloc, yloc, pid, edges, [zplane,] extents). The
    // depth plane (3 words) is threaded only with early-Z.
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    localparam PRIM_DATA_WIDTH = 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS;
`else
    localparam PRIM_DATA_WIDTH = 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS;
`endif

    `STATIC_ASSERT(TILE_LOGSIZE > BLOCK_LOGSIZE, ("invalid parameter"))

    // DCRs

    raster_dcrs_t raster_dcrs;

    VX_raster_dcr #(
        .INSTANCE_ID ($sformatf("%s-dcr", INSTANCE_ID))
    ) raster_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .raster_dcrs(raster_dcrs)
    );

    ///////////////////////////////////////////////////////////////////////////

    // Output from the request
    wire [`VX_RASTER_DIM_BITS-1:0] mem_xloc;
    wire [`VX_RASTER_DIM_BITS-1:0] mem_yloc;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] mem_edges;
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    wire [2:0][`RASTER_DATA_BITS-1:0] mem_zplane;
`endif
    wire [`VX_RASTER_PID_BITS-1:0] mem_pid;

    // Memory unit status
    wire mem_unit_busy;
    wire mem_unit_valid;
    wire mem_unit_ready;

    // Frame kick — the KMU's delegated draw launch (true push). The launch
    // arrives after every config DCR of the draw by command ordering; on the
    // kick the engine starts its own tile/prim load — no consumer pull, no
    // separate begin op. `armed_r` is held from the kick until the engine is
    // fully drained and drives `busy`; `started_r` gates the drain test so
    // `busy` cannot drop in the gap between kick and the mem unit going busy.
    reg mem_unit_start;
    reg armed_r;
    reg started_r;

    assign launch_if.ready = ~armed_r;
    wire frame_kick = launch_if.valid && launch_if.ready;

    // 1-cycle start pulse the cycle after the kick (frames serialize: the host
    // drains the previous frame before issuing the next, so the mem unit is idle).
    always @(posedge clk) begin
        if (reset)
            mem_unit_start <= 1'b0;
        else
            mem_unit_start <= frame_kick;
    end

    // Primitive/tile fetch bus. The rcache carries only the fetch requester;
    // early-Z committed-depth reads go through the ocache (coherent with OM).
    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_WORD_SIZE),
        .TAG_WIDTH (RCACHE_FETCH_TAG_WIDTH)
    ) mem_cache_bus_if [RCACHE_NUM_REQS] ();

    // Memory unit
    VX_raster_mem #(
        .INSTANCE_ID   ($sformatf("%s-mem", INSTANCE_ID)),
        .INSTANCE_IDX  (INSTANCE_IDX),
        .NUM_INSTANCES (NUM_INSTANCES),
        .TILE_LOGSIZE  (TILE_LOGSIZE),
        .QUEUE_SIZE    (MEM_FIFO_DEPTH)
    ) raster_mem (
        .clk          (clk),
        .reset        (reset),

        .start        (mem_unit_start),
        .busy         (mem_unit_busy),

        .dcrs         (raster_dcrs),

        .cache_bus_if (mem_cache_bus_if),

        .valid_out    (mem_unit_valid),
        .xloc_out     (mem_xloc),
        .yloc_out     (mem_yloc),
        .edges_out    (mem_edges),
    `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
        .zplane_out   (mem_zplane),
    `endif
        .pid_out      (mem_pid),
        .ready_out    (mem_unit_ready)
    );

    // Edge function and extents calculation

    wire [2:0][`RASTER_DATA_BITS-1:0] edge_eval;
    wire [2:0][`RASTER_DATA_BITS-1:0] mem_extents;
    wire edge_func_stall;

    VX_raster_extents #(
        .TILE_LOGSIZE (TILE_LOGSIZE)
    ) raster_extents (
        .edges   (mem_edges),
        .extents (mem_extents)
    );

    VX_raster_edge #(
        .LATENCY (EDGE_FUNC_LATENCY)
    ) raster_edge (
        .clk    (clk),
        .reset  (reset),
        .enable (~edge_func_stall),
        .xloc   (mem_xloc),
        .yloc   (mem_yloc),
        .edges  (mem_edges),
        .result (edge_eval)
    );

    wire                            slice_arb_valid_in;
    wire [`VX_RASTER_DIM_BITS-1:0]  slice_arb_xloc;
    wire [`VX_RASTER_DIM_BITS-1:0]  slice_arb_yloc;
    wire [`VX_RASTER_PID_BITS-1:0]  slice_arb_pid;
    wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_arb_edges, slice_arb_edges_e;
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    wire [2:0][`RASTER_DATA_BITS-1:0] slice_arb_zplane;
`endif
    wire [2:0][`RASTER_DATA_BITS-1:0] slice_arb_extents;
    wire                            slice_arb_ready_in;

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    VX_shift_register #(
        .DATAW  (1 + 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS),
        .DEPTH  (EDGE_FUNC_LATENCY),
        .RESETW (1)
    ) edge_func_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~edge_func_stall),
        .data_in  ({mem_unit_valid, mem_xloc, mem_yloc, mem_pid, mem_edges, mem_zplane, mem_extents}),
        .data_out ({slice_arb_valid_in, slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges, slice_arb_zplane, slice_arb_extents})
    );
`else
    VX_shift_register #(
        .DATAW  (1 + 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS),
        .DEPTH  (EDGE_FUNC_LATENCY),
        .RESETW (1)
    ) edge_func_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~edge_func_stall),
        .data_in  ({mem_unit_valid, mem_xloc, mem_yloc, mem_pid, mem_edges, mem_extents}),
        .data_out ({slice_arb_valid_in, slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges, slice_arb_extents})
    );
`endif

    `EDGE_UPDATE (slice_arb_edges_e, slice_arb_edges, edge_eval);

    assign edge_func_stall = slice_arb_valid_in && ~slice_arb_ready_in;

    assign mem_unit_ready = ~edge_func_stall;

    wire [NUM_SLICES-1:0] slice_arb_valid_out;
    wire [NUM_SLICES-1:0][PRIM_DATA_WIDTH-1:0] slice_arb_data_out;
    wire [NUM_SLICES-1:0] slice_arb_ready_out;

    VX_stream_arb #(
        .NUM_OUTPUTS (NUM_SLICES),
        .DATAW       (PRIM_DATA_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) slice_req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (slice_arb_valid_in),
        .ready_in   (slice_arb_ready_in),
    `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
        .data_in    ({slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges_e, slice_arb_zplane, slice_arb_extents}),
    `else
        .data_in    ({slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges_e, slice_arb_extents}),
    `endif
        .data_out   (slice_arb_data_out),
        .valid_out  (slice_arb_valid_out),
        .ready_out  (slice_arb_ready_out),
        `UNUSED_PIN (sel_out)
    );

    // Track in-flight tile data to detect rasterization completion.
    wire no_pending_tiledata;
    wire mem_unit_fire = mem_unit_valid && mem_unit_ready;
    wire [NUM_SLICES-1:0] slice_arb_fire_out = slice_arb_valid_out & slice_arb_ready_out;
    wire [SLICES_BITS-1:0] slice_arb_fire_out_cnt;

    `POP_COUNT(slice_arb_fire_out_cnt, slice_arb_fire_out);

    VX_pending_size #(
        .SIZE  (EDGE_FUNC_LATENCY + 2 * NUM_SLICES),
        .DECRW (SLICES_BITS)
    ) pending_slice_inputs (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_unit_fire),
        .decr  (slice_arb_fire_out_cnt),
        .empty (no_pending_tiledata),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    wire has_pending_inputs = mem_unit_start
                           || mem_unit_busy
                           || mem_unit_valid
                           || ~no_pending_tiledata;

    VX_raster_bus_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) slice_raster_bus_if[NUM_SLICES]();

    VX_raster_bus_if #(
        .NUM_LANES (OUTPUT_QUADS)
    ) raster_bus_tmp_if[1]();

    wire [NUM_SLICES-1:0] slice_valid_in;
    wire [NUM_SLICES-1:0] slice_busy_out;
    wire [NUM_SLICES-1:0] slice_valid_out;

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    wire [NUM_SLICES-1:0] earlyz_busy_out;

    // Per-slice early-Z committed-depth read buses (merged onto this engine's
    // single ocache read port below).
    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_EARLYZ_REQ_TAG_WIDTH)
    ) slice_earlyz_bus_if [NUM_SLICES * OCACHE_NUM_REQS] ();
`endif

    // Generate all slices
    for (genvar slice_id = 0; slice_id < NUM_SLICES; ++slice_id) begin: raster_slices
        wire [`VX_RASTER_DIM_BITS-1:0] slice_xloc_in;
        wire [`VX_RASTER_DIM_BITS-1:0] slice_yloc_in;
        wire [`VX_RASTER_PID_BITS-1:0] slice_pid_in;
        wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_edges_in;
        wire [2:0][`RASTER_DATA_BITS-1:0] slice_extents_in;
        wire slice_ready_in;

        assign slice_valid_in[slice_id] = slice_arb_valid_out[slice_id];
        assign slice_arb_ready_out[slice_id] = slice_ready_in;

        // Slice → (early-Z →) raster bus intermediate stream.
        wire slice_out_valid, slice_out_ready;
        raster_stamp_t [OUTPUT_QUADS-1:0] slice_out_stamps;

    `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
        wire [2:0][`RASTER_DATA_BITS-1:0] slice_zplane_in;
        wire [2:0][`RASTER_DATA_BITS-1:0] slice_out_zplane;
        assign {slice_xloc_in, slice_yloc_in, slice_pid_in, slice_edges_in, slice_zplane_in, slice_extents_in} = slice_arb_data_out[slice_id];
    `else
        assign {slice_xloc_in, slice_yloc_in, slice_pid_in, slice_edges_in, slice_extents_in} = slice_arb_data_out[slice_id];
    `endif

        VX_raster_slice #(
            .INSTANCE_ID     ($sformatf("%s-slice%d", INSTANCE_ID, slice_id)),
            .TILE_LOGSIZE    (TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (BLOCK_LOGSIZE),
            .OUTPUT_QUADS    (OUTPUT_QUADS),
            .QUAD_FIFO_DEPTH (QUAD_FIFO_DEPTH)
        ) raster_slice (
            .clk        (clk),
            .reset      (reset),

            .dcrs       (raster_dcrs),

            .valid_in   (slice_valid_in[slice_id]),
            .xloc_in    (slice_xloc_in),
            .yloc_in    (slice_yloc_in),
            .xmin_in    (raster_dcrs.dst_xmin),
            .xmax_in    (raster_dcrs.dst_xmax),
            .ymin_in    (raster_dcrs.dst_ymin),
            .ymax_in    (raster_dcrs.dst_ymax),
            .edges_in   (slice_edges_in),
        `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
            .zplane_in  (slice_zplane_in),
        `endif
            .pid_in     (slice_pid_in),
            .extents_in (slice_extents_in),
            .ready_in   (slice_ready_in),

            .valid_out  (slice_out_valid),
            .stamps_out (slice_out_stamps),
        `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
            .zplane_out (slice_out_zplane),
        `endif
            .busy_out   (slice_busy_out[slice_id]),
            .ready_out  (slice_out_ready)
        );

        assign slice_valid_out[slice_id] = slice_out_valid;

    `ifdef VX_CFG_RASTER_EARLYZ_ENABLE
        // Early-Z occlusion cull: narrows each wave's coverage against committed
        // depth read from the ocache (elastic, variable-latency). Pass-through
        // when the per-draw earlyz_safe DCR is clear.
        VX_raster_earlyz #(
            .INSTANCE_ID  ($sformatf("%s-earlyz%d", INSTANCE_ID, slice_id)),
            .OUTPUT_QUADS (OUTPUT_QUADS)
        ) raster_earlyz (
            .clk        (clk),
            .reset      (reset),

            .dcrs       (raster_dcrs),

            .cache_bus_if (slice_earlyz_bus_if[slice_id * OCACHE_NUM_REQS +: OCACHE_NUM_REQS]),

            .valid_in   (slice_out_valid),
            .stamps_in  (slice_out_stamps),
            .zplane_in  (slice_out_zplane),
            .ready_in   (slice_out_ready),

            .valid_out  (slice_raster_bus_if[slice_id].req_valid),
            .stamps_out (slice_raster_bus_if[slice_id].req_data.stamps),
            .ready_out  (slice_raster_bus_if[slice_id].req_ready),

            .busy       (earlyz_busy_out[slice_id])
        );
    `else
        // No early-Z: slice output drives the raster bus directly.
        assign slice_raster_bus_if[slice_id].req_valid        = slice_out_valid;
        assign slice_raster_bus_if[slice_id].req_data.stamps  = slice_out_stamps;
        assign slice_out_ready                                = slice_raster_bus_if[slice_id].req_ready;
    `endif
    end

    // The rcache carries only the primitive/tile fetch requester now, so the
    // fetch bus drives the physical rcache ports directly (no requester arb).
    for (genvar p = 0; p < RCACHE_NUM_REQS; ++p) begin : g_rcache_fetch
        `ASSIGN_VX_MEM_BUS_IF (cache_bus_if[p], mem_cache_bus_if[p]);
    end

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // ── Intra-engine early-Z read merge ────────────────────────────────────
    // Merge this engine's NUM_SLICES early-Z depth readers onto its single
    // ocache read port. The arbiter appends slice-select bits above the reader
    // tag so responses demux back to the right slice.
    for (genvar p = 0; p < OCACHE_NUM_REQS; ++p) begin : g_earlyz_merge
        VX_mem_bus_if #(
            .DATA_SIZE (OCACHE_WORD_SIZE),
            .TAG_WIDTH (OCACHE_EARLYZ_REQ_TAG_WIDTH)
        ) merge_in_if [NUM_SLICES] ();

        VX_mem_bus_if #(
            .DATA_SIZE (OCACHE_WORD_SIZE),
            .TAG_WIDTH (OCACHE_EARLYZ_TAG_WIDTH)
        ) merge_out_if [1] ();

        for (genvar s = 0; s < NUM_SLICES; ++s) begin : g_merge_in
            `ASSIGN_VX_MEM_BUS_IF (merge_in_if[s], slice_earlyz_bus_if[s * OCACHE_NUM_REQS + p]);
        end

        VX_mem_bus_arb #(
            .NUM_INPUTS  (NUM_SLICES),
            .NUM_OUTPUTS (1),
            .DATA_SIZE   (OCACHE_WORD_SIZE),
            .TAG_WIDTH   (OCACHE_EARLYZ_REQ_TAG_WIDTH),
            .TAG_SEL_IDX (OCACHE_EARLYZ_REQ_TAG_WIDTH - OCACHE_EARLYZ_SLICE_SEL),
            .ARBITER     ("R"),
            .REQ_OUT_BUF (2),
            .RSP_OUT_BUF (2)
        ) earlyz_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (merge_in_if),
            .bus_out_if (merge_out_if)
        );

        `ASSIGN_VX_MEM_BUS_IF (earlyz_cache_bus_if[p], merge_out_if[0]);
    end
`endif

    VX_raster_bus_arb #(
        .NUM_INPUTS (NUM_SLICES),
        .NUM_LANES  (OUTPUT_QUADS),
        .ARBITER    ("R"),
        .OUT_BUF    (3) // external bus should be registered
    ) raster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (slice_raster_bus_if),
        .bus_out_if (raster_bus_tmp_if)
    );

    // ── fragment warp aggregation + launch ────────────────────────────────
    // The packer used to run once per core, behind the fan-out; it runs once here
    // instead. It compacts sparse covered-quad waves into full warps and hands the
    // launch builder the wave's owner core, so a warp never mixes owners (see
    // VX_raster_packer).
    VX_raster_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES)
    ) packed_bus_if();

    wire [RASTER_DEST_W-1:0] packed_owner;
    wire packer_busy;

    VX_raster_packer #(
        .INSTANCE_ID (`SFORMATF(("%s-packer", INSTANCE_ID))),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES)
    ) packer (
        .clk        (clk),
        .reset      (reset),
        .in_bus_if  (raster_bus_tmp_if[0]),
        .out_bus_if (packed_bus_if),
        .out_owner  (packed_owner),
        .busy       (packer_busy)
    );

    wire launch_busy;

    VX_raster_launch #(
        .INSTANCE_ID (`SFORMATF(("%s-launch", INSTANCE_ID))),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES)
    ) launch (
        .clk             (clk),
        .reset           (reset),
        .dcr_write_valid (dcr_bus_if.req_valid && dcr_bus_if.req_data.rw),
        .dcr_write_addr  (dcr_bus_if.req_data.addr),
        .dcr_write_data  (dcr_bus_if.req_data.data),
        .raster_bus_if   (packed_bus_if),
        .raster_owner_in (packed_owner),
        .kmu_bus_if      (kmu_bus_if),
        .busy            (launch_busy)
    );

    // ── Frame busy / drain (out-of-band; replaces the in-band `done`) ──────
    // The engine is drained when nothing is in the load/edge/slice pipeline and
    // no quad is buffered on the output bus.
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    wire earlyz_idle = ~(| earlyz_busy_out);
`else
    wire earlyz_idle = 1'b1;
`endif

    wire engine_idle = ~has_pending_inputs
                    && ~(| slice_valid_in)
                    && ~(| slice_busy_out)
                    && ~(| slice_valid_out)
                    && earlyz_idle
                    && ~raster_bus_tmp_if[0].req_valid
                    && ~packer_busy
                    && ~launch_busy;

    // Tiles assigned to THIS engine (must match VX_raster_mem's start_tile_count).
    // With NUM_INSTANCES>1 an uneven split can leave an engine with zero tiles: it
    // then never asserts mem_unit_busy, so `started_r` would never set and `busy`
    // (armed_r) would stick high forever, hanging the frame drain. Detect the
    // no-work case so such an engine drains immediately.
    localparam LOG2_NUM_INSTANCES = `CLOG2(NUM_INSTANCES);
    wire [`RASTER_TILE_BITS-1:0] my_tile_count =
        (raster_dcrs.tile_count + `RASTER_TILE_BITS'(NUM_INSTANCES - 1 - INSTANCE_IDX)) >> LOG2_NUM_INSTANCES;
    wire has_no_tiles = (my_tile_count == '0);

    always @(posedge clk) begin
        if (reset) begin
            armed_r   <= 1'b0;
            started_r <= 1'b0;
        end else begin
            if (frame_kick) begin
                armed_r   <= 1'b1;
                started_r <= 1'b0;
            end else begin
                if (mem_unit_busy)
                    started_r <= 1'b1;
                // Clear once the load has begun (started_r) — so `busy` never drops
                // in the kick→mem-busy gap — OR immediately when this engine has no
                // tiles this frame (mem_unit_busy would never pulse).
                if (armed_r && (started_r || has_no_tiles) && engine_idle) begin
                    armed_r   <= 1'b0;
                    started_r <= 1'b0;
                end
            end
        end
    end

    // The frame_kick term covers the one-cycle arm delay so `busy` rises with
    // the kick acceptance.
    assign busy = armed_r | frame_kick;

`ifdef SCOPE
`ifdef DBG_SCOPE_RASTER
    `SCOPE_IO_SWITCH (1);
    wire cache_bus_req_fire_0 = cache_bus_if[0].req_valid && cache_bus_if[0].req_ready;
    wire cache_bus_rsp_fire_0 = cache_bus_if[0].rsp_valid && cache_bus_if[0].rsp_ready;
    wire raster_bus_fire = raster_bus_tmp_if[0].req_valid && raster_bus_tmp_if[0].req_ready;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 7, 12, 5, (
            RCACHE_ADDR_WIDTH + 1 + RCACHE_TAG_WIDTH +
            (RCACHE_WORD_SIZE * 8) + RCACHE_TAG_WIDTH +
            VX_DCR_ADDR_WIDTH + VX_DCR_DATA_WIDTH +
            $bits(raster_stamp_t) +
            $bits(raster_dcrs_t)
        ), {
            cache_bus_if[0].req_valid,
            cache_bus_if[0].req_ready,
            cache_bus_if[0].rsp_valid,
            cache_bus_if[0].rsp_ready,
            raster_bus_tmp_if[0].req_valid,
            raster_bus_tmp_if[0].req_ready,
            mem_unit_busy,
            mem_unit_ready,
            mem_unit_start,
            mem_unit_valid,
            no_pending_tiledata,
            armed_r
        }, {
            cache_bus_req_fire_0,
            cache_bus_rsp_fire_0,
            dcr_bus_if.write_valid,
            raster_bus_fire,
            mem_unit_fire
        }, {
            cache_bus_if[0].req_data.addr,
            cache_bus_if[0].req_data.rw,
            cache_bus_if[0].req_data.tag,
            cache_bus_if[0].rsp_data.data,
            cache_bus_if[0].rsp_data.tag,
            dcr_bus_if.write_addr,
            dcr_bus_if.write_data,
            raster_bus_tmp_if[0].req_data.stamps[0].pos_x,
            raster_bus_tmp_if[0].req_data.stamps[0].pos_y,
            raster_bus_tmp_if[0].req_data.stamps[0].mask,
            raster_bus_tmp_if[0].req_data.stamps[0].pid,
            raster_dcrs.tbuf_addr,
            raster_dcrs.tile_count,
            raster_dcrs.pbuf_addr,
            raster_dcrs.pbuf_stride,
            raster_dcrs.dst_xmin,
            raster_dcrs.dst_xmax,
            raster_dcrs.dst_ymin,
            raster_dcrs.dst_ymax
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED()
`endif
`endif
`ifdef CHIPSCOPE
    ila_raster ila_raster_inst (
        .clk    (clk),
        .probe0 ({cache_bus_if[0].rsp_data.data, cache_bus_if[0].rsp_data.tag, cache_bus_if[0].rsp_ready, cache_bus_if[0].rsp_valid, cache_bus_if[0].req_data.tag, cache_bus_if[0].req_data.addr, cache_bus_if[0].req_data.rw, cache_bus_if[0].req_valid, cache_bus_if[0].req_ready}),
        .probe1 ({no_pending_tiledata, mem_unit_busy, mem_unit_ready, mem_unit_start, mem_unit_valid, armed_r, raster_bus_tmp_if[0].req_valid, raster_bus_tmp_if[0].req_ready})
    );
`endif

`ifdef PERF_ENABLE
    wire [`CLOG2(RCACHE_NUM_REQS+1)-1:0] perf_mem_req_per_cycle;
    wire [`CLOG2(RCACHE_NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;
    wire [`CLOG2(RCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [RCACHE_NUM_REQS-1:0] perf_mem_req_fire;
    for (genvar i = 0; i < RCACHE_NUM_REQS; ++i) begin : g_perf_mem_req_fire
        assign perf_mem_req_fire[i] = cache_bus_if[i].req_valid && cache_bus_if[i].req_ready;
    end

    wire [RCACHE_NUM_REQS-1:0] perf_mem_rsp_fire;
    for (genvar i = 0; i < RCACHE_NUM_REQS; ++i) begin : g_perf_mem_rsp_fire
        assign perf_mem_rsp_fire[i] = cache_bus_if[i].rsp_valid && cache_bus_if[i].rsp_ready;
    end

    `POP_COUNT(perf_mem_req_per_cycle, perf_mem_req_fire);
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_fire);

    reg [PERF_CTR_BITS-1:0] perf_pending_reads;
    assign perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = raster_bus_tmp_if[0].req_valid && ~raster_bus_tmp_if[0].req_ready;

    reg [PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads + PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency + PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_raster_if.mem_reads    = perf_mem_reads;
    assign perf_raster_if.mem_latency  = perf_mem_latency;
    assign perf_raster_if.stall_cycles = perf_stall_cycles;
`endif

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_bus_tmp_if[0].req_valid && raster_bus_tmp_if[0].req_ready) begin
            for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
                `TRACE(1, ("%d: %s-out[%0d]: armed=%b, x=%0d, y=%0d, mask=%0d, pid=%0d\n",
                    $time, INSTANCE_ID, i, armed_r,
                    raster_bus_tmp_if[0].req_data.stamps[i].pos_x, raster_bus_tmp_if[0].req_data.stamps[i].pos_y, raster_bus_tmp_if[0].req_data.stamps[i].mask, raster_bus_tmp_if[0].req_data.stamps[i].pid))
            end
        end
    end
`endif

endmodule
