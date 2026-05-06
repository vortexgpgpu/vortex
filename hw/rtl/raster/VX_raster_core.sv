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

    // Memory interface
    VX_mem_bus_if.master    cache_bus_if [RCACHE_NUM_REQS],

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Outputs
    VX_raster_bus_if.master raster_bus_if
);
    localparam EDGE_FUNC_LATENCY = `LATENCY_IMUL;
    localparam SLICES_BITS = `CLOG2(NUM_SLICES+1);
    localparam START_DELAY = 16; // delay startup to allow for the reset signal to propagate the module hierarchy
    localparam START_DELAY_W = `CLOG2(START_DELAY+1);

    // A primitive data contains (xloc, yloc, pid, edges, extents)
    localparam PRIM_DATA_WIDTH = 2 * `VX_RASTER_DIM_BITS + `VX_RASTER_PID_BITS + 9 * `RASTER_DATA_BITS + 3 * `RASTER_DATA_BITS;

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
    wire [`VX_RASTER_PID_BITS-1:0] mem_pid;

    // Memory unit status
    wire mem_unit_busy;
    wire mem_unit_valid;
    wire mem_unit_ready;

    // Generate start pulse
    reg [START_DELAY_W-1:0] start_cnt;
    reg mem_unit_start;
    reg start_pending;
    reg running;
    always @(posedge clk) begin
        if (reset) begin
            start_cnt <= '0;
            mem_unit_start <= 0;
            start_pending <= 0;
        end else begin
            if (~running && ~reset) begin
                start_cnt  <= START_DELAY_W'(START_DELAY);
                start_pending <= 1'b1;
            end else if (start_cnt != '0) begin
                start_cnt <= start_cnt - 1;
            end
            if (start_cnt == START_DELAY_W'(1'b1)) begin
                mem_unit_start <= 1;
                start_pending <= 0;
            end else begin
                mem_unit_start <= 0;
            end
        end
        running <= ~reset;
    end

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

        .cache_bus_if (cache_bus_if),

        .valid_out    (mem_unit_valid),
        .xloc_out     (mem_xloc),
        .yloc_out     (mem_yloc),
        .edges_out    (mem_edges),
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
    wire [2:0][`RASTER_DATA_BITS-1:0] slice_arb_extents;
    wire                            slice_arb_ready_in;

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
        .data_in    ({slice_arb_xloc, slice_arb_yloc, slice_arb_pid, slice_arb_edges_e, slice_arb_extents}),
        .data_out   (slice_arb_data_out),
        .valid_out  (slice_arb_valid_out),
        .ready_out  (slice_arb_ready_out),
        `UNUSED_PIN (sel_out)
    );

    // track pending tile data
    // this is needed to determine when rasterization has completed

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

    wire has_pending_inputs = start_pending
                           || mem_unit_start
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

    // Generate all slices
    for (genvar slice_id = 0; slice_id < NUM_SLICES; ++slice_id) begin: raster_slices
        wire [`VX_RASTER_DIM_BITS-1:0] slice_xloc_in;
        wire [`VX_RASTER_DIM_BITS-1:0] slice_yloc_in;
        wire [`VX_RASTER_PID_BITS-1:0] slice_pid_in;
        wire [2:0][2:0][`RASTER_DATA_BITS-1:0] slice_edges_in;
        wire [2:0][`RASTER_DATA_BITS-1:0] slice_extents_in;
        wire slice_ready_in;

        assign slice_valid_in[slice_id] = slice_arb_valid_out[slice_id];
        assign {slice_xloc_in, slice_yloc_in, slice_pid_in, slice_edges_in, slice_extents_in} = slice_arb_data_out[slice_id];
        assign slice_arb_ready_out[slice_id] = slice_ready_in;

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
            .pid_in     (slice_pid_in),
            .extents_in (slice_extents_in),
            .ready_in   (slice_ready_in),

            .valid_out  (slice_valid_out[slice_id]),
            .stamps_out (slice_raster_bus_if[slice_id].req_data.stamps),
            .busy_out   (slice_busy_out[slice_id]),
            .ready_out  (slice_raster_bus_if[slice_id].req_ready)
        );

        assign slice_raster_bus_if[slice_id].req_data.done = running
                                                          && ~has_pending_inputs
                                                          && ~(| slice_valid_in)
                                                          && ~(| slice_busy_out)
                                                          && ~(| slice_valid_out);

        assign slice_raster_bus_if[slice_id].req_valid = slice_valid_out[slice_id]
                                                     || slice_raster_bus_if[slice_id].req_data.done;
    end

    VX_raster_arb #(
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

    `ASSIGN_VX_RASTER_BUS_IF (raster_bus_if, raster_bus_tmp_if[0]);

`ifdef SCOPE
`ifdef DBG_SCOPE_RASTER
    `SCOPE_IO_SWITCH (1);
    wire cache_bus_req_fire_0 = cache_bus_if[0].req_valid && cache_bus_if[0].req_ready;
    wire cache_bus_rsp_fire_0 = cache_bus_if[0].rsp_valid && cache_bus_if[0].rsp_ready;
    wire raster_bus_fire = raster_bus_if.req_valid && raster_bus_if.req_ready;
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
            raster_bus_if.req_valid,
            raster_bus_if.req_ready,
            mem_unit_busy,
            mem_unit_ready,
            mem_unit_start,
            mem_unit_valid,
            no_pending_tiledata,
            raster_bus_if.req_data.done
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
            raster_bus_if.req_data.stamps[0].pos_x,
            raster_bus_if.req_data.stamps[0].pos_y,
            raster_bus_if.req_data.stamps[0].mask,
            raster_bus_if.req_data.stamps[0].bcoords,
            raster_bus_if.req_data.stamps[0].pid,
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
        .probe1 ({no_pending_tiledata, mem_unit_busy, mem_unit_ready, mem_unit_start, mem_unit_valid, raster_bus_if.req_data.done, raster_bus_if.req_valid, raster_bus_if.req_ready})
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

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;
    assign perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = raster_bus_if.req_valid && ~raster_bus_if.req_ready && ~raster_bus_if.req_data.done;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads + `PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency + `PERF_CTR_BITS'(perf_pending_reads);
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_raster_if.mem_reads    = perf_mem_reads;
    assign perf_raster_if.mem_latency  = perf_mem_latency;
    assign perf_raster_if.stall_cycles = perf_stall_cycles;
`endif

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_bus_if.req_valid && raster_bus_if.req_ready) begin
            for (integer i = 0; i < OUTPUT_QUADS; ++i) begin
                `TRACE(1, ("%d: %s-out[%0d]: done=%b, x=%0d, y=%0d, mask=%0d, pid=%0d, bcoords={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                    $time, INSTANCE_ID, i, raster_bus_if.req_data.done,
                    raster_bus_if.req_data.stamps[i].pos_x, raster_bus_if.req_data.stamps[i].pos_y, raster_bus_if.req_data.stamps[i].mask, raster_bus_if.req_data.stamps[i].pid,
                    raster_bus_if.req_data.stamps[i].bcoords[0][0], raster_bus_if.req_data.stamps[i].bcoords[1][0], raster_bus_if.req_data.stamps[i].bcoords[2][0],
                    raster_bus_if.req_data.stamps[i].bcoords[0][1], raster_bus_if.req_data.stamps[i].bcoords[1][1], raster_bus_if.req_data.stamps[i].bcoords[2][1],
                    raster_bus_if.req_data.stamps[i].bcoords[0][2], raster_bus_if.req_data.stamps[i].bcoords[1][2], raster_bus_if.req_data.stamps[i].bcoords[2][2],
                    raster_bus_if.req_data.stamps[i].bcoords[0][3], raster_bus_if.req_data.stamps[i].bcoords[1][3], raster_bus_if.req_data.stamps[i].bcoords[2][3]))
            end
        end
    end
`endif

endmodule
