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

// Rasterizer Memory Unit
// Functionality:
//  1. Send memory request to fetch tile header.
//  2. Send memory request to fetch all primitives overlapping the tile.
//  3. Return the primitives with associated tile.

module VX_raster_mem import VX_gpu_pkg::*; import VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter INSTANCE_IDX  = 0,
    parameter NUM_INSTANCES = 1,
    parameter TILE_LOGSIZE  = 5,
    parameter QUEUE_SIZE    = 8
) (
    input wire clk,
    input wire reset,

    // Device configurations
    raster_dcrs_t                       dcrs,

    // Memory interface
    VX_mem_bus_if.master                cache_bus_if [RCACHE_NUM_REQS],

    // Inputs
    input wire                          start,
    output wire                         busy,

    // Outputs
    output wire                         valid_out,
    output wire [`VX_RASTER_PID_BITS-1:0] pid_out,
    output wire [`VX_RASTER_DIM_BITS-1:0] xloc_out,
    output wire [`VX_RASTER_DIM_BITS-1:0] yloc_out,
    output wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges_out,
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Screen-space depth plane {A',B',C'} (Q7.24), fetched alongside the edges
    // for early-Z. Constant per primitive.
    output wire [2:0][`RASTER_DATA_BITS-1:0] zplane_out,
`endif
    input wire                          ready_out
);
    `UNUSED_VAR (dcrs)

    localparam LOG2_NUM_INSTANCES = `CLOG2(NUM_INSTANCES);

    localparam NUM_REQS         = RASTER_MEM_REQS;
    localparam FSM_BITS         = 2;
    localparam FETCH_FLAG_BITS  = 2;
    localparam TAG_WIDTH        = `VX_RASTER_PID_BITS + FETCH_FLAG_BITS;
    localparam W_ADDR_BITS      = (`RASTER_ADDR_BITS + 6) - 2;

    localparam STATE_IDLE       = 2'b00;
    localparam STATE_TILE       = 2'b01;
    localparam STATE_PRIM       = 2'b10;

    localparam FETCH_FLAG_TILE  = 2'b00;
    localparam FETCH_FLAG_PID   = 2'b01;
    localparam FETCH_FLAG_PDATA = 2'b10;

    // Coarse-bin header (rast_bin_header_t):
    //   word0 = {bin_y[31:16], bin_x[15:0]}
    //   word1 = pids_offset (ABSOLUTE u32 index into the sorted-pid array)
    //   word2 = pids_count  (u32)
    // 12 B = 3 words.
    localparam TILE_HEADER_SIZEW = 12 / 4;

    // A primitive data contains (xloc, yloc, pid, edges[, zplane]). The fetch is
    // 9 words (edges) without early-Z, 12 words (edges + depth plane) with it.
    localparam PRIM_DATA_WIDTH = 2 * `VX_RASTER_DIM_BITS + NUM_REQS * `RASTER_DATA_BITS + `VX_RASTER_PID_BITS;

    // Storage to cycle through all primitives and tiles
    reg [W_ADDR_BITS-1:0]       next_tbuf_addr;
    reg [W_ADDR_BITS-1:0]       curr_pbuf_addr;
    reg [`VX_RASTER_PID_BITS-1:0] curr_pid_reqs;
    reg [`VX_RASTER_PID_BITS-1:0] curr_pid_rsps;
    reg [`RASTER_TILE_BITS-1:0] curr_num_tiles;
    reg [`VX_RASTER_DIM_BITS-1:0] curr_xloc;
    reg [`VX_RASTER_DIM_BITS-1:0] curr_yloc;

    // Output buffer
    wire buf_in_valid;
    wire buf_in_ready;

    // Memory request
    reg mem_req_valid, mem_req_valid_qual;
    reg [NUM_REQS-1:0] mem_req_mask;
    reg [NUM_REQS-1:0][W_ADDR_BITS-1:0] mem_req_addr;
    reg [TAG_WIDTH-1:0] mem_req_tag;
    wire mem_req_ready;

    // Memory response
    wire mem_rsp_valid;
    wire [NUM_REQS-1:0][`RASTER_DATA_BITS-1:0] mem_rsp_data;
    wire [TAG_WIDTH-1:0] mem_rsp_tag;
    wire mem_rsp_ready;

     // Primitive info
    wire [W_ADDR_BITS-1:0] pids_addr;
    wire prim_id_rsp_valid;
    wire prim_data_rsp_valid;
    wire prim_addr_rsp_valid;
    wire prim_addr_rsp_ready;
    wire [NUM_REQS-1:0][W_ADDR_BITS-1:0] prim_mem_addr;
    wire [`VX_RASTER_PID_BITS-1:0] primitive_id;

    // Memory fetch FSM

    reg [FSM_BITS-1:0] state;

    wire is_prim_id_req   = (mem_req_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PID);
    wire is_prim_id_rsp   = (mem_rsp_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PID);

    wire is_prim_data_req = (mem_req_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PDATA);
    wire is_prim_data_rsp = (mem_rsp_tag[FETCH_FLAG_BITS-1:0] == FETCH_FLAG_PDATA);

    wire mem_req_fire = mem_req_valid_qual && mem_req_ready;

    wire prim_addr_rsp_fire = prim_addr_rsp_valid && prim_addr_rsp_ready;

    wire prim_data_rsp_fire = prim_data_rsp_valid && mem_rsp_ready;

    // coarse-bin header info (rast_bin_header_t). pids_offset/count are
    // full 32-bit fields.
    wire [15:0] th_tile_pos_x  = mem_rsp_data[0][0  +: 16];
    wire [15:0] th_tile_pos_y  = mem_rsp_data[0][16 +: 16];
    wire [31:0] th_pids_offset = mem_rsp_data[1];
    wire [31:0] th_pids_count  = mem_rsp_data[2];
    `UNUSED_VAR (th_pids_offset)
    `UNUSED_VAR (th_pids_count)

    // calculate tile start info
    wire [`RASTER_TILE_BITS-1:0] start_tile_count = (dcrs.tile_count + `RASTER_TILE_BITS'(NUM_INSTANCES - 1 - INSTANCE_IDX)) >> LOG2_NUM_INSTANCES;
    wire [W_ADDR_BITS-1:0] start_tbuf_addr = {dcrs.tbuf_addr, 4'b0} + W_ADDR_BITS'(INSTANCE_IDX * TILE_HEADER_SIZEW);

    // The sorted-pid array is a single dense block that follows ALL bin headers
    // (num_bins == dcrs.tile_count of them, 3 words each), shared across the N
    // striped instances. pids_offset is an ABSOLUTE word index into it.
    wire [W_ADDR_BITS-1:0] pids_array_base = {dcrs.tbuf_addr, 4'b0}
        + W_ADDR_BITS'(dcrs.tile_count) * W_ADDR_BITS'(TILE_HEADER_SIZEW);

    // calculate address of primitive ids
    assign pids_addr = pids_array_base + W_ADDR_BITS'(th_pids_offset);

    // scheduler FSM
    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
            mem_req_valid <= 0;
        end else begin
            // deassert memory request when fired
            if (mem_req_fire) begin
                mem_req_valid <= 0;
            end

            case (state)
            STATE_IDLE: begin
                // fetch the next tile header
                if (start && (start_tile_count != 0)) begin
                    mem_req_valid <= 1;
                    state <= STATE_TILE;
                end
                mem_req_addr[0] <= start_tbuf_addr;
                mem_req_addr[1] <= start_tbuf_addr + W_ADDR_BITS'(1);
                mem_req_addr[2] <= start_tbuf_addr + W_ADDR_BITS'(2);
                mem_req_mask    <= NUM_REQS'('b111);
                mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_TILE);
                // update tile counters
                next_tbuf_addr  <= start_tbuf_addr + W_ADDR_BITS'(NUM_INSTANCES * TILE_HEADER_SIZEW);
                curr_num_tiles  <= start_tile_count;
            end
            STATE_TILE: begin
                if (mem_rsp_valid) begin
                    // handle tile header response
                    curr_xloc       <= `VX_RASTER_DIM_BITS'(th_tile_pos_x << TILE_LOGSIZE);
                    curr_yloc       <= `VX_RASTER_DIM_BITS'(th_tile_pos_y << TILE_LOGSIZE);
                    if (th_pids_count == '0) begin
                        // Empty bin: no primitives cover this tile. The front
                        // end emits one header per tile over the dense grid, so a
                        // culled / uncovered tile carries pids_count=0. Skip the
                        // pid/prim fetch entirely. Fetching a pid here
                        // would pull a STALE pid from the reused pid pool of a prior
                        // draw (the DCR tbuf/pbuf persist across draws) and rasterize
                        // its primitive.
                        if (curr_num_tiles == 1) begin
                            // last tile — engine drained
                            state <= STATE_IDLE;
                        end else begin
                            // advance to the next tile header
                            state           <= STATE_TILE;
                            mem_req_valid   <= 1;
                            mem_req_mask    <= NUM_REQS'('b111);
                            mem_req_addr[0] <= next_tbuf_addr;
                            mem_req_addr[1] <= next_tbuf_addr + W_ADDR_BITS'(1);
                            mem_req_addr[2] <= next_tbuf_addr + W_ADDR_BITS'(2);
                            mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_TILE);
                            next_tbuf_addr  <= next_tbuf_addr + W_ADDR_BITS'(NUM_INSTANCES * TILE_HEADER_SIZEW);
                        end
                        curr_num_tiles  <= curr_num_tiles - `RASTER_TILE_BITS'(1);
                    end else begin
                        state           <= STATE_PRIM;
                        // fetch next primitive pid
                        mem_req_valid   <= 1;
                        mem_req_addr[0] <= pids_addr;
                        mem_req_mask    <= NUM_REQS'('b1);
                        mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_PID);
                        // set primitive counters
                        curr_pbuf_addr  <= pids_addr;
                        curr_pid_reqs   <= `VX_RASTER_PID_BITS'(th_pids_count);
                        curr_pid_rsps   <= `VX_RASTER_PID_BITS'(th_pids_count);
                    end
                end
            end
            STATE_PRIM: begin
                // handle memory submissions
                if (mem_req_fire) begin
                    if (is_prim_id_req) begin
                        // update pid counters
                        curr_pbuf_addr <= curr_pbuf_addr + W_ADDR_BITS'(1);
                        curr_pid_reqs  <= curr_pid_reqs - `VX_RASTER_PID_BITS'(1);
                    end

                    if ((curr_pid_reqs > 1)
                     || (curr_pid_reqs == 1 && ~is_prim_id_req)) begin
                        // fetch next primitive pid
                        mem_req_valid   <= 1;
                        mem_req_mask    <= NUM_REQS'('b1);
                        mem_req_addr[0] <= curr_pbuf_addr + W_ADDR_BITS'(is_prim_id_req ? 1 : 0);
                        mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_PID);
                    end
                end

                // handle primitive address response (all edge[+zplane] words)
                if (prim_addr_rsp_fire) begin
                    mem_req_valid <= 1;
                    mem_req_mask  <= {NUM_REQS{1'b1}};
                    mem_req_addr  <= prim_mem_addr;
                    mem_req_tag   <= TAG_WIDTH'({primitive_id, FETCH_FLAG_PDATA});
                end

                // handle primitive data response
                if (prim_data_rsp_fire) begin
                    if (curr_pid_rsps == 1) begin
                        if (curr_num_tiles == 1) begin
                            // done, return to idle
                            state <= STATE_IDLE;
                        end else begin
                            // fetch the next tile header
                            state           <= STATE_TILE;
                            mem_req_valid   <= 1;
                            mem_req_mask    <= NUM_REQS'('b111);
                            mem_req_addr[0] <= next_tbuf_addr;
                            mem_req_addr[1] <= next_tbuf_addr + W_ADDR_BITS'(1);
                            mem_req_addr[2] <= next_tbuf_addr + W_ADDR_BITS'(2);
                            mem_req_tag     <= TAG_WIDTH'(FETCH_FLAG_TILE);
                            next_tbuf_addr  <= next_tbuf_addr + W_ADDR_BITS'(NUM_INSTANCES * TILE_HEADER_SIZEW);
                        end
                        // update tile counter
                        curr_num_tiles <= curr_num_tiles - `RASTER_TILE_BITS'(1);
                    end
                    // update pid counter
                    curr_pid_rsps <= curr_pid_rsps - `VX_RASTER_PID_BITS'(1);
                end
            end
            default:;
            endcase
        end
    end

    // Memory scheduler

    // ensure that we have space in the output buffer to prevent memory deadlock
    wire pending_output_full;
    VX_pending_size #(
        .SIZE (QUEUE_SIZE-1)
    ) pending_reads (
        .clk   (clk),
        .reset (reset),
        .incr  (mem_req_fire && is_prim_id_req),
        .decr  (valid_out && ready_out),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (alm_empty),
        .full  (pending_output_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    assign mem_req_valid_qual = mem_req_valid && (~is_prim_id_req || ~pending_output_full);

    // the memory response is for primitive id
    assign prim_id_rsp_valid = mem_rsp_valid && is_prim_id_rsp;

    // the memory response is for primitive data
    assign prim_data_rsp_valid = mem_rsp_valid && is_prim_data_rsp;

    // stall primitive address handling if primitive data fetch stalls
    wire prim_data_req_stall = mem_req_valid && is_prim_data_req && ~mem_req_ready;
    assign prim_addr_rsp_ready = ~prim_data_req_stall || ~prim_addr_rsp_valid;

    // Push primitive data into output buffer
    assign buf_in_valid = prim_data_rsp_valid;

    // stall the memory response
    assign mem_rsp_ready = (~prim_id_rsp_valid || prim_addr_rsp_ready)
                        && (~prim_data_rsp_valid || buf_in_ready);

    wire [NUM_REQS-1:0][RCACHE_ADDR_WIDTH-1:0] mem_req_addr_w;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_mem_req_addr_w
        assign mem_req_addr_w[i] = RCACHE_ADDR_WIDTH'(mem_req_addr[i]);
    end

    wire [NUM_REQS-1:0][RCACHE_WORD_SIZE-1:0] mem_req_byteen;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_mem_req_byteen
        assign mem_req_byteen[i] = {RCACHE_WORD_SIZE{1'b1}};
    end

    // schedule memory request

    VX_lsu_mem_if #(
        .NUM_LANES (RCACHE_NUM_REQS),
        .DATA_SIZE (`RASTER_DATA_BITS / 8),
        .TAG_WIDTH (RCACHE_FETCH_TAG_WIDTH)
    ) lsu_mem_if();

    // Pad the raster tag with leading UUID zeros so the scheduler can
    // slice off the upper UUID_WIDTH bits unconditionally (matches the
    // tex/om pattern). Raster has no instruction UUID — those bits stay '0.
    localparam SCHED_TAG_WIDTH = UUID_WIDTH + TAG_WIDTH;
    wire [SCHED_TAG_WIDTH-1:0] sched_req_tag = {UUID_WIDTH'(0), mem_req_tag};
    wire [SCHED_TAG_WIDTH-1:0] sched_rsp_tag;
    assign mem_rsp_tag = sched_rsp_tag[TAG_WIDTH-1:0];
    `UNUSED_VAR (sched_rsp_tag)

    VX_mem_scheduler #(
        .INSTANCE_ID  ($sformatf("%s-memsched", INSTANCE_ID)),
        .CORE_REQS    (NUM_REQS),
        .MEM_CHANNELS (RCACHE_NUM_REQS),
        .WORD_SIZE    (`RASTER_DATA_BITS / 8),
        .ADDR_WIDTH   (RCACHE_ADDR_WIDTH),
        .USER_WIDTH   (0),
        .TAG_WIDTH    (SCHED_TAG_WIDTH),
        .CORE_QUEUE_SIZE(`VX_CFG_RASTER_MEM_QUEUE_SIZE),
        .UUID_WIDTH   (UUID_WIDTH),
        .RSP_PARTIAL  (0),
        .RW_ENABLE    (0),
        .MEM_OUT_BUF  (3),
        .CORE_OUT_BUF (3)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .core_req_valid (mem_req_valid_qual),
        .core_req_rw    (1'b0),
        .core_req_mask  (mem_req_mask),
        .core_req_byteen(mem_req_byteen),
        .core_req_addr  (mem_req_addr_w),
        .core_req_user  ('0),
        .core_req_data  ('0),
        .core_req_tag   (sched_req_tag),
        .core_req_ready (mem_req_ready),
        `UNUSED_PIN (req_queue_empty),
        `UNUSED_PIN (req_queue_rw_notify),

        // Output response
        .core_rsp_valid (mem_rsp_valid),
        `UNUSED_PIN (core_rsp_mask),
        .core_rsp_data  (mem_rsp_data),
        .core_rsp_tag   (sched_rsp_tag),
        `UNUSED_PIN (core_rsp_sop),
        `UNUSED_PIN (core_rsp_eop),
        .core_rsp_ready (mem_rsp_ready),

        // Memory request
        .mem_req_valid  (lsu_mem_if.req_valid),
        .mem_req_rw     (lsu_mem_if.req_data.rw),
        .mem_req_mask   (lsu_mem_if.req_data.mask),
        .mem_req_byteen (lsu_mem_if.req_data.byteen),
        .mem_req_addr   (lsu_mem_if.req_data.addr),
        `UNUSED_PIN (mem_req_user),
        .mem_req_data   (lsu_mem_if.req_data.data),
        .mem_req_tag    (lsu_mem_if.req_data.tag),
        .mem_req_ready  (lsu_mem_if.req_ready),

        // Memory response
        .mem_rsp_valid  (lsu_mem_if.rsp_valid),
        .mem_rsp_mask   (lsu_mem_if.rsp_data.mask),
        .mem_rsp_data   (lsu_mem_if.rsp_data.data),
        .mem_rsp_tag    (lsu_mem_if.rsp_data.tag),
        .mem_rsp_ready  (lsu_mem_if.rsp_ready)
    );

    // Raster never sets any memory attr; tie off the scheduler-driven LSU bus.
    assign lsu_mem_if.req_data.user = '0;

    VX_lsu_adapter #(
        .NUM_LANES    (RCACHE_NUM_REQS),
        .DATA_SIZE    (4),
        .TAG_WIDTH    (RCACHE_FETCH_TAG_WIDTH),
        .TAG_SEL_BITS (RCACHE_FETCH_TAG_WIDTH),
        .REQ_OUT_BUF  (0),
        .RSP_OUT_BUF  (0)
    ) lsu_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (lsu_mem_if),
        .mem_bus_if (cache_bus_if)
    );

    wire [31:0] prim_mem_offset;
    `UNUSED_VAR (prim_mem_offset)

    VX_multiplier #(
        .A_WIDTH (`RASTER_DATA_BITS),
        .B_WIDTH (`VX_RASTER_STRIDE_BITS),
        .R_WIDTH (32),
        .LATENCY (`LATENCY_IMUL)
    ) multiplier (
        .clk    (clk),
        .enable (prim_addr_rsp_ready),
        .dataa  (mem_rsp_data[0]),
        .datab  (dcrs.pbuf_stride),
        .result (prim_mem_offset)
    );

    // Words 0..8 = edge coefficients, 9..11 = depth plane {A',B',C'} (contiguous
    // in the primitive record right after the edges).
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_prim_mem_addr
        wire [W_ADDR_BITS-1:0] offset = W_ADDR_BITS'(prim_mem_offset[31:2]) + W_ADDR_BITS'(1 * i);
        assign prim_mem_addr[i] = {dcrs.pbuf_addr, 4'b0} + offset;
    end

    VX_shift_register #(
        .DATAW  (1 + `VX_RASTER_PID_BITS),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) mul_shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (prim_addr_rsp_ready),
        .data_in  ({prim_id_rsp_valid,   mem_rsp_data[0][`VX_RASTER_PID_BITS-1:0]}),
        .data_out ({prim_addr_rsp_valid, primitive_id})
    );

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Split the 12-word primitive response into edges (words 0..8) and the
    // depth plane (words 9..11) for the output buffer.
    wire [8:0][`RASTER_DATA_BITS-1:0] rsp_edges  = mem_rsp_data[0 +: 9];
    wire [2:0][`RASTER_DATA_BITS-1:0] rsp_zplane = mem_rsp_data[9 +: 3];

    // Output buffer
    VX_elastic_buffer #(
        .DATAW   (PRIM_DATA_WIDTH),
        .SIZE    (QUEUE_SIZE),
        .OUT_REG (1)
    ) buf_out (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (buf_in_valid),
        .ready_in   (buf_in_ready),
        .data_in    ({curr_xloc, curr_yloc, rsp_edges, rsp_zplane, mem_rsp_tag[FETCH_FLAG_BITS +: `VX_RASTER_PID_BITS]}),
        .data_out   ({xloc_out,  yloc_out,  edges_out,  zplane_out, pid_out}),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );
`else
    // Output buffer (edges only; no depth plane without early-Z).
    VX_elastic_buffer #(
        .DATAW   (PRIM_DATA_WIDTH),
        .SIZE    (QUEUE_SIZE),
        .OUT_REG (1)
    ) buf_out (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (buf_in_valid),
        .ready_in   (buf_in_ready),
        .data_in    ({curr_xloc, curr_yloc, mem_rsp_data, mem_rsp_tag[FETCH_FLAG_BITS +: `VX_RASTER_PID_BITS]}),
        .data_out   ({xloc_out,  yloc_out,  edges_out,   pid_out}),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );
`endif

    assign busy = (state != STATE_IDLE);

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (valid_out && ready_out) begin
            `TRACE(2, ("%d: %s-mem-out: x=%0d, y=%0d, pid=%0d, edge={{0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}, {0x%0h, 0x%0h, 0x%0h}}\n",
                $time, INSTANCE_ID, xloc_out, yloc_out, pid_out,
                edges_out[0][0], edges_out[0][1], edges_out[0][2],
                edges_out[1][0], edges_out[1][1], edges_out[1][2],
                edges_out[2][0], edges_out[2][1], edges_out[2][2]))
        end
    end
`endif

endmodule
