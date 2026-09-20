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

//------------------------------------------------------------------------------
// FEOP accumulator: holds the accumulator fragment of each FEOP
// - Read:  per-element address (0..M*N*TCU_FEOP_STEPS-1)
// - Write: per-element address + data + accumulate flag
//------------------------------------------------------------------------------
`include "VX_define.vh"

`ifdef TCU_OP
// The exact-product datapath (VX_CFG_TCU_TYPE_TFR) uses VX_tcu_op_mul and
// VX_tcu_op_accu instead; this module's BHF/DPI-only parameter lists do not
// elaborate without one of those backends selected.
`ifndef VX_CFG_TCU_TYPE_TFR

module VX_tcu_feop_accu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter int BLOCK_M          = 2,
    parameter int BLOCK_N          = 16,
    parameter int FREC_LATENCY     = 0,
    parameter int FADD_LATENCY     = 1,
    parameter int FRND_LATENCY     = 0,
    parameter int FACC_LATENCY     = FREC_LATENCY + FADD_LATENCY + FRND_LATENCY,
    parameter int XBAR_LATENCY     = 1,
    parameter int XBAR_QUEUE_DEPTH = 1
) (
    input wire       clk,
    input wire       reset,
    input wire       enable,
    input wire [3:0] fmt_d,

    // ---- READ PORT: per-element addressing ----
    input  wire                                  read_en,
    input  wire [BLOCK_M-1:0]                    read_row_valid,
    input  wire [$clog2(TCU_FEOP_STEPS)-1:0]     read_block_idx,
    output wire [BLOCK_M*BLOCK_N-1:0][`VX_CFG_XLEN-1:0] read_data,

    // ---- WRITE PORT: per-element addressing ----
    input  wire                                        write_valid,
    output wire                                        write_ready,
    input  wire [BLOCK_M-1:0][$clog2(TCU_TC_M_OP)-1:0] write_addr_row,
    input  wire [BLOCK_M-1:0]                          write_addr_row_valid,
    input  wire [BLOCK_N-1:0][$clog2(TCU_TC_N_OP)-1:0] write_addr_col,
    input  wire [BLOCK_N-1:0]                          write_addr_col_valid,
    input  wire [BLOCK_M*BLOCK_N-1:0][`VX_CFG_XLEN-1:0]       write_data,
    input  wire                                        overwrite,            // 1: overwrite, 0: accumulate

    output wire                                        accu_ready_to_flush   // Must be empty in order to commence the flushing
);

    localparam LG_BLOCK_M  = $clog2(BLOCK_M);
    localparam LG_BLOCK_N  = $clog2(BLOCK_N);
    localparam int BANKS   = BLOCK_M * BLOCK_N;
    localparam int SLOTS   = TCU_FEOP_STEPS;
    localparam int BANK_AW = $clog2(SLOTS);
    localparam int BANK_BW = $clog2(BANKS);
    localparam [`VX_CFG_XLEN-1:0] ACCU_INIT_VALUE = `VX_CFG_XLEN'(32'hf0ffffff); // Junk values to detect uninitialized reads

    // One crossbar input per FEOP product, one output per accumulator bank.
    localparam int XBAR_INPUTS  = BANKS;
    localparam int XBAR_OUTPUTS = BANKS;
    localparam int XBAR_SELW    = $clog2(XBAR_OUTPUTS);
    localparam int XBAR_DATAW   = 1 + BANK_AW + `VX_CFG_XLEN; // {from_queue, slot, data}
    localparam XBAR_QUEUE_WIDTH = `VX_CFG_XLEN + BANK_AW + BANK_BW; // {data, slot, bank}

    `STATIC_ASSERT (FACC_LATENCY == FREC_LATENCY + FADD_LATENCY + FRND_LATENCY, ("feop_accu: FACC_LATENCY must equal FREC_LATENCY + FADD_LATENCY + FRND_LATENCY"))
    `STATIC_ASSERT (XBAR_INPUTS <= 32, ("feop_accu: crossbar supports at most 32 inputs, got %0d", XBAR_INPUTS))

    initial begin
        `TRACE(1, ("[feop_accu]: Parameters: BLOCK_M: %0d, BLOCK_N: %0d, XBAR_QUEUE_DEPTH: %0d\n", BLOCK_M, BLOCK_N, XBAR_QUEUE_DEPTH))
    end

    wire [XBAR_INPUTS-1:0]  xbar_queue_full;
    wire [XBAR_INPUTS-1:0]  xbar_queue_empty;
    wire [XBAR_INPUTS-1:0]  xbar_ready_in;
    wire                    overwrite_delayed;
    wire [XBAR_OUTPUTS-1:0] xbar_valid_out;
    wire [XBAR_OUTPUTS-1:0] xbar_enable_out;
    wire [XBAR_OUTPUTS-1:0] bank_write_bitmap;
    // Read stage: registered crossbar output; reads its bank slot and feeds the adder.
    reg  [BANKS-1:0]                                rd_valid;
    reg  [BANKS-1:0][BANK_AW-1:0]                   rd_slot;
    reg  [BANKS-1:0][`VX_CFG_XLEN-1:0]              rd_data;
    reg                                             rd_overwrite;
    // Execute stage: registered bank read and product, the adder's inputs.
    reg  [BANKS-1:0]                                ex_valid;
    reg  [BANKS-1:0][BANK_AW-1:0]                   ex_slot;
    reg  [BANKS-1:0][`VX_CFG_XLEN-1:0]              ex_data;
    reg  [BANKS-1:0][`VX_CFG_XLEN-1:0]              ex_rdata;
    reg                                             ex_overwrite;
    // Accumulations in flight: stage k left the execute stage k+1 cycles ago.
    reg  [FACC_LATENCY-1:0][BANKS-1:0]              inflight_valid;
    reg  [FACC_LATENCY-1:0][BANKS-1:0][BANK_AW-1:0] inflight_slot;
    wire [BANKS-1:0][`VX_CFG_XLEN-1:0] bank_rdata;
    reg  [BANKS-1:0][`VX_CFG_XLEN-1:0] result;
    wire [BANKS-1:0][`VX_CFG_XLEN-1:0] result_delayed;

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ADDRESS CONVERSION

    wire [BANKS-1:0][BANK_AW-1:0] write_block;
    wire [BANKS-1:0][BANK_BW-1:0] write_bank;

    for (genvar i = 0; i < BANKS; ++i) begin : g_addr_decode
        wire [$clog2(TCU_TC_M_OP)-1:0] write_row  = write_addr_row[i >> LG_BLOCK_N];
        wire [$clog2(TCU_TC_N_OP)-1:0] write_col  = write_addr_col[i & (BLOCK_N-1)];

        assign write_block[i] = BANK_AW'(((32'(write_row) >> LG_BLOCK_M) << LG_TCU_FEOP_N_STEPS) + (32'(write_col) >> LG_BLOCK_N));
        assign write_bank[i]  = BANK_BW'(((32'(write_row) & (BLOCK_M-1)) << LG_BLOCK_N) + (32'(write_col) & (BLOCK_N-1)));
    end

    // Address decode validity per lane.
    wire [XBAR_INPUTS-1:0] write_lane_valid;
    wire [XBAR_INPUTS-1:0] write_lane_ready = ~xbar_queue_full;
    // Stall only when an active lane's queue is full; ignore full queues on invalid lanes.
    wire [XBAR_INPUTS-1:0] write_lane_blocked = write_lane_valid & xbar_queue_full;
    assign write_ready = enable && (~|write_lane_blocked);
    wire write_fire = write_valid && write_ready;
    wire [XBAR_INPUTS-1:0] write_lane_fire = write_lane_valid & write_lane_ready & {XBAR_INPUTS{write_fire}};

    for (genvar i = 0; i < BLOCK_M; ++i) begin : g_write_lane_valid_d_row
        for (genvar j = 0; j < BLOCK_N; ++j) begin : g_write_lane_valid_d_col
            assign write_lane_valid[(i << LG_BLOCK_N) + j] = write_addr_row_valid[i] && write_addr_col_valid[j];
        end
    end

// ADDRESS CONVERSION
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// QUEUES

    // Ready to flush once drained: queues empty, nothing on (or held at) the
    // crossbar outputs, and no accumulation in flight.
    assign accu_ready_to_flush  = (&xbar_queue_empty) && (~|xbar_valid_out) && (~|rd_valid) && (~|ex_valid) && (~|inflight_valid);

    wire [XBAR_INPUTS-1:0] xbar_use_queue;
    wire [XBAR_INPUTS-1:0] xbar_queue_push;
    wire [XBAR_INPUTS-1:0] xbar_queue_pop;

    wire [XBAR_INPUTS-1:0][`VX_CFG_XLEN-1:0]   xbar_queue_data_out;
    wire [XBAR_INPUTS-1:0][BANK_AW-1:0] xbar_queue_slot_out;
    wire [XBAR_INPUTS-1:0][BANK_BW-1:0] xbar_queue_bank_out;

    for (genvar i = 0; i < XBAR_INPUTS; i++) begin : g_xbar_queue
        wire [XBAR_QUEUE_WIDTH-1:0] xbar_queue_din = {write_data[i], write_block[i], write_bank[i]};
        wire [XBAR_QUEUE_WIDTH-1:0] xbar_queue_dout;

        assign {xbar_queue_data_out[i], xbar_queue_slot_out[i], xbar_queue_bank_out[i]} = xbar_queue_dout;

        assign xbar_use_queue[i]  = ~write_lane_fire[i] && enable && ~xbar_queue_empty[i];
        assign xbar_queue_push[i] = write_lane_fire[i] && ~xbar_ready_in[i]; // If a new input arrives but the xbar is not ready to accept it, push it to the queue
        assign xbar_queue_pop[i]  =  xbar_use_queue[i] &&  xbar_ready_in[i]; // If there is no new input but the xbar is ready to accept data, pop from the queue

        VX_fifo_queue #(
            .DATAW (XBAR_QUEUE_WIDTH),
            .DEPTH (XBAR_QUEUE_DEPTH)
        ) xbar_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (xbar_queue_push[i]),
            .pop      (xbar_queue_pop[i]),
            .data_in  (xbar_queue_din),
            .data_out (xbar_queue_dout),
            .empty    (xbar_queue_empty[i]),
            `UNUSED_PIN(alm_empty),
            .full     (xbar_queue_full[i]),
            `UNUSED_PIN(alm_full),
            `UNUSED_PIN(size)
        );

    end

    always @(posedge clk) begin
        if (~reset && enable && (|xbar_queue_push || |xbar_queue_pop)) begin
            `TRACE(2, ("%t: [feop_accu]: xbar_queue_push=%b, xbar_queue_pop=%b\n", $time, xbar_queue_push, xbar_queue_pop));
        end
        if (~reset && enable && ~write_ready) begin
            `TRACE(1, ("%t: [feop_accu]: xbar queues are full - must stall\n", $time));
        end
        if (~reset && (|xbar_use_queue)) begin
            `TRACE(2, ("%t: [feop_accu]: using queue for inputs: xbar_use_queue=%b\n", $time, xbar_use_queue));
        end
    end

    // overwrite is not stored in the xbar queues; it follows its write through
    // the fixed-latency pipe_overwrite, so an overwrite write must never queue.
    `RUNTIME_ASSERT(~(enable && overwrite && ((|xbar_queue_push) || ~(&xbar_queue_empty))),
        ("%t: *** feop_accu: overwrite asserted while a crossbar queue is in use (push=%b empty=%b); the overwrite flag would misalign with its data", $time, xbar_queue_push, xbar_queue_empty))

// QUEUES
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// CROSSBAR

    wire [XBAR_INPUTS-1:0][XBAR_DATAW-1:0] xbar_data_in;
    wire [XBAR_INPUTS-1:0][XBAR_SELW-1:0]  xbar_sel_in;
    wire [XBAR_INPUTS-1:0]                 xbar_valid_in;

    wire [XBAR_OUTPUTS-1:0][XBAR_DATAW-1:0] xbar_data_out;
    wire [XBAR_OUTPUTS-1:0]                 xbar_ready_out;

    wire [PERF_CTR_BITS-1:0]  xbar_collisions;
    `UNUSED_VAR (xbar_collisions)

    wire [XBAR_OUTPUTS-1:0][BANK_AW-1:0] xbar_slot_out;
    wire [XBAR_OUTPUTS-1:0][`VX_CFG_XLEN-1:0]   xbar_result_out;
    wire [XBAR_OUTPUTS-1:0]              xbar_from_queue_out;

    for (genvar i = 0; i < XBAR_INPUTS; ++i) begin : g_xbar_inputs
        assign xbar_data_in[i]   = xbar_use_queue[i] ? {1'b1, xbar_queue_slot_out[i], xbar_queue_data_out[i]} :
                                                       {1'b0, write_block[i],          write_data[i]};
        assign xbar_sel_in[i]    = xbar_use_queue[i] ? xbar_queue_bank_out[i] : write_bank[i];
        assign xbar_valid_in[i]  = xbar_use_queue[i] || write_lane_fire[i]; // Input comes either from the queue or directly from the FEOP outputs
    end

    // A bank accumulates into a slot by reading it (read stage), adding over
    // FACC_LATENCY cycles from the execute stage, then writing it back. A
    // product whose slot has an accumulation in the read or execute stage, or
    // not yet at its write-back stage, waits at the crossbar output (and so in
    // its input queue). Overwrites read nothing and never wait.
    wire [XBAR_OUTPUTS-1:0] slot_hazard;

    for (genvar i = 0; i < XBAR_OUTPUTS; ++i) begin : g_xbar_outputs
        assign {xbar_from_queue_out[i], xbar_slot_out[i], xbar_result_out[i]} = xbar_data_out[i];

        wire [FACC_LATENCY:0] slot_match;
        assign slot_match[0] = rd_valid[i] && (rd_slot[i] == xbar_slot_out[i]);
        assign slot_match[1] = ex_valid[i] && (ex_slot[i] == xbar_slot_out[i]);
        for (genvar k = 2; k <= FACC_LATENCY; ++k) begin : g_match
            assign slot_match[k] = inflight_valid[k-2][i] && (inflight_slot[k-2][i] == xbar_slot_out[i]);
        end
        assign slot_hazard[i] = ~overwrite_delayed && (|slot_match);

        assign xbar_ready_out[i]  = enable && ~slot_hazard[i];
        assign xbar_enable_out[i] = xbar_valid_out[i] && xbar_ready_out[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS    (XBAR_INPUTS),
        .NUM_OUTPUTS   (XBAR_OUTPUTS),
        .DATAW         (XBAR_DATAW),
        .OUT_BUF       (0),
        .PERF_CTR_BITS (PERF_CTR_BITS)
    ) accu_xbar (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (xbar_valid_in),
        .data_in    (xbar_data_in),
        .sel_in     (xbar_sel_in),
        .ready_in   (xbar_ready_in),
        .valid_out  (xbar_valid_out),
        .data_out   (xbar_data_out),
        `UNUSED_PIN (sel_out),
        .ready_out  (xbar_ready_out),
        .collisions (xbar_collisions)
    );

    // Delays the overwrite control signal, until the data it refers to exit the crossbar
    VX_pipe_register #(
        .DATAW  (1),
        .RESETW (1),
        .DEPTH  (XBAR_LATENCY)
    ) pipe_overwrite (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (overwrite),
        .data_out (overwrite_delayed)
    );

// CROSSBAR
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ADDITION

    /* Control Signals delay */
    wire [XBAR_OUTPUTS-1:0][BANK_AW-1:0] bank_slots;
    `UNUSED_VAR (xbar_from_queue_out)

    // Models FACC latency for the write-back control; every stage is visible
    // to the hazard check above.
    always_ff @(posedge clk) begin
        if (reset) begin
            rd_valid       <= '0;
            rd_overwrite   <= 1'b0;
            ex_valid       <= '0;
            ex_overwrite   <= 1'b0;
            inflight_valid <= '0;
            inflight_slot  <= '0;
        end else if (enable) begin
            rd_valid          <= xbar_enable_out;
            rd_overwrite      <= overwrite_delayed;
            ex_valid          <= rd_valid;
            ex_overwrite      <= rd_overwrite;
            inflight_valid[0] <= ex_valid;
            inflight_slot[0]  <= ex_slot;
            for (int k = 1; k < FACC_LATENCY; ++k) begin
                inflight_valid[k] <= inflight_valid[k-1];
                inflight_slot[k]  <= inflight_slot[k-1];
            end
        end
    end
    always_ff @(posedge clk) begin
        if (enable) begin
            rd_slot  <= xbar_slot_out;
            rd_data  <= xbar_result_out;
            ex_slot  <= rd_slot;
            ex_data  <= rd_data;
            ex_rdata <= bank_rdata;
        end
    end
    assign bank_write_bitmap = inflight_valid[FACC_LATENCY-1];
    assign bank_slots        = inflight_slot[FACC_LATENCY-1];

    reg [63:0] sum64;
`ifdef VX_CFG_TCU_TYPE_DPI
    reg [4:0]  fflags_unused;
    `UNUSED_VAR (sum64[63:32]);
`endif
    wire [BANKS-1:0][`VX_CFG_XLEN-1:0] accum_source;

    for (genvar j = 0; j < BANKS; ++j) begin : g_accum_source
        assign accum_source[j] = ex_overwrite ? `VX_CFG_XLEN'(0) : ex_rdata[j];
    end

`ifdef VX_CFG_TCU_TYPE_BHF

    `UNUSED_VAR (sum64[63:0]);
    wire [BANKS-1:0][31:0] bhf_fadd_result;
    wire [BANKS-1:0][4:0]  bhf_fadd_fflags;
    wire [BANKS-1:0][31:0] int_sum_now;
    wire [BANKS-1:0][31:0] int_sum_delayed;
    for (genvar j = 0; j < BANKS; ++j) begin : g_bhf_fadd
        assign int_sum_now[j] = accum_source[j] + ex_data[j];

        VX_tcu_bhf_fadd #(
            .IN_EXPW      (8),
            .IN_SIGW      (24),
            .IN_REC       (0),
            .OUT_REC      (0),
            .REC_LATENCY  (FREC_LATENCY),
            .ADD_LATENCY  (FADD_LATENCY),
            .RND_LATENCY  (FRND_LATENCY)
        ) fp32_accum_adder (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (3'b000),
            .a      (accum_source[j]),
            .b      (ex_data[j]),
            .y      (bhf_fadd_result[j]),
            .fflags (bhf_fadd_fflags[j])
        );
    end

    VX_pipe_register #(
        .DATAW  (BANKS * 32),
        .DEPTH  (FACC_LATENCY)
    ) pipe_int_sum (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (int_sum_now),
        .data_out (int_sum_delayed)
    );

    `UNUSED_VAR(bhf_fadd_fflags)
`endif

    always_comb begin
        result = '0;
        sum64  = '0;
`ifdef VX_CFG_TCU_TYPE_DPI
        fflags_unused = '0;
`endif

        for (integer j = 0; j < BANKS; ++j) begin
            // Adder operates only if input is valid - energy saving
`ifdef VX_CFG_TCU_TYPE_BHF
            if (~reset && enable && bank_write_bitmap[j]) begin
                if (32'(fmt_d) == TCU_I32_ID) begin // int32
                    result[j] = int_sum_delayed[j];
                end else begin // all floating-point accumulator formats use fp32 accumulation
                    result[j] = bhf_fadd_result[j];
                end
            end
`else
            if (~reset && enable && ex_valid[j]) begin
                if (32'(fmt_d) == TCU_I32_ID) begin // int32
                    sum64 = {{32{accum_source[j][31]}}, accum_source[j]} + {{32{ex_data[j][31]}}, ex_data[j]};
                end else begin // all floating-point accumulator formats use fp32 accumulation
`ifdef VX_CFG_TCU_TYPE_DPI
                    dpi_fadd(enable, int'(0),
                             {32'hffffffff, accum_source[j]},
                             {32'hffffffff, ex_data[j]},
                             3'b0, sum64, fflags_unused);
`endif
                end

                result[j] = sum64[31:0];
            end
`endif
        end
    end

    // Delay the accumulation result by FACC_LATENCY cycles
    VX_pipe_register #(
        .DATAW (`VX_CFG_XLEN * BLOCK_M * BLOCK_N),
`ifdef VX_CFG_TCU_TYPE_DPI
        .DEPTH (FACC_LATENCY)
`elsif VX_CFG_TCU_TYPE_BHF
        .DEPTH (0)
`endif
    ) pipe_acc (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (result),
        .data_out (result_delayed)
    );

// ADDITION
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ACCUMULATOR BUFFER

    // Accumulator buffer: BANKS banks (one per element), depth=SLOTS
    wire [BANKS-1:0]              bank_read_en;
    wire [BANKS-1:0][BANK_AW-1:0] bank_raddr;
    wire [BANKS-1:0]              bank_write_en;
    wire [BANKS-1:0][BANK_AW-1:0] bank_waddr;
    wire [BANKS-1:0][`VX_CFG_XLEN-1:0]   bank_wdata;

    for (genvar b = 0; b < BANKS; ++b) begin : g_bank_ctrl
        /*             read because we are flushing or   read because we are in the gather stage */
        assign bank_read_en[b] = enable && ((read_en && read_row_valid[b >> LG_BLOCK_N]) || (~rd_overwrite && rd_valid[b]));

        assign bank_raddr[b]   = (enable && read_en) ? read_block_idx : rd_slot[b]; // flush read, else accumulate read

        assign bank_write_en[b] =  enable && bank_write_bitmap[b];
        assign bank_waddr[b]    = (enable && bank_write_bitmap[b]) ? bank_slots[b]     : '0;
        assign bank_wdata[b]    = (enable && bank_write_bitmap[b]) ? result_delayed[b] : '0;
    end

    // Flush reads happen only once drained (asserted below), so no bypass.
    for (genvar i = 0; i < BANKS; ++i) begin : g_read_out
        assign read_data[i] = read_en ? bank_rdata[i] : '0;
    end

    for (genvar b = 0; b < BANKS; ++b) begin : g_accu_banks
        VX_dp_ram #(
            .DATAW      (`VX_CFG_XLEN),
            .SIZE       (SLOTS),
            .OUT_REG    (0),
            .RDW_MODE   ("W"), // a write is visible to the next cycle's read (slot_hazard window)
            .RESET_RAM  (1),
            .INIT_VALUE (ACCU_INIT_VALUE)
        ) accu_mem (
            .clk   (clk),
            .reset (reset),
            .read  (bank_read_en[b]),
            .write (bank_write_en[b]),
            .wren  (1'b1),
            .waddr (bank_waddr[b]),
            .wdata (bank_wdata[b]),
            .raddr (bank_raddr[b]),
            .rdata (bank_rdata[b])
        );
    end

// ACCUMULATOR BUFFER
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// DEBUGGING

    // A flush reads the accumulator while no product may be in flight to it.
    `RUNTIME_ASSERT(~(enable && read_en && ((|xbar_enable_out) || (|rd_valid) || (|ex_valid))),
        ("%t: *** feop_accu: flush read while the crossbar is delivering products", $time))
    `RUNTIME_ASSERT(~(enable && read_en && (|bank_write_en)),
        ("%t: *** feop_accu: flush read while accumulator banks are being written", $time))

`ifdef SIMULATION
    // Scoreboard: products accepted at the write port but not yet written.
    integer dbg_pending;
    always @(posedge clk) begin
        if (reset) begin
            dbg_pending <= 0;
        end else begin
            dbg_pending <= dbg_pending + $countones(write_lane_fire) - $countones(bank_write_en);
        end
    end
    `RUNTIME_ASSERT(~(accu_ready_to_flush && (dbg_pending != 0)),
        ("%t: *** feop_accu: reports drained with %0d products pending", $time, dbg_pending))
`endif

    always_ff @(posedge clk) begin
        if (~reset && enable && (|xbar_valid_in || |xbar_enable_out || |bank_write_bitmap)) begin
            `TRACE(2, ("%t: [feop_accu]: |xbar_valid_in=%b, write_block[0]=%0d, |xbar_enable_out=%b, xbar_slot_out[0]=%0d, |bank_write_bitmap=%b\n",
                       $time, |xbar_valid_in, write_block[0], |xbar_enable_out, xbar_slot_out[0], |bank_write_bitmap))
        end
        if (~reset && enable && write_fire) begin
            `TRACE(2, ("%t: [feop_accu]: WRITE req row_valid=%b, col_valid=%b\n",
                       $time, write_addr_row_valid, write_addr_col_valid))
        end
        if (~reset && enable && read_en) begin
            `TRACE(2, ("%t: [feop_accu]: read_en active, read_block_idx=%0d, |xbar_enable_out=%b, ~overwrite_delayed=%b, |xbar_slot_out[0]=%0d\n",
                       $time, read_block_idx, |xbar_enable_out, ~overwrite_delayed, |xbar_slot_out[0]))
        end
    end

endmodule

`endif // !VX_CFG_TCU_TYPE_TFR
`endif // TCU_OP
