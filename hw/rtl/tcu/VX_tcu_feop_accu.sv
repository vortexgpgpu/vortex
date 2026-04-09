//------------------------------------------------------------------------------
// FEOP accumulator: holds the accumulator fragment of each FEOP
// - Read:  per-element address (0..M*N*TCU_FEOP_STEPS-1)
// - Write: per-element address + data + accumulate flag
//------------------------------------------------------------------------------
`include "VX_define.vh"

module VX_tcu_feop_accu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter int BLOCK_M          = 2,
    parameter int BLOCK_N          = 16,
    parameter int FADD_LATENCY     = 1,
    parameter int FRND_LATENCY     = 0, // TODO: Change to 1...
    parameter int FACC_LATENCY     = FADD_LATENCY + FRND_LATENCY,
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
    output wire [BLOCK_M*BLOCK_N-1:0][`XLEN-1:0] read_data,

    // ---- WRITE PORT: per-element addressing ----
    input  wire                                        write_valid,
    output wire                                        write_ready,
    input  wire [BLOCK_M-1:0][$clog2(TCU_TC_M_OP)-1:0] write_addr_row,
    input  wire [BLOCK_M-1:0]                          write_addr_row_valid,
    input  wire [BLOCK_N-1:0][$clog2(TCU_TC_N_OP)-1:0] write_addr_col,
    input  wire [BLOCK_N-1:0]                          write_addr_col_valid,
    input  wire [BLOCK_M*BLOCK_N-1:0][`XLEN-1:0]       write_data,
    input  wire                                        overwrite,            // 1: overwrite, 0: accumulate

    output wire                                        accu_ready_to_flush   // Must be empty in order to commence the flushing
);

    localparam LG_BLOCK_M  = $clog2(BLOCK_M);
    localparam LG_BLOCK_N  = $clog2(BLOCK_N);
    localparam int BANKS   = BLOCK_M * BLOCK_N;
    localparam int SLOTS   = TCU_FEOP_STEPS;
    localparam int BANK_AW = $clog2(SLOTS);
    localparam int BANK_BW = $clog2(BANKS);
    localparam [`XLEN-1:0] ACCU_INIT_VALUE = `XLEN'(32'hf0ffffff); // Junk values to detect uninitialized reads

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
    // If a lane queue is full, only block when xbar cannot take this lane this cycle
    // (e.g. due to bank conflict arbitration).
    // TODO: Try adding the (& ~xbar_ready_in) condition in write_lane_blocked : not easy
    wire [XBAR_INPUTS-1:0] write_lane_blocked = write_lane_valid & xbar_queue_full;
    wire [XBAR_INPUTS-1:0] write_lane_fire = write_lane_valid & write_lane_ready & {XBAR_INPUTS{write_fire}};
    // Stall only when an active lane is blocked; ignore full queues on invalid lanes.
    assign write_ready = enable && (~|write_lane_blocked);
    wire write_fire = write_valid && write_ready;

    for (genvar i = 0; i < BLOCK_M; ++i) begin : g_write_lane_valid_d_row
        for (genvar j = 0; j < BLOCK_N; ++j) begin : g_write_lane_valid_d_col
            assign write_lane_valid[(i << LG_BLOCK_N) + j] = write_addr_row_valid[i] && write_addr_col_valid[j];
        end 
    end

// ADDRESS CONVERSION
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// QUEUES

    localparam XBAR_QUEUE_WIDTH = `XLEN + BANK_AW + BANK_BW; // {data, slot, bank}

    wire [XBAR_INPUTS-1:0] xbar_queue_full;
    wire [XBAR_INPUTS-1:0] xbar_queue_empty;

    /*     ready to flush if: all queues are drained, no data on the xbar output, no data remaining to be written to ACCU */
    assign accu_ready_to_flush  = (&xbar_queue_empty) && (~|xbar_enable_out) && (~|bank_write_bitmap); // Used during flushing to check if all queues are empty before accepting new inputs directly to the crossbar

    wire [XBAR_INPUTS-1:0] xbar_use_queue;
    wire [XBAR_INPUTS-1:0] xbar_queue_push;
    wire [XBAR_INPUTS-1:0] xbar_queue_pop;

    wire [XBAR_INPUTS-1:0][`XLEN-1:0]   xbar_queue_data_out;
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

    `UNUSED_VAR(xbar_collisions)
    always @(posedge clk) begin
        if (~reset && enable && (|xbar_queue_push || |xbar_queue_pop)) begin
            `TRACE(1, ("%t: [feop_accu]: xbar_queue_push=%b, xbar_queue_pop=%b\n", $time, xbar_queue_push, xbar_queue_pop));
        end
        if (~reset && enable && ~write_ready) begin
            `TRACE(1, ("%t: [feop_accu]: xbar queues are full - must stall\n", $time));
        end
        if (|xbar_use_queue) begin
            `TRACE(1, ("%t: [feop_accu]: using queue for inputs: xbar_use_queue=%b\n", $time, xbar_use_queue));
        end
        // if (enable && xbar_collisions > 0) begin
        //     `TRACE(1, ("%t: [feop_accu]: xbar collisions=%0d\n", $time, xbar_collisions));
        // end
    end

// QUEUES
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// CROSSBAR

    localparam int XBAR_INPUTS  = BLOCK_M * BLOCK_N; // 32;
    localparam int XBAR_OUTPUTS = BANKS;             // 32;
    localparam int XBAR_SELW    = $clog2(XBAR_OUTPUTS);
    localparam XBAR_DATAW = 1 + BANK_AW + `XLEN; // {from_queue, slot, data}

    wire [XBAR_INPUTS-1:0][XBAR_DATAW-1:0] xbar_data_in;
    wire [XBAR_INPUTS-1:0][XBAR_SELW-1:0]  xbar_sel_in;
    wire [XBAR_INPUTS-1:0]                 xbar_valid_in;
    wire [XBAR_INPUTS-1:0]                 xbar_ready_in;

    wire [XBAR_OUTPUTS-1:0]                 xbar_valid_out;
    wire [XBAR_OUTPUTS-1:0][XBAR_DATAW-1:0] xbar_data_out;
    wire [XBAR_OUTPUTS-1:0]                 xbar_ready_out;
    wire [XBAR_OUTPUTS-1:0]                 xbar_enable_out;

    wire [PERF_CTR_BITS-1:0]  xbar_collisions;

    wire [XBAR_OUTPUTS-1:0][BANK_AW-1:0] xbar_slot_out;
    wire [XBAR_OUTPUTS-1:0][`XLEN-1:0]   xbar_result_out;
    wire [XBAR_OUTPUTS-1:0]              xbar_from_queue_out;

    for (genvar i = 0; i < XBAR_INPUTS; ++i) begin : g_xbar_inputs
        assign xbar_data_in[i]   = xbar_use_queue[i] ? {1'b1, xbar_queue_slot_out[i], xbar_queue_data_out[i]} :
                                                       {1'b0, write_block[i],          write_data[i]};
        assign xbar_sel_in[i]    = xbar_use_queue[i] ? xbar_queue_bank_out[i] : write_bank[i];
        assign xbar_valid_in[i]  = xbar_use_queue[i] || write_lane_fire[i]; // Input comes either from the queue or directly from the FEOP outputs
    end

    for (genvar i = 0; i < XBAR_OUTPUTS; ++i) begin : g_xbar_outputs
        assign xbar_ready_out[i] = enable;
        assign {xbar_from_queue_out[i], xbar_slot_out[i], xbar_result_out[i]} = xbar_data_out[i];
        assign xbar_enable_out[i] = xbar_valid_out[i] && xbar_ready_out[i];
    end

    generate
        if (XBAR_INPUTS <= 32) begin : g_xbar
            VX_stream_xbar #(
                .NUM_INPUTS    (XBAR_INPUTS),
                .NUM_OUTPUTS   (XBAR_OUTPUTS),
                .DATAW         (XBAR_DATAW),
                .OUT_BUF       (0),
                .PERF_CTR_BITS (PERF_CTR_BITS)
            ) accu_xbar (
                .clk   (clk),
                .reset (reset),

                .valid_in (xbar_valid_in),
                .data_in  (xbar_data_in),
                .sel_in   (xbar_sel_in),
                .ready_in (xbar_ready_in),

                .valid_out (xbar_valid_out),
                .data_out  (xbar_data_out),
                `UNUSED_PIN (sel_out),
                .ready_out (xbar_ready_out),

                .collisions (xbar_collisions)
            );
        end
    endgenerate;

    wire overwrite_delayed;
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
    wire [XBAR_OUTPUTS-1:0] bank_write_bitmap;
    wire [XBAR_OUTPUTS-1:0][BANK_AW-1:0] bank_slots;
    wire [XBAR_OUTPUTS-1:0] bank_from_queue;
    // Models FACC latency for control signals xbar_enable_out, xbar_slot_out
    VX_pipe_register #(
        .DATAW  ($bits(xbar_enable_out) + $bits(xbar_slot_out) + $bits(xbar_from_queue_out)),
        .RESETW ($bits(xbar_enable_out) + $bits(xbar_slot_out) + $bits(xbar_from_queue_out)),
        .DEPTH  (FACC_LATENCY)
    ) pipe_write (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({xbar_enable_out,   xbar_slot_out, xbar_from_queue_out}),
        .data_out ({bank_write_bitmap, bank_slots,    bank_from_queue})
    );    

    reg [63:0] sum64;
`ifdef TCU_TYPE_DPI
    reg [4:0]  fflags_unused;
    `UNUSED_VAR (sum64[63:32]);
`endif
    wire [BANKS-1:0] forwarding_condition;
    wire [BANKS-1:0][`XLEN-1:0] accum_source;

    for (genvar j = 0; j < BANKS; ++j) begin : g_accum_source
        assign forwarding_condition[j] = bank_write_bitmap[j] && (xbar_slot_out[j] == bank_slots[j]);
        assign accum_source[j] = overwrite_delayed ? `XLEN'(0) :
                                 (forwarding_condition[j] ? result_delayed[j] : bank_rdata[j]);
    end

`ifdef TCU_TYPE_BHF

    `UNUSED_VAR (sum64[63:0]);
    wire [BANKS-1:0][31:0] bhf_fadd_result;
    wire [BANKS-1:0][4:0]  bhf_fadd_fflags;
    wire [BANKS-1:0][31:0] int_sum_now;
    wire [BANKS-1:0][31:0] int_sum_delayed;
    for (genvar j = 0; j < BANKS; ++j) begin : g_bhf_fadd
        assign int_sum_now[j] = accum_source[j] + xbar_result_out[j];

        VX_tcu_bhf_fadd #(
            .IN_EXPW      (8),
            .IN_SIGW      (24),
            .IN_REC       (0),
            .OUT_REC      (0),
            .ADD_LATENCY  (FADD_LATENCY),
            .RND_LATENCY  (FRND_LATENCY)
        ) fp32_accum_adder (
            .clk    (clk),
            .reset  (reset),
            .enable (enable),
            .frm    (3'b000),
            .a      (accum_source[j]),
            .b      (xbar_result_out[j]),
            .y      (bhf_fadd_result[j]),
            .fflags (bhf_fadd_fflags[j])
        );
    end

    // TODO: Remove registers
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
`ifdef TCU_TYPE_DPI
        fflags_unused = '0;
`endif

        for (integer j = 0; j < BANKS; ++j) begin
            // Adder operates only if input is valid - energy saving
`ifdef TCU_TYPE_BHF
            if (~reset && enable && bank_write_bitmap[j]) begin
                if (fmt_d == TCU_I32_ID) begin // int32
                    result[j] = int_sum_delayed[j];
                end else begin // all floating-point accumulator formats use fp32 accumulation
                    result[j] = bhf_fadd_result[j];
                end
            end
`else
            if (~reset && enable && xbar_enable_out[j]) begin
                if (~overwrite_delayed && forwarding_condition[j]) begin
                    `TRACE(1, ("%t: [feop_accu]: WRITE forwarding for bank=%0d, slot=%0d\n", $time, j, xbar_slot_out[j]));
                end

                /* Addition */
                if (fmt_d == TCU_I32_ID) begin // int32
                    sum64 = {{32{accum_source[j][31]}}, accum_source[j]} + {{32{xbar_result_out[j][31]}}, xbar_result_out[j]};
                end
                else begin // all floating-point accumulator formats use fp32 accumulation
`ifdef TCU_TYPE_DPI
                    dpi_fadd(enable, int'(0),
                             {32'hffffffff, accum_source[j]},
                             {32'hffffffff, xbar_result_out[j]},
                             3'b0, sum64, fflags_unused);
`endif
                end

                result[j] = sum64[31:0];
            end
`endif
        end
    end


    reg [BLOCK_M*BLOCK_N-1:0][`XLEN-1:0] result;
    reg [BLOCK_M*BLOCK_N-1:0][`XLEN-1:0] result_delayed;
    // Delay the accumulation result by FACC_LATENCY cycles
    VX_pipe_register #(
        .DATAW (`XLEN * BLOCK_M * BLOCK_N),
`ifdef TCU_TYPE_DPI
        .DEPTH (FACC_LATENCY)
`elsif TCU_TYPE_BHF
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
    wire [BANKS-1:0][`XLEN-1:0]   bank_rdata;
    wire [BANKS-1:0]              bank_read_en;
    wire [BANKS-1:0][BANK_AW-1:0] bank_raddr;
    wire [BANKS-1:0]              bank_write_en;
    wire [BANKS-1:0][BANK_AW-1:0] bank_waddr;
    wire [BANKS-1:0][`XLEN-1:0]   bank_wdata;

    for (genvar b = 0; b < BANKS; ++b) begin : g_bank_ctrl
        /*             read because we are flushing or   read because we are in the gather stage */
        assign bank_read_en[b] = enable && ((read_en && read_row_valid[b >> LG_BLOCK_N]) || (~overwrite_delayed && xbar_enable_out[b]));

        assign bank_raddr[b]   = ((enable && read_en)                                ? read_block_idx   : // If flushing, return the block to be flushed
                                  (enable && |xbar_enable_out && ~overwrite_delayed) ? xbar_slot_out[b] : // If gathering, return the elements for accumulation
                                   '0);

        assign bank_write_en[b] =  enable && bank_write_bitmap[b];
        assign bank_waddr[b]    = (enable && bank_write_bitmap[b]) ? bank_slots[b]     : '0;
        assign bank_wdata[b]    = (enable && bank_write_bitmap[b]) ? result_delayed[b] : '0;
    end

    for (genvar i = 0; i < BANKS; ++i) begin : g_read_out
        assign read_data[i] = ((read_en && bank_write_en[i] && (read_block_idx == bank_waddr[i])) ? result_delayed[i] : // TODO: Forwarding here is unnecessary - remove if it causes nuance 
                                read_en                                                           ? bank_rdata[i]     : 
                                '0);
    end
    always_ff @(posedge clk) begin
        integer i;
        if (~reset && enable && read_en) begin
            for (i = 0; i < BANKS; i++) begin
                if (bank_write_en[i] && (read_block_idx == bank_waddr[i])) begin
                    `TRACE(1, ("%t: [feop_accu]: READ forwarding addr (slot:bank) %0h:%0h, read_data=%0d\n", $time, bank_waddr[i], i, read_data[i]));
                end
            end
        end
    end

    for (genvar b = 0; b < BANKS; ++b) begin : g_accu_banks
        VX_dp_ram #(
            .DATAW      (`XLEN),
            .SIZE       (SLOTS),
            .OUT_REG    (0),
            .RDW_MODE   ("W"), // If read and write fall to the same address, write is done first
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
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// DEBUGGING

    // Debug dump placed at end for easy grep in logs
    always_ff @(posedge clk) begin
        integer r;
        integer c;
        integer idx;
        if (~reset && enable && (|xbar_valid_in || |xbar_enable_out || |bank_write_bitmap)) begin
             `TRACE(1, ("%t: [feop_accu]: |xbar_valid_in=%b write_block[0]=%0d,       |xbar_enable_out=%b, xbar_slot_out[0]=%0d,     |bank_write_bitmap=%b\n", $time, |xbar_valid_in, write_block[0], |xbar_enable_out, xbar_slot_out[0], |bank_write_bitmap));
        end
        if (~reset && enable && read_en && (|xbar_enable_out == 1'b1)) begin
            `TRACE(1, ("%t: [feop_accu] ERROR: Flushing and gathering simultaneously\n", $time));
        end
        if (~reset && enable && read_en && (|bank_write_en == 1'b1)) begin
            `TRACE(1, ("%t: [feop_accu] ERROR: Flushing and scattering simultaneously\n", $time));
        end
        if (~reset && enable && write_fire) begin
            `TRACE(1, ("%t: [feop_accu]: WRITE req row_valid=%b col_valid=%b\n",
                       $time, write_addr_row_valid, write_addr_col_valid));
            `TRACE(1, ("%t: [feop_accu]: WRITE row_addrs: ", $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("%0d ", write_addr_row[r]));
            end
            `TRACE(1, ("\n"));
            `TRACE(1, ("%t: [feop_accu]: WRITE col_addrs: ", $time));
            for (c = 0; c < BLOCK_N; ++c) begin
                `TRACE(1, ("%0d ", write_addr_col[c]));
            end
            `TRACE(1, ("\n"));
            `TRACE(1, ("%t: [feop_accu]: WRITE data (%0dx%0d):\n", $time, BLOCK_M, BLOCK_N));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    `TRACE(1, ("0x%0h ", write_data[idx]));
                end
                `TRACE(1, ("\n"));
            end
            `TRACE(1, ("\n"));
        end
        if (~reset && enable && (|bank_write_en)) begin
            `TRACE(1, ("%t: [feop_accu]: ACCU bank writes (blocks):\n",
                        $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    if (bank_write_en[idx]) begin
                        `TRACE(1, ("%02d:%02d ", bank_waddr[idx], idx));
                    end else begin
                        `TRACE(1, ("   -   "));
                    end
                end
                `TRACE(1, ("\n"));
            end
            `TRACE(1, ("%t: [feop_accu]: ACCU bank writes (data):\n", $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    if (bank_write_en[idx]) begin
                        `TRACE(1, ("0x%0h%s ", bank_wdata[idx], bank_from_queue[idx] ? "(Q)" : "(I)"));
                    end else begin
                        `TRACE(1, ("- "));
                    end
                end
                `TRACE(1, ("\n"));
            end
        end
        if (~reset && enable && read_en && (|bank_read_en)) begin
            `TRACE(1, ("%t: [feop_accu]: ACCU bank reads (blocks) from read_en:\n", $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    if (bank_read_en[idx]) begin
                        `TRACE(1, ("%0d:%0d ", bank_raddr[idx], idx));
                    end else begin
                        `TRACE(1, ("- "));
                    end
                end
                `TRACE(1, ("\n"));
            end
            `TRACE(1, ("%t: [feop_accu]: ACCU bank reads (data) from read_en:\n", $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    if (bank_read_en[idx]) begin
                        `TRACE(1, ("0x%0h ", read_data[idx]));
                    end else begin
                        `TRACE(1, ("- "));
                    end
                end
                `TRACE(1, ("\n"));
            end
        end
        if (~reset && enable && ~read_en && (|bank_read_en)) begin
            `TRACE(1, ("%t: [feop_accu]: ACCU bank reads (blocks) for gather:\n", $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    if (bank_read_en[idx]) begin
                        `TRACE(1, ("%0d:%0d ", bank_raddr[idx], idx));
                    end else begin
                        `TRACE(1, ("- "));
                    end
                end
                `TRACE(1, ("\n"));
            end
            `TRACE(1, ("%t: [feop_accu]: ACCU bank reads (data) for gather:\n", $time));
            for (r = 0; r < BLOCK_M; ++r) begin
                `TRACE(1, ("  "));
                for (c = 0; c < BLOCK_N; ++c) begin
                    idx = r * BLOCK_N + c;
                    if (bank_read_en[idx]) begin
                        `TRACE(1, ("0x%0h ", bank_rdata[idx]));
                    end else begin
                        `TRACE(1, ("- "));
                    end
                end
                `TRACE(1, ("\n"));
            end
        end
        if (~reset && enable && read_en) begin
            `TRACE(1, ("%t: [feop_accu]: read_en active, read_block_idx=%0d, |xbar_enable_out=%b, ~overwrite_delayed=%b, |xbar_slot_out[0]=%0d\n",
                       $time, read_block_idx, |xbar_enable_out, ~overwrite_delayed, |xbar_slot_out[0]));        
        end
    end

endmodule
