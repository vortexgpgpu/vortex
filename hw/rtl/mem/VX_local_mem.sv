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

module VX_local_mem import VX_gpu_pkg::*; #(
    parameter `STRING  INSTANCE_ID = "",

    // Size of cache in bytes
    parameter SIZE              = (1024*16*8),

    // Number of Word requests per cycle
    parameter NUM_REQS          = 4,
    // Number of banks
    parameter NUM_BANKS         = 4,

    // Address width
    parameter ADDR_WIDTH        = `CLOG2(SIZE),
    // Size of a word in bytes
    parameter WORD_SIZE         = `VX_CFG_XLEN/8,

    // Request tag size
    parameter TAG_WIDTH         = 16,

    // Enable DMA port
    parameter DMA_ENABLE        = 0,
    parameter DMA_TAG_WIDTH     = 1,

    // Synthesize atomic-op + LR/SC logic in the banks
    parameter AMO_ENABLE        = 0,

    // Response buffer
    parameter OUT_BUF           = 0
 ) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    output lmem_perf_t lmem_perf,
`endif

    // LSU read/write port
    VX_mem_bus_if.slave lsu_bus_if [NUM_REQS],

    // DMA read/write port
    VX_mem_bus_if.slave dma_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam REQ_SEL_BITS    = `CLOG2(NUM_REQS);
    localparam REQ_SEL_WIDTH   = `UP(REQ_SEL_BITS);
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam NUM_WORDS       = SIZE / WORD_SIZE;
    localparam WORDS_PER_BANK  = NUM_WORDS / NUM_BANKS;
    localparam BANK_ADDR_WIDTH = `CLOG2(WORDS_PER_BANK);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);
    // AMO sideband carried to the banks: valid + op + unsigned + hart_id. The
    // operand width comes from the request's byteen; hart_id identifies the
    // reserver so a store-conditional can tell its own reservation from
    // another hart's.
    // Compare-and-swap adds a comparand: the request's data word already
    // carries the swap value, so the third operand needs its own room.
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    localparam AMO_CMP_WIDTH   = `VX_CFG_XLEN;
`else
    localparam AMO_CMP_WIDTH   = 0;
`endif
    localparam AMO_SB_WIDTH    = 1 + $bits(amo_op_e) + 1 + HART_ID_WIDTH + AMO_CMP_WIDTH;
    localparam REQ_DATAW       = 1 + BANK_ADDR_WIDTH + WORD_SIZE + WORD_WIDTH + TAG_WIDTH + AMO_SB_WIDTH;
    localparam RSP_DATAW       = WORD_WIDTH + TAG_WIDTH;

    `STATIC_ASSERT(ADDR_WIDTH == (BANK_ADDR_WIDTH + `CLOG2(NUM_BANKS)), ("invalid parameter"))

    // bank selection

    wire [NUM_REQS-1:0][BANK_SEL_WIDTH-1:0] req_bank_idx;
    if (NUM_BANKS > 1) begin : g_req_bank_idx
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_bank_idxs
            assign req_bank_idx[i] = lsu_bus_if[i].req_data.addr[0 +: BANK_SEL_BITS];
        end
    end else begin : g_req_bank_idx_0
        assign req_bank_idx = 0;
    end

    // bank addressing

    wire [NUM_REQS-1:0][BANK_ADDR_WIDTH-1:0] req_bank_addr;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_bank_addr
        assign req_bank_addr[i] = lsu_bus_if[i].req_data.addr[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    end

    // AMO sideband per request, from the shared mem-bus attr.
    wire [NUM_REQS-1:0]                  req_amo_valid;
    wire [NUM_REQS-1:0][$bits(amo_op_e)-1:0] req_amo_op;
    wire [NUM_REQS-1:0]                  req_amo_unsigned;
    wire [NUM_REQS-1:0][HART_ID_WIDTH-1:0] req_amo_hart_id;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [NUM_REQS-1:0][AMO_CMP_WIDTH-1:0] req_amo_cmp;
`endif
    // Gated by AMO_ENABLE: the widths stay constant and the zeros constant-fold
    // through the crossbar payload (the VX_cache_bank pattern).
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_amo
        mem_bus_attr_t lane_attr;
        assign lane_attr = mem_bus_attr_t'(lsu_bus_if[i].req_data.attr);
        assign req_amo_valid[i]    = AMO_ENABLE ? lane_attr.amo.amo_valid : 1'b0;
        assign req_amo_op[i]       = AMO_ENABLE ? lane_attr.amo.amo_op : '0;
        assign req_amo_unsigned[i] = AMO_ENABLE ? lane_attr.amo.amo_unsigned : 1'b0;
        assign req_amo_hart_id[i]  = AMO_ENABLE ? lane_attr.amo.hart_id : '0;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        assign req_amo_cmp[i]      = AMO_ENABLE ? lane_attr.amo.amo_cmp : '0;
    `endif
        `UNUSED_VAR (lane_attr)
    end

    // bank requests dispatch

    wire [NUM_BANKS-1:0]                    per_bank_req_valid;
    wire [NUM_BANKS-1:0]                    per_bank_req_rw;
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] per_bank_req_addr;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_req_byteen;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]    per_bank_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag;
    wire [NUM_BANKS-1:0]                    per_bank_req_amo_valid;
    wire [NUM_BANKS-1:0][$bits(amo_op_e)-1:0] per_bank_req_amo_op;
    wire [NUM_BANKS-1:0]                    per_bank_req_amo_unsigned;
    wire [NUM_BANKS-1:0][HART_ID_WIDTH-1:0] per_bank_req_amo_hart_id;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [NUM_BANKS-1:0][AMO_CMP_WIDTH-1:0] per_bank_req_amo_cmp;
`endif
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_req_idx;
    wire [NUM_BANKS-1:0]                    per_bank_req_ready;

    wire [NUM_BANKS-1:0][REQ_DATAW-1:0]     per_bank_req_data_aos;

    wire [NUM_REQS-1:0]                 req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_REQS-1:0]                 req_ready_in;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] perf_collisions;
`endif

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_data_in
        assign req_valid_in[i] = lsu_bus_if[i].req_valid;
        assign req_data_in[i] = {
            req_amo_valid[i],
            req_amo_op[i],
            req_amo_unsigned[i],
            req_amo_hart_id[i],
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            req_amo_cmp[i],
        `endif
            lsu_bus_if[i].req_data.rw,
            req_bank_addr[i],
            lsu_bus_if[i].req_data.data,
            lsu_bus_if[i].req_data.byteen,
            lsu_bus_if[i].req_data.tag
        };
        assign lsu_bus_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (REQ_DATAW),
        .PERF_CTR_BITS (PERF_CTR_BITS),
        .ARBITER     ("P"),
        .OUT_BUF     (3) // output should be registered for the data_store addressing
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
    `ifdef PERF_ENABLE
        .collisions (perf_collisions),
    `else
        `UNUSED_PIN (collisions),
    `endif
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .sel_in    (req_bank_idx),
        .ready_in  (req_ready_in),
        .valid_out (per_bank_req_valid),
        .data_out  (per_bank_req_data_aos),
        .sel_out   (per_bank_req_idx),
        .ready_out (per_bank_req_ready)
    );

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_req_data_soa
        assign {
            per_bank_req_amo_valid[i],
            per_bank_req_amo_op[i],
            per_bank_req_amo_unsigned[i],
            per_bank_req_amo_hart_id[i],
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            per_bank_req_amo_cmp[i],
        `endif
            per_bank_req_rw[i],
            per_bank_req_addr[i],
            per_bank_req_data[i],
            per_bank_req_byteen[i],
            per_bank_req_tag[i]
        } = per_bank_req_data_aos[i];
    end

    // banks access (declared here so g_dma_enable can reference per_bank_rsp_data)

    wire [NUM_BANKS-1:0]                 per_bank_rsp_valid;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] per_bank_rsp_data;
    // Back-pressure-safe copy of the LSU response data (see g_data_store).
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] bank_lsu_rsp_data;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_rsp_idx;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]  per_bank_rsp_tag;
    wire [NUM_BANKS-1:0]                 per_bank_rsp_ready;

    // DMA port handshake
    //   rw=0 reads  : accepted when the response pipe-buffer has space.
    //   rw=1 writes : always accepted; no response issued.
    //   DMA has priority over LSU at every bank SRAM.

    wire dma_rsp_buf_ready; // driven by pipe-buffer or tied 0 when disabled

    if (DMA_ENABLE) begin : g_dma_enable
        `UNUSED_VAR (dma_bus_if.req_data.attr)

        assign dma_bus_if.req_ready = dma_bus_if.req_data.rw || dma_rsp_buf_ready;

        wire dma_rd_fire = dma_bus_if.req_valid && ~dma_bus_if.req_data.rw && dma_rsp_buf_ready;

        // Delay tag by 1 cycle to align with SRAM OUT_REG latency
        VX_pipe_buffer #(
            .DATAW (DMA_TAG_WIDTH)
        ) dma_rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (dma_rd_fire),
            .ready_in  (dma_rsp_buf_ready),
            .data_in   (dma_bus_if.req_data.tag),
            .valid_out (dma_bus_if.rsp_valid),
            .data_out  (dma_bus_if.rsp_data.tag),
            .ready_out (dma_bus_if.rsp_ready)
        );

        // Pack all bank SRAM outputs into the read response.
        //
        // Same back-pressure hazard as the LSU side: dma_rsp_buf buffers only
        // the tag/valid; the data (all banks) is the live SRAM OUT_REG. While a
        // DMA read response is back-pressured, an interleaving LSU read
        // (~dma_active) re-drives a bank's OUT_REG and corrupts the held DMA
        // line. Latch the whole line on the first response-valid cycle (rdata
        // is still valid then) and serve the latched copy while stalled. One
        // DMA response spans all banks, so this is a single full-line hold.
        reg  [NUM_BANKS-1:0][WORD_WIDTH-1:0] dma_rsp_hold_data_r;
        reg                                  dma_rsp_hold_valid_r;
        wire dma_rsp_consumed = dma_bus_if.rsp_valid && dma_bus_if.rsp_ready;
        always @(posedge clk) begin
            if (reset) begin
                dma_rsp_hold_valid_r <= 1'b0;
            end else if (dma_rsp_consumed) begin
                dma_rsp_hold_valid_r <= 1'b0;
            end else if (dma_bus_if.rsp_valid && ~dma_rsp_hold_valid_r) begin
                dma_rsp_hold_data_r  <= per_bank_rsp_data;
                dma_rsp_hold_valid_r <= 1'b1;
            end
        end
        for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_dma_rsp_data
            assign dma_bus_if.rsp_data.data[i*WORD_WIDTH +: WORD_WIDTH] =
                dma_rsp_hold_valid_r ? dma_rsp_hold_data_r[i] : per_bank_rsp_data[i];
        end

    end else begin : g_no_dma
        assign dma_rsp_buf_ready    = 1'b0;
        assign dma_bus_if.req_ready = 1'b0;
        assign dma_bus_if.rsp_valid = 1'b0;
        assign dma_bus_if.rsp_data  = '0;
        `UNUSED_VAR (dma_bus_if.req_valid)
        `UNUSED_VAR (dma_bus_if.req_data)
        `UNUSED_VAR (dma_bus_if.rsp_ready)
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_data_store
        wire bank_rsp_valid, bank_rsp_ready;

        // DMA active signals (priority over LSU)
        wire dma_wr_b = DMA_ENABLE
                     && dma_bus_if.req_valid
                     && dma_bus_if.req_data.rw
                     && (|dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]);

        wire dma_rd_b = DMA_ENABLE
                     && dma_bus_if.req_valid
                     && ~dma_bus_if.req_data.rw
                     && dma_rsp_buf_ready;

        wire dma_active = dma_wr_b | dma_rd_b;

        wire lsu_active = per_bank_req_valid[i] && per_bank_req_ready[i];

        // Bank-facing AMO interface, tied off when the A extension is absent so
        // none of the atomic hardware is synthesized (the dcache counterpart is
        // AMO_ENABLE in VX_cache_bank).
        wire                       amo_busy;      // an atomic owns the bank port
        wire                       amo_wb_store;  // owed atomic write-back
        wire [BANK_ADDR_WIDTH-1:0] amo_wb_addr;
        wire [WORD_SIZE-1:0]       amo_wb_byteen;
        wire [WORD_WIDTH-1:0]      amo_wb_data;
        wire                       sc_resolving;  // SC outcome replaces the response
        wire                       sc_fail;

        if (AMO_ENABLE != 0) begin : g_amo

            // Atomics. The LSU sends an AMO with rw=0, so the read returning the old
            // value is the ordinary read path and needs no change. What the bank owes
            // is the write of the computed value.
            //
            // The old word is registered before the ALU sees it. Feeding the ALU
            // straight from the SRAM output puts the BRAM clock-to-out, the adder and
            // the BRAM setup in a single period; the register splits that into
            // BRAM-out-to-flop and flop-through-adder-to-BRAM-in, which is what keeps
            // the atomic adder off the critical path.
            //
            // The port is single-ported, so an atomic owns its bank for three cycles:
            // read, capture, write back.
            reg                       amo_rd_valid_r;
            reg                       amo_wb_valid_r;
            reg [WORD_WIDTH-1:0]      amo_old_r;
            reg [BANK_ADDR_WIDTH-1:0] amo_wb_addr_r;
            reg [WORD_SIZE-1:0]       amo_wb_byteen_r;
            reg [$bits(amo_op_e)-1:0] amo_wb_op_r;
            reg                       amo_wb_unsigned_r;
            reg [WORD_WIDTH-1:0]      amo_wb_rhs_r;
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            reg [AMO_CMP_WIDTH-1:0]   amo_wb_cmp_r;
        `endif

            wire amo_accept = lsu_active && per_bank_req_amo_valid[i];

            // Reservation commit stage. Every accepted request that can touch a
            // reservation -- an atomic, or a plain store -- reaches it one cycle
            // after acceptance, so a single address port serves both. The look-ahead
            // read is driven from the acceptance cycle, which is what makes the
            // registered reservation entry land here.
            wire is_lr_in = per_bank_req_amo_valid[i]
                         && (amo_op_e'(per_bank_req_amo_op[i]) == AMO_OP_LR);
            wire is_sc_in = per_bank_req_amo_valid[i]
                         && (amo_op_e'(per_bank_req_amo_op[i]) == AMO_OP_SC);

            reg                       res_valid_r;
            reg                       res_is_lr_r;
            reg                       res_is_sc_r;
            reg                       res_is_amo_r;
            reg                       res_is_store_r;
            reg [BANK_ADDR_WIDTH-1:0] res_addr_r;
            reg [HART_ID_WIDTH-1:0]   res_hart_id_r;

            always @(posedge clk) begin
                if (reset) begin
                    res_valid_r <= 1'b0;
                end else begin
                    res_valid_r <= lsu_active;
                end
                if (lsu_active) begin
                    res_is_lr_r    <= is_lr_in;
                    res_is_sc_r    <= is_sc_in;
                    res_is_amo_r   <= per_bank_req_amo_valid[i];
                    res_is_store_r <= per_bank_req_rw[i];
                    res_addr_r     <= per_bank_req_addr[i];
                    res_hart_id_r  <= per_bank_req_amo_hart_id[i];
                end
            end

            // A store-conditional fails when its reservation is gone. The outcome is
            // known at the commit stage, one cycle before the write-back, so it can
            // both suppress the write and be returned as this request's response.
            wire res_check;
            assign sc_fail = res_is_sc_r && ~res_check;
            assign sc_resolving = res_valid_r && res_is_sc_r;

            // Everything that ends up writing the word: a plain store, a
            // read-modify-write atomic (which carries rw=0 and so is not a store by
            // the request flag), and a store-conditional that found its reservation.
            wire res_commits_store = res_is_store_r
                                  || (res_is_amo_r && ~res_is_lr_r && ~res_is_sc_r)
                                  || (res_is_sc_r && res_check);

            reg amo_wb_is_lr_r;
            reg amo_wb_sc_fail_r;
            always @(posedge clk) begin
                if (amo_rd_valid_r) begin
                    amo_wb_is_lr_r   <= res_is_lr_r;
                    amo_wb_sc_fail_r <= sc_fail;
                end
            end

            // A load-reserved leaves the word alone, and a failed store-conditional
            // must not store; every other atomic writes its computed value back.
            assign amo_wb_store = amo_wb_valid_r && ~amo_wb_is_lr_r && ~amo_wb_sc_fail_r;
            assign amo_busy = amo_rd_valid_r || amo_wb_valid_r;

            always @(posedge clk) begin
                if (reset) begin
                    amo_rd_valid_r <= 1'b0;
                    amo_wb_valid_r <= 1'b0;
                end else begin
                    amo_rd_valid_r <= amo_accept;
                    amo_wb_valid_r <= amo_rd_valid_r;
                end
                if (amo_rd_valid_r) begin
                    amo_old_r <= per_bank_rsp_data[i];
                end
                if (amo_accept) begin
                    amo_wb_addr_r     <= per_bank_req_addr[i];
                    amo_wb_byteen_r   <= per_bank_req_byteen[i];
                    amo_wb_op_r       <= per_bank_req_amo_op[i];
                    amo_wb_unsigned_r <= per_bank_req_amo_unsigned[i];
                    amo_wb_rhs_r      <= per_bank_req_data[i];
                `ifdef VX_CFG_EXT_ZACAS_ENABLE
                    amo_wb_cmp_r      <= per_bank_req_amo_cmp[i];
                `endif
                end
            end

            // old_word is this cycle's SRAM output -- the same value the response
            // carries back, which is why ret_word is not used here.
            wire [63:0] amo_new_word, amo_ret_word;
            // VX_CFG_AMO_RS_SIZE sizes ONE shared LLC bank's contention; deployed
            // per lmem bank it multiplies by NUM_LMEM_BANKS x NUM_CORES. After the
            // holder-credit protocol a hot word maps to exactly one station, so
            // depth only reduces false eviction between distinct reserved words
            // aliasing in one bank -- and a spurious SC failure is architecturally
            // legal. Clamp the depth and shorten the credit budget to the local
            // round trip instead of inheriting the L1-miss sizing.
            VX_amo_unit #(
                .NUM_RES_ENTRIES  (`MIN(`VX_CFG_AMO_RS_SIZE, 4)),
                .HOLD_CREDIT_BITS (3),
                .LINE_ADDR_BITS   (BANK_ADDR_WIDTH),
                .DATA_WIDTH       (WORD_WIDTH)
            ) amo_unit (
                .clk              (clk),
                .reset            (reset),
                .pipe_stall       (1'b0),
                .compute_op       (amo_op_e'(amo_wb_op_r)),
                .compute_unsigned (amo_wb_unsigned_r),
                .compute_width    (2'd2),
                .compute_old      (64'(amo_old_r)),
                .compute_rhs      (64'(amo_wb_rhs_r)),
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                .compute_cmp      (64'(amo_wb_cmp_r)),
            `else
                .compute_cmp      (64'b0),
            `endif
                .compute_new_word (amo_new_word),
                .compute_ret_word (amo_ret_word),
                // A load-reserved claims the word; a store-conditional gives up its
                // own reservation whether it succeeds or fails; any committing store
                // breaks whoever else holds one.
                .res_reserve      (res_valid_r && res_is_lr_r),
                .res_clear        (res_valid_r && res_is_sc_r),
                .res_invalidate   (res_valid_r && res_commits_store),
                .res_hart_id      (res_hart_id_r),
                .res_line_addr    (res_addr_r),
                .res_line_addr_n  (per_bank_req_addr[i]),
                .res_check        (res_check)
            );
            `UNUSED_VAR (amo_ret_word)
            if (WORD_WIDTH < 64) begin : g_amo_hi_unused
                `UNUSED_VAR (amo_new_word[63:WORD_WIDTH])
            end

            assign amo_wb_addr   = amo_wb_addr_r;
            assign amo_wb_byteen = amo_wb_byteen_r;
            assign amo_wb_data   = amo_new_word[WORD_WIDTH-1:0];

        end else begin : g_no_amo
            assign amo_busy      = 1'b0;
            assign amo_wb_store  = 1'b0;
            assign amo_wb_addr   = '0;
            assign amo_wb_byteen = '0;
            assign amo_wb_data   = '0;
            assign sc_resolving  = 1'b0;
            assign sc_fail       = 1'b0;
            `UNUSED_VAR (per_bank_req_amo_valid[i])
            `UNUSED_VAR (per_bank_req_amo_op[i])
            `UNUSED_VAR (per_bank_req_amo_unsigned[i])
            `UNUSED_VAR (per_bank_req_amo_hart_id[i])
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            `UNUSED_VAR (per_bank_req_amo_cmp[i])
        `endif
        end

        // SRAM address / write-data / write-enable mux: DMA first, then an owed
        // atomic write-back, then the incoming request
        wire [BANK_ADDR_WIDTH-1:0] bank_sram_addr;
        wire [WORD_WIDTH-1:0]      bank_sram_wdata;
        wire [WORD_SIZE-1:0]       bank_sram_wren;

        assign bank_sram_addr  = dma_active   ? BANK_ADDR_WIDTH'(dma_bus_if.req_data.addr)
                               : amo_wb_store ? amo_wb_addr
                                              : per_bank_req_addr[i];
        assign bank_sram_wdata = dma_wr_b     ? dma_bus_if.req_data.data[i*WORD_WIDTH +: WORD_WIDTH]
                               : amo_wb_store ? amo_wb_data
                                              : per_bank_req_data[i];
        assign bank_sram_wren  = dma_wr_b     ? dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]
                               : amo_wb_store ? amo_wb_byteen
                                              : per_bank_req_byteen[i];

        VX_sp_ram #(
            .DATAW (WORD_WIDTH),
            .SIZE  (WORDS_PER_BANK),
            .WRENW (WORD_SIZE),
            .OUT_REG (1),
            .RDW_MODE ("R")
        ) lmem_store (
            .clk   (clk),
            .reset (reset),
            .read  (dma_rd_b || (lsu_active && ~per_bank_req_rw[i])),
            .write (dma_wr_b || amo_wb_store || (lsu_active && per_bank_req_rw[i])),
            .wren  (bank_sram_wren),
            .addr  (bank_sram_addr),
            .wdata (bank_sram_wdata),
            .rdata (per_bank_rsp_data[i])
        );

        // Read-during-write hazard: stalls LSU reads to an address written last cycle
        // (SRAM OUT_REG + RDW_MODE="R" returns stale data on same-cycle read-after-write).
        // DMA reads bypass this check.

        reg [BANK_ADDR_WIDTH-1:0] last_wr_addr;
        reg last_wr_valid;
        always @(posedge clk) begin
            if (reset) begin
                last_wr_valid <= 0;
            end else begin
                last_wr_valid <= dma_wr_b || amo_wb_store || (lsu_active && per_bank_req_rw[i]);
            end
            last_wr_addr <= bank_sram_addr;
        end
        wire is_rdw_hazard = last_wr_valid && ~per_bank_req_rw[i] && (per_bank_req_addr[i] == last_wr_addr);

        // LSU response valid / request ready — blocked by DMA and RDW hazards

        assign bank_rsp_valid = per_bank_req_valid[i]
                             && ~dma_active
                             && ~amo_busy
                             && ~per_bank_req_rw[i]
                             && ~is_rdw_hazard;

        assign per_bank_req_ready[i] = ~dma_active
                                    && ~amo_busy
                                    && (bank_rsp_ready || per_bank_req_rw[i])
                                    && ~is_rdw_hazard;

        // Delay tag/idx to align with SRAM 1-cycle output latency
        VX_pipe_buffer #(
            .DATAW (REQ_SEL_WIDTH + TAG_WIDTH)
        ) bram_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (bank_rsp_valid),
            .ready_in  (bank_rsp_ready),
            .data_in   ({per_bank_req_idx[i], per_bank_req_tag[i]}),
            .data_out  ({per_bank_rsp_idx[i], per_bank_rsp_tag[i]}),
            .valid_out (per_bank_rsp_valid[i]),
            .ready_out (per_bank_rsp_ready[i])
        );

        // A store-conditional returns its outcome, not the word it read: zero
        // for success, one for failure. The outcome is resolved at the commit
        // stage, which is the same cycle the response carries the SRAM output,
        // so it substitutes here rather than needing a response of its own.
        wire [WORD_WIDTH-1:0] bank_rsp_word = sc_resolving
                                            ? WORD_WIDTH'(sc_fail)
                                            : per_bank_rsp_data[i];

        // Back-pressure-safe LSU response data. The bank SRAM OUT_REG
        // (per_bank_rsp_data) is read live by the response xbar, but the
        // response valid/tag is elastic-buffered, so a back-pressured LSU
        // response can sit valid for >1 cycle. During that wait an interleaving
        // DMA read on this bank re-drives the shared OUT_REG, corrupting the
        // held LSU response (WGMMA+DXA issues concurrent LSU A-reads and DMA
        // B-reads to the same banks — the only workload that does). Latch the
        // data on the first response-valid cycle (rdata is still valid then)
        // and serve the latched copy while the response is stalled. The latched
        // copy is the substituted word, so a stalled SC keeps its outcome.
        reg  [WORD_WIDTH-1:0] lsu_rsp_hold_data_r;
        reg                   lsu_rsp_hold_valid_r;
        always @(posedge clk) begin
            if (reset) begin
                lsu_rsp_hold_valid_r <= 1'b0;
            end else if (per_bank_rsp_valid[i] && per_bank_rsp_ready[i]) begin
                lsu_rsp_hold_valid_r <= 1'b0;                     // response consumed
            end else if (per_bank_rsp_valid[i] && ~lsu_rsp_hold_valid_r) begin
                lsu_rsp_hold_data_r  <= bank_rsp_word;            // capture before DMA can clobber
                lsu_rsp_hold_valid_r <= 1'b1;
            end
        end
        assign bank_lsu_rsp_data[i] = lsu_rsp_hold_valid_r ? lsu_rsp_hold_data_r
                                                           : bank_rsp_word;

    end

    // bank responses gather

    wire [NUM_BANKS-1:0][RSP_DATAW-1:0] per_bank_rsp_data_aos;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_rsp_data_aos
        assign per_bank_rsp_data_aos[i] = {bank_lsu_rsp_data[i], per_bank_rsp_tag[i]};
    end

    wire [NUM_REQS-1:0]                 rsp_valid_out;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0]  rsp_data_out;
    wire [NUM_REQS-1:0]                 rsp_ready_out;

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (RSP_DATAW),
        .ARBITER     ("P"), // this priority arbiter has negligeable impact on performance
        .OUT_BUF     (OUT_BUF)
    ) rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .sel_in    (per_bank_rsp_idx),
        .valid_in  (per_bank_rsp_valid),
        .data_in   (per_bank_rsp_data_aos),
        .ready_in  (per_bank_rsp_ready),
        .valid_out (rsp_valid_out),
        .data_out  (rsp_data_out),
        .ready_out (rsp_ready_out),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_lsu_bus_if
        assign lsu_bus_if[i].rsp_valid = rsp_valid_out[i];
        assign lsu_bus_if[i].rsp_data  = rsp_data_out[i];
        assign rsp_ready_out[i] = lsu_bus_if[i].rsp_ready;
    end

`ifdef PERF_ENABLE
    // per cycle: reads, writes
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_reads_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_writes_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_crsp_stall_per_cycle;

    wire [NUM_REQS-1:0] req_rw;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_rw
        assign req_rw[i] = lsu_bus_if[i].req_data.rw;
    end

    wire [NUM_REQS-1:0] perf_reads_per_req, perf_writes_per_req;
    wire [NUM_REQS-1:0] perf_crsp_stall_per_req = rsp_valid_out & ~rsp_ready_out;

    `BUFFER(perf_reads_per_req, req_valid_in & req_ready_in & ~req_rw);
    `BUFFER(perf_writes_per_req, req_valid_in & req_ready_in & req_rw);

    `POP_COUNT(perf_reads_per_cycle, perf_reads_per_req);
    `POP_COUNT(perf_writes_per_cycle, perf_writes_per_req);
    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);

    reg [PERF_CTR_BITS-1:0] perf_reads;
    reg [PERF_CTR_BITS-1:0] perf_writes;
    reg [PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_reads       <= '0;
            perf_writes      <= '0;
            perf_crsp_stalls <= '0;
        end else begin
            perf_reads       <= perf_reads  + PERF_CTR_BITS'(perf_reads_per_cycle);
            perf_writes      <= perf_writes + PERF_CTR_BITS'(perf_writes_per_cycle);
            perf_crsp_stalls <= perf_crsp_stalls + PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign lmem_perf.reads       = perf_reads;
    assign lmem_perf.writes      = perf_writes;
    assign lmem_perf.bank_stalls = perf_collisions;
    assign lmem_perf.crsp_stalls = perf_crsp_stalls;

`endif

`ifdef DBG_TRACE_MEM

    wire [NUM_BANKS-1:0][TAG_WIDTH-UUID_WIDTH-1:0] per_bank_req_tag_value;
    wire [NUM_BANKS-1:0][UUID_WIDTH-1:0] per_bank_req_uuid;

    wire [NUM_BANKS-1:0][TAG_WIDTH-UUID_WIDTH-1:0] per_bank_rsp_tag_value;
    wire [NUM_BANKS-1:0][UUID_WIDTH-1:0] per_bank_rsp_uuid;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_req_uuid
        assign per_bank_req_tag_value[i] = per_bank_req_tag[i][TAG_WIDTH-UUID_WIDTH-1:0];
        assign per_bank_rsp_tag_value[i] = per_bank_rsp_tag[i][TAG_WIDTH-UUID_WIDTH-1:0];
        if (UUID_WIDTH != 0) begin : g_uuid
            assign per_bank_req_uuid[i] = per_bank_req_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
            assign per_bank_rsp_uuid[i] = per_bank_rsp_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin : g_no_uuid
            assign per_bank_req_uuid[i] = 0;
            assign per_bank_rsp_uuid[i] = 0;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_trace
        always @(posedge clk) begin
            if (lsu_bus_if[i].req_valid && lsu_bus_if[i].req_ready) begin
                if (lsu_bus_if[i].req_data.rw) begin
                    `TRACE(2, ("%t: %s core-wr-req[%0d]: addr=0x%0h, byteen=0x%h, data=0x%h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, lsu_bus_if[i].req_data.addr, lsu_bus_if[i].req_data.byteen, lsu_bus_if[i].req_data.data, lsu_bus_if[i].req_data.tag.value, lsu_bus_if[i].req_data.tag.uuid))
                end else begin
                    `TRACE(2, ("%t: %s core-rd-req[%0d]: addr=0x%0h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, lsu_bus_if[i].req_data.addr, lsu_bus_if[i].req_data.tag.value, lsu_bus_if[i].req_data.tag.uuid))
                end
            end
            if (lsu_bus_if[i].rsp_valid && lsu_bus_if[i].rsp_ready) begin
                `TRACE(2, ("%t: %s core-rd-rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n",
                    $time, INSTANCE_ID, i, lsu_bus_if[i].rsp_data.data, lsu_bus_if[i].rsp_data.tag.value, lsu_bus_if[i].rsp_data.tag.uuid))
            end
        end
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_bank_trace
        always @(posedge clk) begin
            if (per_bank_req_valid[i] && per_bank_req_ready[i]) begin
                if (per_bank_req_rw[i]) begin
                    `TRACE(2, ("%t: %s bank-wr-req[%0d]: addr=0x%0h, byteen=0x%h, data=0x%h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, per_bank_req_addr[i], per_bank_req_byteen[i], per_bank_req_data[i], per_bank_req_tag_value[i], per_bank_req_uuid[i]))
                end else begin
                    `TRACE(2, ("%t: %s bank-rd-req[%0d]: addr=0x%0h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, per_bank_req_addr[i], per_bank_req_tag_value[i], per_bank_req_uuid[i]))
                end
            end
            if (per_bank_rsp_valid[i] && per_bank_rsp_ready[i]) begin
                `TRACE(2, ("%t: %s bank-rd-rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n",
                    $time, INSTANCE_ID, i, per_bank_rsp_data[i], per_bank_rsp_tag_value[i], per_bank_rsp_uuid[i]))
            end
        end
    end

`endif

endmodule
