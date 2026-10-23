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

    // 64-bit atomics on a 32-bit hart (AMO64). A wide atomic arrives as two
    // requests on an even/odd port pair -- the issuing thread's lane pair of
    // an LSU block, low word on the even port -- and its two words sit in
    // adjacent banks on the same row. The banks are single-ported and do their
    // atomics word by word, so a wide one is served apart from them by one
    // engine. It first lets the two banks drain what the crossbar already
    // routed to them (the issuing thread's older accesses to the words among
    // them) while it keeps new requests out, then holds off every bank and the
    // DMA port, reads both words, writes the result back and answers on both
    // ports at once. Wide atomics are rare; the exclusive window keeps the
    // common path untouched.
`ifdef VX_CFG_EXT_AMO64_ENABLE
    localparam WIDE_ENABLE = AMO_ENABLE && (`VX_CFG_XLEN == 32);
`else
    localparam WIDE_ENABLE = 0;
`endif
    wire [NUM_REQS-1:0]                  req_wide;
    wire [NUM_REQS-1:0]                  wide_take;
    wire                                 wide_lock;
    wire [NUM_REQS-1:0]                  wide_drain_block;
    wire                                 wide_take_any;
    wire [NUM_BANKS-1:0]                 wide_rd_b;
    wire [NUM_BANKS-1:0]                 wide_wr_b;
    wire [BANK_ADDR_WIDTH-1:0]           wide_row;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] wide_wdata;
    wire [HART_ID_WIDTH-1:0]             wide_hart_id;
    wire [NUM_REQS-1:0]                  wide_rsp_valid;
    wire [NUM_REQS-1:0][WORD_WIDTH-1:0]  wide_rsp_data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   wide_rsp_tag;
    `UNUSED_VAR (wide_hart_id) // read only by the reservation stage

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
        assign req_amo_cmp[i]      = AMO_ENABLE ? lane_attr.amo.amo_cmp[AMO_CMP_WIDTH-1:0] : '0;
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
        assign req_valid_in[i] = lsu_bus_if[i].req_valid && ~req_wide[i] && ~wide_drain_block[i];
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
        assign lsu_bus_if[i].req_ready = req_wide[i] ? wide_take[i] : (req_ready_in[i] && ~wide_drain_block[i]);
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

        assign dma_bus_if.req_ready = (dma_bus_if.req_data.rw || dma_rsp_buf_ready) && ~wide_lock;

        wire dma_rd_fire = dma_bus_if.req_valid && ~dma_bus_if.req_data.rw && dma_rsp_buf_ready && ~wide_lock;

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
                     && (|dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE])
                     && ~wide_lock;

        wire dma_rd_b = DMA_ENABLE
                     && dma_bus_if.req_valid
                     && ~dma_bus_if.req_data.rw
                     && dma_rsp_buf_ready
                     && ~wide_lock;

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
        wire                       sc_lane_hi;    // outcome lane of a resolving .W SC

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

            // A wide atomic's write-back reaches the reservation stage as a
            // plain store, so it breaks a reservation on either word.
            wire res_wide = wide_wr_b[i];
            wire [BANK_ADDR_WIDTH-1:0] res_addr_in = res_wide ? wide_row : per_bank_req_addr[i];
            always @(posedge clk) begin
                if (reset) begin
                    res_valid_r <= 1'b0;
                end else begin
                    res_valid_r <= lsu_active || res_wide;
                end
                if (lsu_active || res_wide) begin
                    res_is_lr_r    <= ~res_wide && is_lr_in;
                    res_is_sc_r    <= ~res_wide && is_sc_in;
                    res_is_amo_r   <= ~res_wide && per_bank_req_amo_valid[i];
                    res_is_store_r <= res_wide || per_bank_req_rw[i];
                    res_addr_r     <= res_addr_in;
                    res_hart_id_r  <= res_wide ? wide_hart_id : per_bank_req_amo_hart_id[i];
                end
            end

            // A store-conditional fails when its reservation is gone. The outcome is
            // known at the commit stage, one cycle before the write-back, so it can
            // both suppress the write and be returned as this request's response.
            wire res_check;
            assign sc_fail = res_is_sc_r && ~res_check;
            assign sc_resolving = res_valid_r && res_is_sc_r;

            // The SC outcome must land in the addressed lane: the LSU extracts
            // the subword by address, so a .W outcome left at bit 0 would read
            // as 0 ("success") for every upper-lane SC, including a failed one
            // whose store was suppressed.
            if (WORD_WIDTH == 64) begin : g_sc_lane
                reg res_lane_hi_r;
                always @(posedge clk) begin
                    if (lsu_active) begin
                        res_lane_hi_r <= ~per_bank_req_byteen[i][0];
                    end
                end
                assign sc_lane_hi = res_lane_hi_r;
            end else begin : g_sc_lane0
                assign sc_lane_hi = 1'b0;
            end

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

            // Lane-align the RMW operands. The bank word is XLEN wide, so on a
            // 64-bit hart a .W atomic addresses one of two 32-bit lanes; the
            // ALU computes right-aligned. Store data arrives lane-positioned
            // (same packing as a plain store), so old and rhs shift down by
            // the lane offset and the result shifts back up under the request
            // byteen. The comparand arrives right-aligned from rs3 and needs
            // no shift. The addressed lane is recovered from the byteen: an
            // atomic is naturally aligned, so a .W to the upper lane never
            // covers byte 0 and a .D covers the full word.
            wire [63:0] amo_old_lane, amo_rhs_lane;
            wire [1:0]  amo_width;
            if (WORD_WIDTH == 64) begin : g_amo_lane
                wire lane_hi = ~amo_wb_byteen_r[0];
                assign amo_width    = (&amo_wb_byteen_r) ? 2'd3 : 2'd2;
                assign amo_old_lane = lane_hi ? {32'h0, amo_old_r[63:32]} : amo_old_r;
                assign amo_rhs_lane = lane_hi ? {32'h0, amo_wb_rhs_r[63:32]} : amo_wb_rhs_r;
                assign amo_wb_data  = lane_hi ? {amo_new_word[31:0], 32'h0} : amo_new_word;
            end else begin : g_amo_word
                assign amo_width    = 2'd2;
                assign amo_old_lane = 64'(amo_old_r);
                assign amo_rhs_lane = 64'(amo_wb_rhs_r);
                assign amo_wb_data  = amo_new_word[WORD_WIDTH-1:0];
            end
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
                .compute_width    (amo_width),
                .compute_old      (amo_old_lane),
                .compute_rhs      (amo_rhs_lane),
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
                .res_line_addr_n  (res_addr_in),
                .res_check        (res_check)
            );
            `UNUSED_VAR (amo_ret_word)
            if (WORD_WIDTH < 64) begin : g_amo_hi_unused
                `UNUSED_VAR (amo_new_word[63:WORD_WIDTH])
            end

            assign amo_wb_addr   = amo_wb_addr_r;
            assign amo_wb_byteen = amo_wb_byteen_r;

        end else begin : g_no_amo
            assign amo_busy      = 1'b0;
            assign amo_wb_store  = 1'b0;
            assign amo_wb_addr   = '0;
            assign amo_wb_byteen = '0;
            assign amo_wb_data   = '0;
            assign sc_resolving  = 1'b0;
            assign sc_fail       = 1'b0;
            assign sc_lane_hi    = 1'b0;
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

        // the wide engine runs only while every other client is held off
        wire wide_b = wide_rd_b[i] || wide_wr_b[i];
        assign bank_sram_addr  = wide_b       ? wide_row
                               : dma_active   ? BANK_ADDR_WIDTH'(dma_bus_if.req_data.addr)
                               : amo_wb_store ? amo_wb_addr
                                              : per_bank_req_addr[i];
        assign bank_sram_wdata = wide_wr_b[i] ? wide_wdata[i]
                               : dma_wr_b     ? dma_bus_if.req_data.data[i*WORD_WIDTH +: WORD_WIDTH]
                               : amo_wb_store ? amo_wb_data
                                              : per_bank_req_data[i];
        assign bank_sram_wren  = wide_wr_b[i] ? {WORD_SIZE{1'b1}}
                               : dma_wr_b     ? dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]
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
            .read  (wide_rd_b[i] || dma_rd_b || (lsu_active && ~per_bank_req_rw[i])),
            .write (wide_wr_b[i] || dma_wr_b || amo_wb_store || (lsu_active && per_bank_req_rw[i])),
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
                last_wr_valid <= wide_wr_b[i] || dma_wr_b || amo_wb_store || (lsu_active && per_bank_req_rw[i]);
            end
            last_wr_addr <= bank_sram_addr;
        end
        wire is_rdw_hazard = last_wr_valid && ~per_bank_req_rw[i] && (per_bank_req_addr[i] == last_wr_addr);

        // LSU response valid / request ready — blocked by DMA and RDW hazards

        assign bank_rsp_valid = per_bank_req_valid[i]
                             && ~dma_active
                             && ~amo_busy
                             && ~wide_lock
                             && ~per_bank_req_rw[i]
                             && ~is_rdw_hazard;

        assign per_bank_req_ready[i] = ~dma_active
                                    && ~amo_busy
                                    && ~wide_lock
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
        // for success, one for failure, positioned in the addressed lane for
        // the LSU's subword extraction. The outcome is resolved at the commit
        // stage, which is the same cycle the response carries the SRAM output,
        // so it substitutes here rather than needing a response of its own.
        wire [WORD_WIDTH-1:0] bank_rsp_word = sc_resolving
                                            ? (sc_lane_hi ? (WORD_WIDTH'(sc_fail) << (WORD_WIDTH/2))
                                                          : WORD_WIDTH'(sc_fail))
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

    if (WIDE_ENABLE != 0) begin : g_wide_amo
        localparam NUM_PAIRS = NUM_REQS / 2;
        localparam PAIR_BITS = `UP(`CLOG2(NUM_PAIRS));
        // Cycles the banks are held before the engine reads: an atomic a bank
        // accepted just before the hold writes back two cycles later, and a
        // bank read in the cycle after a write returns the stale word.
        localparam LOCK_CYCLES = 4;

        `STATIC_ASSERT ((NUM_REQS % 2) == 0, ("wide atomics need even/odd port pairs"))
        `STATIC_ASSERT (NUM_BANKS >= 2, ("wide atomics need two banks for one word pair"))

        localparam STATE_IDLE  = 3'd0;
        localparam STATE_DRAIN = 3'd1;
        localparam STATE_LOCK  = 3'd2;
        localparam STATE_READ  = 3'd3;
        localparam STATE_CAPT  = 3'd4;
        localparam STATE_ALU   = 3'd5;
        localparam STATE_WRITE = 3'd6;
        localparam STATE_RSP   = 3'd7;

        reg [2:0]                 state_r;
        reg [2:0]                 lock_ctr_r;
        reg [PAIR_BITS-1:0]       sel_r;
        reg [ADDR_WIDTH-1:0]      addr_r;
        reg [63:0]                rhs_r;
        reg [63:0]                cmp_r;
        reg [63:0]                old_r;
        reg [63:0]                new_r;
        reg [$bits(amo_op_e)-1:0] op_r;
        reg                       unsigned_r;
        reg [HART_ID_WIDTH-1:0]   hart_id_r;
        reg [TAG_WIDTH-1:0]       tag_lo_r, tag_hi_r;
        reg [1:0]                 rsp_pend_r;

        // per-port views, so a selected pair can be indexed
        mem_bus_attr_t [NUM_REQS-1:0]       port_attr;
        wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] port_addr;
        wire [NUM_REQS-1:0][WORD_WIDTH-1:0] port_data;
        wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  port_tag;
        wire [NUM_REQS-1:0]                 port_rsp_ready;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_port
            assign port_attr[i]      = mem_bus_attr_t'(lsu_bus_if[i].req_data.attr);
            assign port_addr[i]      = lsu_bus_if[i].req_data.addr[ADDR_WIDTH-1:0];
            assign port_data[i]      = lsu_bus_if[i].req_data.data;
            assign port_tag[i]       = lsu_bus_if[i].req_data.tag;
            assign port_rsp_ready[i] = lsu_bus_if[i].rsp_ready;
            assign req_wide[i] = lsu_bus_if[i].req_valid
                              && port_attr[i].amo.amo_valid
                              && port_attr[i].amo.amo_wide;
        end
        `UNUSED_VAR (port_attr)

        // both halves of a pair have arrived
        wire [NUM_PAIRS-1:0] pair_valid;
        for (genvar k = 0; k < NUM_PAIRS; ++k) begin : g_pair_valid
            assign pair_valid[k] = req_wide[2 * k] && req_wide[2 * k + 1];
        end

        wire [PAIR_BITS-1:0] pair_idx;
        wire                 pair_any;
        VX_rr_arbiter #(
            .NUM_REQS (NUM_PAIRS)
        ) pair_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (pair_valid),
            .grant_valid  (pair_any),
            .grant_index  (pair_idx),
            `UNUSED_PIN (grant_onehot),
            .grant_ready  (pair_any && (state_r == STATE_IDLE))
        );

        wire [PAIR_BITS:0] port_lo = {pair_idx, 1'b0};
        wire [PAIR_BITS:0] port_hi = {pair_idx, 1'b1};
        mem_bus_attr_t pair_attr;
        assign pair_attr = port_attr[port_lo];
        `UNUSED_VAR (pair_attr)

        wire [BANK_SEL_WIDTH-1:0] bank_lo = BANK_SEL_WIDTH'(addr_r[0 +: BANK_SEL_BITS]);
        wire [BANK_SEL_WIDTH-1:0] bank_hi = bank_lo + BANK_SEL_WIDTH'(1);
        `RUNTIME_ASSERT (~(state_r != STATE_IDLE) || ~addr_r[0],
            ("%t: wide atomic on an odd word (addr=0x%0h)", $time, addr_r))

        wire [63:0] new_word;
        wire [63:0] ret_word_unused;
        VX_amo_alu #(
            .DATA_WIDTH  (64),
            .ARITH_WIDTH (64)
        ) wide_alu (
            .op          (amo_op_e'(op_r)),
            .is_unsigned (unsigned_r),
            .width       (2'd3),
            .old_word    (old_r),
            .rhs         (rhs_r),
            .cmp         (cmp_r),
            .new_word    (new_word),
            .ret_word    (ret_word_unused)
        );
        `UNUSED_VAR (ret_word_unused)

        wire take = pair_any && (state_r == STATE_IDLE);
        wire [1:0] sel_rsp_ready = {port_rsp_ready[{sel_r, 1'b1}], port_rsp_ready[{sel_r, 1'b0}]};

        always @(posedge clk) begin
            if (reset) begin
                state_r    <= STATE_IDLE;
                lock_ctr_r <= '0;
                rsp_pend_r <= '0;
            end else begin
                case (state_r)
                STATE_IDLE: begin
                    if (take) begin
                        state_r <= STATE_DRAIN;
                    end
                end
                STATE_DRAIN: begin
                    if (~per_bank_req_valid[bank_lo] && ~per_bank_req_valid[bank_hi]) begin
                        state_r    <= STATE_LOCK;
                        lock_ctr_r <= '0;
                    end
                end
                STATE_LOCK: begin
                    lock_ctr_r <= lock_ctr_r + 3'd1;
                    if (lock_ctr_r == 3'(LOCK_CYCLES - 1)) begin
                        state_r <= STATE_READ;
                    end
                end
                STATE_READ:  state_r <= STATE_CAPT;
                STATE_CAPT:  state_r <= STATE_ALU;
                STATE_ALU:   state_r <= STATE_WRITE;
                STATE_WRITE: begin
                    state_r    <= STATE_RSP;
                    rsp_pend_r <= 2'b11;
                end
                default: begin // STATE_RSP
                    rsp_pend_r <= rsp_pend_r & ~sel_rsp_ready;
                    if ((rsp_pend_r & ~sel_rsp_ready) == 2'b00) begin
                        state_r <= STATE_IDLE;
                    end
                end
                endcase
            end
            if (take) begin
                sel_r      <= pair_idx;
                addr_r     <= port_addr[port_lo];
                rhs_r      <= {port_data[port_hi], port_data[port_lo]};
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                cmp_r      <= 64'(pair_attr.amo.amo_cmp);
            `else
                cmp_r      <= '0;
            `endif
                op_r       <= pair_attr.amo.amo_op;
                unsigned_r <= pair_attr.amo.amo_unsigned;
                hart_id_r  <= pair_attr.amo.hart_id;
                tag_lo_r   <= port_tag[port_lo];
                tag_hi_r   <= port_tag[port_hi];
            end
            if (state_r == STATE_CAPT) begin
                old_r <= {per_bank_rsp_data[bank_hi], per_bank_rsp_data[bank_lo]};
            end
            // registered, so the 64-bit adder stays off the BRAM write path
            if (state_r == STATE_ALU) begin
                new_r <= new_word;
            end
        end

        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_take
            assign wide_take[i] = take && (PAIR_BITS'(i / 2) == pair_idx);
            // while draining, the two banks take no new request
            assign wide_drain_block[i] = (state_r == STATE_DRAIN)
                                      && ((req_bank_idx[i] == bank_lo) || (req_bank_idx[i] == bank_hi));
        end
        assign wide_take_any = take;

        assign wide_lock = (state_r == STATE_LOCK)
                        || (state_r == STATE_READ)
                        || (state_r == STATE_CAPT)
                        || (state_r == STATE_ALU)
                        || (state_r == STATE_WRITE);

        // The low word sits in the even bank of the pair, so a bank's half is
        // fixed by its index.
        for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_wide_bank
            wire in_pair = (BANK_SEL_WIDTH'(i) == bank_lo) || (BANK_SEL_WIDTH'(i) == bank_hi);
            assign wide_rd_b[i]  = (state_r == STATE_READ)  && in_pair;
            assign wide_wr_b[i]  = (state_r == STATE_WRITE) && in_pair;
            assign wide_wdata[i] = ((i % 2) != 0) ? new_r[63:32] : new_r[31:0];
        end
        assign wide_row     = addr_r[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
        assign wide_hart_id = hart_id_r;

        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_wide_rsp
            wire is_sel = (PAIR_BITS'(i / 2) == sel_r) && (state_r == STATE_RSP);
            assign wide_rsp_valid[i] = is_sel && rsp_pend_r[i % 2];
            assign wide_rsp_data[i]  = ((i % 2) != 0) ? old_r[63:32] : old_r[31:0];
            assign wide_rsp_tag[i]   = ((i % 2) != 0) ? tag_hi_r : tag_lo_r;
        end

    end else begin : g_no_wide_amo
        assign req_wide       = '0;
        assign wide_take      = '0;
        assign wide_take_any  = 1'b0;
        assign wide_drain_block = '0;
        assign wide_lock      = 1'b0;
        assign wide_rd_b      = '0;
        assign wide_wr_b      = '0;
        assign wide_row       = '0;
        assign wide_wdata     = '0;
        assign wide_hart_id   = '0;
        assign wide_rsp_valid = '0;
        assign wide_rsp_data  = '0;
        assign wide_rsp_tag   = '0;
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

    // a wide atomic's answer takes its two ports ahead of the bank responses
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_lsu_bus_if
        assign lsu_bus_if[i].rsp_valid = wide_rsp_valid[i] || rsp_valid_out[i];
        assign lsu_bus_if[i].rsp_data  = wide_rsp_valid[i] ? {wide_rsp_data[i], wide_rsp_tag[i]} : rsp_data_out[i];
        assign rsp_ready_out[i] = lsu_bus_if[i].rsp_ready && ~wide_rsp_valid[i];
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
    // a wide atomic's two port requests bypass the crossbar; they count as the
    // two reads (atomics carry rw=0) they are at the ports
    reg perf_wide_take;
    always @(posedge clk) begin
        if (reset) begin
            perf_wide_take <= 1'b0;
        end else begin
            perf_wide_take <= wide_take_any;
        end
    end
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
            perf_reads       <= perf_reads  + PERF_CTR_BITS'(perf_reads_per_cycle)
                                              + (perf_wide_take ? PERF_CTR_BITS'(2) : '0);
            perf_writes      <= perf_writes + PERF_CTR_BITS'(perf_writes_per_cycle);
            perf_crsp_stalls <= perf_crsp_stalls + PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign lmem_perf.reads       = perf_reads;
    assign lmem_perf.writes      = perf_writes;
    assign lmem_perf.bank_stalls = perf_collisions;
    assign lmem_perf.crsp_stalls = perf_crsp_stalls;

`else
    `UNUSED_VAR (wide_take_any)
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
