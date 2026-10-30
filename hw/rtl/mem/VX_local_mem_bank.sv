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

// One bank of the local memory (VX_local_mem): its SRAM, the atomic and
// LR/SC logic, the read-during-write and DMA arbitration, and the response
// staging. Every bank of a VX_local_mem is the same module with the same
// parameters, the VX_cache_bank pattern.
module VX_local_mem_bank import VX_gpu_pkg::*; #(
    // Size of a word in bytes
    parameter WORD_SIZE       = `VX_CFG_XLEN/8,
    // Words in the bank
    parameter NUM_WORDS       = 1024,
    // Request tag size
    parameter TAG_WIDTH       = 16,
    // Request index (the LSU port a response returns to)
    parameter REQ_SEL_WIDTH   = 1,
    // DMA port present
    parameter DMA_ENABLE      = 0,
    // Atomic-op + LR/SC logic
    parameter AMO_ENABLE      = 0,

    parameter WORD_WIDTH      = WORD_SIZE * 8,
    parameter BANK_ADDR_WIDTH = `CLOG2(NUM_WORDS)
) (
    input wire clk,
    input wire reset,

    // LSU request
    input wire                          req_valid,
    input wire                          req_rw,
    input wire [BANK_ADDR_WIDTH-1:0]    req_addr,
    input wire [WORD_SIZE-1:0]          req_byteen,
    input wire [WORD_WIDTH-1:0]         req_data,
    input wire [TAG_WIDTH-1:0]          req_tag,
    input wire [REQ_SEL_WIDTH-1:0]      req_idx,
    input wire                          req_amo_valid,
    input wire [$bits(amo_op_e)-1:0]    req_amo_op,
    input wire                          req_amo_unsigned,
    input wire [HART_ID_WIDTH-1:0]      req_amo_hart_id,
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    input wire [`VX_CFG_XLEN-1:0]       req_amo_cmp,
`endif
    output wire                         req_ready,

    // LSU response
    output wire                         rsp_valid,
    output wire [WORD_WIDTH-1:0]        rsp_data,
    output wire [TAG_WIDTH-1:0]         rsp_tag,
    output wire [REQ_SEL_WIDTH-1:0]     rsp_idx,
    input wire                          rsp_ready,

    // The SRAM's read data (the DMA response's word of this bank)
    output wire [WORD_WIDTH-1:0]        rsp_rdata,

    // DMA request (this bank's word of it); DMA has priority over the LSU
    input wire                          dma_req_valid,
    input wire                          dma_req_rw,
    input wire [BANK_ADDR_WIDTH-1:0]    dma_req_addr,
    input wire [WORD_SIZE-1:0]          dma_req_byteen,
    input wire [WORD_WIDTH-1:0]         dma_req_data,
    input wire                          dma_rsp_buf_ready
);
    localparam WORDS_PER_BANK = NUM_WORDS;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    localparam AMO_CMP_WIDTH  = `VX_CFG_XLEN;
`endif

    wire bank_rsp_valid, bank_rsp_ready;

    // DMA active signals (priority over LSU)
    wire dma_wr_b = DMA_ENABLE
                 && dma_req_valid
                 && dma_req_rw
                 && (|dma_req_byteen);

    wire dma_rd_b = DMA_ENABLE
                 && dma_req_valid
                 && ~dma_req_rw
                 && dma_rsp_buf_ready;

    wire dma_active = dma_wr_b | dma_rd_b;

    wire lsu_active = req_valid && req_ready;

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

        wire amo_accept = lsu_active && req_amo_valid;

        // Reservation commit stage. Every accepted request that can touch a
        // reservation -- an atomic, or a plain store -- reaches it one cycle
        // after acceptance, so a single address port serves both. The look-ahead
        // read is driven from the acceptance cycle, which is what makes the
        // registered reservation entry land here.
        wire is_lr_in = req_amo_valid
                     && (amo_op_e'(req_amo_op) == AMO_OP_LR);
        wire is_sc_in = req_amo_valid
                     && (amo_op_e'(req_amo_op) == AMO_OP_SC);

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
                res_is_amo_r   <= req_amo_valid;
                res_is_store_r <= req_rw;
                res_addr_r     <= req_addr;
                res_hart_id_r  <= req_amo_hart_id;
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
                    res_lane_hi_r <= ~req_byteen[0];
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
                amo_old_r <= rsp_rdata;
            end
            if (amo_accept) begin
                amo_wb_addr_r     <= req_addr;
                amo_wb_byteen_r   <= req_byteen;
                amo_wb_op_r       <= req_amo_op;
                amo_wb_unsigned_r <= req_amo_unsigned;
                amo_wb_rhs_r      <= req_data;
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                amo_wb_cmp_r      <= req_amo_cmp;
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
        // legal. Clamp the depth.
        // A credit is spent per refused LR, not per cycle: a warp's LR reaches
        // the bank as one request per lane, back to back, so between a holder's
        // LR and its SC every other hart of the core can be refused, and a
        // refused hart can retry once more. The budget covers two refusals per
        // hart; a smaller one lets one warp's own LR burst displace its holder
        // and a contended retry loop stops making progress.
        VX_amo_unit #(
            .NUM_RES_ENTRIES  (`MIN(`VX_CFG_AMO_RS_SIZE, 4)),
            .HOLD_CREDIT_BITS (`CLOG2(`VX_CFG_NUM_WARPS * `VX_CFG_NUM_THREADS) + 1),
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
            .res_line_addr_n  (req_addr),
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
        `UNUSED_VAR (req_amo_valid)
        `UNUSED_VAR (req_amo_op)
        `UNUSED_VAR (req_amo_unsigned)
        `UNUSED_VAR (req_amo_hart_id)
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        `UNUSED_VAR (req_amo_cmp)
    `endif
    end

    // SRAM address / write-data / write-enable mux: DMA first, then an owed
    // atomic write-back, then the incoming request
    wire [BANK_ADDR_WIDTH-1:0] bank_sram_addr;
    wire [WORD_WIDTH-1:0]      bank_sram_wdata;
    wire [WORD_SIZE-1:0]       bank_sram_wren;

    assign bank_sram_addr  = dma_active   ? dma_req_addr
                           : amo_wb_store ? amo_wb_addr
                                          : req_addr;
    assign bank_sram_wdata = dma_wr_b     ? dma_req_data
                           : amo_wb_store ? amo_wb_data
                                          : req_data;
    assign bank_sram_wren  = dma_wr_b     ? dma_req_byteen
                           : amo_wb_store ? amo_wb_byteen
                                          : req_byteen;

    VX_sp_ram #(
        .DATAW (WORD_WIDTH),
        .SIZE  (WORDS_PER_BANK),
        .WRENW (WORD_SIZE),
        .OUT_REG (1),
        .RDW_MODE ("R")
    ) lmem_store (
        .clk   (clk),
        .reset (reset),
        .read  (dma_rd_b || (lsu_active && ~req_rw)),
        .write (dma_wr_b || amo_wb_store || (lsu_active && req_rw)),
        .wren  (bank_sram_wren),
        .addr  (bank_sram_addr),
        .wdata (bank_sram_wdata),
        .rdata (rsp_rdata)
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
            last_wr_valid <= dma_wr_b || amo_wb_store || (lsu_active && req_rw);
        end
        last_wr_addr <= bank_sram_addr;
    end
    wire is_rdw_hazard = last_wr_valid && ~req_rw && (req_addr == last_wr_addr);

    // LSU response valid / request ready — blocked by DMA and RDW hazards

    assign bank_rsp_valid = req_valid
                         && ~dma_active
                         && ~amo_busy
                         && ~req_rw
                         && ~is_rdw_hazard;

    assign req_ready = ~dma_active
                    && ~amo_busy
                    && (bank_rsp_ready || req_rw)
                    && ~is_rdw_hazard;

    // Delay tag/idx to align with SRAM 1-cycle output latency
    VX_pipe_buffer #(
        .DATAW (REQ_SEL_WIDTH + TAG_WIDTH)
    ) bram_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bank_rsp_valid),
        .ready_in  (bank_rsp_ready),
        .data_in   ({req_idx, req_tag}),
        .data_out  ({rsp_idx, rsp_tag}),
        .valid_out (rsp_valid),
        .ready_out (rsp_ready)
    );

    // A store-conditional returns its outcome, not the word it read: zero
    // for success, one for failure, positioned in the addressed lane for
    // the LSU's subword extraction. The outcome is resolved at the commit
    // stage, which is the same cycle the response carries the SRAM output,
    // so it substitutes here rather than needing a response of its own.
    wire [WORD_WIDTH-1:0] bank_rsp_word = sc_resolving
                                        ? (sc_lane_hi ? (WORD_WIDTH'(sc_fail) << (WORD_WIDTH/2))
                                                      : WORD_WIDTH'(sc_fail))
                                        : rsp_rdata;

    // Back-pressure-safe LSU response data. The bank SRAM OUT_REG
    // (rsp_rdata) is read live by the response xbar, but the
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
        end else if (rsp_valid && rsp_ready) begin
            lsu_rsp_hold_valid_r <= 1'b0;                     // response consumed
        end else if (rsp_valid && ~lsu_rsp_hold_valid_r) begin
            lsu_rsp_hold_data_r  <= bank_rsp_word;            // capture before DMA can clobber
            lsu_rsp_hold_valid_r <= 1'b1;
        end
    end
    assign rsp_data = lsu_rsp_hold_valid_r ? lsu_rsp_hold_data_r
                                           : bank_rsp_word;

endmodule
