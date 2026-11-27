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

// One bank of the warp-wide local memory (VX_lmem_unit): its SRAM, the port A
// mux between the DMA port, an owed atomic write-back and the first engine's
// pass, and, with atomics, the bank's read-modify-write and reservation logic.
// Port B, present with two engines, serves the read engine.
module VX_lmem_bank import VX_gpu_pkg::*; #(
    // Words in the bank
    parameter SIZE            = 1024,
    // Size of a word in bytes
    parameter WORD_SIZE       = `VX_CFG_XLEN/8,
    // A second, read-only port for the read engine
    parameter DUAL            = 0,
    // DMA port present
    parameter DMA_ENABLE      = 0,
    // Synthesize atomic-op + LR/SC logic
    parameter AMO_ENABLE      = 0,
    // the first engine's per-lane payload and common attributes
    parameter PAYLOAD_WIDTH   = 1,
    parameter ATTR_WIDTH      = 1,

    parameter WORD_WIDTH      = WORD_SIZE * 8,
    parameter BANK_ADDR_WIDTH = `CLOG2(SIZE)
) (
    input wire                       clk,
    input wire                       reset,

    // DMA access, its row slice of this bank
    input wire                       dma_fire,
    input wire                       dma_rd_fire,
    input wire                       dma_rw,
    input wire [BANK_ADDR_WIDTH-1:0] dma_addr,
    input wire [WORD_WIDTH-1:0]      dma_wdata,
    input wire [WORD_SIZE-1:0]       dma_byteen,

    // the first engine's pass
    input wire                       a_valid,
    input wire                       a_rw,
    input wire                       a_amo,
    input wire [ATTR_WIDTH-1:0]      a_attr,
    input wire [BANK_ADDR_WIDTH-1:0] a_addr,
    input wire [PAYLOAD_WIDTH-1:0]   a_payload,
    // port A's SRAM output, and the first engine's response word
    output wire [WORD_WIDTH-1:0]     sram_rdata_a,
    output wire [WORD_WIDTH-1:0]     a_rdata,

    // the read engine's pass
    input wire                       b_valid,
    input wire [BANK_ADDR_WIDTH-1:0] b_addr,
    output wire [WORD_WIDTH-1:0]     b_rdata
);
    localparam AMO_OP_WIDTH  = $bits(amo_op_e);
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    localparam AMO_CMP_WIDTH = `VX_CFG_XLEN;
`endif

    // Without the DMA port no DMA access ever reaches the bank.
    wire dma_acc  = (DMA_ENABLE != 0) && dma_fire;
    wire dma_read = (DMA_ENABLE != 0) && dma_rd_fire;
    wire dma_wr_b = dma_acc && dma_rw && (| dma_byteen);

    wire lsu_active = a_valid;

    wire [WORD_WIDTH-1:0]    req_data;
    wire [WORD_SIZE-1:0]     req_byteen;
    wire [HART_ID_WIDTH-1:0] req_hart_id;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [AMO_CMP_WIDTH-1:0] req_amo_cmp;
`endif
    if (AMO_ENABLE != 0) begin : g_payload_amo
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        assign {req_hart_id, req_amo_cmp, req_byteen, req_data} = a_payload;
    `else
        assign {req_hart_id, req_byteen, req_data} = a_payload;
    `endif
    end else begin : g_payload
        assign {req_byteen, req_data} = a_payload;
        assign req_hart_id = '0;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        assign req_amo_cmp = '0;
    `endif
    end

    // Bank-facing AMO interface, tied off when the A extension is absent so
    // none of the atomic hardware is synthesized.
    wire                       amo_wb_store;  // owed atomic write-back
    wire [BANK_ADDR_WIDTH-1:0] amo_wb_addr;
    wire [WORD_SIZE-1:0]       amo_wb_byteen;
    wire [WORD_WIDTH-1:0]      amo_wb_data;
    wire                       sc_resolving;  // SC outcome replaces the response
    wire                       sc_fail;
    wire                       sc_lane_hi;    // outcome lane of a resolving .W SC

    if (AMO_ENABLE != 0) begin : g_amo

        // Atomics. An atomic carries rw=0, so the read returning the old
        // value is the ordinary read path. What the bank owes is the write
        // of the computed value.
        //
        // The old word is registered before the ALU sees it. Feeding the ALU
        // straight from the SRAM output puts the BRAM clock-to-out, the adder
        // and the BRAM setup in a single period; the register splits that
        // into BRAM-out-to-flop and flop-through-adder-to-BRAM-in.
        //
        // An atomic owns its bank for three cycles: read, capture, write back.
        reg                       amo_rd_valid_r;
        reg                       amo_wb_valid_r;
        reg [WORD_WIDTH-1:0]      amo_old_r;
        reg [BANK_ADDR_WIDTH-1:0] amo_wb_addr_r;
        reg [WORD_SIZE-1:0]       amo_wb_byteen_r;
        reg [AMO_OP_WIDTH-1:0]    amo_wb_op_r;
        reg                       amo_wb_unsigned_r;
        reg [WORD_WIDTH-1:0]      amo_wb_rhs_r;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        reg [AMO_CMP_WIDTH-1:0]   amo_wb_cmp_r;
    `endif

        wire [AMO_OP_WIDTH-1:0] req_amo_op = a_attr[1 +: AMO_OP_WIDTH];
        wire req_amo_unsigned = a_attr[0];

        wire amo_accept = lsu_active && a_amo;

        // Reservation commit stage. Every accepted request that can touch a
        // reservation -- an atomic, or a plain store -- reaches it one cycle
        // after acceptance, so a single address port serves both. The
        // look-ahead read is driven from the acceptance cycle, which is what
        // makes the registered reservation entry land here.
        wire is_lr_in = a_amo && (amo_op_e'(req_amo_op) == AMO_OP_LR);
        wire is_sc_in = a_amo && (amo_op_e'(req_amo_op) == AMO_OP_SC);

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
                res_is_amo_r   <= a_amo;
                res_is_store_r <= a_rw;
                res_addr_r     <= a_addr;
                res_hart_id_r  <= req_hart_id;
            end
        end

        // A store-conditional fails when its reservation is gone. The
        // outcome is known at the commit stage, one cycle before the
        // write-back, so it can both suppress the write and be returned as
        // this request's response.
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
        // read-modify-write atomic (which carries rw=0 and so is not a store
        // by the request flag), and a store-conditional that found its
        // reservation.
        wire res_commits_store = res_is_store_r
                              || (res_is_amo_r && ~res_is_lr_r && ~res_is_sc_r)
                              || (res_is_sc_r && res_check);

        reg  amo_wb_is_lr_r;
        reg  amo_wb_sc_fail_r;
        always @(posedge clk) begin
            if (amo_rd_valid_r) begin
                amo_wb_is_lr_r   <= res_is_lr_r;
                amo_wb_sc_fail_r <= sc_fail;
            end
        end

        // A load-reserved leaves the word alone, and a failed
        // store-conditional must not store; every other atomic writes its
        // computed value back.
        assign amo_wb_store = amo_wb_valid_r && ~amo_wb_is_lr_r && ~amo_wb_sc_fail_r;

        always @(posedge clk) begin
            if (reset) begin
                amo_rd_valid_r <= 1'b0;
                amo_wb_valid_r <= 1'b0;
            end else begin
                amo_rd_valid_r <= amo_accept;
                amo_wb_valid_r <= amo_rd_valid_r;
            end
            if (amo_rd_valid_r) begin
                amo_old_r <= sram_rdata_a;
            end
            if (amo_accept) begin
                amo_wb_addr_r     <= a_addr;
                amo_wb_byteen_r   <= req_byteen;
                amo_wb_op_r       <= req_amo_op;
                amo_wb_unsigned_r <= req_amo_unsigned;
                amo_wb_rhs_r      <= req_data;
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                amo_wb_cmp_r      <= req_amo_cmp;
            `endif
            end
        end

        wire [63:0] amo_new_word;
        wire [63:0] amo_ret_word;

        // Lane-align the RMW operands. The bank word is XLEN wide, so on a
        // 64-bit hart a .W atomic addresses one of two 32-bit lanes; the
        // ALU computes right-aligned. Store data arrives lane-positioned
        // (same packing as a plain store), so old and rhs shift down by the
        // lane offset and the result shifts back up under the request
        // byteen. The comparand arrives right-aligned from rs3 and needs no
        // shift. The addressed lane is recovered from the byteen: an atomic
        // is naturally aligned, so a .W to the upper lane never covers byte
        // 0 and a .D covers the full word.
        wire [63:0] amo_old_lane;
        wire [63:0] amo_rhs_lane;
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
        // per lmem bank it multiplies by NUM_LMEM_BANKS x NUM_CORES. After
        // the holder-credit protocol a hot word maps to exactly one station,
        // so depth only reduces false eviction between distinct reserved
        // words aliasing in one bank -- and a spurious SC failure is
        // architecturally legal. Clamp the depth.
        // A credit is spent per refused LR, not per cycle: a warp's LR
        // reaches a bank one lane per pass, so between a holder's LR and its
        // SC every other hart of the core can be refused, and a refused hart
        // can retry once more. The budget covers two refusals per hart; a
        // smaller one lets one warp's own LR burst displace its holder and a
        // contended retry loop stops making progress.
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
            // A load-reserved claims the word; a store-conditional gives up
            // its own reservation whether it succeeds or fails; any committing
            // store breaks whoever else holds one.
            .res_reserve      (res_valid_r && res_is_lr_r),
            .res_clear        (res_valid_r && res_is_sc_r),
            .res_invalidate   (res_valid_r && res_commits_store),
            .res_hart_id      (res_hart_id_r),
            .res_line_addr    (res_addr_r),
            .res_line_addr_n  (a_addr),
            .res_check        (res_check)
        );
        `UNUSED_VAR (amo_ret_word)
        if (WORD_WIDTH < 64) begin : g_amo_hi_unused
            `UNUSED_VAR (amo_new_word[63:WORD_WIDTH])
        end

        assign amo_wb_addr   = amo_wb_addr_r;
        assign amo_wb_byteen = amo_wb_byteen_r;

    end else begin : g_no_amo
        assign amo_wb_store  = 1'b0;
        assign amo_wb_addr   = '0;
        assign amo_wb_byteen = '0;
        assign amo_wb_data   = '0;
        assign sc_resolving  = 1'b0;
        assign sc_fail       = 1'b0;
        assign sc_lane_hi    = 1'b0;
        `UNUSED_VAR ({a_amo, a_attr})
        `UNUSED_VAR (req_hart_id)
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        `UNUSED_VAR (req_amo_cmp)
    `endif
    end

    // Port A address / write-data / write-enable mux: DMA, an owed atomic
    // write-back, or the first engine's pass. They never coincide.
    wire [BANK_ADDR_WIDTH-1:0] bank_sram_addr;
    wire [WORD_WIDTH-1:0]      bank_sram_wdata;
    wire [WORD_SIZE-1:0]       bank_sram_wren;

    assign bank_sram_addr  = dma_acc      ? dma_addr
                           : amo_wb_store ? amo_wb_addr
                                          : a_addr;
    assign bank_sram_wdata = dma_acc      ? dma_wdata
                           : amo_wb_store ? amo_wb_data
                                          : req_data;
    assign bank_sram_wren  = dma_acc      ? dma_byteen
                           : amo_wb_store ? amo_wb_byteen
                                          : req_byteen;

    wire sram_read  = dma_read || (lsu_active && ~a_rw);
    wire sram_write = dma_wr_b || amo_wb_store || (lsu_active && a_rw);

    if (DUAL) begin : g_dual_port
        VX_1rw1r_ram #(
            .DATAW (WORD_WIDTH),
            .SIZE  (SIZE),
            .WRENW (WORD_SIZE)
        ) lmem_store (
            .clk     (clk),
            .reset   (reset),
            .read_a  (sram_read),
            .write_a (sram_write),
            .wren_a  (bank_sram_wren),
            .addr_a  (bank_sram_addr),
            .wdata_a (bank_sram_wdata),
            .rdata_a (sram_rdata_a),
            .read_b  (b_valid),
            .addr_b  (b_addr),
            .rdata_b (b_rdata)
        );
    end else begin : g_single_port
        VX_sp_ram #(
            .DATAW    (WORD_WIDTH),
            .SIZE     (SIZE),
            .WRENW    (WORD_SIZE),
            .OUT_REG  (1),
            .RDW_MODE ("R")
        ) lmem_store (
            .clk   (clk),
            .reset (reset),
            .read  (sram_read),
            .write (sram_write),
            .wren  (bank_sram_wren),
            .addr  (bank_sram_addr),
            .wdata (bank_sram_wdata),
            .rdata (sram_rdata_a)
        );
        assign b_rdata = '0;
        `UNUSED_VAR (b_valid)
        `UNUSED_VAR (b_addr)
    end

    // A store-conditional returns its outcome, not the word it read: zero
    // for success, one for failure, positioned in the addressed lane for
    // the LSU's subword extraction. The outcome is resolved at the commit
    // stage, the cycle the response takes the SRAM output.
    assign a_rdata = sc_resolving
                   ? (sc_lane_hi ? (WORD_WIDTH'(sc_fail) << (WORD_WIDTH/2))
                                 : WORD_WIDTH'(sc_fail))
                   : sram_rdata_a;

endmodule
