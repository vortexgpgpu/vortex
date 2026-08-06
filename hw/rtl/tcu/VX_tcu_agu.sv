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

`ifdef TCU_META_ENABLE

// Warp-level AGU for TCU_LD instructions. Each TCU block can hold one
// transaction, while the LSU request and metadata write ports are shared.

module VX_tcu_agu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_SIZE          = `VX_CFG_NUM_TCU_BLOCKS,
    parameter NUM_LANES           = `VX_CFG_NUM_TCU_LANES
) (
    input wire clk,
    input wire reset,

    // Per-block dispatch observation. The unit feeds INST_TCU_LD
    // instructions here (gated by op_type at the wrapper); other
    // op_types are masked so the AGU never sees them on .valid.
    input  wire [BLOCK_SIZE-1:0]      per_block_ld_valid,
    input  tcu_execute_t              per_block_ld_data  [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0]      per_block_ld_ready,

    // Memory client connection (to VX_lsu_scheduler at VX_core).
    VX_lsu_sched_if.master           client_if,

    // Meta SRAM write port (broadcast — same data lands in every
    // block's VX_tcu_meta so subsequent wmma_sp reads see it
    // regardless of which block the warp resides on).
    output wire                       meta_wr_en,
    output wire [NW_WIDTH-1:0]        meta_wr_wid,
    output wire [4:0]                 meta_wr_idx,
    output wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] meta_wr_data,

    // Result_if hand for the originating block. The wrapper muxes this
    // with tcu_core's normal result_if path.
    output wire [BLOCK_SIZE-1:0]      result_valid,
    output tcu_result_t               result_data        [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0]      result_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_IDX_BITS = (BLOCK_SIZE > 1) ? $clog2(BLOCK_SIZE) : 1;

    logic [BLOCK_SIZE-1:0]       busy_r;
    logic [BLOCK_SIZE-1:0]       issued_r;
    logic [BLOCK_SIZE-1:0]       complete_r;
    tcu_header_t                 header_r        [BLOCK_SIZE];
    logic [4:0]                  slot_r          [BLOCK_SIZE];
    logic [`VX_CFG_XLEN-1:0]     addr_r          [BLOCK_SIZE];
    logic [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0]
                                  rsp_data_r       [BLOCK_SIZE];
    logic [NUM_LANES-1:0]        rsp_lane_done_r  [BLOCK_SIZE];

    wire [BLOCK_SIZE-1:0] req_pending = busy_r & ~issued_r;
    wire [BLOCK_IDX_BITS-1:0] req_block;
    wire [BLOCK_SIZE-1:0] req_grant;
    wire req_valid;

    VX_generic_arbiter #(
        .NUM_REQS (BLOCK_SIZE),
        .TYPE     ("R")
    ) req_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (req_pending),
        .grant_index  (req_block),
        .grant_onehot (req_grant),
        .grant_valid  (req_valid),
        .grant_ready  (client_if.req_ready)
    );

    wire [BLOCK_SIZE-1:0] commit_pending = complete_r;
    wire [BLOCK_IDX_BITS-1:0] commit_block;
    wire [BLOCK_SIZE-1:0] commit_grant;
    wire commit_valid;
    wire commit_ready = result_ready[commit_block];

    VX_generic_arbiter #(
        .NUM_REQS (BLOCK_SIZE),
        .TYPE     ("R")
    ) commit_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (commit_pending),
        .grant_index  (commit_block),
        .grant_onehot (commit_grant),
        .grant_valid  (commit_valid),
        .grant_ready  (commit_ready)
    );

    wire [BLOCK_IDX_BITS-1:0] rsp_block =
        client_if.rsp_data.tag[BLOCK_IDX_BITS-1:0];
    wire rsp_fire = client_if.rsp_valid && client_if.rsp_ready;
    wire req_fire = req_valid && client_if.req_ready;
    wire commit_fire = commit_valid && commit_ready;
    `UNUSED_VAR (req_grant)
    `UNUSED_VAR (commit_grant)

    always @(posedge clk) begin
        if (reset) begin
            busy_r          <= '0;
            issued_r        <= '0;
            complete_r      <= '0;
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) begin
                header_r[bi]        <= '0;
                slot_r[bi]          <= '0;
                addr_r[bi]          <= '0;
                rsp_data_r[bi]      <= '0;
                rsp_lane_done_r[bi] <= '0;
            end
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) begin
                if (per_block_ld_valid[bi] && per_block_ld_ready[bi]) begin
                    busy_r[bi]          <= 1'b1;
                    issued_r[bi]        <= 1'b0;
                    complete_r[bi]      <= 1'b0;
                    header_r[bi]        <= per_block_ld_data[bi].header;
                    slot_r[bi]          <= per_block_ld_data[bi].op_args.tcu.fmt_d;
                    addr_r[bi]          <= per_block_ld_data[bi].rs1_data[0];
                    rsp_data_r[bi]      <= '0;
                    rsp_lane_done_r[bi] <= '0;
                end
            end
            if (req_fire) begin
                issued_r[req_block] <= 1'b1;
            end
            if (rsp_fire) begin
                for (int t = 0; t < NUM_LANES; ++t) begin
                    if (client_if.rsp_data.mask[t]) begin
                        rsp_data_r[rsp_block][t]      <= client_if.rsp_data.data[t];
                        rsp_lane_done_r[rsp_block][t] <= 1'b1;
                    end
                end
                if (client_if.rsp_data.eop
                    && ((rsp_lane_done_r[rsp_block] | client_if.rsp_data.mask)
                        == {NUM_LANES{1'b1}})) begin
                    complete_r[rsp_block] <= 1'b1;
                end
            end
            if (commit_fire) begin
                busy_r[commit_block]          <= 1'b0;
                issued_r[commit_block]        <= 1'b0;
                complete_r[commit_block]      <= 1'b0;
                rsp_lane_done_r[commit_block] <= '0;
            end
        end
    end

    `UNUSED_VAR (client_if.rsp_data.sop)

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_ld_ready
        assign per_block_ld_ready[bi] = ~busy_r[bi];
    end

    // -----------------------------------------------------------------------
    // Memory request: address from rs1 (warp-broadcast). NUM_LANES mask
    // all-ones — we fetch the metadata tile in one warp-level request.
    //
    // Per-lane offset matches the host's pack_metadata_wg layout
    // (sgemm_tcu_wg_sp/main.cpp): the host writes bank `b` and word `c`
    // at index `b * stride + c`. The AGU writes wr_data[T] into
    // VX_tcu_meta at (bank=T%PWD, col_in_group=T/PWD), so lane T must
    // load `meta_base[(T % PWD) * stride + (T / PWD)]`.
    // -----------------------------------------------------------------------
    wire [`VX_CFG_XLEN-1:0] base_addr = addr_r[req_block];
    `UNUSED_VAR (base_addr[`CLOG2(LSU_WORD_SIZE)-1:0])

    wire [LSU_ADDR_WIDTH-1:0] base_word_addr =
        LSU_ADDR_WIDTH'(base_addr[`VX_CFG_XLEN-1:`CLOG2(LSU_WORD_SIZE)]);

    localparam PWD = TCU_META_PER_WARP_DEPTH;
    // Per-thread metadata layout: lane T loads h_meta[slot*NT + T]. The
    // host packs h_meta so the 32-bit word at offset (slot*NT + T) holds
    // the metadata destined for SRAM cell (bank = T%PWD, col = T/PWD).
    // One formula serves both WMMA-SP and WGMMA-SP.

    // LSU bus carries LSU_WORD_SIZE-byte chunks per lane (= XLEN/8 bytes).
    // The metadata layout is contiguous 32-bit words, so when XLEN > 32
    // each LSU response carries XLEN/32 logical 32-bit words and we need
    // to pick the right one. LSU_W32_RATIO = bytes/4, so XLEN=64 → 2.
    localparam LSU_W32_RATIO = LSU_WORD_SIZE / 4;
    localparam LG_W32_RATIO  = (LSU_W32_RATIO > 1) ? $clog2(LSU_W32_RATIO) : 0;

    // is_addr_local route bit — LMEM range check. The metadata buffer is
    // per-warp inside shared memory so LMEM detection is uniform across
    // lanes; compute once from the base address.
`ifdef VX_CFG_LMEM_ENABLE
    wire [LSU_ADDR_WIDTH-1:0] lmem_addr_lo =
        LSU_ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR) >> `CLOG2(LSU_WORD_SIZE));
    wire [LSU_ADDR_WIDTH-1:0] lmem_addr_hi =
        LSU_ADDR_WIDTH'((`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR)
                       + `VX_CFG_XLEN'(1 << `VX_CFG_LMEM_LOG_SIZE))
                       >> `CLOG2(LSU_WORD_SIZE));
    wire base_is_local = (base_word_addr >= lmem_addr_lo)
                      && (base_word_addr <  lmem_addr_hi);
`else
    wire base_is_local = 1'b0;
`endif

    // Per-lane sub-word selector: which 32-bit slice of the LSU response
    // carries the desired metadata word. For XLEN=32 (LSU_W32_RATIO=1)
    // every lane reads exactly one 32-bit word, so half_sel is always 0.
    // For XLEN=64 (LSU_W32_RATIO=2) we round the 32-bit word offset down
    // to an 8-byte LSU address and use the LSB to select low/high half.
    logic [NUM_LANES-1:0][`UP(LG_W32_RATIO)-1:0] half_sel_w;
    // For XLEN=32, LG_W32_RATIO=0 and half_sel_w is never read downstream;
    // tag it unused so Verilator's UNUSEDSIGNAL doesn't fail the build.
    if (LG_W32_RATIO == 0) begin : g_half_sel_unused
        `UNUSED_VAR (half_sel_w)
    end

    lsu_req_data_t req_w;
    always_comb begin
        req_w        = '0;
        req_w.rw     = 1'b0; // load
        req_w.mask   = {NUM_LANES{1'b1}};
        half_sel_w   = '0;
        for (int i = 0; i < NUM_LANES; i++) begin
            automatic mem_bus_attr_t a;
            automatic logic [LSU_ADDR_WIDTH-1:0] word_off_32;
            a = '0;
            a.is_addr_local = base_is_local;
            // base_addr is pre-advanced per TCU_LD slot by the kernel
            // (slot s passes base + s*NT*sizeof(float)), so lane i reads word i
            // relative to base_word_addr.
            word_off_32 = LSU_ADDR_WIDTH'(i);
            // word_off_32 is in 32-bit-word units. The LSU bus addresses
            // LSU_WORD_SIZE-byte chunks (= XLEN/32 × 32-bit words), so we
            // shift down to bus stride and keep the sub-word half for the
            // response extraction below.
            req_w.addr[i]   = base_word_addr + (word_off_32 >> LG_W32_RATIO);
            if (LG_W32_RATIO > 0) begin
                half_sel_w[i] = `UP(LG_W32_RATIO)'(word_off_32[`UP(LG_W32_RATIO)-1:0]);
            end
            req_w.byteen[i] = {LSU_WORD_SIZE{1'b1}};
            req_w.data[i]   = '0;
            req_w.attr[i]   = a;
        end
        req_w.tag = '0;
        req_w.tag[BLOCK_IDX_BITS-1:0] = req_block;
    end
    `UNUSED_PARAM (PWD)
    `STATIC_ASSERT (LSU_CLIENT_TAG_WIDTH >= BLOCK_IDX_BITS, ("LSU client tag cannot encode TCU block"))
    assign client_if.req_valid  = req_valid;
    assign client_if.req_data   = req_w;
    assign client_if.rsp_ready  = busy_r[rsp_block] && issued_r[rsp_block]
                               && !complete_r[rsp_block];

    // -----------------------------------------------------------------------
    // VX_tcu_meta write (broadcast to all blocks' meta instances).
    // -----------------------------------------------------------------------
    assign meta_wr_en   = commit_valid;
    assign meta_wr_wid  = header_r[commit_block].wid;
    assign meta_wr_idx  = slot_r[commit_block];
    // Repack response data into TCU_BLOCK_CAP × XLEN. The downstream
    // meta SRAM consumes only the low 32 bits, so for XLEN=64 we mux the
    // half of the LSU response that holds this lane's logical 32-bit word.
    for (genvar i = 0; i < TCU_BLOCK_CAP; ++i) begin : g_meta_data
        if (i < NUM_LANES) begin : g_in
            if (LG_W32_RATIO == 0) begin : g_w32_passthru
                assign meta_wr_data[i] = `VX_CFG_XLEN'(rsp_data_r[commit_block][i]);
            end else begin : g_w32_pick
                wire [LSU_W32_RATIO-1:0][31:0] rsp_w32_v = rsp_data_r[commit_block][i];
                assign meta_wr_data[i] = `VX_CFG_XLEN'(rsp_w32_v[half_sel_w[i]]);
            end
        end else begin : g_pad
            assign meta_wr_data[i] = '0;
        end
    end

    // -----------------------------------------------------------------------
    // Result drive (commits to scoreboard via writeback, releasing
    // wr_xregs[0]). Only the owner block's result_if fires.
    // tcu_result_t = { header, data[NUM_LANES][XLEN] }.
    // The header is propagated from the latched execute_t header — it
    // already carries wr_xregs and wb=0 from the decoder.
    // -----------------------------------------------------------------------
    tcu_result_t commit_data_w;
    always_comb begin
        commit_data_w        = '0;
        commit_data_w.header = header_r[commit_block];
        commit_data_w.data   = '0;
    end

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result
        assign result_valid[bi] = commit_valid && (commit_block == BLOCK_IDX_BITS'(bi));
        assign result_data[bi]  = commit_data_w;
    end

endmodule

`endif // TCU_META_ENABLE
