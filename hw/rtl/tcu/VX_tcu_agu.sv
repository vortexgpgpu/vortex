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

`ifdef VX_CFG_TCU_SPARSE_ENABLE

// Warp-level AGU for TCU_LD instructions.
//
// One TCU_LD = one memory fetch that lands in one VX_tcu_meta slot
// (selected by op_args.tcu.fmt_d). Software emits N TCU_LD ops to
// fill N slots — same granularity as the META_STORE phase it replaces.
//
// State machine: IDLE → ISSUE → WAIT_RSP → COMMIT.
//   IDLE     : monitor per-block execute_if for INST_TCU_LD.
//   ISSUE    : drive client_if.req with rs1 (base address). Picks the
//              lowest-indexed block with a pending TCU_LD (single AGU,
//              shared across blocks).
//   WAIT_RSP : wait for the matching client_if.rsp.
//   COMMIT   : drive meta_wr_* (broadcast to all blocks' tcu_meta
//              instances) and assert the originating block's result_if.
//              op_args.tcu.fmt_d → meta slot index; warp id from the
//              execute header.
//
// For P2d this is the minimum viable AGU — multi-request stride
// patterns are deferred until profiling shows we need them.

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
    output wire [3:0]                 meta_wr_idx,
    output wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] meta_wr_data,

    // Result_if hand for the originating block. The wrapper muxes this
    // with tcu_core's normal result_if path.
    output wire [BLOCK_SIZE-1:0]      result_valid,
    output tcu_result_t               result_data        [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0]      result_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_IDX_BITS = (BLOCK_SIZE > 1) ? $clog2(BLOCK_SIZE) : 1;

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_ISSUE = 2'd1,
        S_WAIT  = 2'd2,
        S_COMMIT= 2'd3
    } state_e;

    state_e                       state_r;
    logic [BLOCK_IDX_BITS-1:0]    owner_block_r;     // which block's TCU_LD is in flight
    // Latch only the fields we need (avoids unused-bit warnings on the
    // full tcu_execute_t struct).
    tcu_header_t                  owner_header_r;
    logic [3:0]                   owner_slot_r;      // op_args.tcu.fmt_d
    logic [3:0]                   owner_fmt_r;       // op_args.tcu.fmt_s
    logic [`VX_CFG_XLEN-1:0]      owner_addr_r;      // rs1_data[0]
    logic [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] rsp_data_r;
    // Track which lanes have received valid response data. The
    // mem_subsystem may stream responses across multiple cycles when
    // the scheduler coalesces same-line accesses; we transition to
    // S_COMMIT only when all requested lanes have responded.
    logic [NUM_LANES-1:0]         rsp_lane_done_r;

    // -----------------------------------------------------------------------
    // Lowest-block priority pick (single AGU shared across blocks).
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] block_valid_v = per_block_ld_valid;

    logic [BLOCK_IDX_BITS-1:0] first_valid_idx;
    logic                       any_valid;
    always_comb begin
        first_valid_idx = '0;
        any_valid       = 1'b0;
        for (int bi = BLOCK_SIZE-1; bi >= 0; bi--) begin
            if (block_valid_v[bi]) begin
                first_valid_idx = BLOCK_IDX_BITS'(bi);
                any_valid       = 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            state_r          <= S_IDLE;
            owner_block_r    <= '0;
            owner_header_r   <= '0;
            owner_slot_r     <= '0;
            owner_fmt_r      <= '0;
            owner_addr_r     <= '0;
            rsp_data_r       <= '0;
            rsp_lane_done_r  <= '0;
        end else begin
            case (state_r)
                S_IDLE: begin
                    if (any_valid) begin
                        owner_block_r   <= first_valid_idx;
                        owner_header_r  <= per_block_ld_data[first_valid_idx].header;
                        owner_slot_r    <= per_block_ld_data[first_valid_idx].op_args.tcu.fmt_d;
                        owner_fmt_r     <= per_block_ld_data[first_valid_idx].op_args.tcu.fmt_s;
                        owner_addr_r    <= per_block_ld_data[first_valid_idx].rs1_data[0];
                        rsp_lane_done_r <= '0;
                        state_r         <= S_ISSUE;
                    end
                end
                S_ISSUE: begin
                    if (client_if.req_ready) begin
                        state_r <= S_WAIT;
                    end
                end
                S_WAIT: begin
                    if (client_if.rsp_valid) begin
                        // Merge this packet's lane data into rsp_data_r.
                        for (int t = 0; t < NUM_LANES; ++t) begin
                            if (client_if.rsp_data.mask[t]) begin
                                rsp_data_r[t]      <= client_if.rsp_data.data[t];
                                rsp_lane_done_r[t] <= 1'b1;
                            end
                        end
                        // Only commit when EOP arrives AND every requested
                        // lane has its data. With multi-cycle responses
                        // (mem_subsystem may coalesce same-line lanes and
                        // stream the demux), keep waiting until all lanes
                        // accounted for.
                        if (client_if.rsp_data.eop
                            && ((rsp_lane_done_r | client_if.rsp_data.mask) == {NUM_LANES{1'b1}})) begin
                            state_r <= S_COMMIT;
                        end
                    end
                end
                S_COMMIT: begin
                    if (result_ready[owner_block_r]) begin
                        state_r <= S_IDLE;
                    end
                end
                default: state_r <= S_IDLE;
            endcase
        end
    end

    // Unused sub-fields of rsp_data (we consume valid/eop/data only).
    `UNUSED_VAR (client_if.rsp_data.tag)
    `UNUSED_VAR (client_if.rsp_data.sop)

    // -----------------------------------------------------------------------
    // execute_if.ready: only assert for the picked block in S_IDLE when we
    // latch its data (transition to S_ISSUE).
    // -----------------------------------------------------------------------
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_ld_ready
        assign per_block_ld_ready[bi] =
            (state_r == S_IDLE)
            && per_block_ld_valid[bi]
            && (first_valid_idx == BLOCK_IDX_BITS'(bi));
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
    wire [`VX_CFG_XLEN-1:0] base_addr = owner_addr_r;
    `UNUSED_VAR (base_addr[`CLOG2(LSU_WORD_SIZE)-1:0])

    wire [LSU_ADDR_WIDTH-1:0] base_word_addr =
        LSU_ADDR_WIDTH'(base_addr[`VX_CFG_XLEN-1:`CLOG2(LSU_WORD_SIZE)]);

    localparam PWD = TCU_META_PER_WARP_DEPTH;
    `UNUSED_VAR (owner_fmt_r)
    // Per-thread metadata layout: lane T loads h_meta[slot*NT + T]. The
    // host packs h_meta so the 32-bit word at offset (slot*NT + T) holds
    // the metadata destined for SRAM cell (bank = T%PWD, col = T/PWD) —
    // matching the way the old META_STORE phase pulled lane T's f-register
    // and wrote it to the same (bank, col). One formula serves both
    // WMMA-SP (which needs distinct metadata in every SRAM bank) and
    // WGMMA-SP — the previous host_bank/RTL_HALF_K collapse made WMMA-SP
    // impossible because two SRAM banks were forced to share a single
    // host word.

    // LSU bus carries LSU_WORD_SIZE-byte chunks per lane (= XLEN/8 bytes).
    // The metadata layout is contiguous 32-bit words, so when XLEN > 32
    // each LSU response carries XLEN/32 logical 32-bit words and we need
    // to pick the right one. LSU_W32_RATIO = bytes/4, so XLEN=64 → 2.
    localparam LSU_W32_RATIO = LSU_WORD_SIZE / 4;
    localparam LG_W32_RATIO  = (LSU_W32_RATIO > 1) ? $clog2(LSU_W32_RATIO) : 0;

    // is_addr_local route bit — matches VX_lsu_slice.sv:97-99 LMEM range
    // check. The AGU emits a per-warp base address from rs1, but since
    // the metadata buffer is per-warp inside shared memory, the LMEM
    // detection is uniform across lanes; compute once from the base.
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

    lsu_client_req_data_t req_w;
    always_comb begin
        req_w        = '0;
        req_w.rw     = 1'b0; // load
        req_w.mask   = {NUM_LANES{1'b1}};
        half_sel_w   = '0;
        for (int i = 0; i < NUM_LANES; i++) begin
            automatic mem_bus_attr_t a;
            automatic logic [LSU_ADDR_WIDTH-1:0] word_off_32, slot_base;
            a = '0;
            a.is_addr_local = base_is_local;
            // Per-thread layout: lane i loads h_meta[slot * NT + i]. This
            // matches the META_STORE / pack_metadata convention where lane
            // T's word lands at SRAM (bank = T%PWD, col = T/PWD).
            slot_base   = LSU_ADDR_WIDTH'(owner_slot_r) * LSU_ADDR_WIDTH'(NUM_LANES);
            word_off_32 = slot_base + LSU_ADDR_WIDTH'(i);
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
        // Tag pattern: tag width must match LSU_CLIENT_TAG_WIDTH but the
        // AGU doesn't read tags back as anything meaningful — just sized.
        req_w.tag = '0;
    end
    `UNUSED_PARAM (PWD)
    assign client_if.req_valid  = (state_r == S_ISSUE);
    assign client_if.req_data   = req_w;
    assign client_if.rsp_ready  = (state_r == S_WAIT);

    // -----------------------------------------------------------------------
    // VX_tcu_meta write (broadcast to all blocks' meta instances).
    // -----------------------------------------------------------------------
    assign meta_wr_en   = (state_r == S_COMMIT);
    assign meta_wr_wid  = owner_header_r.wid;
    assign meta_wr_idx  = owner_slot_r;
    // Repack response data into TCU_BLOCK_CAP × XLEN. The downstream
    // meta SRAM consumes only the low 32 bits, so for XLEN=64 we mux the
    // half of the LSU response that holds this lane's logical 32-bit word.
    for (genvar i = 0; i < TCU_BLOCK_CAP; ++i) begin : g_meta_data
        if (i < NUM_LANES) begin : g_in
            if (LG_W32_RATIO == 0) begin : g_w32_passthru
                assign meta_wr_data[i] = `VX_CFG_XLEN'(rsp_data_r[i]);
            end else begin : g_w32_pick
                wire [LSU_W32_RATIO-1:0][31:0] rsp_w32_v = rsp_data_r[i];
                assign meta_wr_data[i] = `VX_CFG_XLEN'(rsp_w32_v[half_sel_w[i]]);
            end
        end else begin : g_pad
            assign meta_wr_data[i] = '0;
        end
    end

`ifdef VX_TCU_LD_TRACE
    // P4 trace: META_SRAM write. Format (matches SimX emitter):
    //   META_TRC,wid,bank,col,addr,value
    // Enabled by defining VX_TCU_LD_TRACE on the Verilator command line.
    localparam P4_TRC_PWD = TCU_META_PER_WARP_DEPTH;
    localparam P4_TRC_CPL = TCU_META_COLS_PER_LOAD;
    always @(posedge clk) begin
        if (meta_wr_en) begin
            for (int i = 0; i < NUM_LANES; ++i) begin
                automatic int unsigned tbank = i % P4_TRC_PWD;
                automatic int unsigned tcol_in_grp = i / P4_TRC_PWD;
                automatic int unsigned tcol  = owner_slot_r * P4_TRC_CPL
                                             + tcol_in_grp;
                $write("META_TRC,%0d,%0d,%0d,0x%h,0x%h\n",
                    owner_header_r.wid, tbank, tcol,
                    owner_addr_r, meta_wr_data[i]);
            end
        end
    end
`endif

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
        commit_data_w.header = owner_header_r;
        commit_data_w.data   = '0;
    end

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result
        assign result_valid[bi] = (state_r == S_COMMIT) && (owner_block_r == BLOCK_IDX_BITS'(bi));
        assign result_data[bi]  = commit_data_w;
    end

endmodule

`endif // VX_CFG_TCU_SPARSE_ENABLE
