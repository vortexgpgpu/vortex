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
// transaction. With one memory port the blocks share it; with one port per
// block (NUM_PORTS == BLOCK_SIZE) each block loads through its own LSU
// scheduler, so the loads of a warpgroup's warps proceed in parallel. The
// metadata write port is shared.

module VX_tcu_agu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_SIZE          = `VX_CFG_NUM_TCU_BLOCKS,
    parameter NUM_LANES           = `VX_CFG_NUM_TCU_LANES,
    parameter NUM_PORTS           = 1
) (
    input wire clk,
    input wire reset,

    // Per-block dispatch observation. The unit feeds INST_TCU_LD
    // instructions here (gated by op_type at the wrapper); other
    // op_types are masked so the AGU never sees them on .valid.
    input  wire [BLOCK_SIZE-1:0]      per_block_ld_valid,
    input  tcu_execute_t              per_block_ld_data  [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0]      per_block_ld_ready,

    // Memory client connections (to the LSU schedulers).
    VX_lsu_sched_if.master           client_if [NUM_PORTS],

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
    `STATIC_ASSERT ((NUM_PORTS == 1) || (NUM_PORTS == BLOCK_SIZE), ("invalid parameter: NUM_PORTS=%0d", NUM_PORTS))
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
    wire commit_fire = commit_valid && commit_ready;
    `UNUSED_VAR (commit_grant)

    // -----------------------------------------------------------------------
    // Memory request: address from rs1 (warp-broadcast). NUM_LANES mask
    // all-ones — we fetch the metadata tile in one warp-level request.
    // Lane T loads the 32-bit word at base + 4*T; the host packs h_meta so
    // that word holds the metadata destined for SRAM cell
    // (bank = T%PWD, col = T/PWD). One formula serves WMMA-SP and WGMMA-SP.
    // -----------------------------------------------------------------------
    localparam PWD = TCU_META_PER_WARP_DEPTH;
    `UNUSED_PARAM (PWD)

    // LSU bus carries LSU_WORD_SIZE-byte chunks per lane (= XLEN/8 bytes).
    // The metadata layout is contiguous 32-bit words, so when XLEN > 32
    // each LSU response carries XLEN/32 logical 32-bit words and we need
    // to pick the right one. LSU_W32_RATIO = bytes/4, so XLEN=64 → 2.
    localparam LSU_W32_RATIO = LSU_WORD_SIZE / 4;
    localparam LG_W32_RATIO  = (LSU_W32_RATIO > 1) ? $clog2(LSU_W32_RATIO) : 0;

    // Per-lane sub-word selector: lane i reads 32-bit word i, so its half of
    // the LSU word is fixed by i.
    logic [NUM_LANES-1:0][`UP(LG_W32_RATIO)-1:0] half_sel_w;
    always_comb begin
        half_sel_w = '0;
        for (int i = 0; i < NUM_LANES; i++) begin
            if (LG_W32_RATIO > 0) begin
                half_sel_w[i] = `UP(LG_W32_RATIO)'(i);
            end
        end
    end
    if (LG_W32_RATIO == 0) begin : g_half_sel_unused
        `UNUSED_VAR (half_sel_w)
    end

`ifdef VX_CFG_LMEM_ENABLE
    // Equality on the word-address bits above the window: the base is asserted
    // aligned to the capacity, so the prefix compare is exact and needs no end
    // address -- which is not representable when the window ends at the top of
    // the address space.
    localparam LMEM_WORD_LSB = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);
    localparam [LSU_ADDR_WIDTH-1:0] LMEM_WORD_BASE =
        LSU_ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR) >> `CLOG2(LSU_WORD_SIZE));
    `STATIC_ASSERT (LSU_ADDR_WIDTH > LMEM_WORD_LSB, ("LMEM window wider than the word address space"))
`endif

    function automatic lsu_req_data_t make_req(input logic [LSU_ADDR_WIDTH-1:0] base_word_addr,
                                               input logic [BLOCK_IDX_BITS-1:0] block);
        lsu_req_data_t r;
        logic base_is_local;
    `ifdef VX_CFG_LMEM_ENABLE
        base_is_local = (base_word_addr[LSU_ADDR_WIDTH-1:LMEM_WORD_LSB]
                      == LMEM_WORD_BASE[LSU_ADDR_WIDTH-1:LMEM_WORD_LSB]);
    `else
        base_is_local = 1'b0;
    `endif
        r      = '0;
        r.rw   = 1'b0; // load
        r.mask = {NUM_LANES{1'b1}};
        for (int i = 0; i < NUM_LANES; i++) begin
            mem_bus_attr_t a;
            a = '0;
            a.is_addr_local = base_is_local;
            // base_addr is pre-advanced per TCU_LD slot by the kernel, so lane
            // i reads 32-bit word i relative to it, in LSU-word units.
            r.addr[i]   = base_word_addr + (LSU_ADDR_WIDTH'(i) >> LG_W32_RATIO);
            r.byteen[i] = {LSU_WORD_SIZE{1'b1}};
            r.data[i]   = '0;
            r.attr[i]   = a;
        end
        r.tag = '0;
        r.tag[BLOCK_IDX_BITS-1:0] = block;
        return r;
    endfunction

    `STATIC_ASSERT (LSU_CLIENT_TAG_WIDTH >= BLOCK_IDX_BITS, ("LSU client tag cannot encode TCU block"))

    // Per-block request and response events, from either port structure.
    wire [BLOCK_SIZE-1:0] req_fire_b;
    wire [BLOCK_SIZE-1:0] rsp_fire_b;
    wire [BLOCK_SIZE-1:0] rsp_eop_b;
    wire [NUM_LANES-1:0]  rsp_mask_b [BLOCK_SIZE];
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] rsp_data_b [BLOCK_SIZE];
    wire [`VX_CFG_XLEN-1:0] req_addr_chk [NUM_PORTS];
    wire [NUM_PORTS-1:0] req_fire_chk;

    if (NUM_PORTS == 1) begin : g_shared_port
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
            .grant_ready  (client_if[0].req_ready)
        );
        `UNUSED_VAR (req_grant)

        wire [BLOCK_IDX_BITS-1:0] rsp_block = client_if[0].rsp_data.tag[BLOCK_IDX_BITS-1:0];
        wire req_fire = req_valid && client_if[0].req_ready;
        wire rsp_fire = client_if[0].rsp_valid && client_if[0].rsp_ready;

        for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_ev
            assign req_fire_b[bi] = req_fire && (req_block == BLOCK_IDX_BITS'(bi));
            assign rsp_fire_b[bi] = rsp_fire && (rsp_block == BLOCK_IDX_BITS'(bi));
            assign rsp_eop_b[bi]  = client_if[0].rsp_data.eop;
            assign rsp_mask_b[bi] = client_if[0].rsp_data.mask;
            assign rsp_data_b[bi] = client_if[0].rsp_data.data;
        end

        assign client_if[0].req_valid = req_valid;
        assign client_if[0].req_data  = make_req(LSU_ADDR_WIDTH'(addr_r[req_block] >> `CLOG2(LSU_WORD_SIZE)), req_block);
        assign client_if[0].rsp_ready = busy_r[rsp_block] && issued_r[rsp_block]
                                     && !complete_r[rsp_block];
        assign req_addr_chk[0] = addr_r[req_block];
        assign req_fire_chk[0] = req_fire;
        `UNUSED_VAR (client_if[0].rsp_data.sop)
    end else begin : g_port_per_block
        for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_port
            assign client_if[bi].req_valid = req_pending[bi];
            assign client_if[bi].req_data  = make_req(LSU_ADDR_WIDTH'(addr_r[bi] >> `CLOG2(LSU_WORD_SIZE)), BLOCK_IDX_BITS'(0));
            assign client_if[bi].rsp_ready = busy_r[bi] && issued_r[bi] && !complete_r[bi];

            assign req_fire_b[bi] = req_pending[bi] && client_if[bi].req_ready;
            assign rsp_fire_b[bi] = client_if[bi].rsp_valid && client_if[bi].rsp_ready;
            assign rsp_eop_b[bi]  = client_if[bi].rsp_data.eop;
            assign rsp_mask_b[bi] = client_if[bi].rsp_data.mask;
            assign rsp_data_b[bi] = client_if[bi].rsp_data.data;
            assign req_addr_chk[bi] = addr_r[bi];
            assign req_fire_chk[bi] = req_fire_b[bi];
            `UNUSED_VAR (client_if[bi].rsp_data.sop)
            `UNUSED_VAR (client_if[bi].rsp_data.tag)
        end
    end

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
                if (req_fire_b[bi]) begin
                    issued_r[bi] <= 1'b1;
                end
                if (rsp_fire_b[bi]) begin
                    for (int t = 0; t < NUM_LANES; ++t) begin
                        if (rsp_mask_b[bi][t]) begin
                            rsp_data_r[bi][t]      <= rsp_data_b[bi][t];
                            rsp_lane_done_r[bi][t] <= 1'b1;
                        end
                    end
                    if (rsp_eop_b[bi]
                        && ((rsp_lane_done_r[bi] | rsp_mask_b[bi]) == {NUM_LANES{1'b1}})) begin
                        complete_r[bi] <= 1'b1;
                    end
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

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_ld_ready
        assign per_block_ld_ready[bi] = ~busy_r[bi];
    end

`ifdef VX_CFG_LSU_STACK_INTERLEAVE_ENABLE
    // Metadata is read as a linear warp-wide tile, which an interleaved
    // per-thread stack cannot provide.
    if (`VX_CFG_NUM_THREADS > 1) begin : g_meta_stack_check
        for (genvar pi = 0; pi < NUM_PORTS; ++pi) begin : g_port
            wire [`VX_CFG_XLEN-1:0] meta_end_addr = req_addr_chk[pi] + `VX_CFG_XLEN'(NUM_LANES * 4 - 1);
            `RUNTIME_ASSERT(~req_fire_chk[pi] || (meta_end_addr < STACK_WINDOW_BOTTOM) || (req_addr_chk[pi] >= STACK_WINDOW_TOP),
                ("%t: *** %s TCU metadata inside the thread stack window: addr=0x%0h", $time, INSTANCE_ID, req_addr_chk[pi]))
        end
    end
`else
    `UNUSED_VAR (req_addr_chk)
    `UNUSED_VAR (req_fire_chk)
`endif

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
