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

`ifdef VX_CFG_TCU_TMEM_ENABLE

// TMEM (Tensor Memory) storage and management
//
// TMEM_ALLOC/TMEM_DEALLOC are per-CTA allocation, delegated to
// VX_tcu_tmem_alloc: one block's ALLOC-or-DEALLOC request is arbitrated to
// a single winner per cycle. alloc() only ever reserves a column range —
// it does not initialize TMEM contents, so the allocator has no dependency
// on bank storage.
//
// Storage is BLOCK_SIZE banks, one per cta_rank. Bank r holds the
// TCU_WG_TILE_M-row lane range belonging to cta_rank r. Each bank word
// spans the bank's entire row range (TCU_WG_TILE_M rows x TCU_TC_N cols),
// addressed by column-group (col / TCU_TC_N):
//   - compute reads/writes access a TCU_TC_M-row sub-range of one word,
//     selected by step_m
//   - TMEM_LD/TMEM_ST are NUM_THREADS-wide at a single column
// Every bank is a single-read/single-write port. Losing requesters
// stall and retry next cycle.

module VX_tcu_tmem import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter         BLOCK_SIZE  = `VX_CFG_NUM_TCU_BLOCKS
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_umma_instrs,
    output wire [PERF_CTR_BITS-1:0] perf_tmem_reads,
    output wire [PERF_CTR_BITS-1:0] perf_tmem_writes,
    output wire [PERF_CTR_BITS-1:0] perf_tmem_bank_stalls,
`endif

    // TMEM_ALLOC/DEALLOC/ST/LD management ops (one execute/result_if per
    // block)
    input  wire [BLOCK_SIZE-1:0] mgmt_valid,
    input  tcu_execute_t         mgmt_data  [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0] mgmt_ready,

    output wire [BLOCK_SIZE-1:0] result_valid,
    output tcu_result_t          result_data  [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0] result_ready,

    // UMMA compute read (operand C) — request is this op's tile origin
    // (lane_base/col_base, same value for every cell); response is the
    // whole TC_M x TC_N tile. rd_valid marks a pending request; rd_grant
    // reports whether it won its bank's read port this cycle. The
    // requester must not admit/consume the op unless granted.
    input  wire [BLOCK_SIZE-1:0]                    rd_valid,
    input  wire [TCU_TMEM_LANE_BITS-1:0]            rd_lane_base [BLOCK_SIZE],
    input  wire [TCU_TMEM_COL_BITS-1:0]             rd_col_base  [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0]                    rd_grant,
    output wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0]  rd_data      [BLOCK_SIZE],

    // UMMA compute writeback, driven off the FEDP result once it retires.
    // wr_valid marks pending write request; wr_grant reports whether it won
    // its bank's write port this cycle. The requester must hold retirement
    // until granted.
    input  wire                                    wr_valid     [BLOCK_SIZE],
    input  wire [TCU_TMEM_LANE_BITS-1:0]           wr_lane_base [BLOCK_SIZE],
    input  wire [TCU_TMEM_COL_BITS-1:0]            wr_col_base  [BLOCK_SIZE],
    input  wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] wr_data      [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0]                   wr_grant,

    // abuf A-stripe fill (UMMA TS). Reads one whole bank word: all
    // TCU_WG_TILE_M rows of the CTA's lane range at one column group.
    input  wire [BLOCK_SIZE-1:0]                        afill_valid,
    input  wire [TCU_TMEM_LANE_BITS-1:0]                afill_lane_base [BLOCK_SIZE],
    input  wire [TCU_TMEM_COL_BITS-1:0]                 afill_col_base  [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0]                        afill_grant,
    output wire [TCU_WG_TILE_M-1:0][TCU_TC_N-1:0][31:0] afill_data      [BLOCK_SIZE]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Bank storage: BLOCK_SIZE banks x BANK_WORDS column-groups x
    // TCU_WG_TILE_M rows x TCU_TC_N cols
    // -----------------------------------------------------------------------

    localparam BANK_WORDS  = TCU_TMEM_COLS / TCU_TC_N;
    localparam BANK_ADDR_W = `LOG2UP(BANK_WORDS);
    localparam BANK_IDX_W  = `LOG2UP(BLOCK_SIZE);
    localparam ROW_IDX_W   = `LOG2UP(TCU_WG_TILE_M);
    localparam COL_IDX_W   = `LOG2UP(TCU_TC_N);
    localparam COL_SEL_W   = $clog2(TCU_TC_N);
    // The read port serves three requesters (accumulator read, TMEM_LD, abuf
    // A-stripe fill) while the write port serves two.
    localparam ARB_RD_W    = 3 * BLOCK_SIZE;
    localparam ARB_WR_W    = 2 * BLOCK_SIZE;

    `STATIC_ASSERT ((TCU_TMEM_COLS & (TCU_TMEM_COLS - 1)) == 0, ("VX_CFG_TCU_TMEM_COLS must be a power of 2"))

    // Per-bank storage

    // OUT_REG=1, RDW_MODE="R" makes bank
    // reads synchronous: rdata reflects whichever raddr/read was presented
    // last cycle, one cycle after an arbitration win. Handled by delaying
    // the read grant by one cycle. Writes are unaffected by OUT_REG so the
    // write side is also unchanged. Could introduce a same-cycle read and
    // write to same bank at same address returning stale data. Doesn't
    // apply to the compute path due to existing RAW interlock
    localparam BANK_DATAW = TCU_WG_TILE_M * TCU_TC_N * 32;
    localparam BANK_WRENW = TCU_WG_TILE_M * TCU_TC_N;

    wire [BANK_WRENW-1:0] bank_wren  [BLOCK_SIZE];
    wire [BANK_DATAW-1:0] bank_wdata [BLOCK_SIZE];
    wire [BANK_DATAW-1:0] bank_rdata [BLOCK_SIZE];

    // -----------------------------------------------------------------------
    // Decode every requester into {bank, word, local-row-origin[, local-col]}
    // -----------------------------------------------------------------------

    wire [BANK_IDX_W-1:0]  rd_bank [BLOCK_SIZE];
    wire [BANK_ADDR_W-1:0] rd_word [BLOCK_SIZE];
    wire [ROW_IDX_W-1:0]   rd_row0 [BLOCK_SIZE]; // = step_m*TCU_TC_M (tile row origin)

    wire [BANK_IDX_W-1:0]  wrb_bank [BLOCK_SIZE];
    wire [BANK_ADDR_W-1:0] wrb_word [BLOCK_SIZE];
    wire [ROW_IDX_W-1:0]   wrb_row0 [BLOCK_SIZE];

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_compute_decode
        // TCU_WG_TILE_M and TCU_TC_N are always powers of 2 so
        // lane_base/col_base are bit-sliced. Word uses COL_SEL_W (the true
        // bit count), not COL_IDX_W
        if (BLOCK_SIZE == 1) begin : g_bank_trivial
            assign rd_bank[bi]  = '0;
            assign wrb_bank[bi] = '0;
        end else begin : g_bank_real
            assign rd_bank[bi]  = BANK_IDX_W'(rd_lane_base[bi][TCU_TMEM_LANE_BITS-1:ROW_IDX_W]);
            assign wrb_bank[bi] = BANK_IDX_W'(wr_lane_base[bi][TCU_TMEM_LANE_BITS-1:ROW_IDX_W]);
        end
        assign rd_row0[bi] = ROW_IDX_W'(rd_lane_base[bi][ROW_IDX_W-1:0]);
        assign rd_word[bi] = BANK_ADDR_W'(rd_col_base[bi][TCU_TMEM_COL_BITS-1:COL_SEL_W]);

        assign wrb_row0[bi] = ROW_IDX_W'(wr_lane_base[bi][ROW_IDX_W-1:0]);
        assign wrb_word[bi] = BANK_ADDR_W'(wr_col_base[bi][TCU_TMEM_COL_BITS-1:COL_SEL_W]);
    end

    wire [BANK_IDX_W-1:0]  afill_bank [BLOCK_SIZE];
    wire [BANK_ADDR_W-1:0] afill_word [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]  afill_rd_won;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_afill_decode
        if (BLOCK_SIZE == 1) begin : g_bank_trivial
            assign afill_bank[bi] = '0;
        end else begin : g_bank_real
            assign afill_bank[bi] = BANK_IDX_W'(afill_lane_base[bi][TCU_TMEM_LANE_BITS-1:ROW_IDX_W]);
        end
        assign afill_word[bi] = BANK_ADDR_W'(afill_col_base[bi][TCU_TMEM_COL_BITS-1:COL_SEL_W]);
        // The fill reads whole words, so the lane base must be a bank origin.
        `RUNTIME_ASSERT (~afill_valid[bi] || (afill_lane_base[bi][ROW_IDX_W-1:0] == '0),
            ("%s: A-fill lane base %0d is not a bank origin", INSTANCE_ID, afill_lane_base[bi]))
        `UNUSED_VAR (afill_lane_base[bi][ROW_IDX_W-1:0])
    end

    wire [BLOCK_SIZE-1:0] is_tmem_ld;
    wire [BLOCK_SIZE-1:0] is_tmem_st;
    wire [BANK_IDX_W-1:0]  ldst_bank [BLOCK_SIZE];
    wire [BANK_ADDR_W-1:0] ldst_word [BLOCK_SIZE];
    wire [ROW_IDX_W-1:0]   ldst_row0 [BLOCK_SIZE]; // base lane's local row (thread 0)
    wire [COL_IDX_W-1:0]   ldst_col  [BLOCK_SIZE];

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_ldst_decode
        assign is_tmem_ld[bi] = mgmt_valid[bi] && (mgmt_data[bi].op_type == INST_TCU_TMEM_LD);
        assign is_tmem_st[bi] = mgmt_valid[bi] && (mgmt_data[bi].op_type == INST_TCU_TMEM_ST);

        wire [TCU_TMEM_LANE_BITS-1:0] ldst_lane     = TCU_TMEM_LANE_BITS'(mgmt_data[bi].rs1_data[0][31:16]);
        wire [TCU_TMEM_COL_BITS-1:0]  ldst_addr_col = TCU_TMEM_COL_BITS'(mgmt_data[bi].rs1_data[0][15:0]);

        if (BLOCK_SIZE == 1) begin : g_ldst_bank_trivial
            assign ldst_bank[bi] = '0;
        end else begin : g_ldst_bank_real
            assign ldst_bank[bi] = BANK_IDX_W'(ldst_lane[TCU_TMEM_LANE_BITS-1:ROW_IDX_W]);
        end
        assign ldst_row0[bi] = ROW_IDX_W'(ldst_lane[ROW_IDX_W-1:0]);
        assign ldst_word[bi] = BANK_ADDR_W'(ldst_addr_col[TCU_TMEM_COL_BITS-1:COL_SEL_W]);
        if (COL_SEL_W == 0) begin : g_col_trivial
            assign ldst_col[bi] = '0;
        end else begin : g_col_real
            assign ldst_col[bi] = COL_IDX_W'(ldst_addr_col[COL_SEL_W-1:0]);
        end

        // Bounds check: lane must be physically valid
        logic ldst_col_live;
        always_comb begin
            ldst_col_live = 1'b0;
            for (int e = 0; e < ALLOC_NUM_ENTRIES; ++e) begin
                if (alloc_live_valid[e]
                    && ({1'b0, ldst_addr_col} >= {1'b0, alloc_live_handle[e]})
                    && ({1'b0, ldst_addr_col} < ({1'b0, alloc_live_handle[e]} + {1'b0, alloc_live_ncols[e]}))) begin
                    ldst_col_live = 1'b1;
                end
            end
        end
        `RUNTIME_ASSERT (~(is_tmem_ld[bi] || is_tmem_st[bi]) || (32'(ldst_lane) < 32'(TCU_TMEM_LANES)),
            ("%s: TMEM_LD/ST lane %0d exceeds TCU_TMEM_LANES (%0d)", INSTANCE_ID, ldst_lane, TCU_TMEM_LANES))
        `RUNTIME_ASSERT (~(is_tmem_ld[bi] || is_tmem_st[bi]) || ldst_col_live,
            ("%s: TMEM_LD/ST column %0d not within any active allocation", INSTANCE_ID, ldst_addr_col))
    end

    // -----------------------------------------------------------------------
    // Per-bank READ arbitration: compute-read, TMEM_LD and the abuf A-stripe
    // fill share one unified round-robin pool per bank
    // -----------------------------------------------------------------------

    logic [31:0]            bank_rd_word [BLOCK_SIZE][TCU_WG_TILE_M][TCU_TC_N];
    wire [BANK_ADDR_W-1:0]  bank_raddr [BLOCK_SIZE];
    wire [ARB_RD_W-1:0]     bank_rd_grant_onehot [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]   bank_rd_conflict;

    for (genvar r = 0; r < BLOCK_SIZE; ++r) begin : g_rd_arb
        wire [BLOCK_SIZE-1:0] cmp_req;
        wire [BLOCK_SIZE-1:0] ldst_req;
        wire [BLOCK_SIZE-1:0] afill_req;
        for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_req
            // ~ldst_rd_won/~afill_rd_won: don't re-arbitrate a request that's
            // already being granted this cycle
            assign cmp_req[bi]   = rd_valid[bi] && (rd_bank[bi] == BANK_IDX_W'(r));
            assign ldst_req[bi]  = is_tmem_ld[bi] && (ldst_bank[bi] == BANK_IDX_W'(r)) && ~ldst_rd_won[bi];
            assign afill_req[bi] = afill_valid[bi] && (afill_bank[bi] == BANK_IDX_W'(r)) && ~afill_rd_won[bi];
        end
        wire [ARB_RD_W-1:0] req_vec = {afill_req, ldst_req, cmp_req};

        // grant_ready is tied high: a win is consumed the cycle it is issued
        // (the read grant is registered downstream), and each requester drops
        // its request on winning via its own sticky latch, so the arbiter is
        // never asked to re-grant an outstanding win.
        VX_generic_arbiter #(
            .NUM_REQS (ARB_RD_W),
            .TYPE     ("R")
        ) rd_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (req_vec),
            `UNUSED_PIN (grant_index),
            .grant_onehot (bank_rd_grant_onehot[r]),
            `UNUSED_PIN (grant_valid),
            .grant_ready  (1'b1)
        );

        assign bank_rd_conflict[r] = $countones(req_vec) > 1;

        wire [ARB_RD_W-1:0][BANK_ADDR_W-1:0] arb_rd_word;
        for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_arb_rd_word
            assign arb_rd_word[bi]                  = rd_word[bi];
            assign arb_rd_word[BLOCK_SIZE + bi]     = ldst_word[bi];
            assign arb_rd_word[2 * BLOCK_SIZE + bi] = afill_word[bi];
        end
        VX_onehot_mux #(
            .DATAW (BANK_ADDR_W),
            .N     (ARB_RD_W)
        ) raddr_mux (
            .data_in  (arb_rd_word),
            .sel_in   (bank_rd_grant_onehot[r]),
            .data_out (bank_raddr[r])
        );
    end

    // Bank's rdata this cycle reflects whichever raddr/read was presented
    // last cycle, one cycle after the arbitration win that chose it. Delay
    // the grant by one cycle to match.
    logic [ARB_RD_W-1:0] bank_rd_grant_onehot_d [BLOCK_SIZE];
    always_ff @(posedge clk) begin
        for (integer r = 0; r < BLOCK_SIZE; ++r) begin
            if (reset) bank_rd_grant_onehot_d[r] <= '0;
            else       bank_rd_grant_onehot_d[r] <= bank_rd_grant_onehot[r];
        end
    end

    // Un-flatten VX_dp_ram output back into the [row][col] shape the
    // routing code expects
    for (genvar r = 0; r < BLOCK_SIZE; ++r) begin : g_rd_unflatten
        for (genvar row = 0; row < TCU_WG_TILE_M; ++row) begin : g_row
            for (genvar col = 0; col < TCU_TC_N; ++col) begin : g_col
                assign bank_rd_word[r][row][col] = bank_rdata[r][(row*TCU_TC_N+col)*32 +: 32];
            end
        end
    end

    // Route each bank's winner back out to whichever bi won.
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_rd_route
        assign rd_grant[bi] = bank_rd_grant_onehot_d[rd_bank[bi]][bi];
        for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
            for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j
                assign rd_data[bi][i][j] = bank_rd_word[rd_bank[bi]][rd_row0[bi] + ROW_IDX_W'(i)][j];
            end
        end
    end

    wire [BLOCK_SIZE-1:0] ldst_rd_won;
    logic [31:0] tmem_ld_rd_data [BLOCK_SIZE][`VX_CFG_NUM_THREADS];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_ld_route
        assign ldst_rd_won[bi] = bank_rd_grant_onehot_d[ldst_bank[bi]][BLOCK_SIZE + bi];
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_t
            assign tmem_ld_rd_data[bi][t] = is_tmem_ld[bi]
                ? bank_rd_word[ldst_bank[bi]][ldst_row0[bi] + ROW_IDX_W'(t)][ldst_col[bi]]
                : '0;
        end
    end

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_afill_route
        assign afill_rd_won[bi] = bank_rd_grant_onehot_d[afill_bank[bi]][2 * BLOCK_SIZE + bi];
        assign afill_grant[bi]  = afill_valid[bi] && afill_rd_won[bi];
        for (genvar row = 0; row < TCU_WG_TILE_M; ++row) begin : g_row
            for (genvar col = 0; col < TCU_TC_N; ++col) begin : g_col
                assign afill_data[bi][row][col] = bank_rd_word[afill_bank[bi]][row][col];
            end
        end
    end

    // -----------------------------------------------------------------------
    // Per-bank WRITE arbitration: compute-write vs TMEM_ST plus the actual
    // clocked storage update.
    // -----------------------------------------------------------------------

    wire [ARB_WR_W-1:0] bank_wr_grant_onehot [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] bank_wr_conflict;

    wire [BLOCK_SIZE-1:0] ldst_wr_won;

    // A granted TMEM_ST whose result cannot retire this cycle (result_ready
    // low) keeps mgmt_valid and is_tmem_st asserted, so the arbiter would
    // regrant. Latch the win and drop the request until the op retires.

    reg [BLOCK_SIZE-1:0] ldst_wr_won_r;
    always @(posedge clk) begin
        if (reset) begin
            ldst_wr_won_r <= '0;
        end else begin
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (mgmt_ready[b])       ldst_wr_won_r[b] <= 1'b0;
                else if (ldst_wr_won[b]) ldst_wr_won_r[b] <= 1'b1;
            end
        end
    end

    for (genvar r = 0; r < BLOCK_SIZE; ++r) begin : g_wr_arb
        wire [BLOCK_SIZE-1:0] cmp_req;
        wire [BLOCK_SIZE-1:0] ldst_req;
        for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_req
            assign cmp_req[bi]  = wr_valid[bi] && (wrb_bank[bi] == BANK_IDX_W'(r));
            assign ldst_req[bi] = is_tmem_st[bi] && (ldst_bank[bi] == BANK_IDX_W'(r))
                               && ~ldst_wr_won_r[bi];
        end
        wire [ARB_WR_W-1:0] req_vec = {ldst_req, cmp_req};

        VX_generic_arbiter #(
            .NUM_REQS (ARB_WR_W),
            .TYPE     ("R")
        ) wr_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (req_vec),
            `UNUSED_PIN (grant_index),
            .grant_onehot (bank_wr_grant_onehot[r]),
            `UNUSED_PIN (grant_valid),
            .grant_ready  (1'b1)
        );

        assign bank_wr_conflict[r] = $countones(req_vec) > 1;
    end

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_wr_route
        assign wr_grant[bi] = wr_valid[bi] && bank_wr_grant_onehot[wrb_bank[bi]][bi];
    end
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_st_route
        assign ldst_wr_won[bi] = is_tmem_st[bi] && bank_wr_grant_onehot[ldst_bank[bi]][BLOCK_SIZE + bi];
    end

    // Pre-mux the winning write per bank BEFORE touching bank_mem
    logic [BANK_ADDR_W-1:0] write_word    [BLOCK_SIZE];
    logic [ROW_IDX_W-1:0]   write_row0    [BLOCK_SIZE];
    logic [COL_IDX_W-1:0]   write_col     [BLOCK_SIZE]; // meaningful only when write_is_ldst
    logic                   write_is_ldst [BLOCK_SIZE];
    logic                   write_valid   [BLOCK_SIZE];
    logic [`VX_CFG_NUM_THREADS-1:0] write_tmask [BLOCK_SIZE]; // meaningful only when write_is_ldst
    logic [31:0] write_cdata [BLOCK_SIZE][TCU_TC_M][TCU_TC_N];
    logic [31:0] write_ldata [BLOCK_SIZE][`VX_CFG_NUM_THREADS];

    always_comb begin
        for (integer r = 0; r < BLOCK_SIZE; ++r) begin
            write_word[r]    = '0;
            write_row0[r]    = '0;
            write_col[r]     = '0;
            write_is_ldst[r] = 1'b0;
            write_valid[r]   = 1'b0;
            write_tmask[r]   = '0;
            for (integer i = 0; i < TCU_TC_M; ++i) begin
                for (integer j = 0; j < TCU_TC_N; ++j) begin
                    write_cdata[r][i][j] = '0;
                end
            end
            for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                write_ldata[r][t] = '0;
            end

            for (integer bi = 0; bi < BLOCK_SIZE; ++bi) begin
                // Compute writeback candidate
                if (bank_wr_grant_onehot[r][bi] && wr_valid[bi]) begin
                    write_valid[r] = 1'b1;
                    write_word[r] |= wrb_word[bi];
                    write_row0[r] |= wrb_row0[bi];
                    for (integer i = 0; i < TCU_TC_M; ++i) begin
                        for (integer j = 0; j < TCU_TC_N; ++j) begin
                            write_cdata[r][i][j] |= wr_data[bi][i][j];
                        end
                    end
                end
                // TMEM_ST candidate
                if (bank_wr_grant_onehot[r][BLOCK_SIZE + bi] && is_tmem_st[bi]) begin
                    write_valid[r]   = 1'b1;
                    write_is_ldst[r] = 1'b1;
                    write_word[r]  |= ldst_word[bi];
                    write_row0[r]  |= ldst_row0[bi];
                    write_col[r]   |= ldst_col[bi];
                    write_tmask[r] |= mgmt_data[bi].header.tmask;
                    for (integer t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                        write_ldata[r][t] |= mgmt_data[bi].rs2_data[t][31:0];
                    end
                end
            end
        end
    end

    // Per (row,col), check whether the resolved winner's row0 places it
    // in range, and if so read the corresponding cell out of
    // write_cdata/write_ldata at a dynamic local-row index.
    for (genvar r = 0; r < BLOCK_SIZE; ++r) begin : g_wr_flatten
        for (genvar row = 0; row < TCU_WG_TILE_M; ++row) begin : g_row
            for (genvar col = 0; col < TCU_TC_N; ++col) begin : g_col
                // The borrow of the widened difference is row < row0.
                wire [ROW_IDX_W:0] row_diff = (ROW_IDX_W+1)'(row) - (ROW_IDX_W+1)'(write_row0[r]);
                wire in_range    = write_valid[r] && !row_diff[ROW_IDX_W];
                wire [ROW_IDX_W-1:0] local_row = row_diff[ROW_IDX_W-1:0];
                wire compute_hit = in_range && !write_is_ldst[r]
                                 && (32'(local_row) < 32'(TCU_TC_M));
                wire ldst_hit    = in_range && write_is_ldst[r]
                                 && (COL_IDX_W'(col) == write_col[r])
                                 && (32'(local_row) < 32'(`VX_CFG_NUM_THREADS))
                                 && write_tmask[r][local_row];
                assign bank_wren[r][row*TCU_TC_N+col] = compute_hit || ldst_hit;
                assign bank_wdata[r][(row*TCU_TC_N+col)*32 +: 32] = compute_hit
                    ? write_cdata[r][`LOG2UP(TCU_TC_M)'(local_row)][col]
                    : write_ldata[r][`LOG2UP(`VX_CFG_NUM_THREADS)'(local_row)];
            end
        end
    end

    for (genvar r = 0; r < BLOCK_SIZE; ++r) begin : g_bank_ram
        VX_dp_ram #(
            .DATAW    (BANK_DATAW),
            .SIZE     (BANK_WORDS),
            .WRENW    (BANK_WRENW),
            .OUT_REG  (1),
            .RDW_MODE ("R")
        ) bank_ram (
            .clk   (clk),
            .reset (reset),
            .read  (bank_rd_grant_onehot[r] != '0),
            .write (write_valid[r]),
            .wren  (bank_wren[r]),
            .waddr (write_word[r]),
            .wdata (bank_wdata[r]),
            .raddr (bank_raddr[r]),
            .rdata (bank_rdata[r])
        );
    end

    // -----------------------------------------------------------------------
    // ALLOC/DEALLOC arbitration: one block's request is granted per cycle
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] is_alloc_req;
    wire [BLOCK_SIZE-1:0] is_dealloc_req;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_mgmt_decode
        assign is_alloc_req[bi]   = mgmt_valid[bi] && (mgmt_data[bi].op_type == INST_TCU_TMEM_ALLOC);
        assign is_dealloc_req[bi] = mgmt_valid[bi] && (mgmt_data[bi].op_type == INST_TCU_TMEM_DEALLOC);
    end
    wire [BLOCK_SIZE-1:0] is_alloc_or_dealloc_req = is_alloc_req | is_dealloc_req;

    // Exclude a block from winning arbitration this cycle unless its own
    // retirement (result_ready) can also fire. Masked into the arbiter's
    // input so an unready winner is excluded and the arbiter picks another
    // ready requesting block.
    wire [BLOCK_SIZE-1:0] is_dealloc_req_ready = is_dealloc_req & result_ready;
    wire [BLOCK_SIZE-1:0] is_alloc_req_ready   = is_alloc_req & result_ready;

    // DEALLOC is always arbitrated ahead of ALLOC. An ALLOC that can't be
    // satisfied yet stalls and keeps re-requesting every cycle.
    wire [BLOCK_SIZE-1:0]          dealloc_grant_onehot;
    wire [`LOG2UP(BLOCK_SIZE)-1:0] dealloc_grant_idx;
    wire                           dealloc_grant_valid;
    VX_priority_encoder #(
        .N (BLOCK_SIZE)
    ) dealloc_arb (
        .data_in    (is_dealloc_req_ready),
        .onehot_out (dealloc_grant_onehot),
        .index_out  (dealloc_grant_idx),
        .valid_out  (dealloc_grant_valid)
    );

    wire [BLOCK_SIZE-1:0]          alloc_only_grant_onehot;
    wire [`LOG2UP(BLOCK_SIZE)-1:0] alloc_only_grant_idx;
    wire                           alloc_only_grant_valid;
    VX_priority_encoder #(
        .N (BLOCK_SIZE)
    ) alloc_only_arb (
        .data_in    (is_alloc_req_ready),
        .onehot_out (alloc_only_grant_onehot),
        .index_out  (alloc_only_grant_idx),
        .valid_out  (alloc_only_grant_valid)
    );

    wire                           alloc_grant_valid  = dealloc_grant_valid || alloc_only_grant_valid;
    wire [BLOCK_SIZE-1:0]          alloc_grant_onehot = dealloc_grant_valid ? dealloc_grant_onehot : alloc_only_grant_onehot;
    wire [`LOG2UP(BLOCK_SIZE)-1:0] alloc_grant_idx    = dealloc_grant_valid ? dealloc_grant_idx : alloc_only_grant_idx;

    // Granted request's fields, extracted once from the winning block.
    // req_ncols is meaningful only for ALLOC, req_handle only for DEALLOC.
    wire                          alloc_req_valid      = alloc_grant_valid;
    wire                          alloc_req_is_dealloc = is_dealloc_req[alloc_grant_idx];
    wire [NCTA_WIDTH-1:0]         alloc_req_cta_id     = mgmt_data[alloc_grant_idx].header.cta_id;
    wire [7:0]                    alloc_req_ncols      = mgmt_data[alloc_grant_idx].rs1_data[0][7:0];
    wire [TCU_TMEM_COL_BITS-1:0]  alloc_req_handle     = TCU_TMEM_COL_BITS'(mgmt_data[alloc_grant_idx].rs1_data[0]);

    wire                          alloc_resp_valid;
    wire [TCU_TMEM_COL_BITS-1:0]  alloc_resp_handle;

    localparam ALLOC_NUM_ENTRIES = (`VX_CFG_NUM_WARPS / `VX_CFG_NUM_TCU_BLOCKS) + 1;
    wire                         alloc_live_valid  [ALLOC_NUM_ENTRIES];
    wire [TCU_TMEM_COL_BITS-1:0] alloc_live_handle [ALLOC_NUM_ENTRIES];
    wire [7:0]                   alloc_live_ncols  [ALLOC_NUM_ENTRIES];

    VX_tcu_tmem_alloc #(
        .INSTANCE_ID (`SFORMATF(("%s-alloc", INSTANCE_ID)))
    ) alloc (
        .clk             (clk),
        .reset           (reset),
        .req_valid       (alloc_req_valid),
        .req_is_dealloc  (alloc_req_is_dealloc),
        .req_cta_id      (alloc_req_cta_id),
        .req_ncols       (alloc_req_ncols),
        .req_handle      (alloc_req_handle),
        .resp_valid      (alloc_resp_valid),
        .resp_handle     (alloc_resp_handle),
        .live_valid_out  (alloc_live_valid),
        .live_handle_out (alloc_live_handle),
        .live_ncols_out  (alloc_live_ncols)
    );

    // Bypass execute<->result handshake: same-cycle response, no queueing.
    // ALLOC/DEALLOC only fires for the block the arbiter granted this cycle,
    // gated on VX_tcu_tmem_alloc's resp_valid; TMEM_LD/ST only fire for the
    // block that won its bank's read/write port this cycle. Every other
    // requesting block retries next cycle.
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_result
        wire mgmt_fire = is_alloc_or_dealloc_req[bi] ? (alloc_grant_onehot[bi] && alloc_resp_valid)
                        : is_tmem_ld[bi]             ? ldst_rd_won[bi]
                        : is_tmem_st[bi]             ? (ldst_wr_won[bi] || ldst_wr_won_r[bi])
                        : mgmt_valid[bi];

        assign mgmt_ready[bi]         = mgmt_fire && result_ready[bi];
        assign result_valid[bi]       = mgmt_fire;
        assign result_data[bi].header = mgmt_data[bi].header;
        wire is_tmem_ld_r    = mgmt_data[bi].op_type == INST_TCU_TMEM_LD;
        wire is_tmem_alloc_r = mgmt_data[bi].op_type == INST_TCU_TMEM_ALLOC;
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_tmem_result_t
            if (`VX_CFG_XLEN > 32) begin : g_nanbox
                assign result_data[bi].data[t] = is_tmem_ld_r
                    ? {32'hffffffff, tmem_ld_rd_data[bi][t]}
                    : is_tmem_alloc_r ? `VX_CFG_XLEN'(alloc_resp_handle) : '0;
            end else begin : g_pass
                assign result_data[bi].data[t] = is_tmem_ld_r
                    ? `VX_CFG_XLEN'(tmem_ld_rd_data[bi][t])
                    : is_tmem_alloc_r ? `VX_CFG_XLEN'(alloc_resp_handle) : '0;
            end
        end
    end

`ifdef PERF_ENABLE
    // Elements read/written this cycle: a granted UMMA compute access moves
    // a whole TCU_TC_M x TCU_TC_N tile; a granted TMEM_LD/ST moves one
    // element per active thread.
    logic [PERF_CTR_BITS-1:0] tmem_reads_this_cycle;
    logic [PERF_CTR_BITS-1:0] tmem_writes_this_cycle;
    always_comb begin
        tmem_reads_this_cycle  = '0;
        tmem_writes_this_cycle = '0;
        for (integer bi = 0; bi < BLOCK_SIZE; ++bi) begin
            if (rd_grant[bi])    tmem_reads_this_cycle  += PERF_CTR_BITS'(TCU_TC_M * TCU_TC_N);
            if (ldst_rd_won[bi]) tmem_reads_this_cycle  += PERF_CTR_BITS'($countones(mgmt_data[bi].header.tmask));
            if (wr_grant[bi])    tmem_writes_this_cycle += PERF_CTR_BITS'(TCU_TC_M * TCU_TC_N);
            if (ldst_wr_won[bi]) tmem_writes_this_cycle += PERF_CTR_BITS'($countones(mgmt_data[bi].header.tmask));
        end
    end

    // Bank-conflict stall counting
    logic [PERF_CTR_BITS-1:0] tmem_bank_stalls_r;
    logic [PERF_CTR_BITS-1:0] umma_instrs_r;
    logic [PERF_CTR_BITS-1:0] tmem_reads_r;
    logic [PERF_CTR_BITS-1:0] tmem_writes_r;
    always @(posedge clk) begin
        if (reset) begin
            tmem_bank_stalls_r <= '0;
            umma_instrs_r      <= '0;
            tmem_reads_r       <= '0;
            tmem_writes_r      <= '0;
        end else begin
            tmem_bank_stalls_r <= tmem_bank_stalls_r
                + PERF_CTR_BITS'($countones(bank_rd_conflict))
                + PERF_CTR_BITS'($countones(bank_wr_conflict));
            // UMMA writeback is granted once per completed micro-op
            umma_instrs_r <= umma_instrs_r + PERF_CTR_BITS'($countones(wr_grant));
            tmem_reads_r  <= tmem_reads_r  + tmem_reads_this_cycle;
            tmem_writes_r <= tmem_writes_r + tmem_writes_this_cycle;
        end
    end
    assign perf_umma_instrs      = umma_instrs_r;
    assign perf_tmem_reads       = tmem_reads_r;
    assign perf_tmem_writes      = tmem_writes_r;
    assign perf_tmem_bank_stalls = tmem_bank_stalls_r;
`else
    `UNUSED_VAR ({bank_rd_conflict, bank_wr_conflict})
`endif

endmodule

`endif // VX_CFG_TCU_TMEM_ENABLE
