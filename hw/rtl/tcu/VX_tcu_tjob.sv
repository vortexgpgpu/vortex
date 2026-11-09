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

// Tensor memory and the tensor-job sequencer.
//
// TMEM is BLOCK_SIZE * XM lanes by TMEM_COLS 32-bit columns; block b owns
// lanes [b*XM, (b+1)*XM), M rows of a job. Each block's bank is NT 32-bit
// sub-banks; cell (lane m*TC_M+i, column c) lives in sub-bank
// (TC_N*i + c + TPL*m) mod NT at row (c / TC_N) * M_STEPS + m, which keeps a
// job's C/D fragment and both tmem.ld/st layouts to one row per sub-bank.
//
// A tc.mma op runs its K-tiles over the WGMMA tile buffers and the blocks'
// FEDP arrays without warp instructions: per K-tile it issues every (m, n) step
// on all blocks in lockstep with C read from and D written back to TMEM. A
// tc.cp op moves shared-memory rows into TMEM over the TCU's lowest-priority
// LMEM read port, one NT-word chunk at a time. A CTA's ops run in order, and a
// tc.commit arrives on its barrier once the CTA's earlier ops have completed.

module VX_tcu_tjob import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_busy,
    output wire [PERF_CTR_BITS-1:0] perf_opnd,
    output wire [PERF_CTR_BITS-1:0] perf_other,
    output wire [PERF_CTR_BITS-1:0] perf_idle,
    output wire [PERF_CTR_BITS-1:0] perf_xfers,
`endif

    // TMEM ops and tc ops from each block's dispatch
    input  wire [BLOCK_SIZE-1:0]    op_valid,
    input  tcu_execute_t            op_data [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0]    op_ready,

    output wire [BLOCK_SIZE-1:0]    rsp_valid,
    output tcu_result_t             rsp_data [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0]    rsp_ready,

    // Job MMA steps to the blocks' tcu_cores (presented on all blocks at once)
    output wire                     job_valid,
    output tcu_execute_t            job_data [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0]    core_ready,
    output wire                     job_fire,

    // Tile-buffer requests of the presented step
    output tcu_tbuf_req_t [BLOCK_SIZE-1:0] tbuf_req,

    // A K-tile's A from TMEM, written to each block's abuf storage
    output wire                     a_fill_valid,
    output wire [3:0]               a_fill_k,
    output wire [BLOCK_SIZE-1:0][`VX_CFG_NUM_THREADS-1:0][31:0] a_fill_data,

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Metadata of the presented step: its two TC_K halves
    output wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta,
    output wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta_hi,
`endif

    // Job results from the blocks' tcu_cores, in issue order per block
    input  wire [BLOCK_SIZE-1:0]    jres_valid,
    input  wire [BLOCK_SIZE-1:0][`VX_CFG_NUM_THREADS-1:0][31:0] jres_data,

    VX_tcu_bar_if.master            tcu_bar_if,
    VX_mem_bus_if.master            cp_lmem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NT       = `VX_CFG_NUM_THREADS;
    localparam TC_M     = TCU_TC_M;
    localparam TC_N     = TCU_TC_N;
    localparam M_STEPS  = TCU_WG_M_STEPS;
    localparam XM       = TCU_WG_TILE_M;
    localparam TPL      = NT / XM;
    localparam COLS     = `VX_CFG_TCU_TMEM_COLS;
    localparam LANES    = BLOCK_SIZE * XM;
    localparam DEPTH    = (COLS / TC_N) * M_STEPS;
    localparam ADDRW    = `LOG2UP(DEPTH);
    localparam SUBW     = `LOG2UP(NT);
    localparam LANEBW   = `LOG2UP(XM);
    localparam BLKW     = `LOG2UP(BLOCK_SIZE);
    localparam COLW     = 16;
    localparam NSLOTS   = 8;
    localparam SLOTW    = `LOG2UP(NSLOTS);
    localparam KT_BYTES = TCU_WG_FEDP_K * 4;
    localparam B_BLK_BYTES = TC_N * KT_BYTES;  // a block-major B (k, n) block
    localparam JOBS_PER_CTA = 2;
    localparam TC_COMMITS   = 2;  // tc.commit arrivals per op
    localparam WQ_DEPTH = 16;
    localparam INFLW    = 8;
    localparam A_READS  = TCU_WG_FEDP_K / TPL;
    localparam NBANKS   = `VX_CFG_LMEM_NUM_BANKS;
    localparam CP_ADDRW = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(NBANKS * 4);
    localparam LANEW    = `LOG2UP(LANES);

    `STATIC_ASSERT (NBANKS % NT == 0, ("a copy chunk must sit in one LMEM bank row"))

    `STATIC_ASSERT (TCU_WG_FEDP_K % TPL == 0, ("A from TMEM reads whole K words per lane"))

    `STATIC_ASSERT (TC_M * TC_N == NT, ("C fragment must span the warp"))
    `STATIC_ASSERT (NT % XM == 0, ("TMEM needs NUM_THREADS to be a multiple of WGMMA M"))
    `STATIC_ASSERT (`VX_CFG_XLEN == 32, ("TMEM supports XLEN=32"))

    // -----------------------------------------------------------------------
    // TMEM cell map
    // -----------------------------------------------------------------------

    function automatic logic [SUBW-1:0] cell_sub(input logic [LANEBW-1:0] lane_b, input logic [COLW-1:0] col);
        logic [31:0] m, i;
        m = 32'(lane_b) / TC_M;
        i = 32'(lane_b) % TC_M;
        return SUBW'(TC_N * i + 32'(col) + TPL * m);
    endfunction

    function automatic logic [ADDRW-1:0] cell_addr(input logic [LANEBW-1:0] lane_b, input logic [COLW-1:0] col);
        logic [31:0] m;
        m = 32'(lane_b) / TC_M;
        return ADDRW'((32'(col) / TC_N) * M_STEPS + m);
    endfunction

    // -----------------------------------------------------------------------
    // TMEM banks
    // -----------------------------------------------------------------------

    logic [BLOCK_SIZE-1:0][NT-1:0][ADDRW-1:0] bank_raddr;
    logic [BLOCK_SIZE-1:0][NT-1:0]            bank_write;
    logic [BLOCK_SIZE-1:0][NT-1:0][ADDRW-1:0] bank_waddr;
    logic [BLOCK_SIZE-1:0][NT-1:0][31:0]      bank_wdata;
    wire  [BLOCK_SIZE-1:0][NT-1:0][31:0]      bank_rdata;

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_bank
        for (genvar s = 0; s < NT; ++s) begin : g_sub
            VX_dp_ram #(
                .DATAW    (32),
                .SIZE     (DEPTH),
                .OUT_REG  (1),
                .RDW_MODE ("R")
            ) sub_bank (
                .clk   (clk),
                .reset (reset),
                .read  (1'b1),
                .write (bank_write[b][s]),
                .wren  (1'b1),
                .waddr (bank_waddr[b][s]),
                .wdata (bank_wdata[b][s]),
                .raddr (bank_raddr[b][s]),
                .rdata (bank_rdata[b][s])
            );
        end
    end

    // Per-thread cell selection of one access: sub-bank and row of thread t.
    typedef struct packed {
        logic [NT-1:0][SUBW-1:0]  sub;
        logic [NT-1:0][ADDRW-1:0] addr;
    } tm_access_t;

    // Cells of C/D fragment (m, col0) in one block.
    function automatic tm_access_t frag_access(input logic m, input logic [COLW-1:0] col0);
        tm_access_t acc;
        for (int t = 0; t < NT; ++t) begin
            logic [LANEBW-1:0] lane_b;
            logic [COLW-1:0] col;
            lane_b = LANEBW'(32'(m) * TC_M + t / TC_N);
            col    = col0 + COLW'(t % TC_N);
            acc.sub[t]  = cell_sub(lane_b, col);
            acc.addr[t] = cell_addr(lane_b, col);
        end
        return acc;
    endfunction

    // Cells of tmem.ld/st register r at lane_b0 (a block's first lane).
    function automatic tm_access_t reg_access(input logic row_shape, input logic [4:0] r,
                                              input logic [COLW-1:0] col0);
        tm_access_t acc;
        for (int t = 0; t < NT; ++t) begin
            logic [LANEBW-1:0] lane_b;
            logic [COLW-1:0] col;
            if (row_shape) begin
                lane_b = LANEBW'(32'(r) % XM);
                col    = col0 + COLW'((32'(r) / XM) * NT + t);
            end else begin
                lane_b = LANEBW'(t % XM);
                col    = col0 + COLW'(32'(r) * TPL + t / XM);
            end
            acc.sub[t]  = cell_sub(lane_b, col);
            acc.addr[t] = cell_addr(lane_b, col);
        end
        return acc;
    endfunction

    // Thread routed to each sub-bank (the map is a permutation).
    function automatic logic [NT-1:0][SUBW-1:0] sub_owner(input logic [NT-1:0][SUBW-1:0] sub);
        logic [NT-1:0][SUBW-1:0] owner;
        owner = '0;
        for (int s = 0; s < NT; ++s) begin
            for (int t = 0; t < NT; ++t) begin
                if (sub[t] == SUBW'(s)) owner[s] = SUBW'(t);
            end
        end
        return owner;
    endfunction

    // -----------------------------------------------------------------------
    // Job slots
    // -----------------------------------------------------------------------

    typedef enum logic [1:0] {
        TJ_FREE  = 2'd0,
        TJ_WAIT  = 2'd1,
        TJ_RUN   = 2'd2,
        TJ_DRAIN = 2'd3
    } tj_state_e;

    typedef struct packed {
        // descriptor
        logic [NCTA_WIDTH-1:0]  cta;
        logic [NW_WIDTH-1:0]    wid;
        logic [UUID_WIDTH-1:0]  uuid;
        logic [4:0]             fmt_s;
        logic [4:0]             fmt_d;
        logic [7:0]             n_frag;
        logic                   acc_init;
        logic                   copy;      // tc.cp: k_tiles columns from a_base, lda bytes per lane
        logic [15:0]            k_tiles;
        logic [COLW-1:0]        taddr;
        logic [15:0]            a_base;
        logic [15:0]            b_base;
        logic [15:0]            meta_base;
        logic                   sparse;
        logic                   a_tmem;    // a_base is A's first TMEM column
        logic [15:0]            lda;
        logic [15:0]            ldb;
        // TMEM columns the job touches: D [taddr, d_hi), A [a_base, a_hi), metadata [meta_base, m_hi)
        logic [COLW-1:0]        d_hi;
        logic [COLW-1:0]        a_hi;
        logic [COLW-1:0]        m_hi;
        logic [15:0]            kt;
        logic [COLW-1:0]        a_col;     // K-tile kt's TMEM columns, stepped per K-tile
        logic [COLW-1:0]        meta_col;
        logic [15:0]            meta_cols;
        logic [15:0]            sn_chunk;
        logic [15:0]            b_kt_off;  // K-tile kt's B offset, stepped per K-tile
        logic [15:0]            b_kt_stride;
        // On completion an op (or a bare tc.commit) arrives on its commits,
        // oldest first, one per cycle.
        logic [1:0]             ncommit;
        logic [TC_COMMITS-1:0][31:0]           commit_bar;
        logic [TC_COMMITS-1:0][BAR_SIZE_W-1:0] commit_cnt;  // arrival count - 1
    } tj_slot_t;

    tj_state_e [NSLOTS-1:0] state_r;
    tj_slot_t               slot_r [NSLOTS];
    logic [NSLOTS-1:0][INFLW-1:0] inflight_r;
    // older_r[i][j]: slot i was posted before slot j
    logic [NSLOTS-1:0][NSLOTS-1:0] older_r;

    `IGNORE_UNUSED_BEGIN
    function automatic logic [BAR_ADDR_W-1:0] bar_addr(input logic [31:0] raw);
        logic [NB_WIDTH-1:0] id;
        logic [BAR_ADDR_W-1:0] addr;
        id = NB_WIDTH'(raw[BAR_ID_SHIFT +: NB_WIDTH]);
        addr = BAR_ADDR_W'(id);
        if (NW_BITS != 0) begin
            addr = BAR_ADDR_W'({raw[`UP(NW_BITS)-1:0], NB_BITS'(id)});
        end
        return addr;
    endfunction
    `IGNORE_UNUSED_END

    // CTA heads: a CTA's jobs run in order.
    logic [NSLOTS-1:0] is_head;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            is_head[i] = (state_r[i] != TJ_FREE);
            for (int j = 0; j < NSLOTS; ++j) begin
                if (j != i && state_r[j] != TJ_FREE && slot_r[j].cta == slot_r[i].cta && older_r[j][i])
                    is_head[i] = 1'b0;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Warp ops: TMEM ops and tc ops (one at a time, response next cycle)
    // -----------------------------------------------------------------------

    wire [BLOCK_SIZE-1:0] op_sel_oh;
    wire [BLKW-1:0]       op_sel;
    wire                  op_sel_valid;

    // Counted rather than derived from the slots: op_ready depends on it.
    logic [(1 << NCTA_WIDTH)-1:0][`CLOG2(JOBS_PER_CTA+NSLOTS+1)-1:0] cta_jobs_r;

    wire alloc_valid;

    // An op that needs a slot its CTA cannot have yet lets other blocks' ops by.
    logic [BLOCK_SIZE-1:0] op_eligible;
    always_comb begin
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            automatic logic is_job = (op_data[b].op_type == INST_OP_BITS'(INST_TCU_TC_MMA))
                                  || (op_data[b].op_type == INST_OP_BITS'(INST_TCU_TC_CP));
            op_eligible[b] = op_valid[b]
                          && ~(is_job && ~(alloc_valid && (cta_jobs_r[op_data[b].header.cta_id] < JOBS_PER_CTA)));
        end
    end

    wire [BLKW-1:0] in_sel;
    wire            in_valid;
    VX_priority_encoder #(
        .N (BLOCK_SIZE)
    ) op_pe (
        .data_in    (op_eligible),
        .onehot_out (op_sel_oh),
        .index_out  (in_sel),
        .valid_out  (in_valid)
    );
    `UNUSED_VAR (op_sel_oh)

    // The selected op is registered before it is decoded and accepted: the
    // block select and the TMEM access each get a cycle.
    wire op_fire;
    wire q_load = in_valid && (~op_sel_valid || op_fire);
    logic q_valid_r;
    logic [BLKW-1:0] q_blk_r;
    tcu_execute_t q_op_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            q_valid_r <= 1'b0;
        end else if (q_load) begin
            q_valid_r <= 1'b1;
        end else if (op_fire) begin
            q_valid_r <= 1'b0;
        end
        if (q_load) begin
            q_blk_r <= in_sel;
            q_op_r  <= op_data[in_sel];
        end
    end
    assign op_sel_valid = q_valid_r;
    assign op_sel       = q_blk_r;

    tcu_execute_t op;
    assign op = q_op_r;
    wire op_is_ld   = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_LD));
    wire op_is_st   = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_ST));
    wire op_is_alloc   = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_ALLOC));
    wire op_is_dealloc = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_DEALLOC));
    wire op_is_mma     = (op.op_type == INST_OP_BITS'(INST_TCU_TC_MMA));
    wire op_is_commit  = (op.op_type == INST_OP_BITS'(INST_TCU_TC_COMMIT));
    wire op_is_cp      = (op.op_type == INST_OP_BITS'(INST_TCU_TC_CP));
    wire op_is_idesc   = (op.op_type == INST_OP_BITS'(INST_TCU_TC_IDESC));

    // tc.* operands are scalars, read from thread 0.
    wire [31:0] op_s1 = op.rs1_data[0][31:0];
    wire [31:0] op_s2 = op.rs2_data[0][31:0];
    wire [31:0] op_s3 = op.rs3_data[0][31:0];

    function automatic logic [15:0] meta_row_bits(input logic [4:0] fmt);
        return 16'(TCU_TC_K * 2 * (32 / tcu_fmt_width(fmt)));
    endfunction

    // tc.mma descriptor slots, four per CTA: {metadata, A and D column counts,
    // leading bytes, idesc}; the counts are fixed per slot, so tc.idesc
    // computes them off the tc.mma path.
    localparam DSLOTS = 4 * (1 << NCTA_WIDTH);
    logic [3*COLW+63:0] dslot_r [DSLOTS];
    wire [`LOG2UP(DSLOTS)-1:0] op_dslot = {op.header.cta_id, op.op_args.tcu.fmt_d[1:0]};
    wire [31:0]     op_idesc = dslot_r[op_dslot][31:0];
    wire [31:0]     op_ld    = dslot_r[op_dslot][63:32];
    wire [COLW-1:0] op_d_len = dslot_r[op_dslot][64 +: COLW];
    wire [COLW-1:0] op_a_len = dslot_r[op_dslot][64+COLW +: COLW];
    wire [COLW-1:0] op_m_len = dslot_r[op_dslot][64+2*COLW +: COLW];

    wire [`CLOG2(JOBS_PER_CTA+NSLOTS+1)-1:0] op_cta_jobs = cta_jobs_r[op.header.cta_id];

    logic [NSLOTS-1:0] free_slots;
    for (genvar i = 0; i < NSLOTS; ++i) begin : g_free
        assign free_slots[i] = (state_r[i] == TJ_FREE);
    end
    wire [SLOTW-1:0] alloc_slot;
    VX_priority_encoder #(
        .N (NSLOTS)
    ) alloc_pe (
        .data_in    (free_slots),
        `UNUSED_PIN (onehot_out),
        .index_out  (alloc_slot),
        .valid_out  (alloc_valid)
    );

    // TMEM address of a transfer uop: {lane [31:16], column [15:0]}
    wire [15:0]      op_lane   = op.rs1_data[0][31:16];
    wire [COLW-1:0]  op_col    = op.rs1_data[0][15:0];
    wire [BLKW-1:0]  op_blk    = BLKW'(op_lane / XM);
    wire [4:0]       op_reg    = op.op_args.tcu.fmt_d;
    tm_access_t      op_acc;
    assign op_acc = reg_access(op.op_args.tcu.a_from_smem, op_reg, op_col);
    wire [NT-1:0][SUBW-1:0] op_owner = sub_owner(op_acc.sub);

    // response slot
    logic                    rsp_valid_r;
    logic [BLKW-1:0]         rsp_blk_r;   // dispatching block
    logic [BLKW-1:0]         rsp_bank_r;  // TMEM bank read
    tcu_header_t             rsp_header_r;
    logic                    rsp_ld_fresh_r;
    logic [NT-1:0][SUBW-1:0] rsp_sub_r;
    logic [NT-1:0][31:0]     rsp_hold_r;
    wire rsp_fire = rsp_valid_r && rsp_ready[rsp_blk_r];
    wire rsp_free = ~rsp_valid_r || rsp_fire;

    wire job_rd_any;      // a job C read this cycle
    wire [BLOCK_SIZE-1:0] job_wr;

    // TMEM allocation: one bit per ACOLS-column chunk; a request of n columns
    // (a power of two) takes the lowest free run aligned to n.
    localparam ACOLS   = 32;
    localparam ACHUNKS = COLS / ACOLS;
    localparam ACW     = `UP(`CLOG2(ACHUNKS));
    logic [ACHUNKS-1:0] amap_r;
    wire [31:0] op_ncols = op_is_alloc ? op.rs1_data[0][31:0] : op.rs2_data[0][31:0];
    wire [31:0] op_achunks = op_ncols / ACOLS;
    wire [ACHUNKS-1:0] op_amask = ACHUNKS'((33'(1) << op_achunks) - 1);
    logic [ACHUNKS-1:0] afit;
    always_comb begin
        for (int i = 0; i < ACHUNKS; ++i) begin
            afit[i] = ((32'(i) & (op_achunks - 1)) == 0) && (32'(i) + op_achunks <= ACHUNKS)
                   && ((amap_r & (op_amask << i)) == '0);
        end
    end
    wire [ACW-1:0] afit_idx;
    wire           afit_valid;
    VX_priority_encoder #(
        .N (ACHUNKS)
    ) afit_pe (
        .data_in    (afit),
        `UNUSED_PIN (onehot_out),
        .index_out  (afit_idx),
        .valid_out  (afit_valid)
    );
    wire [31:0] alloc_col = afit_valid ? (32'(afit_idx) * ACOLS) : '1;

    `RUNTIME_ASSERT (~(op_sel_valid && (op_is_mma || op_is_cp || op_is_commit || op_is_idesc)) || op.header.tmask[0],
        ("%t: %s: tc op without thread 0", $time, INSTANCE_ID))

    `RUNTIME_ASSERT (~(op_sel_valid && (op_is_alloc || op_is_dealloc))
                  || (op_ncols >= ACOLS && op_ncols <= COLS && (op_ncols & (op_ncols - 1)) == 0),
        ("%t: %s: TMEM allocation of %0d columns", $time, INSTANCE_ID, op_ncols))

    // A transfer or a dealloc waits only for its CTA's jobs that touch its columns.
    wire             op_row     = op.op_args.tcu.a_from_smem;
    wire [COLW-1:0]  op_xfer_lo = op_col + (op_row ? COLW'((32'(op_reg) / XM) * NT) : COLW'(32'(op_reg) * TPL));
    wire [COLW-1:0]  op_xfer_hi = op_is_dealloc ? COLW'(32'(op_col) + op_ncols)
                                : (op_xfer_lo + (op_row ? COLW'(NT) : COLW'(TPL)));
    wire [COLW-1:0]  op_span_lo = op_is_dealloc ? op_col : op_xfer_lo;
    logic op_conflict;
    always_comb begin
        op_conflict = 1'b0;
        for (int i = 0; i < NSLOTS; ++i) begin
            if (state_r[i] != TJ_FREE && slot_r[i].cta == op.header.cta_id
             && ((op_span_lo < slot_r[i].d_hi && slot_r[i].taddr < op_xfer_hi)
              || (slot_r[i].a_tmem && op_span_lo < slot_r[i].a_hi && COLW'(slot_r[i].a_base) < op_xfer_hi)
              || (slot_r[i].sparse && op_span_lo < slot_r[i].m_hi && COLW'(slot_r[i].meta_base) < op_xfer_hi)))
                op_conflict = 1'b1;
        end
    end
    // tc.commit rides on the CTA's last posted op; once that is done, so is
    // all of the CTA's earlier tensor work.
    logic [(1 << NCTA_WIDTH)-1:0][SLOTW-1:0] young_slot_r;
    logic [(1 << NCTA_WIDTH)-1:0]            young_vld_r;
    wire [SLOTW-1:0] young_i = young_slot_r[op.header.cta_id];
    wire             young_v = young_vld_r[op.header.cta_id];
    wire young_done_fire;  // its completion step runs this cycle
    wire commit_ok = young_v ? ((slot_r[young_i].ncommit != 2'(TC_COMMITS)) && ~young_done_fire)
                             : alloc_valid;

    wire op_can = op_sel_valid && rsp_free
               && (op_is_alloc
                || ((op_is_mma || op_is_cp) && alloc_valid && (op_cta_jobs < JOBS_PER_CTA))
                || (op_is_commit && commit_ok)
                || op_is_idesc
                || (op_is_ld && ~op_conflict)
                || (op_is_st && ~op_conflict && ~job_wr[op_blk])
                || (op_is_dealloc && ~op_conflict));
    assign op_fire = op_can;

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_op_ready
        assign op_ready[b] = q_load && (in_sel == BLKW'(b));
    end

    // decoded apart from op_fire to keep other ops' acceptance off the step path
    wire ld_read  = op_sel_valid && rsp_free && op_is_ld && ~op_conflict;
    wire st_write = op_sel_valid && rsp_free && op_is_st && ~op_conflict && ~job_wr[op_blk];

    `RUNTIME_ASSERT (~(op_fire && (op_is_ld || op_is_st))
                  || ((32'(op_lane) % XM) == 0 && 32'(op_lane) < LANES),
        ("%t: %s: TMEM lane %0d out of range", $time, INSTANCE_ID, op_lane))

    // The posted op as descriptor words. tc.mma: idesc (acc_init in bit 11),
    // k_tiles, taddr, A, B, metadata column, reserved, leading bytes. tc.cp:
    // taddr, source, stride, columns.
    logic [7:0][31:0] post_w;
    always_comb begin
        post_w = '0;
        if (op_is_cp) begin
            post_w[0] = 32'(op_s1[15:0]);
            post_w[1] = op_s2;
            post_w[2] = op_s3;
            post_w[3] = 32'(op_s1[31:16]);
        end else begin
            post_w[0] = {11'b0, op_idesc[20:12], op_s1[31], op_idesc[10:0]};
            post_w[1] = 32'(op_idesc[31:24]);
            post_w[2] = 32'(op_s1[15:0]);
            post_w[3] = op_s2;
            post_w[4] = op_s3;
            post_w[5] = 32'(op_s1[30:16]);
            post_w[6] = 32'({op_idesc[23:21], op_idesc[11]});
            post_w[7] = op_ld;
        end
    end

    always_ff @(posedge clk) begin
        if (op_fire && op_is_idesc) begin
            dslot_r[op_dslot] <= {COLW'(32'(op_s1[31:24]) * ((2 * 32'(meta_row_bits(op_s1[4:0])) + 31) >> 5)),
                                  COLW'(32'(op_s1[31:24]) * TCU_WG_FEDP_K),
                                  COLW'(32'(op_s1[20:13]) * TC_N),
                                  op_s2, op_s1};
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            rsp_valid_r <= 1'b0;
        end else begin
            if (rsp_fire) rsp_valid_r <= 1'b0;
            if (op_fire)  rsp_valid_r <= 1'b1;
        end
        if (op_fire) begin
            rsp_blk_r      <= op_sel;
            rsp_bank_r     <= op_blk;
            rsp_header_r   <= op.header;
            rsp_ld_fresh_r <= op_is_ld;
            rsp_sub_r      <= op_acc.sub;
        end else begin
            rsp_ld_fresh_r <= 1'b0;
        end
    end

    // A load's cells arrive the cycle after its read; hold them until taken.
    logic [NT-1:0][31:0] ld_data;
    always_comb begin
        for (int t = 0; t < NT; ++t) begin
            ld_data[t] = bank_rdata[rsp_bank_r][rsp_sub_r[t]];
        end
    end
    wire [NT-1:0][31:0] rsp_payload = rsp_ld_fresh_r ? ld_data : rsp_hold_r;
    always_ff @(posedge clk) begin
        if (rsp_ld_fresh_r) rsp_hold_r <= ld_data;
        if (op_fire && op_is_alloc) rsp_hold_r <= {NT{alloc_col}};
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            amap_r <= '0;
        end else if (op_fire && op_is_alloc && afit_valid) begin
            amap_r <= amap_r | (op_amask << afit_idx);
        end else if (op_fire && op_is_dealloc) begin
            amap_r <= amap_r & ~(op_amask << (op_col / ACOLS));
        end
    end

    `RUNTIME_ASSERT (~(op_fire && op_is_dealloc)
                  || ((32'(op_col) % op_ncols) == 0
                   && (amap_r & (op_amask << (op_col / ACOLS))) == (op_amask << (op_col / ACOLS))),
        ("%t: %s: TMEM dealloc of %0d columns at column %0d that are not allocated", $time, INSTANCE_ID, op_ncols, op_col))

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_rsp
        assign rsp_valid[b]           = rsp_valid_r && (rsp_blk_r == BLKW'(b));
        assign rsp_data[b].header     = rsp_header_r;
        for (genvar t = 0; t < NT; ++t) begin : g_t
            assign rsp_data[b].data[t] = `VX_CFG_XLEN'(rsp_payload[t]);
        end
    end

    // -----------------------------------------------------------------------
    // Job issue: the step register E presents one (m, n) step to all blocks
    // -----------------------------------------------------------------------

    logic             run_valid_r;
    logic [SLOTW-1:0] run_slot_r;
    logic             run_m_r;
    logic [7:0]       run_n_r;
    logic [3:0]       run_sn_r;
    logic [15:0]      run_choff_r;

    logic             e_valid_r;
    logic [SLOTW-1:0] e_slot_r;
    logic             e_m_r;
    logic [3:0]       e_sn_r;
    logic             e_first_r;
    logic             e_pf_ok_r;
    logic             e_zero_r;
    logic             e_fresh_r;
    logic [BLOCK_SIZE-1:0][NT-1:0][31:0] e_hold_r;

    // Slot fields of the presented step, captured at load: job_fire depends on
    // the descriptors through tbuf_ready, too late for the slot mux and adders.
    logic [UUID_WIDTH-1:0] e_uuid_r;
    logic [NW_WIDTH-1:0]   e_wid_r;
    logic [NCTA_WIDTH-1:0] e_cta_r;
    logic [4:0]       e_fmt_s_r, e_fmt_d_r;
    logic [15:0]      e_lda_r, e_ldb_r;
    logic             e_a_tmem_r, e_sparse_r;
    logic [COLW-1:0]  e_col0_r;
    logic [NT-1:0][SUBW-1:0] e_sub_r;
    logic [BLOCK_SIZE-1:0][15:0] e_a_off_r;
    logic [15:0]      e_b_off_r;

    tj_slot_t rj;
    assign rj = slot_r[run_slot_r];

    assign job_fire = e_valid_r && (&core_ready);
    wire e_advance  = ~e_valid_r || job_fire;

    // K-tile prologue: metadata then A over the read port, all blocks in
    // parallel, once the previous K-tile's last step has left the E register.
    // A lane's K-tile metadata is its two rows, in row order, over `meta_cols`.
    logic [3:0] pro_idx_r;
    wire [3:0]  pro_meta_reads = rj.sparse ? 4'((rj.meta_cols + 16'(TPL) - 16'd1) / 16'(TPL)) : 4'd0;
    wire [3:0]  pro_a_reads    = rj.a_tmem ? 4'(A_READS) : 4'd0;
    wire [3:0]  pro_total      = pro_meta_reads + pro_a_reads;
    wire        pro_pending    = run_valid_r && (pro_idx_r < pro_total);
    wire        pro_go         = pro_pending && e_advance && ~ld_read;
    wire        pro_is_meta    = (pro_idx_r < pro_meta_reads);
    wire [3:0]  pro_a_idx      = pro_idx_r - pro_meta_reads;
    wire [COLW-1:0] pro_col0   = pro_is_meta
        ? (rj.meta_col + COLW'(32'(pro_idx_r) * TPL))
        : (rj.a_col + COLW'(32'(pro_a_idx) * TPL));
    tm_access_t pro_acc;
    assign pro_acc = reg_access(1'b0, 5'd0, pro_col0);

    wire load       = e_advance && run_valid_r && ~ld_read && ~pro_pending;

    wire load_zero      = (rj.kt == 0) && rj.acc_init;
    wire load_m_last    = (run_m_r == 1'(M_STEPS - 1));
    wire load_ktile_end = load_m_last && (run_n_r == rj.n_frag - 8'd1);
    wire load_job_end   = (rj.kt == rj.k_tiles - 16'd1);
    wire [COLW-1:0] load_col0 = rj.taddr + COLW'(32'(run_n_r) * TC_N);
    tm_access_t load_acc;
    assign load_acc = frag_access(run_m_r, load_col0);

    assign job_rd_any = load && ~load_zero;


    // C cells of the presented step, per thread
    logic [BLOCK_SIZE-1:0][NT-1:0][31:0] e_c;
    always_comb begin
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            for (int t = 0; t < NT; ++t) begin
                e_c[b][t] = e_zero_r ? 32'd0
                          : (e_fresh_r ? bank_rdata[b][e_sub_r[t]] : e_hold_r[b][t]);
            end
        end
    end
    always_ff @(posedge clk) begin
        if (e_fresh_r) e_hold_r <= e_c;
    end

    // descriptors of the step being loaded
    wire [15:0] load_a_off = rj.a_base + 16'(32'(rj.kt) * KT_BYTES);
    wire [15:0] load_b_off = rj.b_base + run_choff_r + rj.b_kt_off;

    assign job_valid = e_valid_r;

    // Prologue responses
    logic                    pro_rsp_valid_r;
    logic                    pro_rsp_meta_r;
    logic [3:0]              pro_rsp_idx_r;
    logic [NT-1:0][SUBW-1:0] pro_rsp_sub_r;
    logic [BLOCK_SIZE-1:0][NT-1:0][31:0] pro_cell;
    always_comb begin
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            for (int t = 0; t < NT; ++t) begin
                pro_cell[b][t] = bank_rdata[b][pro_rsp_sub_r[t]];
            end
        end
    end
    assign a_fill_valid = pro_rsp_valid_r && ~pro_rsp_meta_r;
    assign a_fill_k     = 4'(32'(pro_rsp_idx_r) * TPL);
    assign a_fill_data  = pro_cell;

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam META_COLS_MAX = (TCU_WG_FEDP_K * 2 * TCU_MAX_ELT_RATIO + 31) / 32;
    logic [BLOCK_SIZE-1:0][XM-1:0][META_COLS_MAX-1:0][31:0] meta_r;
    always_ff @(posedge clk) begin
        if (pro_rsp_valid_r && pro_rsp_meta_r) begin
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                for (int t = 0; t < NT; ++t) begin
                    meta_r[b][t % XM][32'(pro_rsp_idx_r) * TPL + t / XM] <= pro_cell[b][t];
                end
            end
        end
    end

    // Rows of M step e_m_r, packed as VX_tcu_sp_mux expects for the format.
    logic [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta_lo_c, job_meta_hi_c;
    always_comb begin
        job_meta_lo_c = '0;
        job_meta_hi_c = '0;
        for (int w = TCU_TC_K * 2; w <= TCU_TC_K * 2 * TCU_MAX_ELT_RATIO; w *= 2) begin
            if (meta_row_bits(e_fmt_s_r) == 16'(w)) begin
                for (int b = 0; b < BLOCK_SIZE; ++b) begin
                    for (int i = 0; i < TC_M; ++i) begin
                        for (int j = 0; j < w; ++j) begin
                            job_meta_lo_c[b][i * w + j] = meta_r[b][32'(e_m_r) * TC_M + i][j / 32][j % 32];
                            job_meta_hi_c[b][i * w + j] = meta_r[b][32'(e_m_r) * TC_M + i][(w + j) / 32][(w + j) % 32];
                        end
                    end
                end
            end
        end
    end
    assign job_meta    = job_meta_lo_c;
    assign job_meta_hi = job_meta_hi_c;
`endif

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_job
        always_comb begin
            job_data[b] = '0;
            job_data[b].header.uuid   = e_uuid_r;
            job_data[b].header.wid    = e_wid_r;
            job_data[b].header.cta_id = e_cta_r;
            job_data[b].header.tmask  = '1;
            job_data[b].header.sop    = 1'b1;
            job_data[b].header.eop    = 1'b1;
            job_data[b].header.wb     = 1'b1;
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            job_data[b].op_type       = e_sparse_r ? INST_OP_BITS'(INST_TCU_WGMMA_SP) : INST_OP_BITS'(INST_TCU_WGMMA);
        `else
            job_data[b].op_type       = INST_OP_BITS'(INST_TCU_WGMMA);
        `endif
            job_data[b].op_args.tcu.a_from_smem  = 1'b1;
            job_data[b].op_args.tcu.cd_nregs     = 2'd2;
            job_data[b].op_args.tcu.fmt_s        = e_fmt_s_r;
            job_data[b].op_args.tcu.fmt_d        = e_fmt_d_r;
            job_data[b].op_args.tcu.step_m       = 4'(e_m_r);
            job_data[b].op_args.tcu.step_n       = e_sn_r;
            job_data[b].op_args.tcu.is_first_uop = e_first_r;
            for (int t = 0; t < NT; ++t) begin
                job_data[b].rs3_data[t] = `VX_CFG_XLEN'(e_c[b][t]);
            end

            tbuf_req[b]              = '0;
            tbuf_req[b].valid        = e_valid_r;
            tbuf_req[b].uuid         = e_uuid_r;
            tbuf_req[b].wid          = e_wid_r;
            tbuf_req[b].cta_id       = e_cta_r;
            tbuf_req[b].step_m       = 4'(e_m_r);
            tbuf_req[b].step_n       = e_sn_r;
            tbuf_req[b].cd_nregs     = 2'd2;
            tbuf_req[b].desc_a       = `VX_CFG_XLEN'({e_lda_r, e_a_off_r[b]});
            tbuf_req[b].desc_b       = `VX_CFG_XLEN'({e_ldb_r, e_b_off_r});
            tbuf_req[b].a_is_smem    = ~e_a_tmem_r;
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            tbuf_req[b].is_sparse    = e_sparse_r;
        `endif
            tbuf_req[b].is_first_uop = e_first_r;
            tbuf_req[b].pf_ok        = e_pf_ok_r;
        end
    end

    // D write-back queue per block: fragment (m, col0) and its job slot
    localparam WQ_W = SLOTW + 1 + COLW;
    wire [BLOCK_SIZE-1:0][WQ_W-1:0] wq_head;
    wire [BLOCK_SIZE-1:0]           wq_full;
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_wq
        VX_fifo_queue #(
            .DATAW   (WQ_W),
            .DEPTH   (WQ_DEPTH),
            .LUTRAM  (1)
        ) wq (
            .clk     (clk),
            .reset   (reset),
            .push    (job_fire),
            .pop     (jres_valid[b]),
            .data_in ({e_slot_r, e_m_r, e_col0_r}),
            .data_out(wq_head[b]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            .full    (wq_full[b]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
        `RUNTIME_ASSERT (~(job_fire && wq_full[b]), ("%t: %s: job write queue overflow", $time, INSTANCE_ID))
    end
    assign job_wr = jres_valid;

    // -----------------------------------------------------------------------
    // Copy engine: the oldest tc.cp head, one LMEM read in flight. A response
    // is written in its cycle unless jres/tmem.st hold the bank, else parked.
    // -----------------------------------------------------------------------

    logic [NSLOTS-1:0] want_copy;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            want_copy[i] = (state_r[i] == TJ_WAIT) && is_head[i] && slot_r[i].copy;
        end
    end
    logic [NSLOTS-1:0] copy_oldest;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            copy_oldest[i] = want_copy[i];
            for (int j = 0; j < NSLOTS; ++j) begin
                if (j != i && want_copy[j] && older_r[j][i]) copy_oldest[i] = 1'b0;
            end
        end
    end
    wire [SLOTW-1:0] cpst_i;
    wire cpst_v;
    VX_priority_encoder #(.N (NSLOTS)) pe_copy (.data_in (copy_oldest), `UNUSED_PIN (onehot_out), .index_out (cpst_i), .valid_out (cpst_v));

    logic              cp_active_r;
    logic [SLOTW-1:0]  cp_slot_r;
    logic [LANEW-1:0]  cp_lane_r;     // next request
    logic [15:0]       cp_col_r;
    logic [15:0]       cp_row_r;      // lane * lda
    logic              cp_req_done_r;
    logic              cp_inflight_r;
    logic [LANEW-1:0]  cp_rlane_r;    // in-flight request
    logic [15:0]       cp_rcol_r;
    logic [`LOG2UP(NBANKS)-1:0] cp_rword_r;
    logic              cp_have_r;     // parked response
    logic [LANEW-1:0]  cp_hlane_r;
    logic [15:0]       cp_hcol_r;
    logic [NT-1:0][31:0] cp_hdata_r;

    tj_slot_t cj;
    assign cj = slot_r[cp_slot_r];
    wire cp_start = cpst_v && ~cp_active_r;

    wire [15:0] cp_byte = cj.a_base + cp_row_r + (cp_col_r << 2);
    wire cp_rsp = cp_lmem_if.rsp_valid;
    logic [NT-1:0][31:0] cp_rsp_data;
    always_comb begin
        for (int t = 0; t < NT; ++t) begin
            cp_rsp_data[t] = cp_lmem_if.rsp_data.data[(32'(cp_rword_r) + t) * 32 +: 32];
        end
    end

    // The write candidate: a parked response first, else the arriving one.
    wire [LANEW-1:0] cp_wlane = cp_have_r ? cp_hlane_r : cp_rlane_r;
    wire [15:0]      cp_wcol  = cp_have_r ? cp_hcol_r : cp_rcol_r;
    wire [NT-1:0][31:0] cp_wdata = cp_have_r ? cp_hdata_r : cp_rsp_data;
    wire [BLKW-1:0]  cp_wblk  = BLKW'(32'(cp_wlane) / XM);
    wire cp_wants_wr = cp_have_r || cp_rsp;
    wire cp_port_free = ~jres_valid[cp_wblk] && ~(op_sel_valid && op_is_st && (op_blk == cp_wblk));
    wire cp_wr = cp_wants_wr && cp_port_free;
    wire cp_park = cp_rsp && ~(cp_wr && ~cp_have_r);
    tm_access_t cp_acc;
    assign cp_acc = reg_access(1'b1, 5'(32'(cp_wlane) % XM), cj.taddr + cp_wcol);

    wire cp_issue = cp_active_r && ~cp_req_done_r && (~cp_inflight_r || cp_rsp)
                 && ~cp_have_r && ~cp_park;
    wire cp_fire = cp_issue && cp_lmem_if.req_ready;
    wire cp_last_chunk = (cp_col_r + 16'(NT) == cj.k_tiles);
    wire cp_finish = cp_active_r && cp_req_done_r && ~cp_inflight_r && ~cp_have_r;

    assign cp_lmem_if.req_valid        = cp_issue;
    assign cp_lmem_if.req_data.rw      = 1'b0;
    assign cp_lmem_if.req_data.addr    = CP_ADDRW'(cp_byte >> `CLOG2(NBANKS * 4));
    assign cp_lmem_if.req_data.data    = '0;
    assign cp_lmem_if.req_data.byteen  = '0;
    assign cp_lmem_if.req_data.attr    = '0;
    assign cp_lmem_if.req_data.tag.uuid  = cj.uuid;
    assign cp_lmem_if.req_data.tag.value = '0;
    assign cp_lmem_if.rsp_ready        = 1'b1;
    `UNUSED_VAR (cp_lmem_if.rsp_data.tag)

    always_ff @(posedge clk) begin
        if (reset) begin
            cp_active_r   <= 1'b0;
            cp_inflight_r <= 1'b0;
            cp_have_r     <= 1'b0;
        end else begin
            if (cp_start) begin
                cp_active_r   <= 1'b1;
                cp_slot_r     <= cpst_i;
                cp_lane_r     <= '0;
                cp_col_r      <= '0;
                cp_row_r      <= '0;
                cp_req_done_r <= 1'b0;
            end
            if (cp_fire) begin
                cp_rlane_r <= cp_lane_r;
                cp_rcol_r  <= cp_col_r;
                cp_rword_r <= `LOG2UP(NBANKS)'(cp_byte >> 2);
                if (cp_last_chunk) begin
                    cp_col_r  <= '0;
                    cp_lane_r <= cp_lane_r + LANEW'(1);
                    cp_row_r  <= cp_row_r + cj.lda;
                    if (32'(cp_lane_r) == LANES - 1) cp_req_done_r <= 1'b1;
                end else begin
                    cp_col_r <= cp_col_r + 16'(NT);
                end
            end
            if (cp_fire) begin
                cp_inflight_r <= 1'b1;
            end else if (cp_rsp) begin
                cp_inflight_r <= 1'b0;
            end
            if (cp_park) begin
                cp_have_r  <= 1'b1;
                cp_hlane_r <= cp_rlane_r;
                cp_hcol_r  <= cp_rcol_r;
                cp_hdata_r <= cp_rsp_data;
            end else if (cp_wr && cp_have_r) begin
                cp_have_r <= 1'b0;
            end
            if (cp_finish) begin
                cp_active_r <= 1'b0;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Bank port control
    // -----------------------------------------------------------------------

    tm_access_t wacc [BLOCK_SIZE];
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_wacc
        assign wacc[b] = frag_access(wq_head[b][COLW], wq_head[b][COLW-1:0]);
    end

    always_comb begin
        bank_raddr = '0;
        bank_write = '0;
        bank_waddr = '0;
        bank_wdata = '0;
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            // read: a tmem.ld has priority over the job's C read
            if (ld_read && (op_blk == BLKW'(b))) begin
                for (int s = 0; s < NT; ++s) bank_raddr[b][s] = op_acc.addr[op_owner[s]];
            end else if (pro_go) begin
                for (int t = 0; t < NT; ++t) bank_raddr[b][pro_acc.sub[t]] = pro_acc.addr[t];
            end else begin
                for (int s = 0; s < NT; ++s) begin
                    for (int t = 0; t < NT; ++t) begin
                        if (load_acc.sub[t] == SUBW'(s)) bank_raddr[b][s] = load_acc.addr[t];
                    end
                end
            end
            // write: job results have priority over tmem.st
            if (jres_valid[b]) begin
                for (int t = 0; t < NT; ++t) begin
                    bank_write[b][wacc[b].sub[t]] = 1'b1;
                    bank_waddr[b][wacc[b].sub[t]] = wacc[b].addr[t];
                    bank_wdata[b][wacc[b].sub[t]] = jres_data[b][t];
                end
            end else if (st_write && (op_blk == BLKW'(b))) begin
                for (int t = 0; t < NT; ++t) begin
                    bank_write[b][op_acc.sub[t]] = 1'b1;
                    bank_waddr[b][op_acc.sub[t]] = op_acc.addr[t];
                    bank_wdata[b][op_acc.sub[t]] = op.rs2_data[t][31:0];
                end
            end else if (cp_wr && (cp_wblk == BLKW'(b))) begin
                for (int t = 0; t < NT; ++t) begin
                    bank_write[b][cp_acc.sub[t]] = 1'b1;
                    bank_waddr[b][cp_acc.sub[t]] = cp_acc.addr[t];
                    bank_wdata[b][cp_acc.sub[t]] = cp_wdata[t];
                end
            end
        end
    end
    `UNUSED_VAR (job_rd_any)
    `UNUSED_VAR (op)
    `UNUSED_VAR (post_w)
    `UNUSED_VAR (rj)
    `UNUSED_VAR (e_sparse_r)
    `UNUSED_VAR (cj)

    // -----------------------------------------------------------------------
    // tc.commit arrivals
    // -----------------------------------------------------------------------

    logic [NSLOTS-1:0] want_done;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            want_done[i] = (state_r[i] == TJ_DRAIN) && (inflight_r[i] == 0)
                        && ~(e_valid_r && (e_slot_r == SLOTW'(i)));
        end
    end

    wire [SLOTW-1:0] done_i;
    wire done_v;
    VX_priority_encoder #(.N (NSLOTS)) pe_done (.data_in (want_done), `UNUSED_PIN (onehot_out), .index_out (done_i), .valid_out (done_v));

    tcu_bar_t bar_req;
    always_comb begin
        bar_req         = '0;
        bar_req.addr    = bar_addr(slot_r[done_i].commit_bar[0]);
        bar_req.size_m1 = slot_r[done_i].commit_cnt[0];
    end
    // An op with no commits left completes without a barrier request.
    wire done_silent = done_v && (slot_r[done_i].ncommit == 2'd0);
    wire done_frees  = (slot_r[done_i].ncommit <= 2'd1);
    wire bar_fire    = done_v && (done_silent || tcu_bar_if.req_ready);
    wire op_free     = bar_fire && done_frees;
    assign tcu_bar_if.req_valid = done_v && ~done_silent;
    assign tcu_bar_if.req_data  = bar_req;
    assign young_done_fire = bar_fire && (done_i == young_i);

    // -----------------------------------------------------------------------
    // Scheduling: when the datapath is free, the oldest CTA head tc.mma takes it.
    // -----------------------------------------------------------------------

    logic [NSLOTS-1:0] can_run;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            can_run[i] = (state_r[i] == TJ_WAIT) && is_head[i] && ~slot_r[i].copy;
        end
    end
    logic [NSLOTS-1:0] run_oldest;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            run_oldest[i] = can_run[i];
            for (int j = 0; j < NSLOTS; ++j) begin
                if (j != i && can_run[j] && older_r[j][i]) run_oldest[i] = 1'b0;
            end
        end
    end
    wire [SLOTW-1:0] start_i;
    wire start_v;
    VX_priority_encoder #(.N (NSLOTS)) pe_start (.data_in (run_oldest), `UNUSED_PIN (onehot_out), .index_out (start_i), .valid_out (start_v));
    // An op finishing its last K-tile frees the datapath the same cycle.
    wire run_leaves = load && load_ktile_end && load_job_end;
    wire start = start_v && (~run_valid_r || run_leaves);

    // -----------------------------------------------------------------------
    // State update
    // -----------------------------------------------------------------------

    wire [SLOTW-1:0] post_slot = alloc_slot;
    wire post = op_fire && (op_is_mma || op_is_cp);
    // a tc.commit with no op of its CTA pending gets a slot of its own
    wire commit_new = op_fire && op_is_commit && ~young_v;
    wire commit_add = op_fire && op_is_commit && young_v;

    always_ff @(posedge clk) begin
        if (reset) begin
            state_r     <= '{default: TJ_FREE};
            cta_jobs_r  <= '0;
            young_vld_r <= '0;
            inflight_r  <= '0;
            older_r     <= '0;
            run_valid_r <= 1'b0;
            e_valid_r   <= 1'b0;
            e_fresh_r   <= 1'b0;
            pro_idx_r   <= '0;
            pro_rsp_valid_r <= 1'b0;
        end else begin
            // post
            if (post) begin
                tj_slot_t j;
                j = '0;
                j.cta        = op.header.cta_id;
                j.wid        = op.header.wid;
                j.uuid       = op.header.uuid;
                if (op_is_cp) begin
                    j.copy    = 1'b1;
                    j.taddr   = post_w[0][COLW-1:0];
                    j.a_base  = post_w[1][15:0];
                    j.lda     = post_w[2][15:0];
                    j.k_tiles = post_w[3][15:0];
                    j.d_hi    = j.taddr + COLW'(j.k_tiles);
                end else begin
                    j.fmt_s      = post_w[0][4:0];
                    j.fmt_d      = post_w[0][9:5];
                    j.sparse     = post_w[0][10];
                    j.acc_init   = post_w[0][11];
                    j.a_tmem     = post_w[0][12];
                    j.n_frag     = post_w[0][20:13];
                    j.k_tiles    = post_w[1][15:0];
                    j.taddr      = post_w[2][COLW-1:0];
                    j.a_base     = post_w[3][15:0];
                    j.b_base     = post_w[4][15:0];
                    j.meta_base  = post_w[5][15:0];
                    j.lda        = post_w[7][15:0];
                    j.ldb        = post_w[7][31:16];
                    j.a_col      = COLW'(j.a_base);
                    j.meta_col   = COLW'(j.meta_base);
                    j.meta_cols  = 16'((2 * 32'(meta_row_bits(j.fmt_s)) + 31) >> 5);
                    // Step n of the bbuf/abuf/FEDP is 4 bits and 0 marks a K-tile's
                    // first step, so later steps cycle through 1..15 with the
                    // remainder folded into the B descriptor. Block-major B
                    // (ldb = 0) holds a K-tile's n_frag blocks back to back.
                    j.sn_chunk   = 16'd15 * ((j.ldb != 0) ? (16'(TC_N) * j.ldb) : 16'(B_BLK_BYTES));
                    j.b_kt_stride = (j.ldb != 0) ? 16'(KT_BYTES * (j.sparse ? 2 : 1))
                                                 : 16'(32'(j.n_frag) * B_BLK_BYTES);
                    j.d_hi       = j.taddr + op_d_len;
                    j.a_hi       = COLW'(j.a_base) + op_a_len;
                    j.m_hi       = COLW'(j.meta_base) + op_m_len;
                end
                slot_r[post_slot] <= j;
                state_r[post_slot] <= TJ_WAIT;
                inflight_r[post_slot] <= '0;
                for (int i = 0; i < NSLOTS; ++i) begin
                    older_r[post_slot][i] <= 1'b0;
                    older_r[i][post_slot] <= (state_r[i] != TJ_FREE) && (SLOTW'(i) != post_slot);
                end
            end

            if (commit_new) begin
                tj_slot_t j;
                j = '0;
                j.cta           = op.header.cta_id;
                j.wid           = op.header.wid;
                j.uuid          = op.header.uuid;
                j.ncommit       = 2'd1;
                j.commit_bar[0] = op_s1;
                j.commit_cnt[0] = BAR_SIZE_W'(op_s2 - 32'd1);
                slot_r[post_slot]     <= j;
                state_r[post_slot]    <= TJ_DRAIN;
                inflight_r[post_slot] <= '0;
                for (int i = 0; i < NSLOTS; ++i) begin
                    older_r[post_slot][i] <= 1'b0;
                    older_r[i][post_slot] <= (state_r[i] != TJ_FREE) && (SLOTW'(i) != post_slot);
                end
            end
            if (commit_add) begin
                slot_r[young_i].commit_bar[slot_r[young_i].ncommit[0]] <= op_s1;
                slot_r[young_i].commit_cnt[slot_r[young_i].ncommit[0]] <= BAR_SIZE_W'(op_s2 - 32'd1);
                slot_r[young_i].ncommit <= slot_r[young_i].ncommit + 2'd1;
            end

            for (int c = 0; c < (1 << NCTA_WIDTH); ++c) begin
                cta_jobs_r[c] <= cta_jobs_r[c]
                    + `CLOG2(JOBS_PER_CTA+NSLOTS+1)'((post || commit_new) && op.header.cta_id == NCTA_WIDTH'(c))
                    - `CLOG2(JOBS_PER_CTA+NSLOTS+1)'(op_free && slot_r[done_i].cta == NCTA_WIDTH'(c));
            end
            if (op_free && young_slot_r[slot_r[done_i].cta] == done_i) begin
                young_vld_r[slot_r[done_i].cta] <= 1'b0;
            end
            if (post || commit_new) begin
                young_slot_r[op.header.cta_id] <= post_slot;
                young_vld_r[op.header.cta_id]  <= 1'b1;
            end

            // commit arrivals
            if (bar_fire) begin
                if (done_frees) begin
                    state_r[done_i] <= TJ_FREE;
                end else begin
                    slot_r[done_i].commit_bar[0] <= slot_r[done_i].commit_bar[1];
                    slot_r[done_i].commit_cnt[0] <= slot_r[done_i].commit_cnt[1];
                    slot_r[done_i].ncommit       <= slot_r[done_i].ncommit - 2'd1;
                end
            end

            // steps in flight
            for (int i = 0; i < NSLOTS; ++i) begin
                automatic logic [INFLW-1:0] dec = '0;
                for (int b = 0; b < BLOCK_SIZE; ++b) begin
                    if (jres_valid[b] && wq_head[b][COLW+1 +: SLOTW] == SLOTW'(i)) dec = dec + 1;
                end
                inflight_r[i] <= inflight_r[i] - dec
                               + ((job_fire && e_slot_r == SLOTW'(i)) ? INFLW'(BLOCK_SIZE) : INFLW'(0));
            end

            // K-tile prologue
            pro_rsp_valid_r <= pro_go;
            if (pro_go) begin
                pro_idx_r      <= pro_idx_r + 4'd1;
                pro_rsp_meta_r <= pro_is_meta;
                pro_rsp_idx_r  <= pro_is_meta ? pro_idx_r : pro_a_idx;
                pro_rsp_sub_r  <= pro_acc.sub;
            end
            if (load && load_ktile_end) begin
                pro_idx_r <= 4'd0;
            end

            // step register
            e_fresh_r <= load && ~load_zero;
            if (load) begin
                e_valid_r      <= 1'b1;
                e_slot_r       <= run_slot_r;
                e_m_r          <= run_m_r;
                e_sn_r         <= run_sn_r;
                e_uuid_r       <= rj.uuid;
                e_wid_r        <= rj.wid;
                e_cta_r        <= rj.cta;
                e_fmt_s_r      <= rj.fmt_s;
                e_fmt_d_r      <= rj.fmt_d;
                e_lda_r        <= rj.lda;
                e_ldb_r        <= rj.ldb;
                e_a_tmem_r     <= rj.a_tmem;
                e_sparse_r     <= rj.sparse;
                e_col0_r       <= load_col0;
                e_sub_r        <= load_acc.sub;
                for (int b = 0; b < BLOCK_SIZE; ++b) begin
                    e_a_off_r[b] <= load_a_off + 16'(32'(b) * XM * 32'(rj.lda));
                end
                e_b_off_r      <= load_b_off;
                e_first_r      <= (run_m_r == 1'b0) && (run_n_r == 8'd0);
                e_pf_ok_r      <= (run_n_r != rj.n_frag - 8'd1) && (run_sn_r != 4'd15);
                e_zero_r       <= load_zero;
            end else if (job_fire) begin
                e_valid_r <= 1'b0;
            end

            // running op: walk m (inner), n, then K-tiles
            if (load) begin
                if (~load_m_last) begin
                    run_m_r <= 1'b1;
                end else begin
                    run_m_r <= 1'b0;
                    if (~load_ktile_end) begin
                        run_n_r <= run_n_r + 8'd1;
                        if (run_sn_r == 4'd15) begin
                            run_sn_r    <= 4'd1;
                            run_choff_r <= run_choff_r + rj.sn_chunk;
                        end else begin
                            run_sn_r <= run_sn_r + 4'd1;
                        end
                    end else begin
                        run_n_r     <= 8'd0;
                        run_sn_r    <= 4'd0;
                        run_choff_r <= 16'd0;
                        slot_r[run_slot_r].kt       <= rj.kt + 16'd1;
                        slot_r[run_slot_r].a_col    <= rj.a_col + COLW'(TCU_WG_FEDP_K);
                        slot_r[run_slot_r].b_kt_off <= rj.b_kt_off + rj.b_kt_stride;
                        slot_r[run_slot_r].meta_col <= rj.meta_col + COLW'(rj.meta_cols);
                        if (load_job_end) begin
                            state_r[run_slot_r] <= TJ_DRAIN;
                            run_valid_r <= 1'b0;
                        end
                    end
                end
            end

            if (cp_finish) begin
                state_r[cp_slot_r] <= TJ_DRAIN;
            end

            if (start) begin
                run_valid_r <= 1'b1;
                run_slot_r  <= start_i;
                run_m_r     <= 1'b0;
                run_n_r     <= 8'd0;
                run_sn_r    <= 4'd0;
                run_choff_r <= 16'd0;
                pro_idx_r   <= 4'd0;
                state_r[start_i] <= TJ_RUN;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Descriptor checks
    // -----------------------------------------------------------------------

    // Columns [base, base + cols) must not overlap D, which the op rewrites.
    function automatic logic bad_columns(input logic [31:0] base, input logic [31:0] cols);
        automatic logic [31:0] d_base = 32'(post_w[2][15:0]);
        automatic logic [31:0] d_end  = d_base + 32'(post_w[0][20:13]) * TC_N;
        return (base + cols > COLS) || (base < d_end && d_base < base + cols);
    endfunction
    wire [31:0] post_k_tiles   = 32'(post_w[1][15:0]);
    wire [31:0] post_meta_cols = (2 * 32'(meta_row_bits(post_w[0][4:0])) + 32'd31) >> 5;
    wire [31:0] post_cp_cols   = post_w[3];
    wire mma_bad = (post_w[0][10] && (TCU_WG_SP_FULLK == 0))
                `ifndef VX_CFG_TCU_SPARSE_ENABLE
                    || post_w[0][10]
                `endif
                    || (post_w[0][10] && bad_columns(32'(post_w[5][15:0]), post_k_tiles * post_meta_cols))
                    || (post_w[0][12] && bad_columns(32'(post_w[3][15:0]), post_k_tiles * TCU_WG_FEDP_K))
                    || post_w[1][31:16] != 0 || post_k_tiles == 0 || post_w[0][20:13] == 0
                    || post_w[6] != 0 || post_w[7][15:0] == 0
                    || (post_w[7][31:16] == 0 && (post_w[0][10] || TCU_WG_B_SUB_BLOCKS != 1))
                    || (32'(post_w[2][15:0]) + 32'(post_w[0][20:13]) * TC_N > COLS);
    wire cp_bad = (post_cp_cols == 0) || (post_cp_cols % NT != 0)
               || (post_w[1] % (NT * 4) != 0) || (post_w[2] % (NT * 4) != 0)
               || post_w[0][31:16] != 0 || (32'(post_w[0][15:0]) + post_cp_cols > COLS);
    `RUNTIME_ASSERT (~(post && (op_is_cp ? cp_bad : mma_bad)),
        ("%t: %s: unsupported tensor op (cta=%0d)", $time, INSTANCE_ID, op.header.cta_id))

`ifdef PERF_ENABLE
    // Each cycle counts once: an MMA step, else a running op's operand wait,
    // else a drain or copy, else idle.
    logic any_drain;
    always_comb begin
        any_drain = 1'b0;
        for (int i = 0; i < NSLOTS; ++i) begin
            any_drain |= (state_r[i] == TJ_DRAIN);
        end
    end
    wire perf_is_opnd  = ~job_fire && (run_valid_r || e_valid_r);
    wire perf_is_other = ~job_fire && ~perf_is_opnd && (any_drain || cp_active_r);
    logic [PERF_CTR_BITS-1:0] perf_busy_r, perf_opnd_r, perf_other_r, perf_idle_r, perf_xfers_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            perf_busy_r  <= '0;
            perf_opnd_r  <= '0;
            perf_other_r <= '0;
            perf_idle_r  <= '0;
            perf_xfers_r <= '0;
        end else begin
            if (job_fire)           perf_busy_r  <= perf_busy_r + PERF_CTR_BITS'(BLOCK_SIZE);
            else if (perf_is_opnd)  perf_opnd_r  <= perf_opnd_r + PERF_CTR_BITS'(1);
            else if (perf_is_other) perf_other_r <= perf_other_r + PERF_CTR_BITS'(1);
            else                    perf_idle_r  <= perf_idle_r + PERF_CTR_BITS'(1);
            if (op_fire && (op_is_ld || op_is_st)) perf_xfers_r <= perf_xfers_r + PERF_CTR_BITS'(1);
        end
    end
    assign perf_busy  = perf_busy_r;
    assign perf_opnd  = perf_opnd_r;
    assign perf_other = perf_other_r;
    assign perf_idle  = perf_idle_r;
    assign perf_xfers = perf_xfers_r;
`endif

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (post && op_is_mma)
            `TRACE(3, ("%t: %s tc-mma: slot=%0d, cta=%0d, n_frag=%0d, k_tiles=%0d, taddr=%0d\n",
                $time, INSTANCE_ID, post_slot, op.header.cta_id, post_w[0][20:13], post_w[1][15:0], post_w[2][15:0]))
        if (post && op_is_cp)
            `TRACE(3, ("%t: %s tc-cp: slot=%0d, cta=%0d, cols=%0d, taddr=%0d\n",
                $time, INSTANCE_ID, post_slot, op.header.cta_id, post_w[3], post_w[0][15:0]))
        if (bar_fire && ~done_silent)
            `TRACE(3, ("%t: %s tc-commit: slot=%0d, addr=%0d\n", $time, INSTANCE_ID, done_i, bar_req.addr))
        if (start)
            `TRACE(3, ("%t: %s tj-run: slot=%0d, kt=%0d\n", $time, INSTANCE_ID, start_i, slot_r[start_i].kt))
        if (cp_start)
            `TRACE(3, ("%t: %s tc-cp-run: slot=%0d, cols=%0d\n", $time, INSTANCE_ID, cpst_i, slot_r[cpst_i].k_tiles))
    end
`endif

endmodule

`endif // VX_CFG_TCU_TMEM_ENABLE
