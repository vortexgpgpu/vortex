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
// A job runs the K loop over the WGMMA tile buffers and the blocks' FEDP
// arrays without warp instructions: per K-tile it issues every (m, n) step on
// all blocks in lockstep with C read from and D written back to TMEM. Stages
// are gated by barrier phase completions; CLOSED jobs also issue their stage
// loads to DXA.

module VX_tcu_tjob import VX_gpu_pkg::*, VX_tcu_pkg::*; `ifdef VX_CFG_EXT_DXA_ENABLE import VX_dxa_pkg::*; `endif #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID    = 0,
    parameter BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS
) (
    input wire clk,
    input wire reset,

    // TMEM_LD/ST and TJ_DESC/POST from each block's dispatch
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
    VX_dxa_req_bus_if.master        dxa_req_bus_if
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
    localparam JOBS_PER_CTA = 2;
    localparam WQ_DEPTH = 16;
    localparam INFLW    = 8;
    localparam A_READS  = TCU_WG_FEDP_K / TPL;

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
        logic                   closed;
        logic                   sw_full;
        logic [3:0]             nstages;
        logic [7:0]             kb;
        logic [15:0]            k_tiles;
        logic [COLW-1:0]        taddr;
        logic [15:0]            a_base;
        logic [15:0]            b_base;
        logic [15:0]            meta_base;
        logic                   sparse;
        logic                   a_tmem;    // a_base is A's first TMEM column
        logic [15:0]            stride;
        logic [15:0]            lda;
        logic [15:0]            ldb;
        logic [31:0]            full_bar;
        logic [31:0]            empty_bar;
        logic [31:0]            done_bar;
        logic [3:0]             dxa_slot_a;
        logic [3:0]             dxa_slot_b;
        logic [15:0]            a_coord0;
        logic [15:0]            a_coord1;
        logic [15:0]            b_coord0;
        logic [15:0]            b_coord1;
        logic [15:0]            k_advance;
        logic                   a_kdim;
        logic                   b_kdim;
        // stage use (load side)
        logic [15:0]            kt;
        logic [7:0]             ki;
        logic [3:0]             use_s;
        logic [15:0]            use_off;
        logic [15:0]            use_idx;   // stage fill consumed by the next K-tile
        // stage readiness
        logic                   arrived;
        logic                   stage_ready;
        // stage release
        logic [15:0]            consumed;
        logic                   pend_empty;
        logic [3:0]             pend_empty_s;
        // CLOSED stage fills
        logic [15:0]            fills;
        logic [15:0]            fills_kt;
        logic [1:0]             fill_part;
        logic [3:0]             fill_s;
        logic [15:0]            fill_off;
        logic [15:0]            fill_k;
    } tj_slot_t;

    tj_state_e [NSLOTS-1:0] state_r;
    tj_slot_t               slot_r [NSLOTS];
    logic [NSLOTS-1:0][INFLW-1:0] inflight_r;
    // older_r[i][j]: slot i was posted before slot j
    logic [NSLOTS-1:0][NSLOTS-1:0] older_r;

    `IGNORE_UNUSED_BEGIN
    function automatic logic [BAR_ADDR_W-1:0] bar_addr(input logic [31:0] raw, input logic [3:0] stage);
        logic [NB_WIDTH-1:0] id;
        logic [BAR_ADDR_W-1:0] addr;
        id = NB_WIDTH'(raw[BAR_ID_SHIFT +: NB_WIDTH]) + NB_WIDTH'(stage);
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
    // Warp ops: TMEM_LD/ST, TJ_DESC/POST (one at a time, response next cycle)
    // -----------------------------------------------------------------------

    wire [BLOCK_SIZE-1:0] op_sel_oh;
    wire [BLKW-1:0]       op_sel;
    wire                  op_sel_valid;

    VX_priority_encoder #(
        .N (BLOCK_SIZE)
    ) op_pe (
        .data_in    (op_valid),
        .onehot_out (op_sel_oh),
        .index_out  (op_sel),
        .valid_out  (op_sel_valid)
    );
    `UNUSED_VAR (op_sel_oh)

    tcu_execute_t op;
    assign op = op_data[op_sel];
    wire op_is_ld   = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_LD));
    wire op_is_st   = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_ST));
    wire op_is_desc = (op.op_type == INST_OP_BITS'(INST_TCU_TJ_DESC));
    wire op_is_post = (op.op_type == INST_OP_BITS'(INST_TCU_TJ_POST));

    // Counted rather than derived from the slots: op_ready depends on it.
    logic [(1 << NCTA_WIDTH)-1:0][`CLOG2(JOBS_PER_CTA+NSLOTS+1)-1:0] cta_jobs_r;
    wire [`CLOG2(JOBS_PER_CTA+NSLOTS+1)-1:0] op_cta_jobs = cta_jobs_r[op.header.cta_id];

    logic [NSLOTS-1:0] free_slots;
    for (genvar i = 0; i < NSLOTS; ++i) begin : g_free
        assign free_slots[i] = (state_r[i] == TJ_FREE);
    end
    wire [SLOTW-1:0] alloc_slot;
    wire             alloc_valid;
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

    wire op_no_jobs = (op_cta_jobs == 0);
    wire op_can = op_sel_valid && rsp_free
               && (op_is_desc
                || (op_is_post && alloc_valid && (op_cta_jobs < JOBS_PER_CTA))
                || (op_is_ld && op_no_jobs)
                || (op_is_st && op_no_jobs && ~job_wr[op_blk]));
    wire op_fire = op_can;

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_op_ready
        assign op_ready[b] = op_fire && (op_sel == BLKW'(b));
    end

    wire ld_read = op_fire && op_is_ld;
    wire st_write = op_fire && op_is_st;

    `RUNTIME_ASSERT (~(op_fire && (op_is_ld || op_is_st))
                  || ((32'(op_lane) % XM) == 0 && 32'(op_lane) < LANES),
        ("%t: %s: TMEM lane %0d out of range", $time, INSTANCE_ID, op_lane))

    // descriptor words 0..7 of each warp
    logic [`VX_CFG_NUM_WARPS-1:0][7:0][31:0] desc_buf_r;
    always_ff @(posedge clk) begin
        if (op_fire && op_is_desc) begin
            for (int w = 0; w < 4; ++w) begin
                desc_buf_r[op.header.wid][w]     <= op.rs1_data[w][31:0];
                desc_buf_r[op.header.wid][4 + w] <= op.rs2_data[w][31:0];
            end
        end
    end

    // job descriptor of a TJ_POST
    logic [15:0][31:0] post_w;
    always_comb begin
        for (int w = 0; w < 8; ++w) post_w[w] = desc_buf_r[op.header.wid][w];
        for (int w = 0; w < 4; ++w) begin
            post_w[8 + w]  = op.rs1_data[w][31:0];
            post_w[12 + w] = op.rs2_data[w][31:0];
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
    end

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
    logic [3:0]       e_stage_r;
    logic             e_first_r;
    logic             e_zero_r;
    logic             e_ktile_last_r;
    logic             e_stage_end_r;
    logic             e_fresh_r;
    logic [BLOCK_SIZE-1:0][NT-1:0][31:0] e_hold_r;

    // Slot fields of the presented step, captured at load: job_fire depends on
    // the descriptors through tbuf_ready, too late for the slot mux and adders.
    logic [UUID_WIDTH-1:0] e_uuid_r;
    logic [NW_WIDTH-1:0]   e_wid_r;
    logic [NCTA_WIDTH-1:0] e_cta_r;
    logic [4:0]       e_fmt_s_r, e_fmt_d_r;
    logic [15:0]      e_lda_r, e_ldb_r;
    logic             e_closed_r, e_a_tmem_r, e_sparse_r;
    logic [COLW-1:0]  e_col0_r;
    logic [NT-1:0][SUBW-1:0] e_sub_r;
    logic [BLOCK_SIZE-1:0][15:0] e_a_off_r;
    logic [15:0]      e_b_off_r;

    tj_slot_t rj;
    assign rj = slot_r[run_slot_r];
    tj_slot_t ej;
    assign ej = slot_r[e_slot_r];

    assign job_fire = e_valid_r && (&core_ready);
    wire e_advance  = ~e_valid_r || job_fire;

    // K-tile prologue: metadata then A over the read port, all blocks in
    // parallel, once the previous K-tile's last step has left the E register.
    // A lane's K-tile metadata is its two rows, in row order, over `meta_cols`.
    logic [3:0] pro_idx_r;
    function automatic logic [15:0] meta_row_bits(input logic [4:0] fmt);
        return 16'(TCU_TC_K * 2 * (32 / tcu_fmt_width(fmt)));
    endfunction
    wire [15:0] rj_meta_bits  = 2 * meta_row_bits(rj.fmt_s);
    wire [15:0] rj_meta_cols  = (rj_meta_bits + 16'd31) >> 5;
    wire [3:0]  pro_meta_reads = rj.sparse ? 4'((rj_meta_cols + 16'(TPL) - 16'd1) / 16'(TPL)) : 4'd0;
    wire [3:0]  pro_a_reads    = rj.a_tmem ? 4'(A_READS) : 4'd0;
    wire [3:0]  pro_total      = pro_meta_reads + pro_a_reads;
    wire        pro_pending    = run_valid_r && (pro_idx_r < pro_total);
    wire        pro_go         = pro_pending && e_advance && ~ld_read;
    wire        pro_is_meta    = (pro_idx_r < pro_meta_reads);
    wire [3:0]  pro_a_idx      = pro_idx_r - pro_meta_reads;
    wire [COLW-1:0] pro_col0   = pro_is_meta
        ? (rj.meta_base + COLW'(32'(rj.kt) * 32'(rj_meta_cols)) + COLW'(32'(pro_idx_r) * TPL))
        : (rj.a_base + COLW'(32'(rj.kt) * TCU_WG_FEDP_K) + COLW'(32'(pro_a_idx) * TPL));
    tm_access_t pro_acc;
    assign pro_acc = reg_access(1'b0, 5'd0, pro_col0);

    wire load       = e_advance && run_valid_r && ~ld_read && ~pro_pending;

    wire load_zero      = (rj.kt == 0) && rj.acc_init;
    wire load_m_last    = (run_m_r == 1'(M_STEPS - 1));
    wire load_ktile_end = load_m_last && (run_n_r == rj.n_frag - 8'd1);
    wire load_stage_end = (rj.ki == rj.kb - 8'd1) || (rj.kt == rj.k_tiles - 16'd1);
    wire load_job_end   = (rj.kt == rj.k_tiles - 16'd1);
    wire [COLW-1:0] load_col0 = rj.taddr + COLW'(32'(run_n_r) * TC_N);
    tm_access_t load_acc;
    assign load_acc = frag_access(run_m_r, load_col0);

    assign job_rd_any = load && ~load_zero;

    // Step n of the bbuf/abuf/FEDP is 4 bits and 0 marks a K-tile's first
    // step, so later steps cycle through 1..15 with the remainder folded into
    // the B descriptor.
    wire [15:0] sn_chunk = 16'(TC_N) * rj.ldb * 16'd15;

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
    wire [15:0] load_a_off = rj.a_base + rj.use_off + 16'(32'(rj.ki) * KT_BYTES);
    wire [15:0] load_b_off = rj.b_base + rj.use_off + run_choff_r
                           + ((rj.ldb != 0) ? 16'(32'(rj.ki) * KT_BYTES * (rj.sparse ? 2 : 1)) : 16'd0);

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
            end
        end
    end
    `UNUSED_VAR (job_rd_any)
    `UNUSED_VAR (op)
    `UNUSED_VAR (post_w)
    `UNUSED_VAR (rj)
    `UNUSED_VAR (ej)
    `UNUSED_VAR (dj)
    `UNUSED_VAR (e_sparse_r)

    // -----------------------------------------------------------------------
    // Barrier and DXA requests
    // -----------------------------------------------------------------------

    // per-slot request candidates
    logic [NSLOTS-1:0] want_done, want_empty, want_attach, want_arrive, want_dxa;
    logic [NSLOTS-1:0] e_holds;
    `IGNORE_UNUSED_BEGIN
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            automatic tj_slot_t j = slot_r[i];
            automatic logic fill_open = j.closed && is_head[i] && (state_r[i] != TJ_DRAIN)
                                     && (j.fills_kt < j.k_tiles)
                                     && (j.fills < j.consumed + 16'(j.nstages));
            e_holds[i]     = e_valid_r && (e_slot_r == SLOTW'(i));
            want_done[i]   = (state_r[i] == TJ_DRAIN) && (inflight_r[i] == 0) && ~e_holds[i] && ~j.pend_empty;
            want_empty[i]  = (state_r[i] != TJ_FREE) && j.pend_empty;
            want_attach[i] = fill_open && (j.fill_part == 2'd0);
            want_dxa[i]    = fill_open && (j.fill_part != 2'd0);
            want_arrive[i] = (state_r[i] == TJ_WAIT) && is_head[i] && ~j.sw_full
                          && ~j.arrived && ~j.stage_ready
                          && (~j.closed || (j.fills > j.use_idx));
        end
    end
    `IGNORE_UNUSED_END

    wire [SLOTW-1:0] done_i, empty_i, attach_i, arrive_i, dxa_i;
    wire done_v, empty_v, attach_v, arrive_v, dxa_v;
    VX_priority_encoder #(.N (NSLOTS)) pe_done   (.data_in (want_done),   `UNUSED_PIN (onehot_out), .index_out (done_i),   .valid_out (done_v));
    VX_priority_encoder #(.N (NSLOTS)) pe_empty  (.data_in (want_empty),  `UNUSED_PIN (onehot_out), .index_out (empty_i),  .valid_out (empty_v));
    VX_priority_encoder #(.N (NSLOTS)) pe_attach (.data_in (want_attach), `UNUSED_PIN (onehot_out), .index_out (attach_i), .valid_out (attach_v));
    VX_priority_encoder #(.N (NSLOTS)) pe_arrive (.data_in (want_arrive), `UNUSED_PIN (onehot_out), .index_out (arrive_i), .valid_out (arrive_v));
    VX_priority_encoder #(.N (NSLOTS)) pe_dxa    (.data_in (want_dxa),    `UNUSED_PIN (onehot_out), .index_out (dxa_i),    .valid_out (dxa_v));

    tcu_bar_t        bar_req;
    logic [1:0]      bar_kind; // 0 done, 1 empty, 2 attach, 3 arrive
    logic [SLOTW-1:0] bar_slot;
    always_comb begin
        bar_req  = '0;
        bar_kind = '0;
        bar_slot = '0;
        if (done_v) begin
            bar_kind = 2'd0; bar_slot = done_i;
            bar_req.addr     = bar_addr(slot_r[done_i].done_bar, 4'd0);
            bar_req.is_event = 1'b1;
            bar_req.phase    = 1'b0;
        end else if (empty_v) begin
            bar_kind = 2'd1; bar_slot = empty_i;
            bar_req.addr     = bar_addr(slot_r[empty_i].empty_bar, slot_r[empty_i].pend_empty_s);
            bar_req.size_m1  = BAR_SIZE_W'(1);
        end else if (attach_v) begin
            bar_kind = 2'd2; bar_slot = attach_i;
            bar_req.addr     = bar_addr(slot_r[attach_i].full_bar, slot_r[attach_i].fill_s);
            bar_req.is_event = 1'b1;
            bar_req.phase    = 1'b1;
            bar_req.size_m1  = slot_r[attach_i].a_tmem ? BAR_SIZE_W'(0) : BAR_SIZE_W'(1);
        end else if (arrive_v) begin
            bar_kind = 2'd3; bar_slot = arrive_i;
            bar_req.addr     = bar_addr(slot_r[arrive_i].full_bar, slot_r[arrive_i].use_s);
            bar_req.size_m1  = slot_r[arrive_i].closed ? BAR_SIZE_W'(0) : BAR_SIZE_W'(1);
        end
    end
    wire bar_valid = done_v || empty_v || attach_v || arrive_v;
    wire bar_fire  = bar_valid && tcu_bar_if.req_ready;
    assign tcu_bar_if.req_valid = bar_valid;
    assign tcu_bar_if.req_data  = bar_req;

    // DXA stage loads (CLOSED)
    tj_slot_t dj;
    assign dj = slot_r[dxa_i];
    wire dxa_is_b = (dj.fill_part == 2'd2);
    // k_advance is in dense K; compressed A advances half as far.
    wire [31:0] dxa_k = (dj.sparse && ~dxa_is_b) ? (32'(dj.fill_k) >> 1) : 32'(dj.fill_k);
    dxa_req_data_t dxa_req;
    always_comb begin
        dxa_req = '0;
        dxa_req.core_id   = NC_WIDTH'(CORE_ID);
        dxa_req.uuid      = dj.uuid;
        dxa_req.wid       = dj.wid;
        dxa_req.smem_addr = DXA_SMEM_ADDR_W'((dxa_is_b ? dj.b_base : dj.a_base) + dj.fill_off);
        dxa_req.meta      = ((dj.full_bar + (32'(dj.fill_s) << BAR_ID_SHIFT)) << 4)
                          | 32'(dxa_is_b ? dj.dxa_slot_b : dj.dxa_slot_a);
        if (dxa_is_b) begin
            dxa_req.coords[0] = 32'(dj.b_coord0) + (dj.b_kdim ? 32'd0 : dxa_k);
            dxa_req.coords[1] = 32'(dj.b_coord1) + (dj.b_kdim ? dxa_k : 32'd0);
        end else begin
            dxa_req.coords[0] = 32'(dj.a_coord0) + (dj.a_kdim ? 32'd0 : dxa_k);
            dxa_req.coords[1] = 32'(dj.a_coord1) + (dj.a_kdim ? dxa_k : 32'd0);
        end
    end
    assign dxa_req_bus_if.req_valid = dxa_v;
    assign dxa_req_bus_if.req_data  = dxa_req;
    wire dxa_fire = dxa_v && dxa_req_bus_if.req_ready;

    // -----------------------------------------------------------------------
    // Scheduling: when the datapath is free, the oldest CTA head whose stage
    // is ready takes it for one stage.
    // -----------------------------------------------------------------------

    logic [NSLOTS-1:0] can_run;
    always_comb begin
        for (int i = 0; i < NSLOTS; ++i) begin
            can_run[i] = (state_r[i] == TJ_WAIT) && is_head[i]
                      && (slot_r[i].sw_full || slot_r[i].stage_ready);
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
    // A job finishing its stage frees the datapath the same cycle.
    wire run_leaves = load && load_ktile_end && (load_stage_end);
    wire start = start_v && (~run_valid_r || run_leaves);

    // -----------------------------------------------------------------------
    // State update
    // -----------------------------------------------------------------------

    wire [SLOTW-1:0] post_slot = alloc_slot;
    wire post = op_fire && op_is_post;

    always_ff @(posedge clk) begin
        if (reset) begin
            state_r     <= '{default: TJ_FREE};
            cta_jobs_r  <= '0;
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
                j.fmt_s      = post_w[0][4:0];
                j.fmt_d      = post_w[0][9:5];
                j.acc_init   = post_w[0][11];
                j.n_frag     = post_w[0][20:13];
                j.closed     = (post_w[1][1:0] == 2'd1);
                j.sw_full    = post_w[1][2];
                j.nstages    = post_w[1][7:4];
                j.kb         = post_w[1][15:8];
                j.k_tiles    = post_w[1][31:16];
                j.taddr      = post_w[2][COLW-1:0];
                j.a_base     = post_w[3][15:0];
                j.b_base     = post_w[4][15:0];
                j.meta_base  = post_w[5][15:0];
                j.sparse     = post_w[0][10];
                j.a_tmem     = post_w[0][12];
                j.stride     = post_w[6][15:0];
                j.lda        = post_w[7][15:0];
                j.ldb        = post_w[7][31:16];
                j.full_bar   = post_w[8];
                j.empty_bar  = post_w[9];
                j.done_bar   = post_w[10];
                j.dxa_slot_a = post_w[11][3:0];
                j.dxa_slot_b = post_w[11][7:4];
                j.a_coord0   = post_w[12][15:0];
                j.a_coord1   = post_w[12][31:16];
                j.b_coord0   = post_w[13][15:0];
                j.b_coord1   = post_w[13][31:16];
                j.k_advance  = post_w[14][15:0];
                j.a_kdim     = post_w[14][16];
                j.b_kdim     = post_w[14][17];
                j.use_s      = post_w[15][3:0];
                j.use_off    = 16'(32'(post_w[15][3:0]) * 32'(post_w[6][15:0]));
                j.fill_s     = post_w[15][3:0];
                j.fill_off   = 16'(32'(post_w[15][3:0]) * 32'(post_w[6][15:0]));
                slot_r[post_slot] <= j;
                state_r[post_slot] <= TJ_WAIT;
                inflight_r[post_slot] <= '0;
                for (int i = 0; i < NSLOTS; ++i) begin
                    older_r[post_slot][i] <= 1'b0;
                    older_r[i][post_slot] <= (state_r[i] != TJ_FREE) && (SLOTW'(i) != post_slot);
                end
            end

            for (int c = 0; c < (1 << NCTA_WIDTH); ++c) begin
                cta_jobs_r[c] <= cta_jobs_r[c]
                    + `CLOG2(JOBS_PER_CTA+NSLOTS+1)'(post && op.header.cta_id == NCTA_WIDTH'(c))
                    - `CLOG2(JOBS_PER_CTA+NSLOTS+1)'(bar_fire && bar_kind == 2'd0 && slot_r[bar_slot].cta == NCTA_WIDTH'(c));
            end

            // barrier requests
            if (bar_fire) begin
                case (bar_kind)
                2'd0: state_r[bar_slot] <= TJ_FREE;
                2'd1: slot_r[bar_slot].pend_empty <= 1'b0;
                2'd2: slot_r[bar_slot].fill_part <= slot_r[bar_slot].a_tmem ? 2'd2 : 2'd1;
                default: slot_r[bar_slot].arrived <= 1'b1;
                endcase
            end

            // stage loads
            if (dxa_fire) begin
                if (dj.fill_part == 2'd2) begin
                    slot_r[dxa_i].fill_part <= 2'd0;
                    slot_r[dxa_i].fills     <= dj.fills + 16'd1;
                    slot_r[dxa_i].fills_kt  <= dj.fills_kt + 16'(dj.kb);
                    slot_r[dxa_i].fill_k    <= dj.fill_k + dj.k_advance;
                    if (dj.fill_s == dj.nstages - 4'd1) begin
                        slot_r[dxa_i].fill_s   <= 4'd0;
                        slot_r[dxa_i].fill_off <= 16'd0;
                    end else begin
                        slot_r[dxa_i].fill_s   <= dj.fill_s + 4'd1;
                        slot_r[dxa_i].fill_off <= dj.fill_off + dj.stride;
                    end
                end else begin
                    slot_r[dxa_i].fill_part <= 2'd2;
                end
            end

            // phase completions
            if (tcu_bar_if.flip_valid) begin
                for (int i = 0; i < NSLOTS; ++i) begin
                    if (state_r[i] == TJ_WAIT && slot_r[i].arrived
                     && tcu_bar_if.flip_addr == bar_addr(slot_r[i].full_bar, slot_r[i].use_s)) begin
                        slot_r[i].arrived     <= 1'b0;
                        slot_r[i].stage_ready <= 1'b1;
                    end
                end
            end

            // step fired: count it in flight, release the stage at its end
            if (job_fire && e_ktile_last_r && e_stage_end_r) begin
                slot_r[e_slot_r].consumed <= ej.consumed + 16'd1;
                if (~e_closed_r) begin
                    slot_r[e_slot_r].pend_empty   <= 1'b1;
                    slot_r[e_slot_r].pend_empty_s <= e_stage_r;
                end
            end
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
                e_stage_r      <= rj.use_s;
                e_uuid_r       <= rj.uuid;
                e_wid_r        <= rj.wid;
                e_cta_r        <= rj.cta;
                e_fmt_s_r      <= rj.fmt_s;
                e_fmt_d_r      <= rj.fmt_d;
                e_lda_r        <= rj.lda;
                e_ldb_r        <= rj.ldb;
                e_closed_r     <= rj.closed;
                e_a_tmem_r     <= rj.a_tmem;
                e_sparse_r     <= rj.sparse;
                e_col0_r       <= load_col0;
                e_sub_r        <= load_acc.sub;
                for (int b = 0; b < BLOCK_SIZE; ++b) begin
                    e_a_off_r[b] <= load_a_off + 16'(32'(b) * XM * 32'(rj.lda));
                end
                e_b_off_r      <= load_b_off;
                e_first_r      <= (run_m_r == 1'b0) && (run_n_r == 8'd0);
                e_zero_r       <= load_zero;
                e_ktile_last_r <= load_ktile_end;
                e_stage_end_r  <= load_stage_end;
            end else if (job_fire) begin
                e_valid_r <= 1'b0;
            end

            // running job: walk m (inner), n, then K-tiles of the stage
            if (load) begin
                if (~load_m_last) begin
                    run_m_r <= 1'b1;
                end else begin
                    run_m_r <= 1'b0;
                    if (~load_ktile_end) begin
                        run_n_r <= run_n_r + 8'd1;
                        if (run_sn_r == 4'd15) begin
                            run_sn_r    <= 4'd1;
                            run_choff_r <= run_choff_r + sn_chunk;
                        end else begin
                            run_sn_r <= run_sn_r + 4'd1;
                        end
                    end else begin
                        run_n_r     <= 8'd0;
                        run_sn_r    <= 4'd0;
                        run_choff_r <= 16'd0;
                        slot_r[run_slot_r].kt <= rj.kt + 16'd1;
                        if (rj.ki == rj.kb - 8'd1) begin
                            slot_r[run_slot_r].ki <= 8'd0;
                            slot_r[run_slot_r].use_idx <= rj.use_idx + 16'd1;
                            if (rj.use_s == rj.nstages - 4'd1) begin
                                slot_r[run_slot_r].use_s   <= 4'd0;
                                slot_r[run_slot_r].use_off <= 16'd0;
                            end else begin
                                slot_r[run_slot_r].use_s   <= rj.use_s + 4'd1;
                                slot_r[run_slot_r].use_off <= rj.use_off + rj.stride;
                            end
                        end else begin
                            slot_r[run_slot_r].ki <= rj.ki + 8'd1;
                        end
                        if (load_job_end) begin
                            state_r[run_slot_r] <= TJ_DRAIN;
                            run_valid_r <= 1'b0;
                        end else if (load_stage_end) begin
                            state_r[run_slot_r] <= TJ_WAIT;
                            run_valid_r <= 1'b0;
                        end
                    end
                end
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
                slot_r[start_i].stage_ready <= 1'b0;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Descriptor checks
    // -----------------------------------------------------------------------

    // Columns [base, base + cols) must not overlap D, which the job rewrites.
    function automatic logic bad_columns(input logic [31:0] base, input logic [31:0] cols);
        automatic logic [31:0] d_base = 32'(post_w[2][15:0]);
        automatic logic [31:0] d_end  = d_base + 32'(post_w[0][20:13]) * TC_N;
        return (base + cols > COLS) || (base < d_end && d_base < base + cols);
    endfunction
    wire [31:0] post_k_tiles  = 32'(post_w[1][31:16]);
    wire [31:0] post_meta_cols = (2 * 32'(meta_row_bits(post_w[0][4:0])) + 32'd31) >> 5;
    wire post_bad = post && (
                       (post_w[0][10] && (TCU_WG_SP_FULLK == 0))
                    `ifndef VX_CFG_TCU_SPARSE_ENABLE
                          || post_w[0][10]
                    `endif
                          || (post_w[0][10] && bad_columns(32'(post_w[5][15:0]), post_k_tiles * post_meta_cols))
                          || (post_w[0][12] && bad_columns(32'(post_w[3][15:0]), post_k_tiles * TCU_WG_FEDP_K))
                          || post_w[1][1]                         // unknown mode
                          || (post_w[1][2] && post_w[1][1:0] != 2'd0)
                          || post_w[1][7:4] == 0 || post_w[1][15:8] == 0
                          || post_w[1][31:16] == 0 || post_w[0][20:13] == 0
                          || post_w[7][15:0] == 0 || post_w[7][31:16] == 0
                          || post_w[2][31:16] != 0
                          || (32'(post_w[2][15:0]) + 32'(post_w[0][20:13]) * TC_N > COLS)
                          || (post_w[15][3:0] >= post_w[1][7:4]));
    `RUNTIME_ASSERT (~post_bad, ("%t: %s: unsupported tensor job descriptor (cta=%0d)", $time, INSTANCE_ID, op.header.cta_id))

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (post)
            `TRACE(3, ("%t: %s tj-post: slot=%0d, cta=%0d, n_frag=%0d, k_tiles=%0d, kb=%0d, nstages=%0d, closed=%0d, taddr=%0d\n",
                $time, INSTANCE_ID, post_slot, op.header.cta_id, post_w[0][20:13], post_w[1][31:16], post_w[1][15:8], post_w[1][7:4], post_w[1][1:0], post_w[2][15:0]))
        if (bar_fire)
            `TRACE(3, ("%t: %s tj-bar: slot=%0d, kind=%0d, addr=%0d\n", $time, INSTANCE_ID, bar_slot, bar_kind, bar_req.addr))
        if (dxa_fire)
            `TRACE(3, ("%t: %s tj-fill: slot=%0d, stage=%0d, part=%0d, c0=%0d, c1=%0d\n", $time, INSTANCE_ID, dxa_i, dj.fill_s, dj.fill_part, dxa_req.coords[0], dxa_req.coords[1]))
        if (start)
            `TRACE(3, ("%t: %s tj-run: slot=%0d, kt=%0d\n", $time, INSTANCE_ID, start_i, slot_r[start_i].kt))
    end
`endif

endmodule

`endif // VX_CFG_TCU_TMEM_ENABLE
