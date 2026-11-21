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

// Tensor memory and its MMA engine.
//
// TMEM is BLOCK_SIZE * XM lanes by TMEM_COLS 32-bit columns. Block b owns
// lanes [b*XM, (b+1)*XM), the M rows it computes, stored fragment-major: a
// storage row is one C/D fragment (TC_M lanes by TC_N columns, word
// i*TC_N + j = lane m*TC_M + i, column g*TC_N + j), and the two M steps of a
// column group g sit at row g of two RAMs. An MMA step reads and writes one
// row of one RAM with identity wiring; the transfer shapes are built on the
// tmem.ld/st path only.
//
// tmem.ld/st go to the issuing warp's own block, each block through its own
// port. tc.mma, tc.cp, tc.commit and the allocation ops pass a scalar-only
// port; tc.mma/tc.cp enter one in-order queue, and a tc.commit rides on the
// youngest op still queued or running. The engine runs the head op: a tc.mma
// is one K step, walked (m, n) on all blocks in lockstep over the WGMMA tile
// buffers and the blocks' FEDP arrays; a tc.cp copies pre-packed fragment
// rows, one LMEM bank row per read, through the tile buffers' LMEM port. An
// op leaving the engine posts its commits with the count of steps issued so
// far; a commit arrives on its barrier once that many have been written back.
// Ordering between tmem.ld/st and the MMAs is the software's (commit, barrier
// wait, tmem::wait_st), checked by the SimX contract checker; the only
// hazard checked here is a step whose C fragment is still being written.

module VX_tcu_tmem import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
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

    // MMA steps to the blocks' tcu_cores (presented on all blocks at once)
    output wire                     job_valid,
    output tcu_execute_t            job_data [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0]    core_ready,
    output wire                     job_fire,

    // Tile-buffer requests of the presented step
    output tcu_tbuf_req_t [BLOCK_SIZE-1:0] tbuf_req,

    // A K step's A from TMEM, one fragment group of every lane per read
    output wire                     a_fill_valid,
    output wire [3:0]               a_fill_idx,
    output wire [BLOCK_SIZE-1:0][TCU_WG_TILE_M-1:0][TCU_TC_N-1:0][31:0] a_fill_data,

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Metadata of the presented step: its two TC_K halves
    output wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta,
    output wire [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta_hi,
`endif
`ifdef VX_CFG_TCU_MX_ENABLE
    // MX scales of the presented step: TC_M A rows, then TC_N B columns
    output wire [BLOCK_SIZE-1:0][TCU_TC_M+TCU_TC_N-1:0][TCU_MX_MAX_SF-1:0][7:0] job_sf,
`endif

    // Step results from the blocks' tcu_cores, in issue order per block
    input  wire [BLOCK_SIZE-1:0]    jres_valid,
    input  wire [BLOCK_SIZE-1:0][`VX_CFG_NUM_THREADS-1:0][31:0] jres_data,

    // A CTA's last warp retired: its columns are freed
    input  wire                     cta_exit_valid,
    input  wire [NCTA_WIDTH-1:0]    cta_exit_id,

    VX_tcu_bar_if.master            tcu_bar_if,
    VX_mem_bus_if.master            cp_lmem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NT       = `VX_CFG_NUM_THREADS;
    localparam TC_M     = TCU_TC_M;
    localparam TC_N     = TCU_TC_N;
    localparam M_STEPS  = TCU_WG_M_STEPS;
    localparam XM       = TCU_WG_TILE_M;
    localparam TPL      = NT / XM;              // threads per lane (lane per thread)
    localparam RG       = TC_N / TPL;           // lane-per-thread registers per group
    localparam COLS     = `VX_CFG_TCU_TMEM_COLS;
    localparam LANES    = BLOCK_SIZE * XM;
    localparam GROUPS   = COLS / TC_N;
    localparam GW       = `LOG2UP(GROUPS);
    localparam BLKW     = `LOG2UP(BLOCK_SIZE);
    localparam COLW     = 16;
    localparam OPQ      = 4;                    // tc.mma / tc.cp queue
    localparam OPQW     = `LOG2UP(OPQ);
    localparam CMQ      = 4;                    // posted commits
    localparam CMQW     = `LOG2UP(CMQ);
    localparam WQ_DEPTH = 16;                   // steps in flight
    localparam SCNTW    = 16;                   // step counters
    localparam A_READS  = TCU_WG_FEDP_K / TC_N;
    localparam CP_ROW_WORDS = `VX_CFG_LMEM_NUM_BANKS * (`VX_CFG_XLEN / 32);
    localparam CP_FRAGS = CP_ROW_WORDS / NT;    // fragment rows per LMEM bank row
    localparam CP_ADDRW = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(CP_ROW_WORDS * 4);
    localparam ACOLS    = 32;
    localparam ACHUNKS  = COLS / ACOLS;
    localparam ACW      = `UP(`CLOG2(ACHUNKS));

    `STATIC_ASSERT (TC_M * TC_N == NT, ("C fragment must span the warp"))
    `STATIC_ASSERT (M_STEPS == 2, ("TMEM stores the two M steps of a group in two RAMs"))
    `STATIC_ASSERT (NT % XM == 0, ("TMEM needs NUM_THREADS to be a multiple of WGMMA M"))
    `STATIC_ASSERT (TCU_WG_FEDP_K % TC_N == 0, ("a K step of A from TMEM is whole fragment groups"))
    `STATIC_ASSERT (CP_ROW_WORDS % NT == 0, ("an LMEM bank row holds whole fragment rows"))
    `STATIC_ASSERT ((2 * BLOCK_SIZE) % CP_FRAGS == 0, ("a bank row's fragment rows are in one column group"))
    // MMA steps issue to every block's core in one cycle; a time-multiplexed
    // FEDP array takes a micro-op over several.
    `STATIC_ASSERT (`VX_CFG_TCU_PE_RATIO == 1, ("TMEM requires VX_CFG_TCU_PE_RATIO=1"))

    `IGNORE_UNUSED_BEGIN
    // WGMMA shared-memory descriptor: byte offset below, row stride in words above.
    function automatic logic [31:0] smem_desc(input logic [17:0] off, input logic [15:0] stride_bytes);
        smem_desc = {(32 - TCU_SMEM_DESC_OFF_BITS)'(stride_bytes[15:2]), TCU_SMEM_DESC_OFF_BITS'(off)};
    endfunction

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

    // Metadata rows per K step: one per TC_K words of compressed A.
    localparam SP_ROWS = TCU_WG_FEDP_K / TCU_TC_K;
`ifndef VX_CFG_TCU_SPARSE_ENABLE
    `UNUSED_PARAM (SP_ROWS)
`endif
    function automatic logic [15:0] meta_row_bits(input logic [4:0] fmt);
        return 16'(TCU_TC_K * 2 * (32 / tcu_fmt_width(fmt)));
    endfunction

    // -----------------------------------------------------------------------
    // Storage: per block and M step one RAM of GROUPS fragment rows
    // -----------------------------------------------------------------------

    logic [BLOCK_SIZE-1:0][M_STEPS-1:0]               ram_read;
    logic [BLOCK_SIZE-1:0][M_STEPS-1:0][GW-1:0]       ram_raddr;
    logic [BLOCK_SIZE-1:0][M_STEPS-1:0]               ram_write;
    logic [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0]       ram_wren;
    logic [BLOCK_SIZE-1:0][M_STEPS-1:0][GW-1:0]       ram_waddr;
    wire  [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0][31:0] ram_wdata;
    wire  [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0][31:0] ram_rdata;

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_bank
        for (genvar m = 0; m < M_STEPS; ++m) begin : g_m
            VX_dp_ram #(
                .DATAW    (NT * 32),
                .SIZE     (GROUPS),
                .WRENW    (NT),
                .OUT_REG  (1),
                .RDW_MODE ("R")
            ) mem (
                .clk   (clk),
                .reset (reset),
                .read  (ram_read[b][m]),
                .write (ram_write[b][m]),
                .wren  (ram_wren[b][m]),
                .waddr (ram_waddr[b][m]),
                .wdata (ram_wdata[b][m]),
                .raddr (ram_raddr[b][m]),
                .rdata (ram_rdata[b][m])
            );
        end
    end

    // An allocation is a power-of-two run of ACOLS-column chunks: its size
    // class and chunk mask, decoded as the op is registered.
    localparam NCLS = `CLOG2(ACHUNKS) + 1;
    localparam CLSW = `UP(`CLOG2(NCLS));

    // -----------------------------------------------------------------------
    // Op select: each block's op is registered, then either served by the
    // block's transfer unit or presented to the scalar port.
    // -----------------------------------------------------------------------

    // Only what the ops read is kept: thread 0's rs1, rs2 and rs3 (the tc.*
    // operands and the transfer address). A tmem.st does not enter the slot:
    // it writes from the input, whose data the sender holds until op_ready,
    // in a cycle the slot is empty, so it never passes an earlier op.
    typedef struct packed {
        tcu_header_t             header;
        logic [INST_OP_BITS-1:0] op_type;
        logic [4:0]              fmt_d;      // transfer register index
        logic                    frag;       // transfer shape
        logic [31:0]             s1;
        logic [31:0]             s2;
        logic [31:0]             s3;
    } q_op_t;

    logic [BLOCK_SIZE-1:0] q_valid_r;
    q_op_t                 q_op_r [BLOCK_SIZE];
    logic [BLOCK_SIZE-1:0][CLSW-1:0]    q_cls_r;
    logic [BLOCK_SIZE-1:0][ACHUNKS-1:0] q_amask_r;
    wire  [BLOCK_SIZE-1:0] q_fire;     // the registered op is done with
    wire  [BLOCK_SIZE-1:0] st_valid;   // a tmem.st at the input, the slot empty
    wire  [BLOCK_SIZE-1:0] st_take;    // it writes this cycle
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_q
        q_op_t q_in;
        assign q_in = {op_data[b].header, op_data[b].op_type,
                       op_data[b].op_args.tcu.fmt_d, op_data[b].op_args.tcu.a_from_smem,
                       op_data[b].rs1_data[0][31:0], op_data[b].rs2_data[0][31:0],
                       op_data[b].rs3_data[0][31:0]};
        wire in_st  = (op_data[b].op_type == INST_OP_BITS'(INST_TCU_TMEM_ST));
        wire q_load = op_valid[b] && ~in_st && (~q_valid_r[b] || q_fire[b]);
        assign st_valid[b] = op_valid[b] && in_st && ~q_valid_r[b];
        assign op_ready[b] = q_load || st_take[b];
        wire [31:0] in_nck = ((op_data[b].op_type == INST_OP_BITS'(INST_TCU_TMEM_ALLOC))
                              ? op_data[b].rs1_data[0][31:0] : op_data[b].rs2_data[0][31:0]) / ACOLS;
        always_ff @(posedge clk) begin
            if (reset) begin
                q_valid_r[b] <= 1'b0;
            end else if (q_load) begin
                q_valid_r[b] <= 1'b1;
            end else if (q_fire[b]) begin
                q_valid_r[b] <= 1'b0;
            end
            if (q_load) begin
                q_op_r[b]    <= q_in;
                q_cls_r[b]   <= size_class(in_nck);
                q_amask_r[b] <= ACHUNKS'((33'(1) << in_nck) - 1);
            end
        end
    end

    // Transfer-unit write requests, central-port responses and engine reads
    // meet at the per-block ports below.
    wire job_writing;                      // D write-backs this cycle (all blocks)
    wire [BLOCK_SIZE-1:0] cp_wr_blk;       // a tc.cp row write this cycle
    wire [BLOCK_SIZE-1:0] xf_rd;           // the transfer unit reads both RAMs
    wire [BLOCK_SIZE-1:0] xf_hold;         // a load's response holds the RAM outputs
    wire                  e_cwait;         // a presented step's C holds the RAM outputs
    wire [BLOCK_SIZE-1:0][M_STEPS-1:0] xf_rd_m;
    wire [BLOCK_SIZE-1:0][GW-1:0] xf_rd_g;
    wire [BLOCK_SIZE-1:0] xf_wr;           // the transfer unit writes
    wire [BLOCK_SIZE-1:0][M_STEPS-1:0] xf_wr_m;
    wire [BLOCK_SIZE-1:0][GW-1:0] xf_wr_g;
    wire [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0] xf_wr_en;
    wire [BLOCK_SIZE-1:0] xf_wr_frag;      // the transfer is fragment shaped
    wire [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0][31:0] xf_wd_frag;  // a store's word, fragment
    wire [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0][31:0] xf_wd_lpt;   // a store's word, lane per thread

    // Central-port response, delivered through the issuing block's response slot.
    logic                  c_rsp_valid_r;
    logic [BLKW-1:0]       c_rsp_blk_r;
    tcu_header_t           c_rsp_header_r;
    logic [31:0]           c_rsp_value_r;
    wire [BLOCK_SIZE-1:0]  c_rsp_take;

    // -----------------------------------------------------------------------
    // Per-block transfer unit (tmem.ld/st)
    // -----------------------------------------------------------------------

    wire [BLOCK_SIZE-1:0] is_xfer;
    wire [BLOCK_SIZE-1:0] xfer_fire;

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_xf
        q_op_t op;
        assign op = q_op_r[b];
        `UNUSED_VAR (op)
        // the transfer: a tmem.ld in the slot, or a tmem.st at the input
        // (the slot then empty)
        wire op_ld  = q_valid_r[b] && (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_LD));
        wire op_st  = st_valid[b];
        assign is_xfer[b] = op_ld;
        wire         x_in     = ~q_valid_r[b];
        tcu_header_t x_header;
        assign x_header = x_in ? op_data[b].header : op.header;
        wire [31:0]  x_s1     = x_in ? op_data[b].rs1_data[0][31:0] : op.s1;

        // taddr = {lane [31:16], column [15:0]}: the block's first lane.
        wire [15:0]     lane  = x_s1[31:16];
        wire [COLW-1:0] col0  = x_s1[15:0];
        wire [4:0]      r     = x_in ? op_data[b].op_args.tcu.fmt_d : op.fmt_d;
        wire            frag  = x_in ? op_data[b].op_args.tcu.a_from_smem : op.frag;   // fragment shape
        `RUNTIME_ASSERT (~(op_ld || op_st) || (32'(lane) == b * XM && (32'(col0) % TC_N) == 0),
            ("%t: %s: tmem transfer of lane %0d column %0d from a warp of block %0d", $time, INSTANCE_ID, lane, col0, b))
        // A transfer is warp-wide, as tcgen05.ld/st are .sync.aligned: its
        // address is thread 0's, and a store writes every thread's register.
        `RUNTIME_ASSERT (~(op_ld || op_st) || (x_header.tmask == '1),
            ("%t: %s: tmem transfer with inactive threads (tmask=0x%0h)", $time, INSTANCE_ID, x_header.tmask))

        // Lane per thread: register r is group (col0 + r*TPL) / TC_N of both
        // RAMs; thread t takes lane t % XM, word ((r*TPL + t/XM) % TC_N).
        wire [GW-1:0] tp_g = GW'((32'(col0) + 32'(r) * TPL) / TC_N);
        wire [`UP(`CLOG2(RG))-1:0] tp_sel = `UP(`CLOG2(RG))'(32'(r) % RG);
        // Fragment: register r is the row of M step r % M_STEPS, group
        // col0 / TC_N + r / M_STEPS, thread t its word t (the layout of the
        // WGMMA accumulator registers).
        wire [`UP(`CLOG2(M_STEPS))-1:0] fm = `UP(`CLOG2(M_STEPS))'(32'(r) % M_STEPS);
        wire [GW-1:0] fg = GW'(32'(col0) / TC_N + 32'(r) / M_STEPS);

        // response slot
        logic rsp_valid_r;
        tcu_header_t rsp_header_r;
        logic rsp_ld_r;                          // the response is a load's: its data is
                                                 // the RAM outputs, held by no other read
        // a load's word source, encoded once: RAM m for a fragment
        // (0..M_STEPS-1), else M_STEPS + the lane-per-thread select
        localparam RSRC  = M_STEPS + RG;
        localparam RSRCW = `UP(`CLOG2(RSRC));
        logic [RSRCW-1:0] rd_src_r;
        wire rsp_fire = rsp_valid_r && rsp_ready[b];
        assign xf_hold[b] = rsp_valid_r && rsp_ld_r && ~rsp_ready[b];
        wire rsp_free = (~rsp_valid_r || rsp_fire) && ~(c_rsp_valid_r && c_rsp_blk_r == BLKW'(b));

        // A step's D write-back and a tc.cp row have the write port first.
        wire wr_blocked = job_writing || cp_wr_blk[b];

        // one read or write per register, then the response
        wire rd_go = op_ld && rsp_free && ~e_cwait;
        wire wr_go = op_st && rsp_free && ~wr_blocked;
        assign xfer_fire[b] = rd_go || wr_go;
        assign st_take[b]   = wr_go;

        assign xf_rd[b]    = rd_go;
        assign xf_rd_m[b]  = frag ? (2'b01 << fm) : 2'b11;
        assign xf_rd_g[b]  = frag ? fg : tp_g;
        assign xf_wr[b]    = wr_go;
        assign xf_wr_m[b]  = frag ? (2'b01 << fm) : 2'b11;
        assign xf_wr_g[b]  = frag ? fg : tp_g;
        assign xf_wr_frag[b] = frag;

        // write data and enables
        for (genvar m = 0; m < M_STEPS; ++m) begin : g_wm
            for (genvar i = 0; i < TC_M; ++i) begin : g_wi
                for (genvar j = 0; j < TC_N; ++j) begin : g_wj
                    localparam W = i * TC_N + j;
                    localparam L = m * TC_M + i;          // lane in the block
                    // lane per thread: the thread holding (L, j) for register r
                    // is L + XM * d, d = (j - r*TPL) mod TC_N, if d < TPL. The
                    // data is not gated by the enable: with one candidate
                    // thread per word (NT=32) it is plain wiring.
                    logic tp_en;
                    logic [31:0] tp_data;
                    always_comb begin
                        tp_en   = 1'b0;
                        tp_data = '0;
                        for (int s = 0; s < RG; ++s) begin
                            automatic int d = (j - s * TPL + TC_N) % TC_N;
                            if (d < TPL) begin
                                if (~tp_en || int'(tp_sel) == s) tp_data = op_data[b].rs2_data[L + XM * d][31:0];
                                if (int'(tp_sel) == s) tp_en = 1'b1;
                            end
                        end
                    end
                    assign xf_wr_en[b][m][W]   = frag ? 1'b1 : tp_en;
                    assign xf_wd_frag[b][m][W] = op_data[b].rs2_data[W][31:0];
                    assign xf_wd_lpt[b][m][W]  = tp_data;
                end
            end
        end

        // read data at the RAM outputs: thread t's word is one of RSRC
        // candidates, picked by one encoded select (with M_STEPS = RG = 2, a
        // 4-way select: one 6-input LUT per bit)
        logic [NT-1:0][31:0] rd_value;
        always_comb begin
            for (int t = 0; t < NT; ++t) begin
                rd_value[t] = ram_rdata[b][0][t];
                for (int mm = 1; mm < M_STEPS; ++mm) begin
                    if (int'(rd_src_r) == mm) rd_value[t] = ram_rdata[b][mm][t];
                end
                for (int s = 0; s < RG; ++s) begin
                    if (int'(rd_src_r) == M_STEPS + s) begin
                        rd_value[t] = ram_rdata[b][(t % XM) / TC_M][((t % XM) % TC_M) * TC_N + (s * TPL + t / XM) % TC_N];
                    end
                end
            end
        end

        always_ff @(posedge clk) begin
            if (reset) begin
                rsp_valid_r <= 1'b0;
            end else begin
                if (rsp_fire) rsp_valid_r <= 1'b0;
                if (xfer_fire[b]) rsp_valid_r <= 1'b1;
            end
            if (xfer_fire[b]) begin
                rsp_header_r <= x_header;
                rsp_ld_r     <= op_ld;
            end
            if (rd_go) begin
                rd_src_r <= frag ? RSRCW'(fm) : RSRCW'(32'(M_STEPS) + 32'(tp_sel));
            end
        end

        // a load answers from the RAM outputs, which hold until it is taken
        wire [NT-1:0][31:0] payload = rd_value;
        // the response: this unit's, else a central-port one for this block
        wire c_here = c_rsp_valid_r && (c_rsp_blk_r == BLKW'(b)) && ~rsp_valid_r;
        assign c_rsp_take[b] = c_here && rsp_ready[b];
        assign rsp_valid[b] = rsp_valid_r || c_here;
        tcu_result_t rd;
        always_comb begin
            rd        = '0;
            rd.header = rsp_valid_r ? rsp_header_r : c_rsp_header_r;
            // A central-port result (tmem.alloc's column) goes to thread 0
            // only; the other threads' words are undefined then.
            for (int t = 0; t < NT; ++t) begin
                if (t == 0 && ~rsp_valid_r) begin
                    rd.data[t] = `VX_CFG_XLEN'(c_rsp_value_r);
                end else if (`VX_CFG_XLEN > 32) begin
                    rd.data[t] = `VX_CFG_XLEN'({32'hffffffff, payload[t]});
                end else begin
                    rd.data[t] = `VX_CFG_XLEN'(payload[t]);
                end
            end
        end
        assign rsp_data[b] = rd;
    end

    // -----------------------------------------------------------------------
    // Scalar port: tc.mma, tc.cp, tc.commit, tmem.alloc/dealloc (from thread 0)
    // -----------------------------------------------------------------------

    // Allocation: one bit per ACOLS-column chunk and its owner CTA; per size
    // class, whether a free aligned run exists and where the lowest starts.
    logic [ACHUNKS-1:0] amap_r;
    logic [ACHUNKS-1:0][NCTA_WIDTH-1:0] aown_r;
    wire [NCLS-1:0] cls_avail;
    wire [NCLS-1:0][ACW-1:0] cls_first;
    for (genvar c = 0; c < NCLS; ++c) begin : g_cls
        localparam SZ = 1 << c;
        localparam NR = ACHUNKS / SZ;
        wire [NR-1:0] run_free;
        for (genvar k = 0; k < NR; ++k) begin : g_run
            assign run_free[k] = (amap_r[k * SZ +: SZ] == '0);
        end
        wire [`LOG2UP(NR)-1:0] run_idx;
        VX_priority_encoder #(
            .N (NR)
        ) run_pe (
            .data_in    (run_free),
            `UNUSED_PIN (onehot_out),
            .index_out  (run_idx),
            .valid_out  (cls_avail[c])
        );
        assign cls_first[c] = ACW'(32'(run_idx) * SZ);
    end
    function automatic logic [CLSW-1:0] size_class(input logic [31:0] nck);
        size_class = '0;
        for (int c = 0; c < NCLS; ++c) begin
            if (nck == (32'(1) << c)) size_class = CLSW'(c);
        end
    endfunction

    // op queue: the payload, written once at push, sits in a LUTRAM FIFO; the
    // commits (the low fields), which a later tc.commit can add to the
    // youngest entry, are registers alongside it.
    typedef struct packed {
        logic [1:0]             ncommit;
        logic [1:0][BAR_ADDR_W-1:0] bar;
        logic [1:0][BAR_SIZE_W-1:0] cnt;
    } tc_cm_t;
    typedef struct packed {
        logic                   is_cp;
        logic                   is_marker;     // commits only
        logic [NCTA_WIDTH-1:0]  cta;
        logic [NW_WIDTH-1:0]    wid;
        logic [UUID_WIDTH-1:0]  uuid;
        logic [31:0]            w1;            // tc.mma idesc / tc.cp {cols, taddr}
        logic [31:0]            w2;            // A / tc.cp source
        logic [31:0]            w3;            // B
        logic [1:0]             ncommit;
        logic [1:0][BAR_ADDR_W-1:0] bar;
        logic [1:0][BAR_SIZE_W-1:0] cnt;       // arrival count - 1
    } tc_op_t;

    localparam CMW  = $bits(tc_cm_t);
    localparam PAYW = $bits(tc_op_t) - CMW;
    tc_cm_t           opq_cm_r [OPQ];
    logic [OPQW-1:0]  opq_head_r, opq_tail_r;
    logic [OPQW:0]    opq_count_r;
    wire              opq_pop;
    tc_op_t           run_op_r;
    logic             run_valid_r;
    wire [OPQW-1:0]   opq_last = opq_tail_r - OPQW'(1);

    // The youngest op not out of the engine, which a commit rides on: queued,
    // or a running tc.mma short of its last step (one that may leave this
    // cycle has its commits already being posted).
    wire run_leaves;
    wire run_open;
    wire young_in_q  = (opq_count_r != 0);
    wire young_v     = young_in_q || run_open;
    wire [1:0] young_ncommit = young_in_q ? opq_cm_r[opq_last].ncommit : run_op_r.ncommit;

    // A commit waits while the only queued entry leaves (starts, or posts as
    // a marker): that entry is read out with the commits it had, so one
    // attached in the same cycle would be lost.
    wire q_sole_leaves = opq_pop && (opq_count_r == (OPQW+1)'(1));

    logic [BLOCK_SIZE-1:0] c_elig;
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_celig
        q_op_t op;
        assign op = q_op_r[b];
        `UNUSED_VAR (op)
        wire is_mma   = (op.op_type == INST_OP_BITS'(INST_TCU_TC_MMA));
        wire is_cp    = (op.op_type == INST_OP_BITS'(INST_TCU_TC_CP));
        wire is_cmt   = (op.op_type == INST_OP_BITS'(INST_TCU_TC_COMMIT));
        wire is_alloc = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_ALLOC));
        wire is_free  = (op.op_type == INST_OP_BITS'(INST_TCU_TMEM_DEALLOC));
        // an allocation waits, off the port, until a run of its size is free
        wire fits = cls_avail[q_cls_r[b]];
        wire q_ok = ((is_mma || is_cp) && (opq_count_r < OPQ))
                 || (is_cmt && ~q_sole_leaves && ((young_v && young_ncommit != 2'd2) || (opq_count_r < OPQ)))
                 || (is_alloc && fits)
                 || is_free;
        assign c_elig[b] = q_valid_r[b] && ~is_xfer[b] && q_ok && (~c_rsp_valid_r || (|c_rsp_take));
    end

    wire [BLKW-1:0] c_sel;
    wire            c_go;
    VX_priority_encoder #(
        .N (BLOCK_SIZE)
    ) c_pe (
        .data_in    (c_elig),
        `UNUSED_PIN (onehot_out),
        .index_out  (c_sel),
        .valid_out  (c_go)
    );
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_qfire
        assign q_fire[b] = xfer_fire[b] || (c_go && c_sel == BLKW'(b));
    end

    q_op_t cop;
    assign cop = q_op_r[c_sel];
    `UNUSED_VAR (cop)
    wire c_mma   = (cop.op_type == INST_OP_BITS'(INST_TCU_TC_MMA));
    wire c_cp    = (cop.op_type == INST_OP_BITS'(INST_TCU_TC_CP));
    wire c_cmt   = (cop.op_type == INST_OP_BITS'(INST_TCU_TC_COMMIT));
    wire c_alloc = (cop.op_type == INST_OP_BITS'(INST_TCU_TMEM_ALLOC));
    wire c_free  = (cop.op_type == INST_OP_BITS'(INST_TCU_TMEM_DEALLOC));
    wire [31:0] cs1 = cop.s1;
    wire [31:0] cs2 = cop.s2;
    wire [31:0] cs3 = cop.s3;
    `RUNTIME_ASSERT (~c_go || cop.header.tmask[0], ("%t: %s: tensor-memory op without thread 0", $time, INSTANCE_ID))

    // allocation: the lowest free run aligned to its size
    wire [31:0] c_nck = (c_alloc ? cs1 : cs2) / ACOLS;
    wire [ACHUNKS-1:0] c_amask = q_amask_r[c_sel];
    wire [ACW-1:0] afit_idx = cls_first[q_cls_r[c_sel]];
    wire [ACW-1:0] free_idx = ACW'(cs1 / ACOLS);
    logic free_owned;
    always_comb begin
        free_owned = ((cs1 % (c_nck * ACOLS)) == 0) && ((amap_r & (c_amask << free_idx)) == (c_amask << free_idx));
        for (int i = 0; i < ACHUNKS; ++i) begin
            if (1'(((c_amask << free_idx) >> i) & ACHUNKS'(1))) begin
                free_owned &= (aown_r[i] == cop.header.cta_id);
            end
        end
    end
    `RUNTIME_ASSERT (~(c_go && (c_alloc || c_free)) || (c_nck != 0 && c_nck <= ACHUNKS && (c_nck & (c_nck - 1)) == 0),
        ("%t: %s: TMEM allocation of %0d columns", $time, INSTANCE_ID, c_alloc ? cs1 : cs2))
    `RUNTIME_ASSERT (~(c_go && c_free) || free_owned,
        ("%t: %s: TMEM dealloc of columns the CTA does not own", $time, INSTANCE_ID))

    wire c_attach = c_go && c_cmt && young_v && (young_ncommit != 2'd2);
    wire c_push   = c_go && (c_mma || c_cp || (c_cmt && ~c_attach));

    tc_op_t c_entry;
    always_comb begin
        c_entry            = '0;
        c_entry.is_cp      = c_cp;
        c_entry.is_marker  = c_cmt;
        c_entry.cta        = cop.header.cta_id;
        c_entry.wid        = cop.header.wid;
        c_entry.uuid       = cop.header.uuid;
        c_entry.w1         = cs1;
        c_entry.w2         = cs2;
        c_entry.w3         = cs3;
        c_entry.ncommit    = c_cmt ? 2'd1 : 2'd0;
        c_entry.bar[0]     = bar_addr(cs1);
        c_entry.cnt[0]     = BAR_SIZE_W'(cs2 - 32'd1);
    end

    // -----------------------------------------------------------------------
    // Engine
    // -----------------------------------------------------------------------

    // step counters and posted commits
    logic [SCNTW-1:0] steps_issued_r, steps_done_r;
    logic [BAR_ADDR_W-1:0] cmq_bar_r [CMQ];
    logic [BAR_SIZE_W-1:0] cmq_cnt_r [CMQ];
    logic [SCNTW-1:0]      cmq_tgt_r [CMQ];
    logic [CMQW-1:0] cmq_head_r, cmq_tail_r;
    logic [CMQW:0]   cmq_count_r;

    // decoded running op
    tc_op_t rj;
    assign rj = run_op_r;
    wire [4:0]      r_fmt_s   = rj.w1[4:0];
    wire            r_sparse  = rj.w1[5];
`ifdef VX_CFG_TCU_MX_ENABLE
    wire            r_mx      = tcu_fmt_is_mx(TCU_FMT_WIDTH'(rj.w1[4:0]));
`else
    wire            r_mx      = 1'b0;
`endif
    wire            r_a_tmem  = rj.w1[6];
    wire            r_acc_init= rj.w1[7];
    wire [7:0]      r_n_last  = {1'b0, rj.w1[14:8]};     // n_frag - 1
    wire [GW-1:0]   r_d_g     = GW'(rj.w1[22:15]);
    wire [GW-1:0]   r_side_g  = GW'(rj.w1[31:23] / TC_N);
    wire [GW-1:0]   r_a_g     = GW'(rj.w2 / TC_N);
    wire [17:0]     r_a_off   = 18'(rj.w2[TCU_SMEM_DESC_OFF_BITS-1:0]);
    wire [15:0]     r_lda     = 16'({rj.w2[31:TCU_SMEM_DESC_OFF_BITS], 2'b00});
    wire [17:0]     r_b_off   = 18'(rj.w3[TCU_SMEM_DESC_OFF_BITS-1:0]);
    wire [15:0]     r_ldb     = 16'({rj.w3[31:TCU_SMEM_DESC_OFF_BITS], 2'b00});
    // tc.cp
    wire [GW-1:0]   cp_g0     = GW'(rj.w1[15:0] / TC_N);
    wire [15:0]     cp_rows   = 16'((32'(rj.w1[31:16]) * LANES) / CP_ROW_WORDS);
    wire [17:0]     cp_src    = 18'(rj.w2);

    // walk state
    logic             run_go_r;       // the descriptors are set; prologue and steps follow
    logic             run_m_r;
    logic [7:0]       run_n_r;
    logic [3:0]       pro_idx_r;

    // step register E
    logic             e_valid_r;
    logic             e_m_r;
    logic [7:0]       e_sn_r;
    logic             e_first_r;
    logic             e_zero_r;
    logic [GW-1:0]    e_g_r;
    logic [UUID_WIDTH-1:0] e_uuid_r;
    logic [NW_WIDTH-1:0]   e_wid_r;
    logic [NCTA_WIDTH-1:0] e_cta_r;
    logic [4:0]       e_fmt_s_r;
    logic [15:0]      e_lda_r, e_ldb_r;
    logic             e_a_tmem_r, e_sparse_r;
    logic [BLOCK_SIZE-1:0][17:0] e_a_off_r;
    logic [17:0]      e_b_off_r;

    assign job_fire = e_valid_r && (&core_ready);
    // a presented step's C is at the RAM outputs until it is taken
    assign e_cwait  = e_valid_r && ~e_zero_r && ~(&core_ready);
    wire e_advance  = ~e_valid_r || job_fire;

    // A transfer read on any block, or a load response waiting at a block's
    // RAM outputs, takes the cycle from the engine's reads.
    wire xf_rd_any = (|xf_rd) || (|xf_hold);

    wire is_mma_run = run_valid_r && ~rj.is_cp;
    wire is_cp_run  = run_valid_r && rj.is_cp;

    // prologue: metadata, then A, each a read of both RAMs on all blocks
    wire       r_side    = r_sparse || r_mx;
    wire [3:0] pro_total = (r_side ? 4'd1 : 4'd0) + (r_a_tmem ? 4'(A_READS) : 4'd0);
    wire       pro_pending = is_mma_run && run_go_r && (pro_idx_r < pro_total);
    wire       pro_go   = pro_pending && ~xf_rd_any && ~e_cwait;
    wire       pro_meta = r_side && (pro_idx_r == 4'd0);
    wire [3:0] pro_a_idx = pro_idx_r - (r_side ? 4'd1 : 4'd0);
    wire [GW-1:0] pro_g = pro_meta ? r_side_g : (r_a_g + GW'(pro_a_idx));

    // MX B-scale read of a step (below)
    wire                          sfb_rd;
    wire [`UP(`CLOG2(BLOCK_SIZE))-1:0] sfb_blk;
    wire [GW-1:0]                 sfb_g;

    // in-flight write-backs, for the row hazard: (m, group) per step
    logic [WQ_DEPTH-1:0]           wb_v_r;
    logic [WQ_DEPTH-1:0][GW:0]     wb_key_r;
    logic [`LOG2UP(WQ_DEPTH)-1:0]  wb_head_r, wb_tail_r;
    wire [GW-1:0] load_g = r_d_g + GW'(run_n_r);
    logic row_hazard;
    always_comb begin
        row_hazard = 1'b0;
        for (int i = 0; i < WQ_DEPTH; ++i) begin
            if (wb_v_r[i] && wb_key_r[i] == {run_m_r, load_g}) row_hazard = 1'b1;
        end
    end

    wire load_m_last = (run_m_r == 1'(M_STEPS - 1));
    wire load_last   = load_m_last && (run_n_r == r_n_last);
    assign run_open  = is_mma_run && ~load_last;
    // the op's commits go out with its last step
    wire cmq_room2   = (cmq_count_r + (CMQW+1)'(rj.ncommit)) <= CMQ;
    wire load = e_advance && is_mma_run && run_go_r && ~pro_pending && ~xf_rd_any
             && (r_acc_init || ~row_hazard) && (~load_last || cmq_room2);

    // tc.cp: drain the write-backs, then one bank-row read at a time
    logic [15:0] cp_req_r, cp_rsp_r;
    logic        cp_inflight_r;
    wire cp_drained = (steps_issued_r == steps_done_r);
    wire cp_issue = is_cp_run && run_go_r && cp_drained && (cp_req_r < cp_rows) && ~cp_inflight_r;
    wire cp_fire  = cp_issue && cp_lmem_if.req_ready;
    wire cp_rsp   = cp_lmem_if.rsp_valid;
    wire cp_done  = is_cp_run && run_go_r && (cp_rsp_r == cp_rows) && cmq_room2;

    assign cp_lmem_if.req_valid        = cp_issue;
    assign cp_lmem_if.req_data.rw      = 1'b0;
    assign cp_lmem_if.req_data.addr    = CP_ADDRW'((cp_src + 18'(cp_req_r) * 18'(CP_ROW_WORDS * 4)) >> `CLOG2(CP_ROW_WORDS * 4));
    assign cp_lmem_if.req_data.data    = '0;
    assign cp_lmem_if.req_data.byteen  = '0;
    assign cp_lmem_if.req_data.attr    = '0;
    assign cp_lmem_if.req_data.tag.uuid  = rj.uuid;
    assign cp_lmem_if.req_data.tag.value = '0;
    assign cp_lmem_if.rsp_ready        = 1'b1;
    `UNUSED_VAR (cp_lmem_if.rsp_data.tag)

    // a response's fragment rows: blob fragment f = row * CP_FRAGS + q is
    // M step f % 2 of block (f / 2) % BLOCK_SIZE in group f / (2 * BLOCK_SIZE)
    wire [31:0] cp_f0 = 32'(cp_rsp_r) * CP_FRAGS;
    wire [GW-1:0] cp_g = cp_g0 + GW'(cp_f0 / (2 * BLOCK_SIZE));
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_cpw
        logic hit;
        always_comb begin
            hit = 1'b0;
            for (int q = 0; q < CP_FRAGS; ++q) begin
                if (((cp_f0 + 32'(q)) / 2) % BLOCK_SIZE == b) hit = 1'b1;
            end
        end
        assign cp_wr_blk[b] = cp_rsp && hit;
    end

    // the next op: when the engine is free, or its op leaves this cycle
    assign run_leaves = (load && load_last) || cp_done;
    wire eng_free   = ~run_valid_r || run_leaves;
    wire [PAYW-1:0] head_pay;
    VX_fifo_queue #(
        .DATAW  (PAYW),
        .DEPTH  (OPQ),
        .LUTRAM (1)
    ) opq_pay (
        .clk     (clk),
        .reset   (reset),
        .push    (c_push),
        .pop     (opq_pop),
        .data_in (c_entry[CMW +: PAYW]),
        .data_out(head_pay),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    tc_op_t head;
    assign head = {head_pay, opq_cm_r[opq_head_r]};
    wire head_v      = (opq_count_r != 0);
    wire head_marker = head_v && head.is_marker;
    wire marker_post = eng_free && head_marker && ((cmq_count_r + (CMQW+1)'(head.ncommit)) <= CMQ) && ~run_leaves;
    wire start       = eng_free && head_v && ~head.is_marker;
    assign opq_pop   = start || marker_post;

    assign job_valid = e_valid_r;
    `UNUSED_VAR (e_sparse_r)

    // C cells of the presented step: identity from the RAM of its M step,
    // whose outputs hold them until the step is taken (e_cwait keeps transfer
    // and prologue reads off the RAMs meanwhile).
    logic [BLOCK_SIZE-1:0][NT-1:0][31:0] e_c;
    always_comb begin
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            for (int t = 0; t < NT; ++t) begin
                e_c[b][t] = e_zero_r ? 32'd0 : ram_rdata[b][e_m_r][t];
            end
        end
    end

    // prologue responses
    logic pro_rsp_valid_r, pro_rsp_meta_r;
    logic [3:0] pro_rsp_idx_r;
    assign a_fill_valid = pro_rsp_valid_r && ~pro_rsp_meta_r;
    assign a_fill_idx   = pro_rsp_idx_r;
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_afill
        for (genvar l = 0; l < XM; ++l) begin : g_l
            for (genvar j = 0; j < TC_N; ++j) begin : g_j
                assign a_fill_data[b][l][j] = ram_rdata[b][l / TC_M][(l % TC_M) * TC_N + j];
            end
        end
    end

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Metadata, latched once per op: a lane's group words hold, from bit 0,
    // the VX_tcu_sp_mux rows of the step's lower then upper TC_K words.
    logic [BLOCK_SIZE-1:0][XM-1:0][TC_N*32-1:0] meta_r;
    always_ff @(posedge clk) begin
        if (pro_rsp_valid_r && pro_rsp_meta_r) begin
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                for (int l = 0; l < XM; ++l) begin
                    meta_r[b][l] <= ram_rdata[b][l / TC_M][(l % TC_M) * TC_N +: TC_N];
                end
            end
        end
    end

    // Rows of M step e_m_r, packed as VX_tcu_sp_mux expects for the format.
    logic [BLOCK_SIZE-1:0][TCU_MAX_META_BLOCK_WIDTH-1:0] job_meta_lo_c, job_meta_hi_c;
    always_comb begin
        job_meta_lo_c = '0;
        job_meta_hi_c = '0;
        for (int lw = 0; lw <= $clog2(TCU_MAX_ELT_RATIO); ++lw) begin
            int w;
            w = (TCU_TC_K * 2) << lw;
            if (meta_row_bits(e_fmt_s_r) == 16'(w)) begin
                for (int b = 0; b < BLOCK_SIZE; ++b) begin
                    for (int i = 0; i < TC_M; ++i) begin
                        for (int j = 0; j < w; ++j) begin
                            job_meta_lo_c[b][i * w + j] = meta_r[b][32'(e_m_r) * TC_M + i][j];
                            if (SP_ROWS > 1) begin
                                job_meta_hi_c[b][i * w + j] = meta_r[b][32'(e_m_r) * TC_M + i][w + j];
                            end
                        end
                    end
                end
            end
        end
    end
    assign job_meta    = job_meta_lo_c;
    assign job_meta_hi = job_meta_hi_c;
`endif

`ifdef VX_CFG_TCU_MX_ENABLE
    // MX scales, SF = TCU_MX_MAX_SF bytes per A row and B column of a step
    // (scale s covers the step's K elements from s * fedpK_elems / SF).
    // A: lane l, columns side + s/4, byte s%4; latched by the prologue.
    // B: byte q = col * SF + s of a dense stream of NT*4-byte rows; row r is
    // M step 1 of block r % BLOCK_SIZE at group side/TC_N + 1 + r / BLOCK_SIZE.
    // A step (m = 0, n) whose n starts a row reads it on the RAM the step
    // leaves free, and every block takes it.
    localparam SF       = TCU_MX_MAX_SF;
    localparam SF_WORDS = (SF + 3) / 4;
    localparam SFB_NPR  = (NT * 4) / (TC_N * SF);     // n groups per B-scale row
    `STATIC_ASSERT (SFB_NPR >= 1 && (SFB_NPR & (SFB_NPR - 1)) == 0, ("a B-scale row holds whole n groups"))
    `STATIC_ASSERT (SF_WORDS <= TC_N, ("a row's A scales fit one fragment group"))
    localparam NPOSW    = `UP(`CLOG2(SFB_NPR));
    localparam BLKW_SF  = `UP(`CLOG2(BLOCK_SIZE));

    wire [7:0]         sfb_row = run_n_r / 8'(SFB_NPR);
    assign sfb_rd  = r_mx && (run_m_r == 1'b0) && ((run_n_r % 8'(SFB_NPR)) == 8'd0);
    assign sfb_blk = BLKW_SF'(32'(sfb_row) % BLOCK_SIZE);
    assign sfb_g   = r_side_g + GW'(1) + GW'(32'(sfb_row) / BLOCK_SIZE);

    logic [BLOCK_SIZE-1:0][XM-1:0][SF_WORDS-1:0][31:0] sfa_r;
    logic [NT-1:0][31:0] sfb_r;
    logic                sfb_rd_r;
    logic [BLKW_SF-1:0]  sfb_blk_r;
    logic [NPOSW-1:0]    e_npos_r;
    always_ff @(posedge clk) begin
        if (pro_rsp_valid_r && pro_rsp_meta_r) begin
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                for (int l = 0; l < XM; ++l) begin
                    for (int w = 0; w < SF_WORDS; ++w) begin
                        sfa_r[b][l][w] <= ram_rdata[b][l / TC_M][(l % TC_M) * TC_N + w];
                    end
                end
            end
        end
        sfb_rd_r <= load && sfb_rd;
        if (load) begin
            sfb_blk_r <= sfb_blk;
            e_npos_r  <= NPOSW'(run_n_r % 8'(SFB_NPR));
        end
        if (sfb_rd_r) begin
            sfb_r <= ram_rdata[sfb_blk_r][1];
        end
    end
    wire [NT-1:0][31:0] sfb_cur = sfb_rd_r ? ram_rdata[sfb_blk_r][1] : sfb_r;
    wire [NT*4-1:0][7:0] sfb_bytes = sfb_cur;

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_sf
        for (genvar i = 0; i < TC_M; ++i) begin : g_a
            for (genvar sx = 0; sx < SF; ++sx) begin : g_s
                wire [SF_WORDS-1:0][3:0][7:0] row0 = sfa_r[b][i];
                wire [SF_WORDS-1:0][3:0][7:0] row1 = sfa_r[b][TC_M + i];
                assign job_sf[b][i][sx] = e_m_r ? row1[sx / 4][sx % 4] : row0[sx / 4][sx % 4];
            end
        end
        for (genvar j = 0; j < TC_N; ++j) begin : g_b
            for (genvar sx = 0; sx < SF; ++sx) begin : g_s
                assign job_sf[b][TC_M + j][sx] = sfb_bytes[(32'(e_npos_r) * TC_N + j) * SF + sx];
            end
        end
    end
`else
    assign sfb_rd  = 1'b0;
    assign sfb_blk = '0;
    assign sfb_g   = '0;
`endif

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_job
        tcu_execute_t  jd;
        tcu_tbuf_req_t tr;
        always_comb begin
            jd = '0;
            jd.header.uuid   = e_uuid_r;
            jd.header.wid    = e_wid_r;
            jd.header.cta_id = e_cta_r;
            jd.header.tmask  = '1;
            jd.header.sop    = 1'b1;
            jd.header.eop    = 1'b1;
            jd.header.wb     = 1'b1;
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            jd.op_type       = e_sparse_r ? INST_OP_BITS'(INST_TCU_WGMMA_SP) : INST_OP_BITS'(INST_TCU_WGMMA);
        `else
            jd.op_type       = INST_OP_BITS'(INST_TCU_WGMMA);
        `endif
            jd.op_args.tcu.a_from_smem  = 1'b1;
            jd.op_args.tcu.cd_nregs     = 2'd2;
            jd.op_args.tcu.fmt_s        = e_fmt_s_r;
            jd.op_args.tcu.fmt_d        = (e_fmt_s_r >= 5'(TCU_I32_ID)) ? 5'(TCU_I32_ID) : 5'(TCU_FP32_ID);
            jd.op_args.tcu.step_m       = 4'(e_m_r);
            jd.op_args.tcu.step_n       = 4'(e_sn_r);   // the core takes its low bits
            jd.op_args.tcu.is_first_uop = e_first_r;
            for (int t = 0; t < NT; ++t) begin
                jd.rs3_data[t] = `VX_CFG_XLEN'(e_c[b][t]);
            end

            tr              = '0;
            tr.valid        = e_valid_r;
            tr.uuid         = e_uuid_r;
            tr.wid          = e_wid_r;
            tr.cta_id       = e_cta_r;
            tr.step_m       = 4'(e_m_r);
            tr.step_n       = TCU_TBUF_N_W'(e_sn_r);
            tr.cd_nregs     = 2'd2;
            tr.desc_a       = `VX_CFG_XLEN'(smem_desc(e_a_off_r[b], e_lda_r));
            tr.desc_b       = `VX_CFG_XLEN'(smem_desc(e_b_off_r, e_ldb_r));
            tr.a_is_smem    = ~e_a_tmem_r;
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            tr.is_sparse    = e_sparse_r;
        `endif
            tr.is_first_uop = e_first_r;
        end
        assign job_data[b] = jd;
        assign tbuf_req[b] = tr;
    end

    // D write-back queue per block: the fragment's M step and group
    wire [BLOCK_SIZE-1:0][GW:0] wq_head;
    wire [BLOCK_SIZE-1:0]       wq_full;
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_wq
        VX_fifo_queue #(
            .DATAW   (GW + 1),
            .DEPTH   (WQ_DEPTH),
            .LUTRAM  (1)
        ) wq (
            .clk     (clk),
            .reset   (reset),
            .push    (job_fire),
            .pop     (jres_valid[b]),
            .data_in ({e_m_r, e_g_r}),
            .data_out(wq_head[b]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            .full    (wq_full[b]),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
        `RUNTIME_ASSERT (~(job_fire && wq_full[b]), ("%t: %s: step write queue overflow", $time, INSTANCE_ID))
    end
    // blocks run in lockstep: their write-backs land together
    assign job_writing = jres_valid[0];
    `UNUSED_VAR (jres_valid)

    // -----------------------------------------------------------------------
    // Port control
    // -----------------------------------------------------------------------

    // The write data is a 4-way select per RAM, its select encoded once
    // per RAM, so each data bit is one 6-input LUT: D write-back, a tc.cp
    // row, or a store in either shape.
    localparam WS_WB = 2'd0, WS_CP = 2'd1, WS_FRAG = 2'd2, WS_LPT = 2'd3;
    logic [BLOCK_SIZE-1:0][M_STEPS-1:0][1:0]          ram_wsel;
    logic [BLOCK_SIZE-1:0][M_STEPS-1:0][NT-1:0][31:0] cp_wdata;

    always_comb begin
        ram_read  = '0;
        ram_raddr = '0;
        ram_write = '0;
        ram_wren  = '0;
        ram_waddr = '0;
        ram_wsel  = '0;
        cp_wdata  = '0;
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            for (int m = 0; m < M_STEPS; ++m) begin
                // read: a transfer, else the prologue, else the step's C
                if (xf_rd[b]) begin
                    ram_read[b][m]  = xf_rd_m[b][m];
                    ram_raddr[b][m] = xf_rd_g[b];
                end else if (pro_go) begin
                    ram_read[b][m]  = 1'b1;
                    ram_raddr[b][m] = pro_g;
                end else if (load && ~r_acc_init && (32'(run_m_r) == m)) begin
                    ram_read[b][m]  = 1'b1;
                    ram_raddr[b][m] = load_g;
                end else if (load && sfb_rd && (32'(sfb_blk) == b) && (m == 1)) begin
                    // MX: a B-scale row, on the RAM the M-step-0 step leaves free
                    ram_read[b][m]  = 1'b1;
                    ram_raddr[b][m] = sfb_g;
                end
                // tc.cp: the fragment row of the LMEM bank row bound for this
                // RAM (with one fragment per bank row, plain wiring)
                for (int q = 0; q < CP_FRAGS; ++q) begin
                    if (q == 0 || (((cp_f0 + 32'(q)) / 2) % BLOCK_SIZE == b && (cp_f0 + 32'(q)) % 2 == m)) begin
                        cp_wdata[b][m] = cp_lmem_if.rsp_data.data[q * NT * 32 +: NT * 32];
                    end
                end
                // write: D write-back, else a tc.cp row, else a transfer;
                // the data select is a store's when nothing writes
                ram_wsel[b][m] = xf_wr_frag[b] ? WS_FRAG : WS_LPT;
                if (jres_valid[b] && (32'(wq_head[b][GW]) == m)) begin
                    ram_write[b][m] = 1'b1;
                    ram_wren[b][m]  = '1;
                    ram_waddr[b][m] = wq_head[b][GW-1:0];
                    ram_wsel[b][m]  = WS_WB;
                end else if (cp_wr_blk[b]) begin
                    ram_wsel[b][m] = WS_CP;
                    for (int q = 0; q < CP_FRAGS; ++q) begin
                        if (((cp_f0 + 32'(q)) / 2) % BLOCK_SIZE == b && (cp_f0 + 32'(q)) % 2 == m) begin
                            ram_write[b][m] = 1'b1;
                            ram_wren[b][m]  = '1;
                            ram_waddr[b][m] = cp_g;
                        end
                    end
                end else if (xf_wr[b] && xf_wr_m[b][m]) begin
                    ram_write[b][m] = 1'b1;
                    ram_wren[b][m]  = xf_wr_en[b][m];
                    ram_waddr[b][m] = xf_wr_g[b];
                end
            end
        end
    end

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_wd
        for (genvar m = 0; m < M_STEPS; ++m) begin : g_wdm
            wire [1:0] ws = ram_wsel[b][m];
            for (genvar w = 0; w < NT; ++w) begin : g_wdw
                assign ram_wdata[b][m][w] = ws[1] ? (ws[0] ? xf_wd_lpt[b][m][w] : xf_wd_frag[b][m][w])
                                                  : (ws[0] ? cp_wdata[b][m][w]  : jres_data[b][w]);
            end
        end
    end

    // -----------------------------------------------------------------------
    // Commit arrivals
    // -----------------------------------------------------------------------

    wire [SCNTW-1:0] cmq_head_tgt = cmq_tgt_r[cmq_head_r];
    wire [SCNTW-1:0] cmq_slack = steps_done_r - cmq_head_tgt;   // >= 0 once its steps are done
    wire cm_ready = (cmq_count_r != 0) && ~cmq_slack[SCNTW-1];
    tcu_bar_t bar_req;
    always_comb begin
        bar_req         = '0;
        bar_req.addr    = cmq_bar_r[cmq_head_r];
        bar_req.size_m1 = cmq_cnt_r[cmq_head_r];
    end
    assign tcu_bar_if.req_valid = cm_ready;
    assign tcu_bar_if.req_data  = bar_req;
    wire cm_fire = cm_ready && tcu_bar_if.req_ready;

    // the commits posted this cycle: by the op leaving the engine, or a marker
    wire [1:0] post_n = run_leaves ? rj.ncommit : (marker_post ? head.ncommit : 2'd0);
    tc_op_t post_src;
    assign post_src = run_leaves ? rj : head;
    `UNUSED_VAR (post_src)
    // the step leaving with load_last counts toward its own commits
    wire [SCNTW-1:0] post_tgt = steps_issued_r + SCNTW'(load && load_last);

    // -----------------------------------------------------------------------
    // State update
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            opq_head_r     <= '0;
            opq_tail_r     <= '0;
            opq_count_r    <= '0;
            run_valid_r    <= 1'b0;
            run_go_r       <= 1'b0;
            e_valid_r      <= 1'b0;
            pro_idx_r      <= '0;
            pro_rsp_valid_r<= 1'b0;
            steps_issued_r <= '0;
            steps_done_r   <= '0;
            cmq_head_r     <= '0;
            cmq_tail_r     <= '0;
            cmq_count_r    <= '0;
            wb_v_r         <= '0;
            wb_head_r      <= '0;
            wb_tail_r      <= '0;
            amap_r         <= '0;
            c_rsp_valid_r  <= 1'b0;
            cp_req_r       <= '0;
            cp_rsp_r       <= '0;
            cp_inflight_r  <= 1'b0;
        end else begin
            // central port
            if (c_go) begin
                c_rsp_valid_r <= 1'b1;
            end else if (|c_rsp_take) begin
                c_rsp_valid_r <= 1'b0;
            end
            if (c_go && c_alloc) begin
                amap_r <= amap_r | (c_amask << afit_idx);
                for (int i = 0; i < ACHUNKS; ++i) begin
                    if (1'(((c_amask << afit_idx) >> i) & ACHUNKS'(1))) aown_r[i] <= cop.header.cta_id;
                end
            end else if (c_go && c_free) begin
                amap_r <= amap_r & ~(c_amask << free_idx);
            end else if (cta_exit_valid) begin
                for (int i = 0; i < ACHUNKS; ++i) begin
                    if (aown_r[i] == cta_exit_id) amap_r[i] <= 1'b0;
                end
            end

            // op queue
            if (c_push) begin
                opq_cm_r[opq_tail_r] <= c_entry[CMW-1:0];
                opq_tail_r <= opq_tail_r + OPQW'(1);
            end
            if (c_attach) begin
                if (young_in_q) begin
                    opq_cm_r[opq_last].bar[young_ncommit[0]] <= bar_addr(cs1);
                    opq_cm_r[opq_last].cnt[young_ncommit[0]] <= BAR_SIZE_W'(cs2 - 32'd1);
                    opq_cm_r[opq_last].ncommit <= young_ncommit + 2'd1;
                end else begin
                    run_op_r.bar[young_ncommit[0]] <= bar_addr(cs1);
                    run_op_r.cnt[young_ncommit[0]] <= BAR_SIZE_W'(cs2 - 32'd1);
                    run_op_r.ncommit <= young_ncommit + 2'd1;
                end
            end
            if (opq_pop) begin
                opq_head_r <= opq_head_r + OPQW'(1);
            end
            opq_count_r <= opq_count_r + (OPQW+1)'(c_push) - (OPQW+1)'(opq_pop);

            // posted commits
            if (post_n != 0) begin
                for (int c = 0; c < 2; ++c) begin
                    if (c < int'(post_n)) begin
                        cmq_bar_r[cmq_tail_r + CMQW'(c)] <= post_src.bar[c];
                        cmq_cnt_r[cmq_tail_r + CMQW'(c)] <= post_src.cnt[c];
                        cmq_tgt_r[cmq_tail_r + CMQW'(c)] <= post_tgt;
                    end
                end
                cmq_tail_r <= cmq_tail_r + CMQW'(post_n);
            end
            if (cm_fire) begin
                cmq_head_r <= cmq_head_r + CMQW'(1);
            end
            cmq_count_r <= cmq_count_r + (CMQW+1)'(post_n) - (CMQW+1)'(cm_fire);

            // steps and their write-backs
            if (job_fire) begin
                steps_issued_r <= steps_issued_r + SCNTW'(1);
                wb_v_r[wb_tail_r]   <= 1'b1;
                wb_key_r[wb_tail_r] <= {e_m_r, e_g_r};
                wb_tail_r <= wb_tail_r + 1;
            end
            if (jres_valid[0]) begin
                steps_done_r <= steps_done_r + SCNTW'(1);
                wb_v_r[wb_head_r] <= 1'b0;
                wb_head_r <= wb_head_r + 1;
            end

            // prologue
            pro_rsp_valid_r <= pro_go;
            if (pro_go) begin
                pro_idx_r      <= pro_idx_r + 4'd1;
                pro_rsp_meta_r <= pro_meta;
                pro_rsp_idx_r  <= pro_a_idx;
            end

            // step register
            if (load) begin
                e_valid_r      <= 1'b1;
                e_m_r          <= run_m_r;
                e_sn_r         <= run_n_r;
                e_g_r          <= load_g;
                e_uuid_r       <= rj.uuid;
                e_wid_r        <= rj.wid;
                e_cta_r        <= rj.cta;
                e_fmt_s_r      <= r_fmt_s;
                e_lda_r        <= r_lda;
                e_ldb_r        <= r_ldb;
                e_a_tmem_r     <= r_a_tmem;
                e_sparse_r     <= r_sparse;
                for (int b = 0; b < BLOCK_SIZE; ++b) begin
                    e_a_off_r[b] <= r_a_off + 18'(32'(b) * XM * 32'(r_lda));
                end
                e_b_off_r      <= r_b_off;
                e_first_r      <= (run_m_r == 1'b0) && (run_n_r == 8'd0);
                e_zero_r       <= r_acc_init;
            end else if (job_fire) begin
                e_valid_r <= 1'b0;
            end

            // walk m (inner), then n
            if (load) begin
                if (~load_m_last) begin
                    run_m_r <= 1'b1;
                end else begin
                    run_m_r <= 1'b0;
                    if (~load_last) begin
                        run_n_r <= run_n_r + 8'd1;
                    end
                end
            end

            // tc.cp
            if (cp_fire) begin
                cp_req_r      <= cp_req_r + 16'd1;
                cp_inflight_r <= 1'b1;
            end else if (cp_rsp) begin
                cp_inflight_r <= 1'b0;
            end
            if (cp_rsp) begin
                cp_rsp_r <= cp_rsp_r + 16'd1;
            end

            // the engine
            if (run_leaves) begin
                run_valid_r <= 1'b0;
                run_go_r    <= 1'b0;
            end
            if (run_valid_r && ~run_go_r) begin
                run_go_r <= 1'b1;   // the descriptors take a cycle
            end
            if (start) begin
                run_valid_r <= 1'b1;
                run_go_r    <= 1'b0;
                run_op_r    <= head;
                run_m_r     <= 1'b0;
                run_n_r     <= 8'd0;
                pro_idx_r   <= 4'd0;
                cp_req_r    <= '0;
                cp_rsp_r    <= '0;
            end
        end
        if (c_go) begin
            c_rsp_blk_r    <= c_sel;
            c_rsp_header_r <= cop.header;
            c_rsp_value_r  <= c_alloc ? (32'(afit_idx) * ACOLS) : 32'd0;
        end
    end

    // -----------------------------------------------------------------------
    // Descriptor checks
    // -----------------------------------------------------------------------

    wire [31:0] c_d_end  = 32'(cs1[22:15]) * TC_N + (32'(cs1[14:8]) + 1) * TC_N;
    wire mma_bad = 1'b0
                `ifndef VX_CFG_TCU_SPARSE_ENABLE
                    || cs1[5]
                `endif
                    || (c_d_end > COLS)
                    || (cs1[5] && ((32'(cs1[31:23]) % TC_N) != 0 || 32'(cs1[31:23]) + TC_N > COLS))
                    || (cs1[6] && ((cs2 % TC_N) != 0 || cs2 + TCU_WG_FEDP_K > COLS))
                    || (~cs1[6] && cs2[31:TCU_SMEM_DESC_OFF_BITS] == 0)
                    || (cs3[31:TCU_SMEM_DESC_OFF_BITS] == 0);
    wire cp_bad = (cs1[31:16] == 0) || (32'(cs1[31:16]) % TC_N != 0) || (32'(cs1[15:0]) % TC_N != 0)
               || (cs2 % (CP_ROW_WORDS * 4) != 0) || ((32'(cs1[31:16]) * LANES) % CP_ROW_WORDS != 0)
               || (32'(cs1[15:0]) + 32'(cs1[31:16]) > COLS);
    `RUNTIME_ASSERT (~(c_push && (c_mma ? mma_bad : (c_cp ? cp_bad : 1'b0))),
        ("%t: %s: unsupported tensor op (cta=%0d)", $time, INSTANCE_ID, cop.header.cta_id))

`ifdef PERF_ENABLE
    // Each cycle counts once: an MMA step, else a running op's operand wait,
    // else posted commits or a copy, else idle.
    wire perf_is_opnd  = ~job_fire && (is_mma_run || e_valid_r);
    wire perf_is_other = ~job_fire && ~perf_is_opnd && ((cmq_count_r != 0) || is_cp_run);
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
            perf_xfers_r <= perf_xfers_r + PERF_CTR_BITS'($countones(xfer_fire));
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
        if (c_push && c_mma)
            `TRACE(3, ("%t: %s tc-mma: cta=%0d, idesc=0x%0h, a=0x%0h, b=0x%0h\n",
                $time, INSTANCE_ID, cop.header.cta_id, cs1, cs2, cs3))
        if (c_push && c_cp)
            `TRACE(3, ("%t: %s tc-cp: cta=%0d, cols=%0d, taddr=%0d\n",
                $time, INSTANCE_ID, cop.header.cta_id, cs1[31:16], cs1[15:0]))
        if (cm_fire)
            `TRACE(3, ("%t: %s tc-commit: addr=%0d\n", $time, INSTANCE_ID, bar_req.addr))
        if (start)
            `TRACE(3, ("%t: %s tc-run: cta=%0d\n", $time, INSTANCE_ID, head.cta))
    end
`endif

endmodule

`endif // VX_CFG_TCU_TMEM_ENABLE
