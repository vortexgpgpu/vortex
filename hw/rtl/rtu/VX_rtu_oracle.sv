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

// VX_rtu_oracle — the visit-order oracle: of two opaque triangle hits within
// the near window of each other (VX_rtu_near_t), which one the source BVH's
// traversal keeps. The source traversal (the Vulkan reference) walks its
// binary tree depth-first, nearer child box first, child 0 on equal entry
// distances, tests each box in F32 against the hit committed when the box's
// parent is visited, and commits only a strictly nearer hit. So of two such
// hits the first-reached wins at equal t, and a nearer hit reached second is
// lost when a box on its own path, below where the two paths part, enters at
// or past the first one's t. The driver lays the source trees out as
// visit-order tables (TLAS in the scene, BLAS tables reached by scene offset):
//
//   TLAS { n_leaves, n_nodes, nodes_off, leaf_stride }
//     leaf[i] @ +16 + i*stride : { parent << 1 | side, _, _, _, world->object 3x4 }
//     node[j] @ +nodes_off + j*64 : { child box 0, child box 1, parent << 1 | side, depth }
//   BLAS { n_nodes, _, nodes_off, 32 }
//     node[j] @ +nodes_off + j*32 : { own box, parent << 1 | side, depth }
//
// a triangle's box being its vertices' F32 min/max. Both leaves climb to
// their lowest common ancestor, every box on the way tested against the other
// hit's t; the two child boxes under it are ordered by their entry distance.
// Across instances the TLAS is climbed with the world ray, and a lost-by-cull
// verdict also climbs the nearer hit's own BLAS path with the reference's
// object-space ray, rebuilt from the table's matrix in its own op order.
//
// One request at a time; the requesting context parks until `done`. The
// engine is a small sequencer around one F32 add/mul unit (IEEE, subnormals),
// three serial correctly-rounded reciprocals and a 64-word LUTRAM register
// file. Table words are fetched a line at a time on the scheduler's memory
// port, under the requesting context's tag, by a fetch unit that runs beside
// the sequencer, so a climb step's parent fetch overlaps its box test. Cost is
// O(tree depth) table reads and box tests, paid only on near ties.

`include "VX_define.vh"

module VX_rtu_oracle import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter CTX_TAG_W = 1,
    parameter ADDRW     = `VX_CFG_MEM_ADDR_WIDTH,
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8,
    parameter FMA_LAT   = RTU_LATENCY_FMA
) (
    input  wire                  clk,
    input  wire                  reset,

    // request: hit a (the new one) against hit b (the committed one)
    input  wire                  req_valid,
    output wire                  req_ready,
    input  wire [CTX_TAG_W-1:0]  req_ctx,
    input  wire [ADDRW-1:0]      req_scene,
    input  wire [2:0][31:0]      req_wo,       // world ray
    input  wire [2:0][31:0]      req_wd,
    input  wire [31:0]           req_tlas,     // TLAS table (scene offset)
    input  wire [31:0]           req_a_t,
    input  wire [31:0]           req_a_inst,   // TLAS leaf rank
    input  wire [31:0]           req_a_ps,     // parent << 1 | side in its BLAS table
    input  wire [31:0]           req_a_tab,    // its BLAS table (scene offset)
    input  wire [8:0][31:0]      req_a_v,      // v0.xyz, v1.xyz, v2.xyz
    input  wire [31:0]           req_b_t,
    input  wire [31:0]           req_b_inst,
    input  wire [31:0]           req_b_ps,
    input  wire [31:0]           req_b_tab,
    input  wire [8:0][31:0]      req_b_v,

    // verdict: true when the source traversal keeps hit a
    output wire                  done_valid,
    output wire [CTX_TAG_W-1:0]  done_ctx,
    output wire                  done_keep,

    // table fetch, one line in flight
    output wire                  mem_req_valid,
    output wire [ADDRW-1:0]      mem_req_addr,
    input  wire                  mem_req_ready,
    input  wire                  mem_rsp_valid,
    input  wire [LINE_BITS-1:0]  mem_rsp_data
);
    localparam LINE_BYTES = LINE_BITS / 8;
    localparam WPL        = LINE_BYTES / 4;       // words per line
    localparam WIDXW      = `CLOG2(2 * WPL);
    localparam LSELW      = `CLOG2(LINE_BYTES);

    `STATIC_ASSERT((LINE_BYTES >= 64), ("table fetches assume >= 64-byte lines"))

    localparam [31:0] ROOT  = 32'hffffffff;
    localparam [31:0] F_INF = 32'h7f800000;
    localparam [31:0] F_MAX = 32'h7f7fffff;

    // Climb steps one verdict may take. Valid tables reach the root in tree
    // depth steps; a malformed table (a parent cycle) must still not hang the
    // context, so past the cap the verdict falls to the nearer hit.
    localparam MAX_CLIMBS = 4096;
    localparam CLIMBW     = `CLOG2(MAX_CLIMBS + 1);

    // ── register file map ─────────────────────────────────────────────
    localparam [5:0] RA_WO  = 6'd0,    // world ray o, d
                     RA_WD  = 6'd3,
                     RA_RO  = 6'd6,    // current ray o, d, 1/d
                     RA_RD  = 6'd9,
                     RA_RI  = 6'd12,
                     RA_BA  = 6'd15,   // hit a / b vertex boxes (min.xyz, max.xyz)
                     RA_BB  = 6'd21,
                     RA_CB0 = 6'd27,   // climb boxes
                     RA_CB1 = 6'd33,
                     RA_M   = 6'd27,   // object-ray matrix (aliases the climb boxes)
                     RA_S   = 6'd39,   // slab distances
                     RA_X   = 6'd45;   // the object ray's products

    // ── sequencer states ──────────────────────────────────────────────
    localparam [6:0]
        S_IDLE  = 7'd0,  S_CAP   = 7'd1,  S_BOX1  = 7'd2,  S_BOX2  = 7'd3,
        S_DISP  = 7'd5,  S_DONE  = 7'd6,
        // same instance
        T_S1    = 7'd8,  T_S2    = 7'd9,  T_S3    = 7'd10, T_S4    = 7'd11,
        T_S5    = 7'd12, T_S6    = 7'd13, T_S7    = 7'd14,
        // different instances
        T_D0    = 7'd16, T_D1    = 7'd17, T_D2    = 7'd18, T_D3    = 7'd19,
        T_D3B   = 7'd20, T_D4    = 7'd21, T_D5    = 7'd22, T_D6    = 7'd23,
        T_D7    = 7'd24, T_D8    = 7'd25, T_D9    = 7'd26, T_D10   = 7'd27,
        T_D11   = 7'd28, T_D12   = 7'd29,
        // a hit's BLAS path cull
        P_0     = 7'd32, P_1     = 7'd33, P_2     = 7'd34, P_3     = 7'd35,
        // lowest common ancestor
        L_0     = 7'd40, L_1     = 7'd41, L_2     = 7'd42,
        // one climb step
        U_0     = 7'd44, U_1     = 7'd45, U_2     = 7'd46, U_3     = 7'd47,
        // box test
        B_SUB   = 7'd48, B_MUL   = 7'd49, B_RED   = 7'd50, B_FIN   = 7'd51,
        // object ray
        O_0     = 7'd56, O_1     = 7'd57, O_2     = 7'd58, O_3     = 7'd59,
        O_4     = 7'd60, O_5     = 7'd61, O_6     = 7'd62, O_7     = 7'd63,
        O_8     = 7'd64,
        // reciprocals
        R_LD0   = 7'd68, R_LD1   = 7'd69, R_IT    = 7'd70, R_WR    = 7'd71,
        // fetch wait / move / copy / multiply / drain
        F_WAIT  = 7'd72, M_MV    = 7'd76, C_CP    = 7'd77, X_MUL   = 7'd78,
        W_DRAIN = 7'd79;

    // ── F32 helpers (C fmin/fmax, IEEE ordered compares) ──────────────
    function automatic logic f_nan(input logic [30:0] a);
        f_nan = (a[30:23] == 8'hff) && (a[22:0] != 23'd0);
    endfunction
    function automatic logic f_eq(input logic [31:0] a, input logic [31:0] b);
        f_eq = !f_nan(a[30:0]) && !f_nan(b[30:0])
            && ((a == b) || ((a[30:0] == 31'd0) && (b[30:0] == 31'd0)));
    endfunction
    function automatic logic f_lt(input logic [31:0] a, input logic [31:0] b);
        if (f_nan(a[30:0]) || f_nan(b[30:0]) || ((a[30:0] == 31'd0) && (b[30:0] == 31'd0))) begin
            f_lt = 1'b0;
        end else if (a[31] != b[31]) begin
            f_lt = a[31];
        end else if (!a[31]) begin
            f_lt = (a[30:0] < b[30:0]);
        end else begin
            f_lt = (a[30:0] > b[30:0]);
        end
    endfunction
    function automatic logic [31:0] f_min(input logic [31:0] a, input logic [31:0] b);
        f_min = f_nan(a[30:0]) ? b : (f_nan(b[30:0]) ? a : (f_lt(b, a) ? b : a));
    endfunction
    function automatic logic [31:0] f_max(input logic [31:0] a, input logic [31:0] b);
        f_max = f_nan(a[30:0]) ? b : (f_nan(b[30:0]) ? a : (f_lt(a, b) ? b : a));
    endfunction
    function automatic logic [5:0] mod3(input logic [4:0] k);
        mod3 = (k >= 5'd3) ? 6'(k - 5'd3) : 6'(k);
    endfunction

    // the scene offset of a table node
    function automatic logic [31:0] node_off(input logic [31:0] tab, input logic [31:0] noff,
                                             input logic tl, input logic [31:0] ps);
        node_off = tab + noff + (tl ? ((ps >> 1) << 6) : ((ps >> 1) << 5));
    endfunction

    // ── state ─────────────────────────────────────────────────────────
    reg [6:0]            state;
    reg [6:0]            f_ret, b_ret, u_ret, l_ret, o_ret, r_ret, w_ret, m_ret, c_ret, x_ret;

    reg [CTX_TAG_W-1:0]  ctx_r;
    reg [ADDRW-1:0]      scene_r;
    reg [23:0][31:0]     cap;
    reg [31:0]           tlas_r, ta, tb, ia, ib, psa, psb, taba, tabb;
    reg [31:0]           tab_r, noff_r, stride_r;
    reg                  is_tlas;
    reg [31:0]           ps0, ps1, dep0, dep1;
    reg                  cul0, cul1;
    reg                  cur;          // the climb UP moves
    reg [31:0]           upt;          // ... and the t its boxes are culled against
    reg                  a_first, keep, ph;
    reg [31:0]           key0;
    reg [CLIMBW-1:0]     climbs;

    // box test
    reg [5:0]            bt_base;
    reg [31:0]           bt_tmax;
    reg                  bt_nan;
    reg [31:0]           bt_lo, bt_hi, bt_key;
    reg                  bt_pass;
    reg [2:0]            bt_wbc;       // slab differences written back so far
    reg [4:0]            k;            // the running routine counter

    // object ray / multiply
    reg [31:0]           o_inst;
    reg [31:0]           mul_a, mul_b, mul_p;

    // move / copy
    reg [5:0]            mv_dst, cp_src;
    reg [4:0]            mv_k0, mv_n;

    // ── register file: two LUTRAM copies for two read ports ──────────
    reg  [5:0]  ra, rb;
    wire [31:0] rda, rdb;
    reg         fsm_we;
    reg  [5:0]  fsm_wa;
    reg  [31:0] fsm_wd;
    wire        wb_v;
    wire [5:0]  wb_dst;
    wire [31:0] fma_res;
    wire        rf_we = wb_v || fsm_we;
    wire [5:0]  rf_wa = wb_v ? wb_dst : fsm_wa;
    wire [31:0] rf_wd = wb_v ? fma_res : fsm_wd;

    VX_dp_ram #(
        .DATAW   (32),
        .SIZE    (64),
        .LUTRAM  (1),
        .OUT_REG (0)
    ) rf_a (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (rf_we),
        .wren  (1'b1),
        .waddr (rf_wa),
        .wdata (rf_wd),
        .raddr (ra),
        .rdata (rda)
    );
    VX_dp_ram #(
        .DATAW   (32),
        .SIZE    (64),
        .LUTRAM  (1),
        .OUT_REG (0)
    ) rf_b (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (rf_we),
        .wren  (1'b1),
        .waddr (rf_wa),
        .wdata (rf_wd),
        .raddr (rb),
        .rdata (rdb)
    );

    // ── F32 add / sub / mul (IEEE, subnormals) ───────────────────────
    reg        fma_issue;
    reg [1:0]  fma_kind;   // 0: add, 1: sub, 2: mul
    reg [5:0]  fma_dst;
    reg [5:0]  fma_pend;

    VX_fma_unit #(
        .LATENCY        (FMA_LAT),
        .USE_DSP        (`VX_CFG_RTU_USE_DSP),
        .SUBNORM_ENABLE (1),
        .EXCEPT_ENABLE  (1)
    ) fma (
        .clk     (clk),
        .reset   (reset),
        .enable  (1'b1),
        .mask    (fma_issue),
        .op_type ((fma_kind == 2'd2) ? INST_FPU_MUL : INST_FPU_ADD),
        .fmt     ((fma_kind == 2'd1) ? INST_FMT_BITS'(2'b10) : INST_FMT_BITS'(2'b00)),
        .frm     (INST_FRM_RNE),
        .dataa   (rda),
        .datab   (rdb),
        .datac   (32'd0),
        .result  (fma_res),
        `UNUSED_PIN (fflags)
    );

    VX_shift_register #(
        .DATAW  (1 + 6),
        .RESETW (1),
        .DEPTH  (FMA_LAT)
    ) fma_tags (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({fma_issue, fma_dst}),
        .data_out ({wb_v, wb_dst})
    );

    // ── fetch unit: f_n words at scene offset f_off, beside the sequencer ──
    localparam [2:0] FS_IDLE = 3'd0, FS_REQ0 = 3'd1, FS_RSP0 = 3'd2,
                     FS_REQ1 = 3'd3, FS_RSP1 = 3'd4;
    reg [2:0]            fs_state;
    reg                  fs_start;
    reg [31:0]           f_off;
    reg [4:0]            f_n;
    reg [ADDRW-1:0]      f_line;
    reg [LSELW-3:0]      f_w0;
    reg                  f_two;
    reg [LINE_BITS-1:0]  lb0, lb1;

    // tables are word aligned: the low address bits are always zero
    wire [ADDRW-1:0] f_addr = scene_r + ADDRW'(f_off);
    `UNUSED_VAR (f_addr[1:0])

    always_ff @(posedge clk) begin
        if (reset) begin
            fs_state <= FS_IDLE;
        end else begin
            case (fs_state)
            FS_IDLE: begin
                if (fs_start) begin
                    f_line   <= {f_addr[ADDRW-1:LSELW], LSELW'(0)};
                    f_w0     <= f_addr[LSELW-1:2];
                    f_two    <= (32'(f_addr[LSELW-1:2]) + 32'(f_n)) > WPL;
                    fs_state <= FS_REQ0;
                end
            end
            FS_REQ0: if (mem_req_ready) fs_state <= FS_RSP0;
            FS_RSP0: begin
                if (mem_rsp_valid) begin
                    lb0      <= mem_rsp_data;
                    fs_state <= f_two ? FS_REQ1 : FS_IDLE;
                end
            end
            FS_REQ1: if (mem_req_ready) fs_state <= FS_RSP1;
            FS_RSP1: begin
                if (mem_rsp_valid) begin
                    lb1      <= mem_rsp_data;
                    fs_state <= FS_IDLE;
                end
            end
            default: fs_state <= FS_IDLE;
            endcase
        end
    end
    wire fs_done = (fs_state == FS_IDLE) && !fs_start;

    assign mem_req_valid = (fs_state == FS_REQ0) || (fs_state == FS_REQ1);
    assign mem_req_addr  = (fs_state == FS_REQ0) ? f_line : (f_line + ADDRW'(LINE_BYTES));

    // fetched words: one read port over the two lines
    reg  [4:0]   fw_k;
    wire [2*LINE_BITS-1:0] lbs = {lb1, lb0};
    wire [WIDXW-1:0] fw_idx = WIDXW'(f_w0) + WIDXW'(fw_k);
    wire [31:0]  fw = lbs[32'(fw_idx) * 32 +: 32];

    // ── reciprocals: 2^50 / significand, 28 quotient bits, three lanes ─
    reg [2:0]             rc_spec;     // lane result is special (rc_sval)
    reg [2:0][31:0]       rc_sval;
    reg [2:0][23:0]       dv_m;
    reg [2:0][10:0]       dv_e;        // signed exponent of the operand's significand LSB
    reg [2:0]             dv_s;
    reg [2:0][24:0]       dv_rem;
    reg [2:0][27:0]       dv_q;
    reg [4:0]             dv_it;

    // operand decode: 1/0 -> FLT_MAX (the reference's zero-direction
    // reciprocal), 1/inf -> 0, 1/NaN -> NaN
    function automatic logic [68:0] rc_decode(input logic [31:0] x);
        logic [7:0]  e;
        logic [22:0] f;
        logic [4:0]  msb;
        logic        spec;
        logic [31:0] sval;
        logic [23:0] m;
        logic [10:0] ex;
        e = x[30:23];
        f = x[22:0];
        msb = '0;
        for (integer i = 0; i < 23; ++i) begin
            if (f[i]) msb = 5'(i);
        end
        spec = (x[30:0] == 31'd0) || (e == 8'hff);
        sval = (x[30:0] == 31'd0) ? F_MAX
             : ((f != 23'd0) ? (x | 32'h00400000) : {x[31], 31'd0});
        if (e == 8'h00) begin
            m  = 24'(f) << (5'd23 - msb);
            ex = 11'({6'd0, msb}) - 11'd172;
        end else begin
            m  = {1'b1, f};
            ex = 11'({3'd0, e}) - 11'd150;
        end
        rc_decode = {spec, sval, x[31], m, ex};
    endfunction
    wire [68:0] rc_dec_a = rc_decode(rda);
    wire [68:0] rc_dec_b = rc_decode(rdb);

    wire [1:0]  rc_w   = 2'(k);
    wire [30:0] rc_mag;
    VX_rtu_f32_round #(
        .WB (28),
        .EW (11)
    ) rc_round (
        .mag    (dv_q[rc_w]),
        .exp    (-11'sd50 - $signed(dv_e[rc_w])),
        .sticky (dv_rem[rc_w] != 25'd0),
        .result (rc_mag)
    );

    // ── sequencer ─────────────────────────────────────────────────────
    // O_4's product index k = (d ? 9 : 0) + i*3 + j
    wire [4:0] o4_kk = (k >= 5'd9) ? (k - 5'd9) : k;
    wire [5:0] o4_i  = (o4_kk >= 5'd6) ? 6'd2 : ((o4_kk >= 5'd3) ? 6'd1 : 6'd0);
    wire [5:0] o4_j  = 6'(o4_kk) - o4_i * 6'd3;

    // S_BOX: hit k/3, axis k%3, straight off the captured vertices
    wire [4:0]  bx_h = (k >= 5'd3) ? 5'd9 : 5'd0;
    wire [4:0]  bx_a = 5'(mod3(k));
    wire [31:0] bx_v0 = cap[bx_h + bx_a];
    wire [31:0] bx_v1 = cap[bx_h + 5'd3 + bx_a];
    wire [31:0] bx_v2 = cap[bx_h + 5'd6 + bx_a];

    // combinational controls
    always @(*) begin
        ra        = '0;
        rb        = '0;
        fma_issue = 1'b0;
        fma_kind  = 2'd0;
        fma_dst   = '0;
        fsm_we    = 1'b0;
        fsm_wa    = '0;
        fsm_wd    = '0;
        fw_k      = '0;
        case (state)
        S_CAP: begin
            fsm_we = 1'b1;
            fsm_wa = RA_WO + 6'(k);
            fsm_wd = cap[0];
        end
        S_BOX1: begin
            fsm_we = 1'b1;
            fsm_wa = ((k >= 5'd3) ? RA_BB : RA_BA) + 6'(bx_a);
            fsm_wd = f_min(bx_v0, f_min(bx_v1, bx_v2));
        end
        S_BOX2: begin
            fsm_we = 1'b1;
            fsm_wa = ((k >= 5'd3) ? RA_BB : RA_BA) + 6'd3 + 6'(bx_a);
            fsm_wd = f_max(bx_v0, f_max(bx_v1, bx_v2));
        end
        B_SUB: begin
            ra        = bt_base + 6'(k);
            rb        = RA_RO + mod3(k);
            fma_issue = 1'b1;
            fma_kind  = 2'd1;
            fma_dst   = RA_S + 6'(k);
        end
        B_MUL: begin
            // each product issues as soon as its difference is written back
            ra        = RA_S + 6'(k);
            rb        = RA_RI + mod3(k);
            fma_issue = (5'(bt_wbc) > k);
            fma_kind  = 2'd2;
            fma_dst   = RA_S + 6'(k);
        end
        B_RED: begin
            ra = RA_S + 6'(k);
            rb = RA_S + 6'd3 + 6'(k);
        end
        O_4: begin
            // products: X[i*3+j] = wo[j] * m[i][j], X[9+i*3+j] = wd[j] * m[i][j]
            ra        = ((k >= 5'd9) ? RA_WD : RA_WO) + o4_j;
            rb        = RA_M + o4_i * 6'd4 + o4_j;
            fma_issue = 1'b1;
            fma_kind  = 2'd2;
            fma_dst   = RA_X + 6'(k);
        end
        O_5: begin
            // ro[i] = m[i][3] + P[i][0];  rd[i] = Q[i][0] + Q[i][1]
            if (k < 5'd3) begin
                ra      = RA_M + 6'(k) * 6'd4 + 6'd3;
                rb      = RA_X + 6'(k) * 6'd3;
                fma_dst = RA_RO + 6'(k);
            end else begin
                ra      = RA_X + 6'd9 + 6'(k - 5'd3) * 6'd3;
                rb      = RA_X + 6'd9 + 6'(k - 5'd3) * 6'd3 + 6'd1;
                fma_dst = RA_RD + 6'(k - 5'd3);
            end
            fma_issue = 1'b1;
        end
        O_6: begin
            // ro[i] += P[i][1];  rd[i] += Q[i][2]
            if (k < 5'd3) begin
                ra      = RA_RO + 6'(k);
                rb      = RA_X + 6'(k) * 6'd3 + 6'd1;
                fma_dst = RA_RO + 6'(k);
            end else begin
                ra      = RA_RD + 6'(k - 5'd3);
                rb      = RA_X + 6'd9 + 6'(k - 5'd3) * 6'd3 + 6'd2;
                fma_dst = RA_RD + 6'(k - 5'd3);
            end
            fma_issue = 1'b1;
        end
        O_7: begin
            // ro[i] += P[i][2]
            ra        = RA_RO + 6'(k);
            rb        = RA_X + 6'(k) * 6'd3 + 6'd2;
            fma_dst   = RA_RO + 6'(k);
            fma_issue = 1'b1;
        end
        R_LD0: begin
            ra = RA_RD;
            rb = RA_RD + 6'd1;
        end
        R_LD1: begin
            ra = RA_RD + 6'd2;
        end
        R_WR: begin
            fsm_we = 1'b1;
            fsm_wa = RA_RI + 6'(k);
            fsm_wd = rc_spec[rc_w] ? rc_sval[rc_w] : {dv_s[rc_w], rc_mag};
        end
        M_MV: begin
            fw_k   = mv_k0 + k;
            fsm_we = 1'b1;
            fsm_wa = mv_dst + 6'(k);
            fsm_wd = fw;
        end
        C_CP: begin
            ra     = cp_src + 6'(k);
            fsm_we = 1'b1;
            fsm_wa = mv_dst + 6'(k);
            fsm_wd = rda;
        end
        T_D3B: fw_k = 5'd1;
        U_2:   fw_k = is_tlas ? 5'd12 : 5'd6;
        T_D6, T_D10: fw_k = 5'd13;
        default:;
        endcase
    end

    `RUNTIME_ASSERT(!(wb_v && fsm_we), ("%t: rtu oracle: register-file write conflict", $time))

    wire [31:0] mul_sum = mul_p + (mul_a[0] ? mul_b : 32'd0);

    always_ff @(posedge clk) begin
        fs_start <= 1'b0;
        if (reset) begin
            state    <= S_IDLE;
            fma_pend <= '0;
        end else begin
            fma_pend <= fma_pend + 6'(fma_issue) - 6'(wb_v);
            if ((state == B_SUB) || (state == B_MUL)) begin
                bt_wbc <= bt_wbc + 3'(wb_v);
            end else begin
                bt_wbc <= '0;
            end

            case (state)
            S_IDLE: begin
                if (req_valid) begin
                    ctx_r   <= req_ctx;
                    scene_r <= req_scene;
                    tlas_r  <= req_tlas;
                    ta      <= req_a_t;   tb   <= req_b_t;
                    ia      <= req_a_inst; ib  <= req_b_inst;
                    psa     <= req_a_ps;  psb  <= req_b_ps;
                    taba    <= req_a_tab; tabb <= req_b_tab;
                    cap     <= {req_b_v, req_a_v, req_wd, req_wo};
                    k       <= '0;
                    climbs  <= '0;
                    state   <= S_CAP;
                end
            end
            S_CAP: begin
                // world ray into the register file; the vertices stay captured
                cap <= cap >> 32;
                k   <= k + 5'd1;
                if (k == 5'd5) begin
                    k     <= '0;
                    state <= S_BOX1;
                end
            end
            S_BOX1: begin
                state <= S_BOX2;
            end
            S_BOX2: begin
                k     <= k + 5'd1;
                state <= S_BOX1;
                if (k == 5'd5) begin
                    k     <= '0;
                    state <= S_DISP;
                end
            end
            S_DISP: begin
                state <= (ia == ib) ? T_S1 : T_D0;
            end

            // ── same instance: climb its BLAS table ──────────────────
            T_S1: begin
                if ((psa == ROOT) || (psb == ROOT)) begin
                    keep  <= f_lt(ta, tb) || f_eq(ta, tb);
                    state <= S_DONE;
                end else begin
                    o_inst <= ia;
                    o_ret  <= T_S2;
                    state  <= O_0;
                end
            end
            T_S2: begin
                is_tlas  <= 1'b0;
                tab_r    <= taba;
                f_off    <= taba + 32'd8;
                f_n      <= 5'd1;
                fs_start <= 1'b1;
                f_ret    <= T_S3;
                state    <= F_WAIT;
            end
            T_S3: begin
                noff_r   <= fw;
                ps0      <= psa;
                f_off    <= node_off(tab_r, fw, 1'b0, psa) + 32'd28;
                fs_start <= 1'b1;
                f_ret    <= T_S4;
                state    <= F_WAIT;
            end
            T_S4: begin
                dep0     <= fw + 32'd1;
                ps1      <= psb;
                f_off    <= node_off(tab_r, noff_r, 1'b0, psb) + 32'd28;
                fs_start <= 1'b1;
                f_ret    <= T_S5;
                state    <= F_WAIT;
            end
            T_S5: begin
                dep1   <= fw + 32'd1;
                cul0   <= 1'b0;
                cul1   <= 1'b0;
                mv_dst <= RA_CB0;
                cp_src <= RA_BA;
                mv_n   <= 5'd12;
                c_ret  <= T_S6;
                state  <= C_CP;
            end
            T_S6: begin
                l_ret <= T_S7;
                state <= L_0;
            end
            T_S7: begin
                keep  <= f_eq(ta, tb) ? a_first
                       : f_lt(ta, tb) ? (a_first || !cul0)
                                      : (a_first && cul1);
                state <= S_DONE;
            end

            // ── different instances: climb the TLAS table ────────────
            T_D0: begin
                mv_dst <= RA_RO;        // world ray o, d -> current ray
                cp_src <= RA_WO;
                mv_n   <= 5'd6;
                c_ret  <= T_D1;
                state  <= C_CP;
            end
            T_D1: begin
                // the TLAS header fetch runs beside the reciprocals
                is_tlas  <= 1'b1;
                tab_r    <= tlas_r;
                f_off    <= tlas_r + 32'd8;
                f_n      <= 5'd2;
                fs_start <= 1'b1;
                r_ret    <= T_D2;
                state    <= R_LD0;
            end
            T_D2: begin
                f_ret <= T_D3;
                state <= F_WAIT;
            end
            T_D3: begin
                noff_r <= fw;
                state  <= T_D3B;
            end
            T_D3B: begin
                stride_r <= fw;
                mul_a    <= ia;
                mul_b    <= fw;
                mul_p    <= '0;
                x_ret    <= T_D4;
                state    <= X_MUL;
            end
            T_D4: begin
                f_off    <= tlas_r + 32'd16 + mul_p;
                f_n      <= 5'd1;
                fs_start <= 1'b1;
                f_ret    <= T_D5;
                state    <= F_WAIT;
            end
            T_D5: begin
                ps0 <= fw;
                if (fw == ROOT) begin
                    dep0  <= '0;
                    state <= T_D7;
                end else begin
                    f_off    <= node_off(tab_r, noff_r, 1'b1, fw);
                    f_n      <= 5'd14;
                    fs_start <= 1'b1;
                    f_ret    <= T_D6;
                    state    <= F_WAIT;
                end
            end
            T_D6: begin
                dep0   <= fw + 32'd1;
                mv_dst <= RA_CB0;
                mv_k0  <= ps0[0] ? 5'd6 : 5'd0;
                mv_n   <= 5'd6;
                m_ret  <= T_D7;
                state  <= M_MV;
            end
            T_D7: begin
                mul_a <= ib;
                mul_b <= stride_r;
                mul_p <= '0;
                x_ret <= T_D8;
                state <= X_MUL;
            end
            T_D8: begin
                f_off    <= tlas_r + 32'd16 + mul_p;
                f_n      <= 5'd1;
                fs_start <= 1'b1;
                f_ret    <= T_D9;
                state    <= F_WAIT;
            end
            T_D9: begin
                ps1 <= fw;
                if (fw == ROOT) begin
                    dep1  <= '0;
                    state <= T_D11;
                end else begin
                    f_off    <= node_off(tab_r, noff_r, 1'b1, fw);
                    f_n      <= 5'd14;
                    fs_start <= 1'b1;
                    f_ret    <= T_D10;
                    state    <= F_WAIT;
                end
            end
            T_D10: begin
                dep1   <= fw + 32'd1;
                mv_dst <= RA_CB1;
                mv_k0  <= ps1[0] ? 5'd6 : 5'd0;
                mv_n   <= 5'd6;
                m_ret  <= T_D11;
                state  <= M_MV;
            end
            T_D11: begin
                cul0  <= 1'b0;
                cul1  <= 1'b0;
                l_ret <= T_D12;
                state <= L_0;
            end
            T_D12: begin
                state <= S_DONE;
                if (f_eq(ta, tb)) begin
                    keep <= a_first;
                end else if (f_lt(ta, tb)) begin
                    if (a_first) begin
                        keep <= 1'b1;
                    end else if (cul0) begin
                        keep <= 1'b0;
                    end else begin
                        ph    <= 1'b0;
                        state <= P_0;
                    end
                end else begin
                    if (!a_first) begin
                        keep <= 1'b0;
                    end else if (cul1) begin
                        keep <= 1'b1;
                    end else begin
                        ph    <= 1'b1;
                        state <= P_0;
                    end
                end
            end

            // ── the nearer hit's BLAS path, culled against the other t ──
            P_0: begin
                o_inst <= ph ? ib : ia;
                o_ret  <= P_1;
                state  <= O_0;
            end
            P_1: begin
                is_tlas  <= 1'b0;
                tab_r    <= ph ? tabb : taba;
                f_off    <= (ph ? tabb : taba) + 32'd8;
                f_n      <= 5'd1;
                fs_start <= 1'b1;
                f_ret    <= P_2;
                state    <= F_WAIT;
            end
            P_2: begin
                noff_r <= fw;
                ps0    <= ph ? psb : psa;
                cul0   <= 1'b0;
                cur    <= 1'b0;
                upt    <= ph ? ta : tb;
                mv_dst <= RA_CB0;
                cp_src <= ph ? RA_BB : RA_BA;
                mv_n   <= 5'd6;
                c_ret  <= P_3;
                state  <= C_CP;
            end
            P_3: begin
                // the verdict only needs whether some box fails
                if ((ps0 == ROOT) || cul0) begin
                    keep  <= ph ? cul0 : !cul0;
                    state <= S_DONE;
                end else begin
                    u_ret <= P_3;
                    state <= U_0;
                end
            end

            // ── climb both leaves to their lowest common ancestor ────
            L_0: begin
                u_ret <= L_0;
                if (dep0 > dep1) begin
                    cur   <= 1'b0;
                    upt   <= tb;
                    state <= U_0;
                end else if (dep1 > dep0) begin
                    cur   <= 1'b1;
                    upt   <= ta;
                    state <= U_0;
                end else if (ps0[31:1] != ps1[31:1]) begin
                    cur   <= 1'b0;
                    upt   <= tb;
                    state <= U_0;
                end else begin
                    bt_base <= RA_CB0;
                    b_ret   <= L_1;
                    state   <= B_SUB;
                end
            end
            L_1: begin
                key0    <= bt_key;
                bt_base <= RA_CB1;
                b_ret   <= L_2;
                state   <= B_SUB;
            end
            L_2: begin
                // child 1 first only on a strictly nearer entry
                a_first <= (ps0[0] == f_lt(ps0[0] ? key0 : bt_key, ps0[0] ? bt_key : key0));
                state   <= l_ret;
            end

            // ── one climb step: test the box, move to the parent ─────
            U_0: begin
                // the parent link's fetch runs beside the box test
                climbs   <= climbs + CLIMBW'(1);
                if (climbs == CLIMBW'(MAX_CLIMBS)) begin
                    keep <= f_lt(ta, tb);
                end
                bt_base  <= cur ? RA_CB1 : RA_CB0;
                bt_tmax  <= upt;
                f_off    <= node_off(tab_r, noff_r, is_tlas, cur ? ps1 : ps0);
                f_n      <= is_tlas ? 5'd14 : 5'd8;
                fs_start <= (climbs != CLIMBW'(MAX_CLIMBS));
                b_ret    <= U_1;
                state    <= (climbs == CLIMBW'(MAX_CLIMBS)) ? S_DONE : B_SUB;
            end
            U_1: begin
                if (!bt_pass) begin
                    if (cur) cul1 <= 1'b1; else cul0 <= 1'b1;
                end
                f_ret <= U_2;
                state <= F_WAIT;
            end
            U_2: begin
                if (cur) begin
                    ps1  <= fw;
                    dep1 <= dep1 - 32'd1;
                end else begin
                    ps0  <= fw;
                    dep0 <= dep0 - 32'd1;
                end
                mv_dst <= cur ? RA_CB1 : RA_CB0;
                mv_n   <= 5'd6;
                m_ret  <= u_ret;
                if (!is_tlas) begin
                    // a BLAS node holds its own box, read with its parent link
                    mv_k0 <= 5'd0;
                    state <= M_MV;
                end else if (fw == ROOT) begin
                    state <= u_ret;
                end else begin
                    mv_k0    <= fw[0] ? 5'd6 : 5'd0;
                    f_off    <= node_off(tab_r, noff_r, 1'b1, fw);
                    f_n      <= 5'd12;
                    fs_start <= 1'b1;
                    f_ret    <= U_3;
                    state    <= F_WAIT;
                end
            end
            U_3: begin
                state <= M_MV;
            end

            // ── box test (the reference's slab test) ─────────────────
            B_SUB: begin
                if (k == 5'd0) begin
                    bt_nan <= f_nan(rda[30:0]);
                end
                k <= k + 5'd1;
                if (k == 5'd5) begin
                    k     <= '0;
                    state <= B_MUL;
                end
            end
            B_MUL: begin
                if (fma_issue) begin
                    k <= k + 5'd1;
                    if (k == 5'd5) begin
                        k     <= '0;
                        w_ret <= B_RED;
                        state <= W_DRAIN;
                    end
                end
            end
            B_RED: begin
                bt_lo <= (k == 5'd0) ? f_min(rda, rdb) : f_max(bt_lo, f_min(rda, rdb));
                bt_hi <= (k == 5'd0) ? f_max(rda, rdb) : f_min(bt_hi, f_max(rda, rdb));
                k     <= k + 5'd1;
                if (k == 5'd2) begin
                    k     <= '0;
                    state <= B_FIN;
                end
            end
            B_FIN: begin
                // hi >= fmax(0, lo); an empty (NaN) box is never entered
                logic hit;
                logic [31:0] lo0;
                lo0 = f_max(32'd0, bt_lo);
                hit = !bt_nan && (f_lt(lo0, bt_hi) || f_eq(lo0, bt_hi));
                bt_key  <= hit ? bt_lo : F_INF;
                bt_pass <= hit && f_lt(bt_lo, bt_tmax);
                state   <= b_ret;
            end

            // ── the reference's object-space ray for TLAS leaf o_inst ─
            O_0: begin
                f_off    <= tlas_r + 32'd12;
                f_n      <= 5'd1;
                fs_start <= 1'b1;
                f_ret    <= O_1;
                state    <= F_WAIT;
            end
            O_1: begin
                stride_r <= fw;
                mul_a    <= o_inst;
                mul_b    <= fw;
                mul_p    <= '0;
                x_ret    <= O_2;
                state    <= X_MUL;
            end
            O_2: begin
                f_off    <= tlas_r + 32'd32 + mul_p;
                f_n      <= 5'd12;
                fs_start <= 1'b1;
                f_ret    <= O_3;
                state    <= F_WAIT;
            end
            O_3: begin
                mv_dst <= RA_M;
                mv_k0  <= '0;
                mv_n   <= 5'd12;
                m_ret  <= O_4;
                state  <= M_MV;
            end
            O_4: begin
                k <= k + 5'd1;
                if (k == 5'd17) begin
                    k     <= '0;
                    w_ret <= O_5;
                    state <= W_DRAIN;
                end
            end
            O_5: begin
                k <= k + 5'd1;
                if (k == 5'd5) begin
                    k     <= '0;
                    w_ret <= O_6;
                    state <= W_DRAIN;
                end
            end
            O_6: begin
                k <= k + 5'd1;
                if (k == 5'd5) begin
                    k     <= '0;
                    w_ret <= O_7;
                    state <= W_DRAIN;
                end
            end
            O_7: begin
                k <= k + 5'd1;
                if (k == 5'd2) begin
                    k     <= '0;
                    w_ret <= O_8;
                    state <= W_DRAIN;
                end
            end
            O_8: begin
                r_ret <= o_ret;
                state <= R_LD0;
            end

            // ── 1/d per axis, correctly rounded ──────────────────────
            R_LD0: begin
                {rc_spec[0], rc_sval[0], dv_s[0], dv_m[0], dv_e[0]} <= rc_dec_a;
                {rc_spec[1], rc_sval[1], dv_s[1], dv_m[1], dv_e[1]} <= rc_dec_b;
                state <= R_LD1;
            end
            R_LD1: begin
                {rc_spec[2], rc_sval[2], dv_s[2], dv_m[2], dv_e[2]} <= rc_dec_a;
                for (integer i = 0; i < 3; ++i) begin
                    dv_rem[i] <= 25'h400000;   // 2^22: the dividend's leading bits
                    dv_q[i]   <= '0;
                end
                dv_it <= '0;
                state <= R_IT;
            end
            R_IT: begin
                for (integer i = 0; i < 3; ++i) begin
                    if ({dv_rem[i][23:0], 1'b0} >= 25'(dv_m[i])) begin
                        dv_rem[i] <= {dv_rem[i][23:0], 1'b0} - 25'(dv_m[i]);
                        dv_q[i]   <= {dv_q[i][26:0], 1'b1};
                    end else begin
                        dv_rem[i] <= {dv_rem[i][23:0], 1'b0};
                        dv_q[i]   <= {dv_q[i][26:0], 1'b0};
                    end
                end
                dv_it <= dv_it + 5'd1;
                if (dv_it == 5'd27) begin
                    k     <= '0;
                    state <= R_WR;
                end
            end
            R_WR: begin
                k <= k + 5'd1;
                if (k == 5'd2) begin
                    k     <= '0;
                    state <= r_ret;
                end
            end

            // ── leaf routines ────────────────────────────────────────
            F_WAIT: begin
                if (fs_done) begin
                    state <= f_ret;
                end
            end
            M_MV, C_CP: begin
                k <= k + 5'd1;
                if ((k + 5'd1) == mv_n) begin
                    k     <= '0;
                    state <= (state == M_MV) ? m_ret : c_ret;
                end
            end
            X_MUL: begin
                mul_p <= mul_sum;
                mul_a <= mul_a >> 1;
                mul_b <= mul_b << 1;
                if (mul_a[31:1] == 31'd0) begin
                    state <= x_ret;
                end
            end
            W_DRAIN: begin
                if (fma_pend == 6'd0) begin
                    state <= w_ret;
                end
            end
            S_DONE: begin
                state <= S_IDLE;
            end
            default: begin
                state <= S_IDLE;
            end
            endcase
        end
    end

    assign req_ready  = (state == S_IDLE);
    assign done_valid = (state == S_DONE);
    assign done_ctx   = ctx_r;
    assign done_keep  = keep;

endmodule
