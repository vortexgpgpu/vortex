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

// VX_tex_unit — per-core SFU PE that decodes vx_tex / vx_tex4 SFU ops, emits
// tex_bus_if request(s) to the cluster-shared TEX core, and forwards the
// returned texels to result_if / the shared graphics window.
//
//   vx_tex  (legacy) : u,v,lod from rs1/rs2/rs3; texel -> rd.
//   vx_tex4 single   : u,v from the window (rs2 base), lod from rs1; texel ->
//                      window[out_slot] + rd (sync handle).
//   vx_tex4 quad     : one thread owns a 2x2 quad; u[0..3],v[0..3] from the
//                      window (rs2 base, 8 slots), rs1 = texture dims
//                      {logh,logw}; the unit computes ONE integer mip LOD from
//                      the quad derivatives (vx_tex_lod.h formula, via VX_lzc)
//                      and issues the four fragments sequentially, each sampled
//                      at that LOD, writing texel[F] -> window[out_slot+F]; the
//                      4th retires rd (sync handle).

`include "VX_tex_define.vh"

module VX_tex_unit import VX_gpu_pkg::*, VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter CONS_RD_PORTS = 2
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces (sfu_execute_t / sfu_result_t)
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // Cluster-side texture bus (master)
    VX_tex_bus_if.master    tex_bus_if,

    // Shared graphics-window access (vx_tex4): read the (u,v) payload at issue,
    // write the sampled texel back on response. Wired to VX_gfx_window in
    // VX_sfu_unit; idle for legacy vx_tex.
    VX_gfx_win_rd_if.master                                        cons_rd_if,
    VX_gfx_win_wr_if.master                                           cons_wr_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)
    localparam REQ_QUEUE_BITS = `LOG2UP(`VX_CFG_TEX_REQ_QUEUE_SIZE);
    localparam LANE_BITS   = `CLOG2(NUM_LANES);
    localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam SLOT_BITS   = `CLOG2(`VX_RT_SLOT_COUNT);
    localparam RHO_W       = 48;                 // 32-bit |dcoord| << up to 15
    localparam LZC_W       = `CLOG2(RHO_W);

    wire is_tex4 = execute_if.data.op_args.tex.is_tex4;
    wire is_quad = is_tex4 && execute_if.data.op_args.tex.mode;
    wire [PID_W-1:0]       in_pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0] in_tbase = THREAD_BITS'(in_pid) << LANE_BITS;

    // ── quad fragment sequencer (one fragment in flight at a time) ─────────
    reg [1:0] q_frag;     // current fragment 0..3
    reg       q_issued;   // current fragment issued, awaiting its response

    // ── window read: 8 contiguous slots from the rs2 base ─────────────────
    wire [SLOT_BITS-1:0] in_slot = execute_if.data.rs2_data[0][SLOT_BITS-1:0];
    assign cons_rd_if.req.wid   = execute_if.data.header.wid;
    assign cons_rd_if.req.tbase = in_tbase;
    for (genvar p = 0; p < CONS_RD_PORTS; ++p) begin : g_cons_rd_slot
        assign cons_rd_if.req.slot[p] = in_slot + SLOT_BITS'(p);
    end

    // ── quad LOD: integer mip from the 2x2 derivatives (vx_tex_lod.h) ──────
    wire [3:0] logw = execute_if.data.rs1_data[0][3:0];   // rs1 = {logh, _, logw}
    wire [3:0] logh = execute_if.data.rs1_data[0][19:16];
    wire [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] quad_lod;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_quad_lod
        // frags: 0=(x,y) 1=(x+1,y) 2=(x,y+1) 3=(x+1,y+1). u[k]=port k, v[k]=port 4+k.
        wire signed [31:0] du1 = $signed(cons_rd_if.data[1][i]) - $signed(cons_rd_if.data[0][i]);
        wire signed [31:0] du2 = $signed(cons_rd_if.data[2][i]) - $signed(cons_rd_if.data[0][i]);
        wire signed [31:0] dv1 = $signed(cons_rd_if.data[5][i]) - $signed(cons_rd_if.data[4][i]);
        wire signed [31:0] dv2 = $signed(cons_rd_if.data[6][i]) - $signed(cons_rd_if.data[4][i]);
        wire [31:0] au1 = du1[31] ? (~du1 + 1'b1) : du1;
        wire [31:0] au2 = du2[31] ? (~du2 + 1'b1) : du2;
        wire [31:0] av1 = dv1[31] ? (~dv1 + 1'b1) : dv1;
        wire [31:0] av2 = dv2[31] ? (~dv2 + 1'b1) : dv2;
        wire [RHO_W-1:0] gux = RHO_W'(au1) << logw;
        wire [RHO_W-1:0] guy = RHO_W'(au2) << logw;
        wire [RHO_W-1:0] gvx = RHO_W'(av1) << logh;
        wire [RHO_W-1:0] gvy = RHO_W'(av2) << logh;
        wire [RHO_W-1:0] mu  = (gux > guy) ? gux : guy;
        wire [RHO_W-1:0] mv  = (gvx > gvy) ? gvx : gvy;
        wire [RHO_W-1:0] rho = (mu > mv) ? mu : mv;
        wire [LZC_W-1:0] lzc;
        wire             nz;
        VX_lzc #(.N(RHO_W)) u_lzc (.data_in(rho), .data_out(lzc), .valid_out(nz));
        // msb = (RHO_W-1) - lzc; lod = clamp(msb - VX_TEX_FXD_FRAC, 0, LOD_MAX).
        wire signed [8:0] lod_s = $signed(9'(RHO_W-1-`VX_TEX_FXD_FRAC)) - $signed({3'b0, lzc});
        assign quad_lod[i] = (~nz || lod_s < 0) ? '0
                           : (lod_s > $signed(9'(`VX_TEX_LOD_MAX))) ? `VX_TEX_LOD_BITS'(`VX_TEX_LOD_MAX)
                           : lod_s[`VX_TEX_LOD_BITS-1:0];
    end

    // ── per-fragment coords + lod ─────────────────────────────────────────
    wire [1:0][NUM_LANES-1:0][31:0]            sfu_exe_coords;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] sfu_exe_lod;
    wire [`VX_TEX_STAGE_BITS-1:0]              sfu_exe_stage;
    assign sfu_exe_stage = execute_if.data.op_args.tex.stage;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sfu_exe_coords
        assign sfu_exe_coords[0][i] = is_quad ? cons_rd_if.data[{1'b0, q_frag}][i]
                                    : is_tex4 ? cons_rd_if.data[0][i]
                                              : execute_if.data.rs1_data[i][31:0];
        assign sfu_exe_coords[1][i] = is_quad ? cons_rd_if.data[{1'b1, q_frag}][i]   // ports 4..7
                                    : is_tex4 ? cons_rd_if.data[1][i]
                                              : execute_if.data.rs2_data[i][31:0];
        assign sfu_exe_lod[i]       = is_quad ? quad_lod[i]
                                    : is_tex4 ? execute_if.data.rs1_data[i][0 +: `VX_TEX_LOD_BITS]
                                              : execute_if.data.rs3_data[i][0 +: `VX_TEX_LOD_BITS];
    end

    // ── tag-store echo (round-trips header + window writeback info) ────────
    typedef struct packed {
        logic [NW_WIDTH-1:0]                            wid;
        logic [NUM_LANES-1:0]                           tmask;
        logic [PID_W-1:0]                               pid;
        logic                                           sop;
        logic                                           eop;
        logic [PC_BITS-1:0]                             PC;
        logic                                           wb;
        logic [NUM_XREGS-1:0]                           wr_xregs;
        logic [NUM_REGS_BITS-1:0]                       rd;
        logic [BYTESEL_BITS-1:0]                        bytesel;
        logic                                           is_tex4;
        logic                                           is_quad;
        logic [1:0]                                     frag;
        logic [SLOT_BITS-1:0]                           out_slot;
    } header_echo_t;

    header_echo_t in_echo, out_echo;
    assign in_echo.wid       = execute_if.data.header.wid;
    assign in_echo.tmask     = execute_if.data.header.tmask;
    assign in_echo.pid       = execute_if.data.header.pid;
    assign in_echo.sop       = execute_if.data.header.sop;
    assign in_echo.eop       = execute_if.data.header.eop;
    assign in_echo.PC        = execute_if.data.header.PC;
    assign in_echo.wb        = execute_if.data.header.wb;
    assign in_echo.wr_xregs  = execute_if.data.header.wr_xregs;
    assign in_echo.rd        = execute_if.data.header.rd;
    assign in_echo.bytesel   = execute_if.data.header.bytesel;
    assign in_echo.is_tex4   = is_tex4;
    assign in_echo.is_quad   = is_quad;
    assign in_echo.frag      = q_frag;
    assign in_echo.out_slot  = execute_if.data.op_args.tex.out_slot[SLOT_BITS-1:0];

    wire [REQ_QUEUE_BITS-1:0] mdata_waddr, mdata_raddr;
    wire mdata_full;

    // ── request submit ────────────────────────────────────────────────────
    // Single / vx_tex: one request, accepted immediately. Quad: four requests
    // issued one at a time (~q_issued gates the next); the macro-op is accepted
    // only when its 4th response retires.
    wire resp_fire = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;
    wire valid_in, ready_in;
    assign valid_in = (is_quad ? (execute_if.valid && ~q_issued) : execute_if.valid) && ~mdata_full;
    wire req_fire = valid_in && ready_in;

    assign execute_if.ready = is_quad
        ? (out_echo.is_quad && (out_echo.frag == 2'd3) && resp_fire)  // 4th frag retired
        : (ready_in && ~mdata_full);

    wire mdata_push = req_fire;          // acquire a tag per issued request
    wire mdata_pop  = resp_fire;         // release a tag per response

    always @(posedge clk) begin
        if (reset) begin
            q_frag   <= 2'd0;
            q_issued <= 1'b0;
        end else begin
            if (is_quad && req_fire) q_issued <= 1'b1;
            // advance only on THIS quad's fragment responses (a prior single's
            // in-flight response must not move the sequencer).
            if (out_echo.is_quad && resp_fire) begin
                q_issued <= 1'b0;
                q_frag   <= (q_frag == 2'd3) ? 2'd0 : (q_frag + 2'd1);
            end
        end
    end

    VX_index_buffer #(
        .DATAW ($bits(header_echo_t)),
        .SIZE  (`VX_CFG_TEX_REQ_QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (mdata_push),
        .write_addr   (mdata_waddr),
        .write_data   (in_echo),
        .read_data    (out_echo),
        .read_addr    (mdata_raddr),
        .release_en   (mdata_pop),
        .full         (mdata_full),
        `UNUSED_PIN (empty)
    );

    wire [TEX_REQ_TAG_WIDTH-1:0] req_tag = {execute_if.data.header.uuid, mdata_waddr};

    VX_elastic_buffer #(
        .DATAW   (NUM_LANES * (1 + 2 * 32 + `VX_TEX_LOD_BITS) + `VX_TEX_STAGE_BITS + TEX_REQ_TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (2) // external bus should be registered
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({execute_if.data.header.tmask, sfu_exe_coords, sfu_exe_lod, sfu_exe_stage, req_tag}),
        .data_out  ({tex_bus_if.req_data.mask, tex_bus_if.req_data.coords, tex_bus_if.req_data.lod, tex_bus_if.req_data.stage, tex_bus_if.req_data.tag}),
        .valid_out (tex_bus_if.req_valid),
        .ready_out (tex_bus_if.req_ready)
    );

    // ── response ──────────────────────────────────────────────────────────
    assign mdata_raddr = tex_bus_if.rsp_data.tag[0 +: REQ_QUEUE_BITS];

    wire [UUID_WIDTH-1:0]      rsp_uuid = tex_bus_if.rsp_data.tag[REQ_QUEUE_BITS +: UUID_WIDTH];
    wire [NUM_LANES-1:0][31:0] rsp_texels = tex_bus_if.rsp_data.texels;

    // Reassemble the result header from echo + uuid.
    sfu_result_t rsp_data_in;
    assign rsp_data_in.header.uuid     = rsp_uuid;
    assign rsp_data_in.header.wid      = out_echo.wid;
    assign rsp_data_in.header.tmask    = out_echo.tmask;
    assign rsp_data_in.header.pid      = out_echo.pid;
    assign rsp_data_in.header.sop      = out_echo.sop;
    assign rsp_data_in.header.eop      = out_echo.eop;
    assign rsp_data_in.header.PC       = out_echo.PC;
    assign rsp_data_in.header.wb       = out_echo.wb;
    assign rsp_data_in.header.wr_xregs = out_echo.wr_xregs;
    assign rsp_data_in.header.rd       = out_echo.rd;
    assign rsp_data_in.header.bytesel  = out_echo.bytesel;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        assign rsp_data_in.data[i] = `VX_CFG_XLEN'(rsp_texels[i]);
    end

    // Single (and vx_tex) retire on their one response; quad retires only on the
    // 4th fragment — frags 0..2 just land their texel in the window.
    wire to_rsp_buf = out_echo.is_quad ? (out_echo.frag == 2'd3) : 1'b1;
    wire rsp_buf_rdy;
    assign tex_bus_if.rsp_ready = to_rsp_buf ? rsp_buf_rdy : 1'b1;

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_bus_if.rsp_valid && to_rsp_buf),
        .ready_in  (rsp_buf_rdy),
        .data_in   (rsp_data_in),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    // vx_tex4: land each fragment's texel in the window output slot as its
    // response is consumed (slot = out_slot + frag). For frags 0..2 the response
    // is accepted unconditionally; the 4th (and single mode) lands as the op
    // retires, so a handle-chained GETW sees the complete result window.
    assign cons_wr_if.valid    = resp_fire && out_echo.is_tex4;
    assign cons_wr_if.data.wid   = out_echo.wid;
    assign cons_wr_if.data.tbase = THREAD_BITS'(out_echo.pid) << LANE_BITS;
    assign cons_wr_if.data.mask  = out_echo.tmask;
    assign cons_wr_if.data.slot  = out_echo.out_slot + SLOT_BITS'(out_echo.frag);
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cons_wr
        assign cons_wr_if.data.data[i] = rsp_texels[i];
    end

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (req_fire) begin
            `TRACE(1, ("%d: %s tex-req: wid=%0d, PC=0x%0h, tmask=%b, stage=%0d, quad=%b, frag=%0d, ibuf_idx=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, sfu_exe_stage, is_quad, q_frag, mdata_waddr, execute_if.data.header.uuid))
        end
        if (resp_fire) begin
            `TRACE(1, ("%d: %s tex-rsp: wid=%0d, tmask=%b, frag=%0d, out_slot=%0d, ibuf_idx=%0d (#%0d)\n",
                $time, INSTANCE_ID, out_echo.wid, out_echo.tmask, out_echo.frag, out_echo.out_slot,
                mdata_raddr, rsp_uuid))
        end
    end
`endif

endmodule
