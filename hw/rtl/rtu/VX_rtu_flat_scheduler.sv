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

// VX_rtu_flat_scheduler — flat triangle-list traversal control (the
// RTU_BVH_WIDTH==0 build). Holds one ray context per lane and time-multiplexes
// a single shared ray-triangle datapath across them: it linearly scans all
// triangles of a flat scene, keeping the closest opaque hit within
// [t_min, best_t). No BVH nodes, stack, box PE, or ray-setup reciprocal — the
// Möller-Trumbore tri PE consumes the ray directly. Mirrors the SimX FlatWalker;
// it presents the same scheduler↔core interface as VX_rtu_scheduler so
// VX_rtu_core can instantiate either by configured width.
//
// Like the BVH scheduler it runs a SELECT/EXEC micro-step pipeline and parks a
// context across the two long-latency operations (a cache-line fetch and a
// tri-PE test), tagging each by context id so responses route back. The flat
// triangle record (40 B, v0@0/v1@12/v2@24/flags@36, no leaf header) is fetched
// and byte-aligned exactly like a BVH leaf, since records straddle cache lines.

`include "VX_define.vh"

module VX_rtu_flat_scheduler import VX_gpu_pkg::*, VX_fpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_CTX   = 4,
    parameter LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8,
    parameter CTX_TAG_W = `LOG2UP(NUM_CTX)
) (
    input  wire        clk,
    input  wire        reset,

    input  wire                       start,
    input  wire [NUM_CTX-1:0]         mask,
    input  rtu_ray_t [NUM_CTX-1:0]    rays,
    output wire                       busy,
    output wire                       done,

    output wire [NUM_CTX-1:0]         res_hit,
    output wire [NUM_CTX-1:0][31:0]   res_t,
    output wire [NUM_CTX-1:0][31:0]   res_u,
    output wire [NUM_CTX-1:0][31:0]   res_v,
    output wire [NUM_CTX-1:0][31:0]   res_prim,
    output wire [NUM_CTX-1:0][31:0]   res_geom,

    output wire                              mem_req_valid,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [CTX_TAG_W-1:0]              mem_req_tag,
    input  wire                              mem_req_ready,
    input  wire                              mem_rsp_valid,
    input  wire [LINE_BITS-1:0]              mem_rsp_data,
    input  wire [CTX_TAG_W-1:0]              mem_rsp_tag,
    output wire                              mem_rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BUF_BITS = RTU_FLAT_LINES * LINE_BITS;
    localparam LB       = RTU_FLAT_LINES_BITS;

    // per-context FSM states
    localparam [3:0] CS_DONE     = 4'd0,   // retired (also idle lanes)
                     CS_HDR_REQ  = 4'd1,   // issue scene-header fetch
                     CS_HDR_WAIT = 4'd2,   // park: header line (triangle_count)
                     CS_REQ0     = 4'd3,   // issue triangle line 0
                     CS_RSP0     = 4'd4,   // park: line 0
                     CS_REQN     = 4'd5,   // issue triangle line N
                     CS_RSPN     = 4'd6,   // park: line N
                     CS_TRI_FEED = 4'd7,   // stream triangle to tri PE
                     CS_TRI_WAIT = 4'd8,   // park: tri result
                     CS_NEXT     = 4'd9;   // advance to next triangle / terminate

    // ── per-context state ─────────────────────────────────────────────
    reg [NUM_CTX-1:0][3:0]                cstate;
    rtu_ray_t [NUM_CTX-1:0]               ray_r;
    reg [NUM_CTX-1:0][31:0]               best_t;
    reg [NUM_CTX-1:0]                     hit_r;
    reg [NUM_CTX-1:0][31:0]               hit_t_r, hit_u_r, hit_v_r, hit_prim_r;
    reg [NUM_CTX-1:0][31:0]               tri_idx, tri_count, cur_off;
    reg [NUM_CTX-1:0][BUF_BITS-1:0]       f_buf;
    reg [NUM_CTX-1:0][LB-1:0]             f_idx, f_total, f_slot;
    reg [NUM_CTX-1:0]                     line_ready, tri_ready;
    reg [NUM_CTX-1:0]                     tri_hit_p, tri_back_p;
    reg [NUM_CTX-1:0][31:0]               tri_t_p, tri_u_p, tri_v_p, tri_prim_p;

    reg                   running, done_r;
    reg [CTX_TAG_W-1:0]   cc;

    // ── micro-step pipeline: SELECT then EXEC ─────────────────────────
    reg                   phase;
    localparam PH_SELECT = 1'b0, PH_EXEC = 1'b1;
    reg [CTX_TAG_W-1:0]   sel_q;
    rtu_ray_t             ray_q;
    reg [BUF_BITS-1:0]    fbuf_q;
    reg [31:0]            curoff_q, bestt_q, triidx_q, tricount_q;
    reg [3:0]             cstate_q;
    reg [LB-1:0]          fidx_q, ftotal_q;

    // ── runnable predicate ────────────────────────────────────────────
    wire [NUM_CTX-1:0] runnable;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_runnable
        reg r;
        always @(*) begin
            case (cstate[i])
                CS_DONE:     r = 1'b0;
                CS_HDR_WAIT,
                CS_RSP0,
                CS_RSPN:     r = line_ready[i];
                CS_TRI_WAIT: r = tri_ready[i];
                default:     r = 1'b1;
            endcase
        end
        assign runnable[i] = r;
    end

    reg [CTX_TAG_W-1:0] sel;
    reg                 sel_valid;
    always @(*) begin
        sel       = cc;
        sel_valid = 1'b0;
        for (integer off = NUM_CTX-1; off >= 0; off = off - 1) begin
            integer cand;
            cand = (32'(cc) + off) % NUM_CTX;
            if (runnable[cand]) begin
                sel       = CTX_TAG_W'(cand);
                sel_valid = 1'b1;
            end
        end
    end
    wire exec = (phase == PH_EXEC);

    // ── combinational decode of the EXEC snapshot ─────────────────────
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] struct_addr = ray_q.scene_base + curoff_q;
    wire [RTU_LINE_SEL_BITS-1:0]      f_off   = struct_addr[RTU_LINE_SEL_BITS-1:0];
    wire [RTU_LINE_SEL_BITS+2:0]      f_shift = {f_off, 3'b000};
    wire [BUF_BITS-1:0]               f_aligned = fbuf_q >> f_shift;

    wire [2:0][31:0] tri_v0, tri_v1, tri_v2;
    for (genvar a = 0; a < 3; ++a) begin : g_tri_v
        assign tri_v0[a] = f_aligned[(RTU_FLAT_OFF_V0 + 4*a)*8 +: 32];
        assign tri_v1[a] = f_aligned[(RTU_FLAT_OFF_V1 + 4*a)*8 +: 32];
        assign tri_v2[a] = f_aligned[(RTU_FLAT_OFF_V2 + 4*a)*8 +: 32];
    end

    wire [31:0] f_off32 = 32'(f_off);
    wire [LB-1:0] tri_lines =
        LB'(((f_off32 + RTU_FLAT_DEC_BYTES - 1) >> RTU_LINE_SEL_BITS) + 1);
    // header line 0: word0 = triangle_count.
    wire [31:0] hdr_count = f_aligned[31:0];

    // ── tri PE: tagged by context id so results route back ────────────
    wire        tri_valid_in = exec && (cstate_q == CS_TRI_FEED);
    wire        tri_valid_out, tri_hit, tri_back;
    wire [CTX_TAG_W-1:0] tri_tag_out;
    wire [31:0] tri_t, tri_u, tri_v;
    VX_rtu_tri_pe #(.TAG_WIDTH (CTX_TAG_W)) tri_pe (
        .clk (clk), .reset (reset), .enable (1'b1), .valid_in (tri_valid_in),
        .tag_in (sel_q),
        .origin (ray_q.origin), .dir (ray_q.dir),
        .v0 (tri_v0), .v1 (tri_v1), .v2 (tri_v2),
        .t_min (ray_q.t_min), .t_max (bestt_q),
        .valid_out (tri_valid_out), .tag_out (tri_tag_out), .hit (tri_hit),
        .t (tri_t), .u (tri_u), .v (tri_v), .back_facing (tri_back)
    );

    // ── memory request (single shared port, tagged by context) ────────
    wire fetch_issue = (cstate_q == CS_HDR_REQ)
                    || (cstate_q == CS_REQ0)
                    || (cstate_q == CS_REQN);
    assign mem_req_valid = exec && fetch_issue;
    assign mem_req_tag   = sel_q;
    assign mem_req_addr  = (cstate_q == CS_HDR_REQ) ? ray_q.scene_base
                         : (cstate_q == CS_REQ0)    ? struct_addr
                         : (struct_addr + (`VX_CFG_MEM_ADDR_WIDTH'(fidx_q) << RTU_LINE_SEL_BITS));
    assign mem_rsp_ready = 1'b1;
    wire mem_req_fire = mem_req_valid && mem_req_ready;

    // face-culling on the committed candidate (matches SimX classify_tri_hit).
    wire cull_back  = (ray_q.flags & `VX_RT_FLAG_CULL_BACK_FACING)  != 0;
    wire cull_front = (ray_q.flags & `VX_RT_FLAG_CULL_FRONT_FACING) != 0;
    wire skip_tris  = (ray_q.flags & `VX_RT_FLAG_SKIP_TRIANGLES)    != 0;

    wire [NUM_CTX-1:0] ctx_done;
    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_ctx_done
        assign ctx_done[i] = (cstate[i] == CS_DONE);
    end
    wire all_done = &ctx_done;

    integer k;
    always @(posedge clk) begin
        if (reset) begin
            running <= 1'b0;
            done_r  <= 1'b0;
            cc      <= '0;
            phase   <= PH_SELECT;
            for (k = 0; k < NUM_CTX; k = k + 1) begin
                cstate[k]     <= CS_DONE;
                line_ready[k] <= 1'b0;
                tri_ready[k]  <= 1'b0;
            end
        end else begin
            done_r <= 1'b0;

            if (!running && start) begin
                running <= 1'b1;
                cc      <= '0;
                phase   <= PH_SELECT;
                for (k = 0; k < NUM_CTX; k = k + 1) begin
                    ray_r[k]      <= rays[k];
                    best_t[k]     <= rays[k].t_max;
                    cur_off[k]    <= '0;
                    tri_idx[k]    <= '0;
                    tri_count[k]  <= '0;
                    line_ready[k] <= 1'b0;
                    tri_ready[k]  <= 1'b0;
                    hit_r[k]      <= 1'b0;
                    hit_t_r[k]    <= rays[k].t_max;
                    hit_u_r[k]    <= '0;
                    hit_v_r[k]    <= '0;
                    hit_prim_r[k] <= '0;
                    cstate[k]     <= mask[k] ? CS_HDR_REQ : CS_DONE;
                end
            end

            // async line-fetch response → route to its context
            if (mem_rsp_valid) begin
                f_buf[mem_rsp_tag][f_slot[mem_rsp_tag] * LINE_BITS +: LINE_BITS] <= mem_rsp_data;
                line_ready[mem_rsp_tag] <= 1'b1;
            end

            // async tri-PE result → route to its context
            if (tri_valid_out) begin
                tri_ready[tri_tag_out]  <= 1'b1;
                tri_hit_p[tri_tag_out]  <= tri_hit;
                tri_back_p[tri_tag_out] <= tri_back;
                tri_t_p[tri_tag_out]    <= tri_t;
                tri_u_p[tri_tag_out]    <= tri_u;
                tri_v_p[tri_tag_out]    <= tri_v;
            end

            if (running) begin
                if (phase == PH_SELECT) begin
                    if (sel_valid) begin
                        sel_q      <= sel;
                        cc         <= sel;
                        ray_q      <= ray_r[sel];
                        fbuf_q     <= f_buf[sel];
                        curoff_q   <= cur_off[sel];
                        bestt_q    <= best_t[sel];
                        cstate_q   <= cstate[sel];
                        fidx_q     <= f_idx[sel];
                        ftotal_q   <= f_total[sel];
                        triidx_q   <= tri_idx[sel];
                        tricount_q <= tri_count[sel];
                        phase      <= PH_EXEC;
                    end
                end else begin
                    phase <= PH_SELECT;
                    case (cstate_q)
                    CS_HDR_REQ: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_HDR_WAIT;
                        end
                    end
                    CS_HDR_WAIT: begin
                        if (line_ready[sel_q]) begin
                            if (hdr_count == 32'd0 || skip_tris) begin
                                cstate[sel_q] <= CS_DONE;
                            end else begin
                                tri_count[sel_q] <= hdr_count;
                                tri_idx[sel_q]   <= '0;
                                cur_off[sel_q]   <= 32'(RTU_SCENE_HDR_BYTES);
                                cstate[sel_q]    <= CS_REQ0;
                            end
                        end
                    end
                    CS_REQ0: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= '0;
                            line_ready[sel_q] <= 1'b0;
                            f_total[sel_q]    <= tri_lines;
                            cstate[sel_q]     <= CS_RSP0;
                        end
                    end
                    CS_RSP0: begin
                        if (line_ready[sel_q]) begin
                            if (ftotal_q == LB'(1)) begin
                                cstate[sel_q] <= CS_TRI_FEED;
                            end else begin
                                f_idx[sel_q]  <= LB'(1);
                                cstate[sel_q] <= CS_REQN;
                            end
                        end
                    end
                    CS_REQN: begin
                        if (mem_req_fire) begin
                            f_slot[sel_q]     <= fidx_q;
                            line_ready[sel_q] <= 1'b0;
                            cstate[sel_q]     <= CS_RSPN;
                        end
                    end
                    CS_RSPN: begin
                        if (line_ready[sel_q]) begin
                            if ((fidx_q + LB'(1)) == ftotal_q) begin
                                cstate[sel_q] <= CS_TRI_FEED;
                            end else begin
                                f_idx[sel_q]  <= fidx_q + LB'(1);
                                cstate[sel_q] <= CS_REQN;
                            end
                        end
                    end
                    CS_TRI_FEED: begin
                        tri_prim_p[sel_q] <= triidx_q;  // flat prim_id = triangle index
                        cstate[sel_q]     <= CS_TRI_WAIT;
                    end
                    CS_TRI_WAIT: begin
                        if (tri_ready[sel_q]) begin
                            tri_ready[sel_q] <= 1'b0;
                            // commit closest opaque hit; honor face culling.
                            if (tri_hit_p[sel_q]
                                && !(tri_back_p[sel_q] && cull_back)
                                && !(!tri_back_p[sel_q] && cull_front)
                                && (tri_t_p[sel_q] < bestt_q)) begin
                                best_t[sel_q]     <= tri_t_p[sel_q];
                                hit_r[sel_q]      <= 1'b1;
                                hit_t_r[sel_q]    <= tri_t_p[sel_q];
                                hit_u_r[sel_q]    <= tri_u_p[sel_q];
                                hit_v_r[sel_q]    <= tri_v_p[sel_q];
                                hit_prim_r[sel_q] <= tri_prim_p[sel_q];
                            end
                            cstate[sel_q] <= CS_NEXT;
                        end
                    end
                    CS_NEXT: begin
                        if ((triidx_q + 32'd1) == tricount_q) begin
                            cstate[sel_q] <= CS_DONE;
                        end else begin
                            tri_idx[sel_q] <= triidx_q + 32'd1;
                            cur_off[sel_q] <= curoff_q + 32'(RTU_TRI_STRIDE);
                            cstate[sel_q]  <= CS_REQ0;
                        end
                    end
                    default:;
                    endcase
                end
            end

            if (running && all_done) begin
                running <= 1'b0;
                done_r  <= 1'b1;
            end
        end
    end

`ifdef DBG_TRACE_RTU
    always @(posedge clk) begin
        if (tri_valid_out) begin
            `TRACE(2, ("%t: %s rtu-flat-tri: ctx=%0d, hit=%0d, t=0x%0h\n",
                $time, INSTANCE_ID, tri_tag_out, tri_hit, tri_t))
        end
        if (done_r) begin
            `TRACE(1, ("%t: %s rtu-flat-done\n", $time, INSTANCE_ID))
        end
    end
`endif

    for (genvar i = 0; i < NUM_CTX; ++i) begin : g_res
        assign res_hit[i]  = hit_r[i];
        assign res_t[i]    = hit_t_r[i];
        assign res_u[i]    = hit_u_r[i];
        assign res_v[i]    = hit_v_r[i];
        assign res_prim[i] = hit_prim_r[i];
        assign res_geom[i] = 32'd0;   // flat scenes report geometry 0
    end

    assign busy = running;
    assign done = done_r;

endmodule
