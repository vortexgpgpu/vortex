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

// VX_om_unit — per-core SFU PE for vx_om4 (the sole OM op). One thread owns a
// 2x2 quad: its four colours/depths live in the shared graphics window (staged
// by SETW) at slots base..base+3 (colour) and base+4..base+7 (depth). rs1 is the
// quad descriptor (cov_mask[3:0], quad origin qx@[4 +: 14] / qy@[18 +: 13], face
// in bit 31); rs2 is the window slot base. The unit reads the payload window and
// emits one om_bus request per covered sub-pixel F (skipping sub-pixels no lane
// covers), each carrying all active lanes' fragment-F data, then retires the op
// (rd=x0, fire-and-forget — no response, no window writeback). VX_om_core / the
// om_bus are unchanged: they already take a per-lane masked {pos,colour,depth,face}.

`include "VX_om_define.vh"

module VX_om_unit import VX_gpu_pkg::*, VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter CONS_RD_PORTS = 8
) (
    input wire clk,
    input wire reset,

    // SFU PE-style interfaces
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // Shared graphics-window read ports (driven via the VX_sfu_unit mux): the
    // unit fetches colour[0..3] (ports 0..3) and depth[0..3] (ports 4..7) for
    // all lanes from the contiguous slot window at the rs2 base.
    VX_gfx_win_rd_if.master                                        cons_rd_if,

    // Cluster-side OM bus (master)
    VX_om_bus_if.master     om_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    localparam LANE_BITS   = `CLOG2(NUM_LANES);
    localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam SLOT_BITS   = `CLOG2(`VX_RT_SLOT_COUNT);
    localparam QX_BITS     = `VX_RASTER_DIM_BITS-1;   // quad-x field width (14)
    localparam QY_BITS     = `VX_RASTER_DIM_BITS-2;   // quad-y field width (13; bit 31 = face)

    wire [PID_W-1:0]       in_pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0] in_tbase = THREAD_BITS'(in_pid) << LANE_BITS;

    // ── window read: 8 contiguous slots from the rs2 base ─────────────────
    wire [SLOT_BITS-1:0] in_slot = execute_if.data.rs2_data[0][SLOT_BITS-1:0];
    assign cons_rd_if.req.wid   = execute_if.data.header.wid;
    assign cons_rd_if.req.tbase = in_tbase;
    for (genvar p = 0; p < CONS_RD_PORTS; ++p) begin : g_cons_rd_slot
        assign cons_rd_if.req.slot[p] = in_slot + SLOT_BITS'(p);
    end

    // ── per-lane descriptor decode (rs1) ──────────────────────────────────
    wire [NUM_LANES-1:0] act = execute_if.data.header.tmask[in_tbase +: NUM_LANES];
    wire [NUM_LANES-1:0][3:0]         cov;
    wire [NUM_LANES-1:0][QX_BITS-1:0] qx;
    wire [NUM_LANES-1:0][QY_BITS-1:0] qy;
    wire [NUM_LANES-1:0]              face;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_desc
        wire [31:0] desc = execute_if.data.rs1_data[i][31:0];
        assign cov[i]  = desc[3:0];
        assign qx[i]   = desc[4 +: QX_BITS];
        assign qy[i]   = desc[4 + QX_BITS +: QY_BITS];
        assign face[i] = desc[31];
    end

    // ── 4-sub-pixel sequencer (one om_bus request per covered sub-pixel) ───
    reg [1:0] q_frag;     // current sub-pixel 0..3
    reg       last_sent;  // sub-pixel 3 handled, retire stalled on result_if.ready

    // current sub-pixel's per-lane coverage and payload
    wire [NUM_LANES-1:0]                        fcov;
    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0]   fpos_x;
    wire [NUM_LANES-1:0][`VX_OM_DIM_BITS-1:0]   fpos_y;
    wire [NUM_LANES-1:0][31:0]                  fcolor;
    wire [NUM_LANES-1:0][`VX_OM_DEPTH_BITS-1:0] fdepth;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_frag
        assign fcov[i]   = act[i] & cov[i][q_frag];
        assign fpos_x[i] = `VX_OM_DIM_BITS'({qx[i], q_frag[0]});   // (qx<<1)|F[0]
        assign fpos_y[i] = `VX_OM_DIM_BITS'({qy[i], q_frag[1]});   // (qy<<1)|F[1]
        assign fcolor[i] = cons_rd_if.data[{1'b0, q_frag}][i];        // ports 0..3
        assign fdepth[i] = cons_rd_if.data[{1'b1, q_frag}][i][`VX_OM_DEPTH_BITS-1:0]; // ports 4..7
    end
    wire frag_any = |fcov;

    // Push the current covered sub-pixel into the OM-bus request buffer (skipped
    // for free when empty, or when sub-pixel 3 is already buffered and we are
    // only waiting to retire). The elastic buffer (SIZE 2, OUT_REG 2) registers
    // the om_bus handshake exactly as the gfx-v1 single-fragment path did — the
    // sequencer just feeds it one sub-pixel at a time.
    wire push = execute_if.valid && ~last_sent && frag_any;
    wire buf_ready;
    VX_elastic_buffer #(
        .DATAW   (UUID_WIDTH + NUM_LANES * (1 + 2 * `VX_OM_DIM_BITS + 32 + `VX_OM_DEPTH_BITS + 1)),
        .SIZE    (2),
        .OUT_REG (2)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (push),
        .ready_in  (buf_ready),
        .data_in   ({execute_if.data.header.uuid, fcov, fpos_x, fpos_y, fcolor, fdepth, face}),
        .data_out  ({om_bus_if.req_data.uuid, om_bus_if.req_data.mask,
                     om_bus_if.req_data.pos_x, om_bus_if.req_data.pos_y,
                     om_bus_if.req_data.color, om_bus_if.req_data.depth, om_bus_if.req_data.face}),
        .valid_out (om_bus_if.req_valid),
        .ready_out (om_bus_if.req_ready)
    );

    // Current sub-pixel handled this cycle: an empty one is skipped for free; a
    // covered one needs the request buffer to accept it.
    wire frag_handled = execute_if.valid && ~last_sent && (~frag_any || (push && buf_ready));
    wire is_last = (q_frag == 2'd3);

    // Retire once sub-pixel 3 is handled; no return data (rd=x0).
    sfu_result_t rsp_data_in;
    assign rsp_data_in.header = execute_if.data.header;
    assign rsp_data_in.data   = '0;
    assign result_if.valid = last_sent || (frag_handled && is_last);
    assign result_if.data  = rsp_data_in;
    wire retire = result_if.valid && result_if.ready;
    assign execute_if.ready = retire;

    always @(posedge clk) begin
        if (reset) begin
            q_frag    <= 2'd0;
            last_sent <= 1'b0;
        end else begin
            if (frag_handled && ~is_last) q_frag <= q_frag + 2'd1;
            if (frag_handled && is_last && ~result_if.ready) last_sent <= 1'b1;
            if (retire) begin
                q_frag    <= 2'd0;
                last_sent <= 1'b0;
            end
        end
    end

`ifdef DBG_TRACE_OM
    always @(posedge clk) begin
        if (push && buf_ready) begin
            `TRACE(1, ("%d: %s om-req: wid=%0d, PC=0x%0h, frag=%0d, cov=%b (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                q_frag, fcov, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
