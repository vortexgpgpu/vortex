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

// VX_tex_unit — per-core SFU PE that decodes the vx_tex SFU op, emits one
// tex_bus_if request to the cluster-shared TEX core, and forwards the returned
// texel to result_if.
//
//   vx_tex : u,v,lod from rs1/rs2/rs3; texel -> rd. (R4-type.)
//
// The unit does not touch the graphics window. It used to also serve vx_tex4, a
// windowed form whose quad mode gave one thread a whole 2x2 quad: eight (u,v)
// operands — more than any RISC-V encoding holds — spilled into window slots, a
// hardware mip-LOD tree over the quad derivatives, and a sequencer that walked
// the four fragments one at a time while holding execute_if. That cost the window
// two full RAM mirrors and serialised four samples behind one op. A shader that
// owns a quad now computes the same integer LOD itself with vx_tex_quad_lod()
// (sw/common/vx_tex_lod.h — already the bit-exact source of truth this RTL was
// written against) and issues four independent vx_tex ops, which the scoreboard
// pipelines.

`include "VX_tex_define.vh"

module VX_tex_unit import VX_gpu_pkg::*, VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // SFU PE-style request/response interfaces (sfu_execute_t / sfu_result_t)
    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

    // Cluster-side texture bus (master)
    VX_tex_bus_if.master    tex_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)
    localparam REQ_QUEUE_BITS = `LOG2UP(`VX_CFG_TEX_REQ_QUEUE_SIZE);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);

    // ── coords + lod (all from registers) ─────────────────────────────────
    wire [1:0][NUM_LANES-1:0][31:0]        sfu_exe_coords;
    wire [NUM_LANES-1:0][TEX_LOD_BITS-1:0] sfu_exe_lod;
    wire [TEX_STAGE_BITS-1:0]              sfu_exe_stage;
    assign sfu_exe_stage = execute_if.data.op_args.tex.stage;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sfu_exe_coords
        assign sfu_exe_coords[0][i] = execute_if.data.rs1_data[i][31:0];              // u
        assign sfu_exe_coords[1][i] = execute_if.data.rs2_data[i][31:0];              // v
        assign sfu_exe_lod[i]       = execute_if.data.rs3_data[i][0 +: TEX_LOD_BITS]; // lod
    end

    // ── tag-store echo (round-trips the header past the texel round-trip) ──
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

    wire [REQ_QUEUE_BITS-1:0] mdata_waddr, mdata_raddr;
    wire mdata_full;

    // ── request submit ────────────────────────────────────────────────────
    // One request per op, accepted as soon as the bus and the tag store allow.
    wire resp_fire = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;
    wire valid_in, ready_in;
    assign valid_in = execute_if.valid && ~mdata_full;
    wire req_fire = valid_in && ready_in;

    assign execute_if.ready = ready_in && ~mdata_full;

    wire mdata_push = req_fire;          // acquire a tag per issued request
    wire mdata_pop  = resp_fire;         // release a tag per response

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
        .DATAW   (NUM_LANES * (1 + 2 * 32 + TEX_LOD_BITS) + TEX_STAGE_BITS + TEX_REQ_TAG_WIDTH),
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

    // The texel goes straight to rd; there is no window writeback to gate on.
    wire rsp_buf_rdy;
    assign tex_bus_if.rsp_ready = rsp_buf_rdy;

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_bus_if.rsp_valid),
        .ready_in  (rsp_buf_rdy),
        .data_in   (rsp_data_in),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (req_fire) begin
            `TRACE(1, ("%d: %s tex-req: wid=%0d, PC=0x%0h, tmask=%b, stage=%0d, ibuf_idx=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, sfu_exe_stage, mdata_waddr, execute_if.data.header.uuid))
        end
        if (resp_fire) begin
            `TRACE(1, ("%d: %s tex-rsp: wid=%0d, tmask=%b, ibuf_idx=%0d (#%0d)\n",
                $time, INSTANCE_ID, out_echo.wid, out_echo.tmask, mdata_raddr, rsp_uuid))
        end
    end
`endif

endmodule
