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

// VX_tex_unit — per-core SFU PE that decodes vx_tex SFU ops, emits a
// tex_bus_if request to the cluster-shared TEX unit, and forwards the
// returned texels to the result_if pipeline.
//
// Header fields live under `execute_if.data.header.*` and
// `op_args.tex.stage` carries the texture stage in funct2 of the
// CUSTOM1 R4-type encoding.

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
    output wire [NW_WIDTH-1:0]                                     cons_rd_wid,
    output wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                  cons_rd_tbase,
    output wire [CONS_RD_PORTS-1:0][`CLOG2(`VX_RT_SLOT_COUNT)-1:0] cons_rd_slot,
    input  wire [CONS_RD_PORTS-1:0][NUM_LANES-1:0][31:0]           cons_rd_data,
    output wire                                                    cons_wr_en,
    output wire [NW_WIDTH-1:0]                                     cons_wr_wid,
    output wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                  cons_wr_tbase,
    output wire [NUM_LANES-1:0]                                    cons_wr_mask,
    output wire [`CLOG2(`VX_RT_SLOT_COUNT)-1:0]                    cons_wr_slot,
    output wire [NUM_LANES-1:0][31:0]                              cons_wr_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)
    localparam REQ_QUEUE_BITS = `LOG2UP(`VX_CFG_TEX_REQ_QUEUE_SIZE);
    localparam LANE_BITS   = `CLOG2(NUM_LANES);
    localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);
    localparam PID_W       = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);

    wire is_tex4 = execute_if.data.op_args.tex.is_tex4;
    wire [PID_W-1:0]       in_pid   = execute_if.data.header.pid;
    wire [THREAD_BITS-1:0] in_tbase = THREAD_BITS'(in_pid) << LANE_BITS;

    // Stash header bits in a tag-indexed buffer so they round-trip with
    // the texture response.

    wire [1:0][NUM_LANES-1:0][31:0]            sfu_exe_coords;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] sfu_exe_lod;
    wire [`VX_TEX_STAGE_BITS-1:0]              sfu_exe_stage;

    // Header echo (uuid travels in tag; remaining fields go through tag-store)
    typedef struct packed {
        logic [NW_WIDTH-1:0]                                           wid;
        logic [NUM_LANES-1:0]                                          tmask;
        logic [`LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES)-1:0]                  pid;
        logic                                                          sop;
        logic                                                          eop;
        logic [PC_BITS-1:0]                                            PC;
        logic                                                          wb;
        logic [NUM_XREGS-1:0]                                          wr_xregs;
        logic [NUM_REGS_BITS-1:0]                                      rd;
        logic [BYTESEL_BITS-1:0]                                       bytesel;
        logic                                                          is_tex4;
        logic [`CLOG2(`VX_RT_SLOT_COUNT)-1:0]                          out_slot;
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
    assign in_echo.out_slot  = execute_if.data.op_args.tex.out_slot[`CLOG2(`VX_RT_SLOT_COUNT)-1:0];

    wire [REQ_QUEUE_BITS-1:0] mdata_waddr, mdata_raddr;
    wire mdata_full;

    assign sfu_exe_stage = execute_if.data.op_args.tex.stage;

    // vx_tex4 sources u,v from the window read port and lod from rs1; legacy
    // vx_tex sources u,v from rs1/rs2 and lod from rs3.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sfu_exe_coords
        assign sfu_exe_coords[0][i] = is_tex4 ? cons_rd_data[0][i] : execute_if.data.rs1_data[i][31:0];
        assign sfu_exe_coords[1][i] = is_tex4 ? cons_rd_data[1][i] : execute_if.data.rs2_data[i][31:0];
        assign sfu_exe_lod[i]       = is_tex4 ? execute_if.data.rs1_data[i][0 +: `VX_TEX_LOD_BITS]
                                              : execute_if.data.rs3_data[i][0 +: `VX_TEX_LOD_BITS];
    end

    // vx_tex4 window read: the input slot base is the warp-uniform rs2 value;
    // u at in_slot, v at in_slot+1, for the issuing warp.
    wire [`CLOG2(`VX_RT_SLOT_COUNT)-1:0] in_slot = execute_if.data.rs2_data[0][`CLOG2(`VX_RT_SLOT_COUNT)-1:0];
    assign cons_rd_wid     = execute_if.data.header.wid;
    assign cons_rd_tbase   = in_tbase;
    assign cons_rd_slot[0] = in_slot;
    assign cons_rd_slot[1] = in_slot + 1'b1;

    wire mdata_push = execute_if.valid && execute_if.ready;
    wire mdata_pop  = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;

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

    // ---- submit texture request -------------------------------------------

    wire valid_in, ready_in;
    assign valid_in = execute_if.valid && ~mdata_full;
    assign execute_if.ready = ready_in && ~mdata_full;

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

    // ---- handle texture response ------------------------------------------

    assign mdata_raddr = tex_bus_if.rsp_data.tag[0 +: REQ_QUEUE_BITS];

    wire [UUID_WIDTH-1:0]      rsp_uuid = tex_bus_if.rsp_data.tag[REQ_QUEUE_BITS +: UUID_WIDTH];
    wire [NUM_LANES-1:0][31:0] rsp_texels = tex_bus_if.rsp_data.texels;

    // Reassemble the result header from echo + uuid carried in the tag.
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

    wire                                 rsp_is_tex4;
    wire [`CLOG2(`VX_RT_SLOT_COUNT)-1:0] rsp_out_slot;
    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t) + 1 + `CLOG2(`VX_RT_SLOT_COUNT)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_bus_if.rsp_valid),
        .ready_in  (tex_bus_if.rsp_ready),
        .data_in   ({rsp_data_in, out_echo.is_tex4, out_echo.out_slot}),
        .data_out  ({result_if.data, rsp_is_tex4, rsp_out_slot}),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    // vx_tex4: also land the texel in the window output slot as the op retires.
    // The rd writeback (above) is the scoreboard sync handle a chained GETW
    // depends on, so the window write has committed by the time GETW reads it.
    wire result_fire = result_if.valid && result_if.ready;
    assign cons_wr_en    = result_fire && rsp_is_tex4;
    assign cons_wr_wid   = result_if.data.header.wid;
    assign cons_wr_tbase = THREAD_BITS'(result_if.data.header.pid) << LANE_BITS;
    assign cons_wr_mask  = result_if.data.header.tmask;
    assign cons_wr_slot  = rsp_out_slot;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cons_wr
        assign cons_wr_data[i] = result_if.data.data[i][31:0];
    end

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: %s tex-req: wid=%0d, PC=0x%0h, tmask=%b, stage=%0d, ibuf_idx=%0d (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.PC,
                execute_if.data.header.tmask, sfu_exe_stage, mdata_waddr, execute_if.data.header.uuid))
        end
        if (tex_bus_if.rsp_valid && tex_bus_if.rsp_ready) begin
            `TRACE(1, ("%d: %s tex-rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, ibuf_idx=%0d (#%0d)\n",
                $time, INSTANCE_ID, out_echo.wid, out_echo.PC, out_echo.tmask, out_echo.rd,
                mdata_raddr, rsp_uuid))
        end
    end
`endif

endmodule
