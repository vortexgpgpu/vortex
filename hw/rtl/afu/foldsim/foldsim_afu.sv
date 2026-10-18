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

`include "foldsim_sim.vh"
`include "foldsim_kernel.vh"
`include "foldsim_ports.vh"

// The foldsim kernel: the simulator CIRCT composes of Vortex (foldsim_sim, of
// the target foldsim_target), a memory endpoint on each of its AXI4 ports and
// the control endpoint on its DCR, start and busy signals. Its ports are the
// native V80 kernel's, but for the host-memory master it does not need: one
// device-memory master per memory port of the target (FOLDSIM_MEM_PORTS, from
// foldsim_kernel.vh, with the ports' id width), each its own way into device
// memory.
module foldsim_afu #(
    parameter C_S_AXI_CONTROL_ADDR_WIDTH = 16,
    parameter C_S_AXI_CONTROL_DATA_WIDTH = 32,
    parameter C_M_AXI_MEM_ID_WIDTH       = 1,
    parameter C_M_AXI_MEM_DATA_WIDTH     = 512,
    parameter C_M_AXI_MEM_ADDR_WIDTH     = 64,
    // The target's AXI4 ports.
    parameter TARGET_ID_W                = `FOLDSIM_MEM_ID_W,
    parameter TARGET_ADDR_W              = 64,
    parameter DCR_ADDR_W                 = 12,
    parameter DCR_DATA_W                 = 32
) (
    input  wire                                    ap_clk,
    input  wire                                    ap_rst_n,

`define FOLDSIM_AXI_MEM(i) \
    output wire                                    m_axi_mem_``i``_awvalid, \
    input  wire                                    m_axi_mem_``i``_awready, \
    output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]       m_axi_mem_``i``_awaddr, \
    output wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_``i``_awid, \
    output wire [7:0]                              m_axi_mem_``i``_awlen, \
    output wire                                    m_axi_mem_``i``_wvalid, \
    input  wire                                    m_axi_mem_``i``_wready, \
    output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]       m_axi_mem_``i``_wdata, \
    output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0]     m_axi_mem_``i``_wstrb, \
    output wire                                    m_axi_mem_``i``_wlast, \
    output wire                                    m_axi_mem_``i``_arvalid, \
    input  wire                                    m_axi_mem_``i``_arready, \
    output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]       m_axi_mem_``i``_araddr, \
    output wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_``i``_arid, \
    output wire [7:0]                              m_axi_mem_``i``_arlen, \
    input  wire                                    m_axi_mem_``i``_rvalid, \
    output wire                                    m_axi_mem_``i``_rready, \
    input  wire [C_M_AXI_MEM_DATA_WIDTH-1:0]       m_axi_mem_``i``_rdata, \
    input  wire                                    m_axi_mem_``i``_rlast, \
    input  wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_``i``_rid, \
    input  wire [1:0]                              m_axi_mem_``i``_rresp, \
    input  wire                                    m_axi_mem_``i``_bvalid, \
    output wire                                    m_axi_mem_``i``_bready, \
    input  wire [1:0]                              m_axi_mem_``i``_bresp, \
    input  wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_``i``_bid

    `FOLDSIM_REPEAT(`FOLDSIM_MEM_PORTS, FOLDSIM_AXI_MEM, FOLDSIM_COMMA),

    input  wire                                    s_axi_control_awvalid,
    output wire                                    s_axi_control_awready,
    input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_awaddr,
    input  wire                                    s_axi_control_wvalid,
    output wire                                    s_axi_control_wready,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_wdata,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_wstrb,
    input  wire                                    s_axi_control_arvalid,
    output wire                                    s_axi_control_arready,
    input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_araddr,
    output wire                                    s_axi_control_rvalid,
    input  wire                                    s_axi_control_rready,
    output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_rdata,
    output wire [1:0]                              s_axi_control_rresp,
    output wire                                    s_axi_control_bvalid,
    input  wire                                    s_axi_control_bready,
    output wire [1:0]                              s_axi_control_bresp
);
`undef FOLDSIM_AXI_MEM
    localparam DATA_W = C_M_AXI_MEM_DATA_WIDTH;
    localparam PORTS  = `FOLDSIM_MEM_PORTS;

    // A registered reset, so it reaches the whole kernel in one cycle's time.
    reg reset;
    always @(posedge ap_clk)
        reset <= ~ap_rst_n;

    // ---- The simulator's channels, named after the target's ports ----
`define FOLDSIM_CH(name, w) \
    wire name``_valid, name``_ready; \
    wire [w-1:0] name``_bits;

    `FOLDSIM_CH(reset, 1)
    `FOLDSIM_CH(dcr_req_valid, 1)
    `FOLDSIM_CH(dcr_req_rw, 1)
    `FOLDSIM_CH(dcr_req_addr, DCR_ADDR_W)
    `FOLDSIM_CH(dcr_req_data, DCR_DATA_W)
    `FOLDSIM_CH(start, 1)
    `FOLDSIM_CH(dcr_rsp_valid, 1)
    `FOLDSIM_CH(dcr_rsp_data, DCR_DATA_W)
    `FOLDSIM_CH(busy, 1)
`undef FOLDSIM_CH

    // A memory port's channels, by port.
`define FOLDSIM_MCH(name, w) \
    wire         mem_``name``_valid [PORTS]; \
    wire         mem_``name``_ready [PORTS]; \
    wire [w-1:0] mem_``name``_bits  [PORTS];

    `FOLDSIM_MCH(awready, 1)
    `FOLDSIM_MCH(wready, 1)
    `FOLDSIM_MCH(arready, 1)
    `FOLDSIM_MCH(rvalid, 1)
    `FOLDSIM_MCH(rdata, DATA_W)
    `FOLDSIM_MCH(rlast, 1)
    `FOLDSIM_MCH(rid, TARGET_ID_W)
    `FOLDSIM_MCH(rresp, 2)
    `FOLDSIM_MCH(bvalid, 1)
    `FOLDSIM_MCH(bresp, 2)
    `FOLDSIM_MCH(bid, TARGET_ID_W)
    `FOLDSIM_MCH(awvalid, 1)
    `FOLDSIM_MCH(awaddr, TARGET_ADDR_W)
    `FOLDSIM_MCH(awid, TARGET_ID_W)
    `FOLDSIM_MCH(awlen, 8)
    `FOLDSIM_MCH(wvalid, 1)
    `FOLDSIM_MCH(wdata, DATA_W)
    `FOLDSIM_MCH(wstrb, DATA_W/8)
    `FOLDSIM_MCH(wlast, 1)
    `FOLDSIM_MCH(arvalid, 1)
    `FOLDSIM_MCH(araddr, TARGET_ADDR_W)
    `FOLDSIM_MCH(arid, TARGET_ID_W)
    `FOLDSIM_MCH(arlen, 8)
    `FOLDSIM_MCH(rready, 1)
    `FOLDSIM_MCH(bready, 1)
`undef FOLDSIM_MCH

    // Device memory, by port.
    wire                   hbm_awvalid [PORTS], hbm_awready [PORTS];
    wire [63:0]            hbm_awaddr  [PORTS];
    wire [7:0]             hbm_awlen   [PORTS];
    wire                   hbm_wvalid  [PORTS], hbm_wready [PORTS], hbm_wlast [PORTS];
    wire [DATA_W-1:0]      hbm_wdata   [PORTS];
    wire [DATA_W/8-1:0]    hbm_wstrb   [PORTS];
    wire                   hbm_bvalid  [PORTS], hbm_bready [PORTS];
    wire                   hbm_arvalid [PORTS], hbm_arready [PORTS];
    wire [63:0]            hbm_araddr  [PORTS];
    wire [7:0]             hbm_arlen   [PORTS];
    wire                   hbm_rvalid  [PORTS], hbm_rready [PORTS];
    wire [DATA_W-1:0]      hbm_rdata   [PORTS];

`define FOLDSIM_HBM(i) \
    assign m_axi_mem_``i``_awvalid = hbm_awvalid[i]; \
    assign hbm_awready[i]          = m_axi_mem_``i``_awready; \
    assign m_axi_mem_``i``_awaddr  = hbm_awaddr[i]; \
    assign m_axi_mem_``i``_awid    = '0; \
    assign m_axi_mem_``i``_awlen   = hbm_awlen[i]; \
    assign m_axi_mem_``i``_wvalid  = hbm_wvalid[i]; \
    assign hbm_wready[i]           = m_axi_mem_``i``_wready; \
    assign m_axi_mem_``i``_wdata   = hbm_wdata[i]; \
    assign m_axi_mem_``i``_wstrb   = hbm_wstrb[i]; \
    assign m_axi_mem_``i``_wlast   = hbm_wlast[i]; \
    assign hbm_bvalid[i]           = m_axi_mem_``i``_bvalid; \
    assign m_axi_mem_``i``_bready  = hbm_bready[i]; \
    assign m_axi_mem_``i``_arvalid = hbm_arvalid[i]; \
    assign hbm_arready[i]          = m_axi_mem_``i``_arready; \
    assign m_axi_mem_``i``_araddr  = hbm_araddr[i]; \
    assign m_axi_mem_``i``_arid    = '0; \
    assign m_axi_mem_``i``_arlen   = hbm_arlen[i]; \
    assign hbm_rvalid[i]           = m_axi_mem_``i``_rvalid; \
    assign m_axi_mem_``i``_rready  = hbm_rready[i]; \
    assign hbm_rdata[i]            = m_axi_mem_``i``_rdata

    `FOLDSIM_REPEAT(`FOLDSIM_MEM_PORTS, FOLDSIM_HBM, FOLDSIM_SEMI);
`undef FOLDSIM_HBM

`define FOLDSIM_IN(name) \
    .name``_valid (name``_valid), .name``_ready (name``_ready), .name``_bits (name``_bits)
`define FOLDSIM_MIN(i, name) \
    .m_axi_mem_``i``_``name``_valid (mem_``name``_valid[i]), \
    .m_axi_mem_``i``_``name``_ready (mem_``name``_ready[i]), \
    .m_axi_mem_``i``_``name``_bits  (mem_``name``_bits[i])
`define FOLDSIM_SIM_MEM(i) \
    `FOLDSIM_MIN(i, awready), `FOLDSIM_MIN(i, wready), `FOLDSIM_MIN(i, arready), \
    `FOLDSIM_MIN(i, rvalid), `FOLDSIM_MIN(i, rdata), `FOLDSIM_MIN(i, rlast), `FOLDSIM_MIN(i, rid), \
    `FOLDSIM_MIN(i, rresp), `FOLDSIM_MIN(i, bvalid), `FOLDSIM_MIN(i, bresp), `FOLDSIM_MIN(i, bid), \
    `FOLDSIM_MIN(i, awvalid), `FOLDSIM_MIN(i, awaddr), `FOLDSIM_MIN(i, awid), `FOLDSIM_MIN(i, awlen), \
    `FOLDSIM_MIN(i, wvalid), `FOLDSIM_MIN(i, wdata), `FOLDSIM_MIN(i, wstrb), `FOLDSIM_MIN(i, wlast), \
    `FOLDSIM_MIN(i, arvalid), `FOLDSIM_MIN(i, araddr), `FOLDSIM_MIN(i, arid), `FOLDSIM_MIN(i, arlen), \
    `FOLDSIM_MIN(i, rready), `FOLDSIM_MIN(i, bready)

    wire [`FOLDSIM_NUM_CLASSES-1:0] profile_finish /*verilator public_flat_rd*/;

    foldsim_sim sim (
        .hostClock (ap_clk),
        .hostReset (reset),
        .profile_finish (profile_finish),
        `FOLDSIM_REPEAT(`FOLDSIM_MEM_PORTS, FOLDSIM_SIM_MEM, FOLDSIM_COMMA),
        `FOLDSIM_IN(reset),
        `FOLDSIM_IN(dcr_req_valid),
        `FOLDSIM_IN(dcr_req_rw),
        `FOLDSIM_IN(dcr_req_addr),
        `FOLDSIM_IN(dcr_req_data),
        `FOLDSIM_IN(start),
        `FOLDSIM_IN(dcr_rsp_valid),
        `FOLDSIM_IN(dcr_rsp_data),
        `FOLDSIM_IN(busy)
    );
`undef FOLDSIM_SIM_MEM
`undef FOLDSIM_MIN

    // Each memory endpoint's bank model: read slots, and a ring of 2^9 DRAM
    // clocks (MEM_CAPS reports both).
    localparam MEM_SLOTS = 32;
    localparam MEM_HLOG2 = 9;
    // The memory endpoints' token ends register have and done, so their finish
    // starts at their own state, not at the simulator's handshake. A token then
    // counts a host cycle later; s2w4i2's host cycles do not move.
    localparam MEM_STAGED = 1;
    // Ports, engines and device-memory masters: one of each per port.
    localparam [31:0] MEM_CAPS = {4'd0, 4'(PORTS), 4'(PORTS), 8'(MEM_SLOTS), 4'(MEM_HLOG2 - 4), 8'(PORTS)};

    wire [31:0] mem_read_latency, mem_write_latency;
    wire [7:0]  mem_max_reads, mem_max_writes;
    wire [63:0] mem_base;
    wire        mem_model_bank;
    wire [31:0] mem_clock, mem_queue;
    wire [15:0] mem_latency, mem_wmark;
    wire [12:0] mem_geom;
    wire [18*12-1:0] mem_timing;
    wire [PORTS*8*64-1:0] mem_stats;
    wire [1:0]  mem_refresh;
    wire [15:0] mem_refi, mem_rfc;
    wire [PORTS-1:0] mem_idle, mem_error, mem_stall;
    wire [PORTS-1:0] mem_reset_valid, mem_reset_ready, mem_reset_bits;

`define FOLDSIM_MEM(name) \
    .name``_valid (mem_``name``_valid[p]), .name``_ready (mem_``name``_ready[p]), \
    .name``_bits (mem_``name``_bits[p])

    for (genvar p = 0; p < PORTS; ++p) begin : g_mem
        foldsim_mem #(
            .ID_W     (TARGET_ID_W),
            .ADDR_W   (TARGET_ADDR_W),
            .DATA_W   (DATA_W),
            .SLOTS    (MEM_SLOTS),
            .HLOG2    (MEM_HLOG2),
            .STAGED   (MEM_STAGED),
            .CH_SHIFT (`FOLDSIM_MEM_CH_SHIFT)
        ) mem (
            .clk           (ap_clk),
            .reset         (reset),
            .read_latency  (mem_read_latency),
            .write_latency (mem_write_latency),
            .max_reads     (mem_max_reads),
            .max_writes    (mem_max_writes),
            .base          (mem_base),
            .model_bank    (mem_model_bank),
            .cfg_data_rate (mem_clock[15:0]),
            .cfg_core_clock(mem_clock[31:16]),
            .cfg_latency   (mem_latency),
            .cfg_queue     (mem_queue[15:0]),
            .cfg_rq        (mem_queue[23:16]),
            .cfg_wq        (mem_queue[31:24]),
            .cfg_wr_high   (mem_wmark[7:0]),
            .cfg_wr_low    (mem_wmark[15:8]),
            .cfg_geom      (mem_geom),
            .cfg_t         (mem_timing),
            .cfg_refresh   (mem_refresh),
            .cfg_refi      (mem_refi),
            .cfg_rfc       (mem_rfc),
            .stats         (mem_stats[p*8*64 +: 8*64]),
            .idle          (mem_idle[p]),
            .error         (mem_error[p]),
            .stall         (mem_stall[p]),
            .treset_valid  (mem_reset_valid[p]),
            .treset_ready  (mem_reset_ready[p]),
            .treset_bits   (mem_reset_bits[p]),
            `FOLDSIM_MEM(awvalid), `FOLDSIM_MEM(awaddr), `FOLDSIM_MEM(awid), `FOLDSIM_MEM(awlen),
            `FOLDSIM_MEM(wvalid), `FOLDSIM_MEM(wdata), `FOLDSIM_MEM(wstrb), `FOLDSIM_MEM(wlast),
            `FOLDSIM_MEM(arvalid), `FOLDSIM_MEM(araddr), `FOLDSIM_MEM(arid), `FOLDSIM_MEM(arlen),
            `FOLDSIM_MEM(rready), `FOLDSIM_MEM(bready),
            `FOLDSIM_MEM(awready), `FOLDSIM_MEM(wready), `FOLDSIM_MEM(arready),
            `FOLDSIM_MEM(rvalid), `FOLDSIM_MEM(rdata), `FOLDSIM_MEM(rlast), `FOLDSIM_MEM(rid), `FOLDSIM_MEM(rresp),
            `FOLDSIM_MEM(bvalid), `FOLDSIM_MEM(bresp), `FOLDSIM_MEM(bid),
            .m_awvalid (hbm_awvalid[p]),
            .m_awready (hbm_awready[p]),
            .m_awaddr  (hbm_awaddr[p]),
            .m_awlen   (hbm_awlen[p]),
            .m_wvalid  (hbm_wvalid[p]),
            .m_wready  (hbm_wready[p]),
            .m_wdata   (hbm_wdata[p]),
            .m_wstrb   (hbm_wstrb[p]),
            .m_wlast   (hbm_wlast[p]),
            .m_bvalid  (hbm_bvalid[p]),
            .m_bready  (hbm_bready[p]),
            .m_arvalid (hbm_arvalid[p]),
            .m_arready (hbm_arready[p]),
            .m_araddr  (hbm_araddr[p]),
            .m_arlen   (hbm_arlen[p]),
            .m_rvalid  (hbm_rvalid[p]),
            .m_rready  (hbm_rready[p]),
            .m_rdata   (hbm_rdata[p])
        );
    end
`undef FOLDSIM_MEM

    foldsim_ctrl #(
        .DCR_ADDR_W  (DCR_ADDR_W),
        .DCR_DATA_W  (DCR_DATA_W),
        .NUM_CLASSES (`FOLDSIM_NUM_CLASSES),
        .MEM_PORTS   (PORTS),
        .MEM_CAPS    (MEM_CAPS)
    ) ctrl (
        .clk               (ap_clk),
        .reset             (reset),
        .s_awvalid         (s_axi_control_awvalid),
        .s_awready         (s_axi_control_awready),
        .s_awaddr          (s_axi_control_awaddr[11:0]),
        .s_wvalid          (s_axi_control_wvalid),
        .s_wready          (s_axi_control_wready),
        .s_wdata           (s_axi_control_wdata),
        .s_bvalid          (s_axi_control_bvalid),
        .s_bready          (s_axi_control_bready),
        .s_arvalid         (s_axi_control_arvalid),
        .s_arready         (s_axi_control_arready),
        .s_araddr          (s_axi_control_araddr[11:0]),
        .s_rvalid          (s_axi_control_rvalid),
        .s_rready          (s_axi_control_rready),
        .s_rdata           (s_axi_control_rdata),
        .mem_read_latency  (mem_read_latency),
        .mem_write_latency (mem_write_latency),
        .mem_max_reads     (mem_max_reads),
        .mem_max_writes    (mem_max_writes),
        .mem_base          (mem_base),
        .mem_model_bank    (mem_model_bank),
        .mem_clock         (mem_clock),
        .mem_latency       (mem_latency),
        .mem_queue         (mem_queue),
        .mem_geom          (mem_geom),
        .mem_wmark         (mem_wmark),
        .mem_timing        (mem_timing),
        .mem_refresh       (mem_refresh),
        .mem_refi          (mem_refi),
        .mem_rfc           (mem_rfc),
        .mem_stats         (mem_stats),
        .mem_idle          (&mem_idle),
        .mem_error         (|mem_error),
        .mem_stall         (|mem_stall),
        .profile_finish    (profile_finish),
        .mem_reset_valid   (mem_reset_valid),
        .mem_reset_ready   (mem_reset_ready),
        .mem_reset_bits    (mem_reset_bits),
        `FOLDSIM_IN(reset),
        `FOLDSIM_IN(dcr_req_valid),
        `FOLDSIM_IN(dcr_req_rw),
        `FOLDSIM_IN(dcr_req_addr),
        `FOLDSIM_IN(dcr_req_data),
        `FOLDSIM_IN(start),
        `FOLDSIM_IN(dcr_rsp_valid),
        `FOLDSIM_IN(dcr_rsp_data),
        `FOLDSIM_IN(busy)
    );
    assign s_axi_control_rresp = 2'b00;
    assign s_axi_control_bresp = 2'b00;

`undef FOLDSIM_IN

endmodule
