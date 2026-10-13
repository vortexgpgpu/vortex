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

// The foldsim kernel: the simulator CIRCT composes of Vortex (foldsim_sim, of
// the target foldsim_target), the memory endpoint on its AXI4 port and the
// control endpoint on its DCR, start and busy signals. Its ports are the
// native V80 kernel's, but for the host-memory master it does not need.
module foldsim_afu #(
    parameter C_S_AXI_CONTROL_ADDR_WIDTH = 16,
    parameter C_S_AXI_CONTROL_DATA_WIDTH = 32,
    parameter C_M_AXI_MEM_ID_WIDTH       = 1,
    parameter C_M_AXI_MEM_DATA_WIDTH     = 512,
    parameter C_M_AXI_MEM_ADDR_WIDTH     = 64,
    // The target's AXI4 port.
    parameter TARGET_ID_W                = 4,
    parameter TARGET_ADDR_W              = 64,
    parameter DCR_ADDR_W                 = 12,
    parameter DCR_DATA_W                 = 32
) (
    input  wire                                    ap_clk,
    input  wire                                    ap_rst_n,

    output wire                                    m_axi_mem_0_awvalid,
    input  wire                                    m_axi_mem_0_awready,
    output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]       m_axi_mem_0_awaddr,
    output wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_0_awid,
    output wire [7:0]                              m_axi_mem_0_awlen,
    output wire                                    m_axi_mem_0_wvalid,
    input  wire                                    m_axi_mem_0_wready,
    output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]       m_axi_mem_0_wdata,
    output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0]     m_axi_mem_0_wstrb,
    output wire                                    m_axi_mem_0_wlast,
    output wire                                    m_axi_mem_0_arvalid,
    input  wire                                    m_axi_mem_0_arready,
    output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]       m_axi_mem_0_araddr,
    output wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_0_arid,
    output wire [7:0]                              m_axi_mem_0_arlen,
    input  wire                                    m_axi_mem_0_rvalid,
    output wire                                    m_axi_mem_0_rready,
    input  wire [C_M_AXI_MEM_DATA_WIDTH-1:0]       m_axi_mem_0_rdata,
    input  wire                                    m_axi_mem_0_rlast,
    input  wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_0_rid,
    input  wire [1:0]                              m_axi_mem_0_rresp,
    input  wire                                    m_axi_mem_0_bvalid,
    output wire                                    m_axi_mem_0_bready,
    input  wire [1:0]                              m_axi_mem_0_bresp,
    input  wire [C_M_AXI_MEM_ID_WIDTH-1:0]         m_axi_mem_0_bid,

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
    localparam DATA_W = C_M_AXI_MEM_DATA_WIDTH;

    // A registered reset, so it reaches the whole kernel in one cycle's time.
    reg reset;
    always @(posedge ap_clk)
        reset <= ~ap_rst_n;

    // ---- The simulator's channels, named after the target's ports ----
`define FOLDSIM_CH(name, w) \
    wire name``_valid, name``_ready; \
    wire [w-1:0] name``_bits;

    `FOLDSIM_CH(reset, 1)
    `FOLDSIM_CH(m_axi_mem_0_awready, 1)
    `FOLDSIM_CH(m_axi_mem_0_wready, 1)
    `FOLDSIM_CH(m_axi_mem_0_arready, 1)
    `FOLDSIM_CH(m_axi_mem_0_rvalid, 1)
    `FOLDSIM_CH(m_axi_mem_0_rdata, DATA_W)
    `FOLDSIM_CH(m_axi_mem_0_rlast, 1)
    `FOLDSIM_CH(m_axi_mem_0_rid, TARGET_ID_W)
    `FOLDSIM_CH(m_axi_mem_0_rresp, 2)
    `FOLDSIM_CH(m_axi_mem_0_bvalid, 1)
    `FOLDSIM_CH(m_axi_mem_0_bresp, 2)
    `FOLDSIM_CH(m_axi_mem_0_bid, TARGET_ID_W)
    `FOLDSIM_CH(dcr_req_valid, 1)
    `FOLDSIM_CH(dcr_req_rw, 1)
    `FOLDSIM_CH(dcr_req_addr, DCR_ADDR_W)
    `FOLDSIM_CH(dcr_req_data, DCR_DATA_W)
    `FOLDSIM_CH(start, 1)
    `FOLDSIM_CH(m_axi_mem_0_awvalid, 1)
    `FOLDSIM_CH(m_axi_mem_0_awaddr, TARGET_ADDR_W)
    `FOLDSIM_CH(m_axi_mem_0_awid, TARGET_ID_W)
    `FOLDSIM_CH(m_axi_mem_0_awlen, 8)
    `FOLDSIM_CH(m_axi_mem_0_wvalid, 1)
    `FOLDSIM_CH(m_axi_mem_0_wdata, DATA_W)
    `FOLDSIM_CH(m_axi_mem_0_wstrb, DATA_W/8)
    `FOLDSIM_CH(m_axi_mem_0_wlast, 1)
    `FOLDSIM_CH(m_axi_mem_0_arvalid, 1)
    `FOLDSIM_CH(m_axi_mem_0_araddr, TARGET_ADDR_W)
    `FOLDSIM_CH(m_axi_mem_0_arid, TARGET_ID_W)
    `FOLDSIM_CH(m_axi_mem_0_arlen, 8)
    `FOLDSIM_CH(m_axi_mem_0_rready, 1)
    `FOLDSIM_CH(m_axi_mem_0_bready, 1)
    `FOLDSIM_CH(dcr_rsp_valid, 1)
    `FOLDSIM_CH(dcr_rsp_data, DCR_DATA_W)
    `FOLDSIM_CH(busy, 1)
`undef FOLDSIM_CH

`define FOLDSIM_IN(name) \
    .name``_valid (name``_valid), .name``_ready (name``_ready), .name``_bits (name``_bits)

    wire [`FOLDSIM_NUM_CLASSES-1:0] profile_finish /*verilator public_flat_rd*/;

    foldsim_sim sim (
        .hostClock (ap_clk),
        .hostReset (reset),
        .profile_finish (profile_finish),
        `FOLDSIM_IN(reset),
        `FOLDSIM_IN(m_axi_mem_0_awready),
        `FOLDSIM_IN(m_axi_mem_0_wready),
        `FOLDSIM_IN(m_axi_mem_0_arready),
        `FOLDSIM_IN(m_axi_mem_0_rvalid),
        `FOLDSIM_IN(m_axi_mem_0_rdata),
        `FOLDSIM_IN(m_axi_mem_0_rlast),
        `FOLDSIM_IN(m_axi_mem_0_rid),
        `FOLDSIM_IN(m_axi_mem_0_rresp),
        `FOLDSIM_IN(m_axi_mem_0_bvalid),
        `FOLDSIM_IN(m_axi_mem_0_bresp),
        `FOLDSIM_IN(m_axi_mem_0_bid),
        `FOLDSIM_IN(dcr_req_valid),
        `FOLDSIM_IN(dcr_req_rw),
        `FOLDSIM_IN(dcr_req_addr),
        `FOLDSIM_IN(dcr_req_data),
        `FOLDSIM_IN(start),
        `FOLDSIM_IN(m_axi_mem_0_awvalid),
        `FOLDSIM_IN(m_axi_mem_0_awaddr),
        `FOLDSIM_IN(m_axi_mem_0_awid),
        `FOLDSIM_IN(m_axi_mem_0_awlen),
        `FOLDSIM_IN(m_axi_mem_0_wvalid),
        `FOLDSIM_IN(m_axi_mem_0_wdata),
        `FOLDSIM_IN(m_axi_mem_0_wstrb),
        `FOLDSIM_IN(m_axi_mem_0_wlast),
        `FOLDSIM_IN(m_axi_mem_0_arvalid),
        `FOLDSIM_IN(m_axi_mem_0_araddr),
        `FOLDSIM_IN(m_axi_mem_0_arid),
        `FOLDSIM_IN(m_axi_mem_0_arlen),
        `FOLDSIM_IN(m_axi_mem_0_rready),
        `FOLDSIM_IN(m_axi_mem_0_bready),
        `FOLDSIM_IN(dcr_rsp_valid),
        `FOLDSIM_IN(dcr_rsp_data),
        `FOLDSIM_IN(busy)
    );

`define FOLDSIM_MEM(name) \
    .name``_valid (m_axi_mem_0_``name``_valid), .name``_ready (m_axi_mem_0_``name``_ready), \
    .name``_bits (m_axi_mem_0_``name``_bits)

    wire [31:0] mem_read_latency, mem_write_latency;
    wire [7:0]  mem_max_reads, mem_max_writes;
    wire [63:0] mem_base;
    wire        mem_idle, mem_error, mem_stall;

    foldsim_mem #(
        .ID_W   (TARGET_ID_W),
        .ADDR_W (TARGET_ADDR_W),
        .DATA_W (DATA_W)
    ) mem (
        .clk           (ap_clk),
        .reset         (reset),
        .read_latency  (mem_read_latency),
        .write_latency (mem_write_latency),
        .max_reads     (mem_max_reads),
        .max_writes    (mem_max_writes),
        .base          (mem_base),
        .idle          (mem_idle),
        .error         (mem_error),
        .stall         (mem_stall),
        `FOLDSIM_MEM(awvalid), `FOLDSIM_MEM(awaddr), `FOLDSIM_MEM(awid), `FOLDSIM_MEM(awlen),
        `FOLDSIM_MEM(wvalid), `FOLDSIM_MEM(wdata), `FOLDSIM_MEM(wstrb), `FOLDSIM_MEM(wlast),
        `FOLDSIM_MEM(arvalid), `FOLDSIM_MEM(araddr), `FOLDSIM_MEM(arid), `FOLDSIM_MEM(arlen),
        `FOLDSIM_MEM(rready), `FOLDSIM_MEM(bready),
        `FOLDSIM_MEM(awready), `FOLDSIM_MEM(wready), `FOLDSIM_MEM(arready),
        `FOLDSIM_MEM(rvalid), `FOLDSIM_MEM(rdata), `FOLDSIM_MEM(rlast), `FOLDSIM_MEM(rid), `FOLDSIM_MEM(rresp),
        `FOLDSIM_MEM(bvalid), `FOLDSIM_MEM(bresp), `FOLDSIM_MEM(bid),
        .m_awvalid (m_axi_mem_0_awvalid),
        .m_awready (m_axi_mem_0_awready),
        .m_awaddr  (m_axi_mem_0_awaddr),
        .m_awlen   (m_axi_mem_0_awlen),
        .m_wvalid  (m_axi_mem_0_wvalid),
        .m_wready  (m_axi_mem_0_wready),
        .m_wdata   (m_axi_mem_0_wdata),
        .m_wstrb   (m_axi_mem_0_wstrb),
        .m_wlast   (m_axi_mem_0_wlast),
        .m_bvalid  (m_axi_mem_0_bvalid),
        .m_bready  (m_axi_mem_0_bready),
        .m_arvalid (m_axi_mem_0_arvalid),
        .m_arready (m_axi_mem_0_arready),
        .m_araddr  (m_axi_mem_0_araddr),
        .m_arlen   (m_axi_mem_0_arlen),
        .m_rvalid  (m_axi_mem_0_rvalid),
        .m_rready  (m_axi_mem_0_rready),
        .m_rdata   (m_axi_mem_0_rdata)
    );
    assign m_axi_mem_0_awid = '0;
    assign m_axi_mem_0_arid = '0;

    foldsim_ctrl #(
        .DCR_ADDR_W  (DCR_ADDR_W),
        .DCR_DATA_W  (DCR_DATA_W),
        .NUM_CLASSES (`FOLDSIM_NUM_CLASSES)
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
        .mem_idle          (mem_idle),
        .mem_error         (mem_error),
        .mem_stall         (mem_stall),
        .profile_finish    (profile_finish),
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
`undef FOLDSIM_MEM

endmodule
