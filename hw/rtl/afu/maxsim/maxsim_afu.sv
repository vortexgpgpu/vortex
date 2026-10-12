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

`include "maxsim_sim.vh"

// The maxsim kernel: the simulator CIRCT composes of Vortex (maxsim_sim, of
// the target VX_firesim_wrap), the memory endpoint on its AXI4 port and the
// control endpoint on its DCR, start and busy signals. Its ports are the
// native V80 kernel's, but for the host-memory master it does not need.
module maxsim_afu #(
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
`define MAXSIM_CH(name, w) \
    wire name``_valid, name``_ready; \
    wire [w-1:0] name``_bits;

    `MAXSIM_CH(reset, 1)
    `MAXSIM_CH(m_axi_mem_0_awready, 1)
    `MAXSIM_CH(m_axi_mem_0_wready, 1)
    `MAXSIM_CH(m_axi_mem_0_arready, 1)
    `MAXSIM_CH(m_axi_mem_0_rvalid, 1)
    `MAXSIM_CH(m_axi_mem_0_rdata, DATA_W)
    `MAXSIM_CH(m_axi_mem_0_rlast, 1)
    `MAXSIM_CH(m_axi_mem_0_rid, TARGET_ID_W)
    `MAXSIM_CH(m_axi_mem_0_rresp, 2)
    `MAXSIM_CH(m_axi_mem_0_bvalid, 1)
    `MAXSIM_CH(m_axi_mem_0_bresp, 2)
    `MAXSIM_CH(m_axi_mem_0_bid, TARGET_ID_W)
    `MAXSIM_CH(dcr_req_valid, 1)
    `MAXSIM_CH(dcr_req_rw, 1)
    `MAXSIM_CH(dcr_req_addr, DCR_ADDR_W)
    `MAXSIM_CH(dcr_req_data, DCR_DATA_W)
    `MAXSIM_CH(start, 1)
    `MAXSIM_CH(m_axi_mem_0_awvalid, 1)
    `MAXSIM_CH(m_axi_mem_0_awaddr, TARGET_ADDR_W)
    `MAXSIM_CH(m_axi_mem_0_awid, TARGET_ID_W)
    `MAXSIM_CH(m_axi_mem_0_awlen, 8)
    `MAXSIM_CH(m_axi_mem_0_wvalid, 1)
    `MAXSIM_CH(m_axi_mem_0_wdata, DATA_W)
    `MAXSIM_CH(m_axi_mem_0_wstrb, DATA_W/8)
    `MAXSIM_CH(m_axi_mem_0_wlast, 1)
    `MAXSIM_CH(m_axi_mem_0_arvalid, 1)
    `MAXSIM_CH(m_axi_mem_0_araddr, TARGET_ADDR_W)
    `MAXSIM_CH(m_axi_mem_0_arid, TARGET_ID_W)
    `MAXSIM_CH(m_axi_mem_0_arlen, 8)
    `MAXSIM_CH(m_axi_mem_0_rready, 1)
    `MAXSIM_CH(m_axi_mem_0_bready, 1)
    `MAXSIM_CH(dcr_rsp_valid, 1)
    `MAXSIM_CH(dcr_rsp_data, DCR_DATA_W)
    `MAXSIM_CH(busy, 1)
`undef MAXSIM_CH

`define MAXSIM_IN(name) \
    .name``_valid (name``_valid), .name``_ready (name``_ready), .name``_bits (name``_bits)

    wire [`MAXSIM_NUM_CLASSES-1:0] profile_finish /*verilator public_flat_rd*/;

    maxsim_sim sim (
        .hostClock (ap_clk),
        .hostReset (reset),
        .profile_finish (profile_finish),
        `MAXSIM_IN(reset),
        `MAXSIM_IN(m_axi_mem_0_awready),
        `MAXSIM_IN(m_axi_mem_0_wready),
        `MAXSIM_IN(m_axi_mem_0_arready),
        `MAXSIM_IN(m_axi_mem_0_rvalid),
        `MAXSIM_IN(m_axi_mem_0_rdata),
        `MAXSIM_IN(m_axi_mem_0_rlast),
        `MAXSIM_IN(m_axi_mem_0_rid),
        `MAXSIM_IN(m_axi_mem_0_rresp),
        `MAXSIM_IN(m_axi_mem_0_bvalid),
        `MAXSIM_IN(m_axi_mem_0_bresp),
        `MAXSIM_IN(m_axi_mem_0_bid),
        `MAXSIM_IN(dcr_req_valid),
        `MAXSIM_IN(dcr_req_rw),
        `MAXSIM_IN(dcr_req_addr),
        `MAXSIM_IN(dcr_req_data),
        `MAXSIM_IN(start),
        `MAXSIM_IN(m_axi_mem_0_awvalid),
        `MAXSIM_IN(m_axi_mem_0_awaddr),
        `MAXSIM_IN(m_axi_mem_0_awid),
        `MAXSIM_IN(m_axi_mem_0_awlen),
        `MAXSIM_IN(m_axi_mem_0_wvalid),
        `MAXSIM_IN(m_axi_mem_0_wdata),
        `MAXSIM_IN(m_axi_mem_0_wstrb),
        `MAXSIM_IN(m_axi_mem_0_wlast),
        `MAXSIM_IN(m_axi_mem_0_arvalid),
        `MAXSIM_IN(m_axi_mem_0_araddr),
        `MAXSIM_IN(m_axi_mem_0_arid),
        `MAXSIM_IN(m_axi_mem_0_arlen),
        `MAXSIM_IN(m_axi_mem_0_rready),
        `MAXSIM_IN(m_axi_mem_0_bready),
        `MAXSIM_IN(dcr_rsp_valid),
        `MAXSIM_IN(dcr_rsp_data),
        `MAXSIM_IN(busy)
    );

`define MAXSIM_MEM(name) \
    .name``_valid (m_axi_mem_0_``name``_valid), .name``_ready (m_axi_mem_0_``name``_ready), \
    .name``_bits (m_axi_mem_0_``name``_bits)

    wire [31:0] mem_read_latency, mem_write_latency;
    wire [7:0]  mem_max_reads, mem_max_writes;
    wire [63:0] mem_base;
    wire        mem_idle, mem_error, mem_stall;

    maxsim_mem #(
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
        `MAXSIM_MEM(awvalid), `MAXSIM_MEM(awaddr), `MAXSIM_MEM(awid), `MAXSIM_MEM(awlen),
        `MAXSIM_MEM(wvalid), `MAXSIM_MEM(wdata), `MAXSIM_MEM(wstrb), `MAXSIM_MEM(wlast),
        `MAXSIM_MEM(arvalid), `MAXSIM_MEM(araddr), `MAXSIM_MEM(arid), `MAXSIM_MEM(arlen),
        `MAXSIM_MEM(rready), `MAXSIM_MEM(bready),
        `MAXSIM_MEM(awready), `MAXSIM_MEM(wready), `MAXSIM_MEM(arready),
        `MAXSIM_MEM(rvalid), `MAXSIM_MEM(rdata), `MAXSIM_MEM(rlast), `MAXSIM_MEM(rid), `MAXSIM_MEM(rresp),
        `MAXSIM_MEM(bvalid), `MAXSIM_MEM(bresp), `MAXSIM_MEM(bid),
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

    maxsim_ctrl #(
        .DCR_ADDR_W  (DCR_ADDR_W),
        .DCR_DATA_W  (DCR_DATA_W),
        .NUM_CLASSES (`MAXSIM_NUM_CLASSES)
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
        `MAXSIM_IN(reset),
        `MAXSIM_IN(dcr_req_valid),
        `MAXSIM_IN(dcr_req_rw),
        `MAXSIM_IN(dcr_req_addr),
        `MAXSIM_IN(dcr_req_data),
        `MAXSIM_IN(start),
        `MAXSIM_IN(dcr_rsp_valid),
        `MAXSIM_IN(dcr_rsp_data),
        `MAXSIM_IN(busy)
    );
    assign s_axi_control_rresp = 2'b00;
    assign s_axi_control_bresp = 2'b00;

`undef MAXSIM_IN
`undef MAXSIM_MEM

endmodule
