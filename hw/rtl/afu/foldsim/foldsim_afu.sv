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

`define FOLDSIM_COMMA ,
`define FOLDSIM_SEMI  ;

// The foldsim kernel: the simulator CIRCT composes of Vortex (foldsim_sim, of
// the target foldsim_target), a memory endpoint on each of its AXI4 ports and
// the control endpoint on its DCR, start and busy signals. Its ports are the
// native V80 kernel's, but for the host-memory master it does not need: one
// device-memory master per memory port of the target (FOLDSIM_MEM_PORTS, from
// foldsim_kernel.vh, with the ports' id width), each its own way into device
// memory, or, where the simulator folds its memory ports (FOLDSIM_PORTS), one
// master in all for the folded endpoint (FOLDSIM_MEM_MASTERS). A simulator
// with virtual memories (FOLDSIM_VMEM_COUNT, from
// foldsim_sim.vh) has a foldsim_vmem controller for each, with a master of its
// own after the memory ports' (FOLDSIM_VMEM_REPEAT), or, where
// foldsim_kernel.vh gives fewer (FOLDSIM_VMEM_MASTERS), controller k on master
// k mod FOLDSIM_VMEM_MASTERS with the others there (foldsim_vmem_share); with
// no master left beside the memory ports' (FOLDSIM_VMEM_ON_PORTS), controller k
// shares the memory ports' master k mod FOLDSIM_MEM_MASTERS with its endpoint. With
// FOLDSIM_VMEM_GROUPS (foldsim --vmem-group), the controllers are groups: group
// g's members (FOLDSIM_VMEM_G<g>_REPEAT) share one foldsim_vmem_group (one
// foldsim_vmem alone), and FOLDSIM_VMEM_REPEAT lists the groups.
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

    // Not every input is used: responses' resp, the memory ports' ids and
    // rlast (their transactions are single beats of id 0), and the control
    // port's strobes and upper address bits (its registers are whole words).
    /* verilator lint_off UNUSEDSIGNAL */
    `FOLDSIM_MEM_REPEAT0(FOLDSIM_AXI_MEM, FOLDSIM_COMMA)
    `FOLDSIM_MEM_REPEAT1(FOLDSIM_AXI_MEM, FOLDSIM_COMMA)
    `FOLDSIM_MEM_REPEAT2(FOLDSIM_AXI_MEM, FOLDSIM_COMMA)
    `FOLDSIM_MEM_REPEAT3(FOLDSIM_AXI_MEM, FOLDSIM_COMMA),
`ifdef FOLDSIM_VMEM_COUNT
`ifndef FOLDSIM_VMEM_ON_PORTS
`define FOLDSIM_AXI_VMEM(p, k) `FOLDSIM_AXI_MEM(p)
`ifdef FOLDSIM_VMEM_MASTERS
    `FOLDSIM_VMEM_MASTER_REPEAT(FOLDSIM_AXI_VMEM, FOLDSIM_COMMA),
`else
    `FOLDSIM_VMEM_REPEAT(FOLDSIM_AXI_VMEM, FOLDSIM_COMMA),
`endif
`undef FOLDSIM_AXI_VMEM
`endif
`endif

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
    /* verilator lint_on UNUSEDSIGNAL */
);
`undef FOLDSIM_AXI_MEM
    localparam DATA_W = C_M_AXI_MEM_DATA_WIDTH;
    localparam PORTS  = `FOLDSIM_MEM_PORTS;
    // The memory ports' device-memory masters (m_axi_mem_0 ..).
    localparam MEM_MASTERS = `FOLDSIM_MEM_MASTERS;
`ifdef FOLDSIM_PORTS
    if (MEM_MASTERS != 1) begin : g_masters_check
        $error("foldsim_afu: folded memory ports have one master, not %0d", MEM_MASTERS);
    end
`else
    if (MEM_MASTERS != PORTS) begin : g_masters_check
        $error("foldsim_afu: %0d memory ports on %0d masters", PORTS, MEM_MASTERS);
    end
`endif
    // The memory endpoints offer R and AR's ready for either fire of R in the
    // cycle before (foldsim_mem's SPEC): those channels carry both values. At
    // depth 2, for the fires of the two cycles before, and AW's and W's ready
    // too: four values each.
`ifdef FOLDSIM_MEM_SPEC
    localparam MEM_SPEC = `FOLDSIM_MEM_SPEC;
`else
    localparam MEM_SPEC = 0;
`endif
`ifndef FOLDSIM_PORTS
    localparam NC = (MEM_SPEC == 2) ? 4 : ((MEM_SPEC != 0) ? 2 : 1);
    localparam NW = (MEM_SPEC == 2) ? 4 : 1;
`endif

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

    // A memory port's channels, by port (with folded ports, the simulator has
    // none: the records carry them).
`ifndef FOLDSIM_PORTS
`define FOLDSIM_MCH(name, w) \
    wire         mem_``name``_valid [PORTS]; \
    wire         mem_``name``_ready [PORTS]; \
    wire [w-1:0] mem_``name``_bits  [PORTS];

    `FOLDSIM_MCH(awready, NW)
    `FOLDSIM_MCH(wready, NW)
    `FOLDSIM_MCH(arready, NC)
    `FOLDSIM_MCH(rvalid, NC)
    `FOLDSIM_MCH(rdata, NC*DATA_W)
    `FOLDSIM_MCH(rlast, NC)
    `FOLDSIM_MCH(rid, NC*TARGET_ID_W)
    `FOLDSIM_MCH(rresp, NC*2)
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
`endif

    // Device memory, by master of the memory ports.
    wire                   hbm_awvalid [MEM_MASTERS], hbm_awready [MEM_MASTERS];
    wire [63:0]            hbm_awaddr  [MEM_MASTERS];
    wire [7:0]             hbm_awlen   [MEM_MASTERS];
    wire                   hbm_wvalid  [MEM_MASTERS], hbm_wready [MEM_MASTERS], hbm_wlast [MEM_MASTERS];
    wire [DATA_W-1:0]      hbm_wdata   [MEM_MASTERS];
    wire [DATA_W/8-1:0]    hbm_wstrb   [MEM_MASTERS];
    wire                   hbm_bvalid  [MEM_MASTERS], hbm_bready [MEM_MASTERS];
    wire                   hbm_arvalid [MEM_MASTERS], hbm_arready [MEM_MASTERS];
    wire [63:0]            hbm_araddr  [MEM_MASTERS];
    wire [7:0]             hbm_arlen   [MEM_MASTERS];
    wire                   hbm_rvalid  [MEM_MASTERS], hbm_rready [MEM_MASTERS];
    wire [DATA_W-1:0]      hbm_rdata   [MEM_MASTERS];

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

`ifndef FOLDSIM_VMEM_ON_PORTS
    `FOLDSIM_MEM_REPEAT0(FOLDSIM_HBM, FOLDSIM_SEMI)
    `FOLDSIM_MEM_REPEAT1(FOLDSIM_HBM, FOLDSIM_SEMI)
    `FOLDSIM_MEM_REPEAT2(FOLDSIM_HBM, FOLDSIM_SEMI)
    `FOLDSIM_MEM_REPEAT3(FOLDSIM_HBM, FOLDSIM_SEMI);
`endif
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

`ifdef FOLDSIM_ADVANCE
    localparam BUBBLES = 1;
`else
    localparam BUBBLES = 0;
`endif
    // Idle slots at the host's rate: a host cycle is one when a pseudo-random
    // byte (a 16-bit maximal LFSR) is below it; at 0, never.
    wire [7:0] bubble;
    reg [15:0] bubble_lfsr;
    reg        advance;
    always @(posedge ap_clk) begin
        if (reset) begin
            bubble_lfsr <= 16'hACE1;
            advance     <= 1'b1;
        end else begin
            bubble_lfsr <= {1'b0, bubble_lfsr[15:1]} ^ (bubble_lfsr[0] ? 16'hB400 : 16'h0);
            advance     <= (bubble_lfsr[7:0] >= bubble);
        end
    end

`ifdef FOLDSIM_VMEM_COUNT
    // ---- The virtual memories: a controller each, on its own master ----
    // The simulator issues them nothing while vmem_on is low: until the
    // runtime has placed their region (FOLDSIM_REG_VMEM_BASE), it sees no room.
    localparam [63:0] VMEM_BYTES = `FOLDSIM_VMEM_BYTES;
    localparam VMEM_QUEUE       = 32;
    // Reads in flight per controller, on as many ids as the master has (up
    // to that many), in order per id.
    localparam VMEM_OUTSTANDING = 64;
    // Memory k's host-cache rows: FOLDSIM_VMEM_<k>_CACHE_ROWS where
    // foldsim_kernel.vh says the headers give one per memory, else
    // FOLDSIM_VMEM_CACHE_ROWS (0: none).
`ifdef FOLDSIM_VMEM_PER_MEMORY_ROWS
`define FOLDSIM_VMEM_ROWS(k) (`FOLDSIM_VMEM_``k``_CACHE_ROWS)
`else
`ifdef FOLDSIM_VMEM_CACHE_ROWS
    localparam VMEM_CACHE_ROWS  = `FOLDSIM_VMEM_CACHE_ROWS;
`else
    localparam VMEM_CACHE_ROWS  = 0;
`endif
`define FOLDSIM_VMEM_ROWS(k) VMEM_CACHE_ROWS
`endif
    // Known-zero rows (foldsim_vmem KNOWN_ZERO), on unless 0.
`ifdef FOLDSIM_VMEM_KNOWN_ZERO
    localparam VMEM_KNOWN_ZERO  = `FOLDSIM_VMEM_KNOWN_ZERO;
`else
    localparam VMEM_KNOWN_ZERO  = 1;
`endif
    // Controller k's row bytes in memory, known-zero bits (on or off) and
    // their block (log2 rows): the compiler's (foldsim_sim.vh, or vmem_ports.py's
    // defaults in foldsim_kernel.vh: the v6 layout, a bit per row); a
    // kernel.vh before layout v7 gives the defaults. Groups: G<g>.
`ifdef FOLDSIM_VMEM_LAYOUT_V7
`define FOLDSIM_VMEM_RB(k)  (`FOLDSIM_VMEM_``k``_ROW_BYTES)
`define FOLDSIM_VMEM_KZ(k)  (`FOLDSIM_VMEM_``k``_KNOWN_ZERO)
`define FOLDSIM_VMEM_KZS(k) (`FOLDSIM_VMEM_``k``_KZ_SHIFT)
`else
`define FOLDSIM_VMEM_RB(k)  ((((`FOLDSIM_VMEM_``k``_DW + 7) / 8) > 64) ? (1 << $clog2((`FOLDSIM_VMEM_``k``_DW + 7) / 8)) : 64)
`define FOLDSIM_VMEM_KZ(k)  1
`define FOLDSIM_VMEM_KZS(k) 0
`endif
    if (C_M_AXI_MEM_DATA_WIDTH != 512) begin : g_vmem_width
        $error("foldsim_afu: the virtual memories' controllers have 512-bit masters");
    end
    wire [63:0] vmem_base;
    wire        vmem_on, vmem_zero, vmem_move;
    // Each controller is still moving its region (foldsim_vmem `moving`).
`ifdef FOLDSIM_VMEM_GROUPS
    localparam VMEM_CTRLS = `FOLDSIM_VMEM_GROUPS;
`else
    localparam VMEM_CTRLS = `FOLDSIM_VMEM_COUNT;
`endif
    wire [VMEM_CTRLS-1:0] vmem_moving;

    // Controller k's master: m_axi_mem_<p>, or its slot of the sharing.
`ifdef FOLDSIM_VMEM_ON_PORTS
`define FOLDSIM_VMEM_SHARED
`endif
`ifdef FOLDSIM_VMEM_MASTERS
`define FOLDSIM_VMEM_SHARED
`endif
`ifdef FOLDSIM_VMEM_SHARED
    localparam VMEM_N = VMEM_CTRLS;
    localparam VMEM_ID_W = C_M_AXI_MEM_ID_WIDTH;
    // A controller's reads of one id in flight, for the sharing's route of R.
    localparam VMEM_ID_READS = (VMEM_ID_W >= 6) ? 1 : VMEM_OUTSTANDING >> VMEM_ID_W;
    wire                 vmc_awvalid [VMEM_N], vmc_awready [VMEM_N];
    wire [63:0]          vmc_awaddr  [VMEM_N];
    wire [VMEM_ID_W-1:0] vmc_awid    [VMEM_N];
    wire [7:0]           vmc_awlen   [VMEM_N];
    wire                 vmc_wvalid  [VMEM_N], vmc_wready [VMEM_N], vmc_wlast [VMEM_N];
    wire [511:0]         vmc_wdata   [VMEM_N];
    wire [63:0]          vmc_wstrb   [VMEM_N];
    wire                 vmc_bvalid  [VMEM_N], vmc_bready [VMEM_N];
    wire [VMEM_ID_W-1:0] vmc_bid     [VMEM_N];
    wire                 vmc_arvalid [VMEM_N], vmc_arready [VMEM_N];
    wire [63:0]          vmc_araddr  [VMEM_N];
    wire [VMEM_ID_W-1:0] vmc_arid    [VMEM_N];
    wire [7:0]           vmc_arlen   [VMEM_N];
    wire                 vmc_rvalid  [VMEM_N], vmc_rready [VMEM_N], vmc_rlast [VMEM_N];
    wire [511:0]         vmc_rdata   [VMEM_N];
    wire [VMEM_ID_W-1:0] vmc_rid     [VMEM_N];
`undef FOLDSIM_VMEM_SHARED
`define FOLDSIM_VMEM_AXI(p, k, s) vmc_``s[k]
`else
`define FOLDSIM_VMEM_AXI(p, k, s) m_axi_mem_``p``_``s
`endif

    // Memory k on master p. Its region's base is registered: base + the
    // region's offset is static while requests flow. Its host cache has
    // FOLDSIM_VMEM_ROWS(k) lines, at most a line per row of its image.
`define FOLDSIM_VMEM(p, k) \
    wire                                vm``k``_rd_valid, vm``k``_wr_valid, vm``k``_rsp_valid; \
    wire [`FOLDSIM_VMEM_``k``_IW-1:0]   vm``k``_rd_inst, vm``k``_wr_inst, vm``k``_rsp_inst; \
    wire [`FOLDSIM_VMEM_``k``_AW-1:0]   vm``k``_rd_row, vm``k``_wr_row; \
    wire [`FOLDSIM_VMEM_``k``_DW-1:0]   vm``k``_wr_data, vm``k``_rsp_data; \
    wire [`FOLDSIM_VMEM_``k``_MW-1:0]   vm``k``_wr_mask; \
    wire [$clog2(VMEM_QUEUE+1)-1:0]     vm``k``_room; \
    wire [15:0]                         vm``k``_free = vmem_on ? 16'(vm``k``_room) : 16'd0; \
    wire                                vm``k``_moving; \
    assign vmem_moving[k] = vm``k``_moving; \
    reg  [63:0]                         vm``k``_base; \
    always @(posedge ap_clk) \
        vm``k``_base <= vmem_base + `FOLDSIM_VMEM_``k``_BASE; \
    foldsim_vmem #( \
        .IW             (`FOLDSIM_VMEM_``k``_IW), \
        .AW             (`FOLDSIM_VMEM_``k``_AW), \
        .DW             (`FOLDSIM_VMEM_``k``_DW), \
        .MW             (`FOLDSIM_VMEM_``k``_MW), \
        .RDW_READ_FIRST (`FOLDSIM_VMEM_``k``_RDWRF), \
        .QUEUE          (VMEM_QUEUE), \
        .AXI_ID_W       (C_M_AXI_MEM_ID_WIDTH), \
        .OUTSTANDING    (VMEM_OUTSTANDING), \
        .INSTANCES      (`FOLDSIM_VMEM_``k``_INSTANCES), \
        .CACHE_ROWS     ((`FOLDSIM_VMEM_ROWS(k) < (1 << (`FOLDSIM_VMEM_``k``_AW + $clog2(`FOLDSIM_VMEM_``k``_INSTANCES)))) \
                         ? `FOLDSIM_VMEM_ROWS(k) : (1 << (`FOLDSIM_VMEM_``k``_AW + $clog2(`FOLDSIM_VMEM_``k``_INSTANCES)))), \
        .KNOWN_ZERO     (VMEM_KNOWN_ZERO & `FOLDSIM_VMEM_KZ(k)), \
        .KZ_SHIFT       (`FOLDSIM_VMEM_KZS(k)), \
        .ROW_BYTES      (`FOLDSIM_VMEM_RB(k)) \
    ) vmem_``k ( \
        .clk       (ap_clk), \
        .reset     (reset), \
        .rd_valid  (vm``k``_rd_valid), \
        .rd_inst   (vm``k``_rd_inst), \
        .rd_row    (vm``k``_rd_row), \
        .wr_valid  (vm``k``_wr_valid), \
        .wr_inst   (vm``k``_wr_inst), \
        .wr_row    (vm``k``_wr_row), \
        .wr_data   (vm``k``_wr_data), \
        .wr_mask   (vm``k``_wr_mask), \
        .rsp_valid (vm``k``_rsp_valid), \
        .rsp_inst  (vm``k``_rsp_inst), \
        .rsp_data  (vm``k``_rsp_data), \
        .free      (vm``k``_room), \
        .base      (vm``k``_base), \
        .zero      (vmem_zero), \
        .move      (vmem_move), \
        .moving    (vm``k``_moving), \
        .m_awvalid (`FOLDSIM_VMEM_AXI(p, k, awvalid)), \
        .m_awready (`FOLDSIM_VMEM_AXI(p, k, awready)), \
        .m_awaddr  (`FOLDSIM_VMEM_AXI(p, k, awaddr)), \
        .m_awid    (`FOLDSIM_VMEM_AXI(p, k, awid)), \
        .m_awlen   (`FOLDSIM_VMEM_AXI(p, k, awlen)), \
        .m_wvalid  (`FOLDSIM_VMEM_AXI(p, k, wvalid)), \
        .m_wready  (`FOLDSIM_VMEM_AXI(p, k, wready)), \
        .m_wdata   (`FOLDSIM_VMEM_AXI(p, k, wdata)), \
        .m_wstrb   (`FOLDSIM_VMEM_AXI(p, k, wstrb)), \
        .m_wlast   (`FOLDSIM_VMEM_AXI(p, k, wlast)), \
        .m_bvalid  (`FOLDSIM_VMEM_AXI(p, k, bvalid)), \
        .m_bready  (`FOLDSIM_VMEM_AXI(p, k, bready)), \
        .m_bid     (`FOLDSIM_VMEM_AXI(p, k, bid)), \
        .m_arvalid (`FOLDSIM_VMEM_AXI(p, k, arvalid)), \
        .m_arready (`FOLDSIM_VMEM_AXI(p, k, arready)), \
        .m_araddr  (`FOLDSIM_VMEM_AXI(p, k, araddr)), \
        .m_arid    (`FOLDSIM_VMEM_AXI(p, k, arid)), \
        .m_arlen   (`FOLDSIM_VMEM_AXI(p, k, arlen)), \
        .m_rvalid  (`FOLDSIM_VMEM_AXI(p, k, rvalid)), \
        .m_rready  (`FOLDSIM_VMEM_AXI(p, k, rready)), \
        .m_rdata   (`FOLDSIM_VMEM_AXI(p, k, rdata)), \
        .m_rid     (`FOLDSIM_VMEM_AXI(p, k, rid)), \
        .m_rlast   (`FOLDSIM_VMEM_AXI(p, k, rlast)) \
    )

`ifdef FOLDSIM_VMEM_GROUPS
`undef FOLDSIM_VMEM
`undef FOLDSIM_VMEM_ROWS
`undef FOLDSIM_VMEM_RB
`undef FOLDSIM_VMEM_KZ
`undef FOLDSIM_VMEM_KZS
`ifdef FOLDSIM_VMEM_LAYOUT_V7
`define FOLDSIM_VMEM_RB(g)  (`FOLDSIM_VMEM_G``g``_ROW_BYTES)
`define FOLDSIM_VMEM_KZ(g)  (`FOLDSIM_VMEM_G``g``_KNOWN_ZERO)
`define FOLDSIM_VMEM_KZS(g) (`FOLDSIM_VMEM_G``g``_KZ_SHIFT)
`else
`define FOLDSIM_VMEM_RB(g)  ((((`FOLDSIM_VMEM_G``g``_DW + 7) / 8) > 64) ? (1 << $clog2((`FOLDSIM_VMEM_G``g``_DW + 7) / 8)) : 64)
`define FOLDSIM_VMEM_KZ(g)  1
`define FOLDSIM_VMEM_KZS(g) 0
`endif
`ifdef FOLDSIM_VMEM_PER_MEMORY_ROWS
`define FOLDSIM_VMEM_ROWS(g) (`FOLDSIM_VMEM_G``g``_CACHE_ROWS)
`else
`define FOLDSIM_VMEM_ROWS(g) VMEM_CACHE_ROWS
`endif
    // Each memory's ports, as the simulator has them.
`define FOLDSIM_VMEM_WIRES(g, i, k) \
    wire                                vm``k``_rd_valid, vm``k``_wr_valid, vm``k``_rsp_valid; \
    wire [`FOLDSIM_VMEM_``k``_IW-1:0]   vm``k``_rd_inst, vm``k``_wr_inst, vm``k``_rsp_inst; \
    wire [`FOLDSIM_VMEM_``k``_AW-1:0]   vm``k``_rd_row, vm``k``_wr_row; \
    wire [`FOLDSIM_VMEM_``k``_DW-1:0]   vm``k``_wr_data, vm``k``_rsp_data; \
    wire [`FOLDSIM_VMEM_``k``_MW-1:0]   vm``k``_wr_mask; \
    wire [15:0]                         vm``k``_free
    `FOLDSIM_VMEM_MEMBERS(FOLDSIM_VMEM_WIRES, FOLDSIM_SEMI);
`undef FOLDSIM_VMEM_WIRES
    // Member i of group g is memory k: its ports at the controller's widths
    // (the group takes each member's own bits), its answers and room back.
`define FOLDSIM_VMEM_MEMBER(g, i, k) \
    assign vg``g``_rd_valid[i] = vm``k``_rd_valid; \
    assign vg``g``_rd_inst[i]  = vm``k``_rd_inst; \
    assign vg``g``_rd_row[i]   = `FOLDSIM_VMEM_G``g``_AW'(vm``k``_rd_row); \
    assign vg``g``_wr_valid[i] = vm``k``_wr_valid; \
    assign vg``g``_wr_inst[i]  = vm``k``_wr_inst; \
    assign vg``g``_wr_row[i]   = `FOLDSIM_VMEM_G``g``_AW'(vm``k``_wr_row); \
    assign vg``g``_wr_data[i]  = `FOLDSIM_VMEM_G``g``_DW'(vm``k``_wr_data); \
    assign vg``g``_wr_mask[i]  = `FOLDSIM_VMEM_G``g``_MW'(vm``k``_wr_mask); \
    assign vm``k``_rsp_valid   = vg``g``_rsp_valid[i]; \
    assign vm``k``_rsp_inst    = vg``g``_rsp_inst; \
    assign vm``k``_rsp_data    = vg``g``_rsp_data[`FOLDSIM_VMEM_``k``_DW-1:0]; \
    assign vm``k``_free        = vmem_on ? vg``g``_free[i] : 16'd0
`define FOLDSIM_VMEM_MAW(g, i, k)  `FOLDSIM_VMEM_``k``_AW
`define FOLDSIM_VMEM_MDW(g, i, k)  `FOLDSIM_VMEM_``k``_DW
`define FOLDSIM_VMEM_MMW(g, i, k)  `FOLDSIM_VMEM_``k``_MW
`define FOLDSIM_VMEM_MROW(g, i, k) `FOLDSIM_VMEM_``k``_GROW
    // Group g on master p: its members' FIFOs and one controller, or the
    // controller alone for one member. Its region's base is registered.
`define FOLDSIM_VMEM(p, g) \
    localparam VG``g``_N  = `FOLDSIM_VMEM_G``g``_N; \
    localparam VG``g``_AW = `FOLDSIM_VMEM_G``g``_AW; \
    localparam VG``g``_IN = `FOLDSIM_VMEM_G``g``_INSTANCES; \
    localparam int VG``g``_MAW [VG``g``_N]  = '{`FOLDSIM_VMEM_G``g``_REPEAT(FOLDSIM_VMEM_MAW, FOLDSIM_COMMA)}; \
    localparam int VG``g``_MDW [VG``g``_N]  = '{`FOLDSIM_VMEM_G``g``_REPEAT(FOLDSIM_VMEM_MDW, FOLDSIM_COMMA)}; \
    localparam int VG``g``_MMW [VG``g``_N]  = '{`FOLDSIM_VMEM_G``g``_REPEAT(FOLDSIM_VMEM_MMW, FOLDSIM_COMMA)}; \
    localparam int VG``g``_MROW [VG``g``_N] = '{`FOLDSIM_VMEM_G``g``_REPEAT(FOLDSIM_VMEM_MROW, FOLDSIM_COMMA)}; \
    wire                                  vg``g``_rd_valid [VG``g``_N], vg``g``_wr_valid [VG``g``_N]; \
    wire [`FOLDSIM_VMEM_G``g``_IW-1:0]    vg``g``_rd_inst  [VG``g``_N], vg``g``_wr_inst [VG``g``_N]; \
    wire [VG``g``_AW-1:0]                 vg``g``_rd_row   [VG``g``_N], vg``g``_wr_row [VG``g``_N]; \
    wire [`FOLDSIM_VMEM_G``g``_DW-1:0]    vg``g``_wr_data  [VG``g``_N]; \
    wire [`FOLDSIM_VMEM_G``g``_MW-1:0]    vg``g``_wr_mask  [VG``g``_N]; \
    wire                                  vg``g``_rsp_valid [VG``g``_N]; \
    wire [`FOLDSIM_VMEM_G``g``_IW-1:0]    vg``g``_rsp_inst; \
    wire [`FOLDSIM_VMEM_G``g``_DW-1:0]    vg``g``_rsp_data; \
    wire [15:0]                           vg``g``_free [VG``g``_N]; \
    `FOLDSIM_VMEM_G``g``_REPEAT(FOLDSIM_VMEM_MEMBER, FOLDSIM_SEMI); \
    reg  [63:0]                           vg``g``_base; \
    always @(posedge ap_clk) \
        vg``g``_base <= vmem_base + `FOLDSIM_VMEM_G``g``_BASE; \
    localparam VG``g``_CACHE = (`FOLDSIM_VMEM_ROWS(g) < (1 << (VG``g``_AW + $clog2(VG``g``_IN)))) \
                             ? `FOLDSIM_VMEM_ROWS(g) : (1 << (VG``g``_AW + $clog2(VG``g``_IN))); \
    if (VG``g``_N == 1) begin : g_vg``g \
        wire [$clog2(VMEM_QUEUE+1)-1:0] room; \
        assign vg``g``_free[0] = 16'(room); \
        foldsim_vmem #( \
            .IW             (`FOLDSIM_VMEM_G``g``_IW), \
            .AW             (VG``g``_AW), \
            .DW             (`FOLDSIM_VMEM_G``g``_DW), \
            .MW             (`FOLDSIM_VMEM_G``g``_MW), \
            .RDW_READ_FIRST (`FOLDSIM_VMEM_G``g``_RDWRF), \
            .QUEUE          (VMEM_QUEUE), \
            .AXI_ID_W       (C_M_AXI_MEM_ID_WIDTH), \
            .OUTSTANDING    (VMEM_OUTSTANDING), \
            .INSTANCES      (VG``g``_IN), \
            .CACHE_ROWS     (VG``g``_CACHE), \
            .KNOWN_ZERO     (VMEM_KNOWN_ZERO & `FOLDSIM_VMEM_KZ(g)), \
            .KZ_SHIFT       (`FOLDSIM_VMEM_KZS(g)), \
            .ROW_BYTES      (`FOLDSIM_VMEM_RB(g)) \
        ) vmem ( \
            .clk       (ap_clk), \
            .reset     (reset), \
            .rd_valid  (vg``g``_rd_valid[0]), \
            .rd_inst   (vg``g``_rd_inst[0]), \
            .rd_row    (vg``g``_rd_row[0]), \
            .wr_valid  (vg``g``_wr_valid[0]), \
            .wr_inst   (vg``g``_wr_inst[0]), \
            .wr_row    (vg``g``_wr_row[0]), \
            .wr_data   (vg``g``_wr_data[0]), \
            .wr_mask   (vg``g``_wr_mask[0]), \
            .rsp_valid (vg``g``_rsp_valid[0]), \
            .rsp_inst  (vg``g``_rsp_inst), \
            .rsp_data  (vg``g``_rsp_data), \
            .free      (room), \
            .base      (vg``g``_base), \
            .zero      (vmem_zero), \
            .move      (vmem_move), \
            .moving    (vmem_moving[g]), \
            `FOLDSIM_VMEM_AXI_PORTS(p, g) \
        ); \
    end else begin : g_vg``g \
        wire [$clog2(2 * VMEM_GROUP_FIFO + 1)-1:0] room [VG``g``_N]; \
        for (genvar i = 0; i < VG``g``_N; ++i) begin : g_room \
            assign vg``g``_free[i] = 16'(room[i]); \
        end \
        foldsim_vmem_group #( \
            .N              (VG``g``_N), \
            .IW             (`FOLDSIM_VMEM_G``g``_IW), \
            .INSTANCES      (VG``g``_IN), \
            .AW             (VG``g``_AW), \
            .DW             (`FOLDSIM_VMEM_G``g``_DW), \
            .MW             (`FOLDSIM_VMEM_G``g``_MW), \
            .MAW            (VG``g``_MAW), \
            .MDW            (VG``g``_MDW), \
            .MMW            (VG``g``_MMW), \
            .MROW           (VG``g``_MROW), \
            .RDW_READ_FIRST (`FOLDSIM_VMEM_G``g``_RDWRF), \
            .QUEUE          (VMEM_QUEUE), \
            .AXI_ID_W       (C_M_AXI_MEM_ID_WIDTH), \
            .OUTSTANDING    (VMEM_OUTSTANDING), \
            .CACHE_ROWS     (VG``g``_CACHE), \
            .KNOWN_ZERO     (VMEM_KNOWN_ZERO & `FOLDSIM_VMEM_KZ(g)), \
            .KZ_SHIFT       (`FOLDSIM_VMEM_KZS(g)), \
            .ROW_BYTES      (`FOLDSIM_VMEM_RB(g)), \
            .FIFO           (VMEM_GROUP_FIFO) \
        ) group ( \
            .clk       (ap_clk), \
            .reset     (reset), \
            .rd_valid  (vg``g``_rd_valid), \
            .rd_inst   (vg``g``_rd_inst), \
            .rd_row    (vg``g``_rd_row), \
            .wr_valid  (vg``g``_wr_valid), \
            .wr_inst   (vg``g``_wr_inst), \
            .wr_row    (vg``g``_wr_row), \
            .wr_data   (vg``g``_wr_data), \
            .wr_mask   (vg``g``_wr_mask), \
            .rsp_valid (vg``g``_rsp_valid), \
            .rsp_inst  (vg``g``_rsp_inst), \
            .rsp_data  (vg``g``_rsp_data), \
            .free      (room), \
            .base      (vg``g``_base), \
            .zero      (vmem_zero), \
            .move      (vmem_move), \
            .moving    (vmem_moving[g]), \
            `FOLDSIM_VMEM_AXI_PORTS(p, g) \
        ); \
    end
`define FOLDSIM_VMEM_AXI_PORTS(p, k) \
            .m_awvalid (`FOLDSIM_VMEM_AXI(p, k, awvalid)), \
            .m_awready (`FOLDSIM_VMEM_AXI(p, k, awready)), \
            .m_awaddr  (`FOLDSIM_VMEM_AXI(p, k, awaddr)), \
            .m_awid    (`FOLDSIM_VMEM_AXI(p, k, awid)), \
            .m_awlen   (`FOLDSIM_VMEM_AXI(p, k, awlen)), \
            .m_wvalid  (`FOLDSIM_VMEM_AXI(p, k, wvalid)), \
            .m_wready  (`FOLDSIM_VMEM_AXI(p, k, wready)), \
            .m_wdata   (`FOLDSIM_VMEM_AXI(p, k, wdata)), \
            .m_wstrb   (`FOLDSIM_VMEM_AXI(p, k, wstrb)), \
            .m_wlast   (`FOLDSIM_VMEM_AXI(p, k, wlast)), \
            .m_bvalid  (`FOLDSIM_VMEM_AXI(p, k, bvalid)), \
            .m_bready  (`FOLDSIM_VMEM_AXI(p, k, bready)), \
            .m_bid     (`FOLDSIM_VMEM_AXI(p, k, bid)), \
            .m_arvalid (`FOLDSIM_VMEM_AXI(p, k, arvalid)), \
            .m_arready (`FOLDSIM_VMEM_AXI(p, k, arready)), \
            .m_araddr  (`FOLDSIM_VMEM_AXI(p, k, araddr)), \
            .m_arid    (`FOLDSIM_VMEM_AXI(p, k, arid)), \
            .m_arlen   (`FOLDSIM_VMEM_AXI(p, k, arlen)), \
            .m_rvalid  (`FOLDSIM_VMEM_AXI(p, k, rvalid)), \
            .m_rready  (`FOLDSIM_VMEM_AXI(p, k, rready)), \
            .m_rdata   (`FOLDSIM_VMEM_AXI(p, k, rdata)), \
            .m_rid     (`FOLDSIM_VMEM_AXI(p, k, rid)), \
            .m_rlast   (`FOLDSIM_VMEM_AXI(p, k, rlast))
    // A member's FIFO: entries for the simulator's lead (five) and a burst.
    localparam VMEM_GROUP_FIFO = 16;
`endif
    `FOLDSIM_VMEM_REPEAT(FOLDSIM_VMEM, FOLDSIM_SEMI);
`undef FOLDSIM_VMEM
`undef FOLDSIM_VMEM_ROWS
`undef FOLDSIM_VMEM_RB
`undef FOLDSIM_VMEM_KZ
`undef FOLDSIM_VMEM_KZS
`undef FOLDSIM_VMEM_AXI
`ifdef FOLDSIM_VMEM_GROUPS
`undef FOLDSIM_VMEM_AXI_PORTS
`undef FOLDSIM_VMEM_MEMBER
`undef FOLDSIM_VMEM_MAW
`undef FOLDSIM_VMEM_MDW
`undef FOLDSIM_VMEM_MMW
`undef FOLDSIM_VMEM_MROW
`endif

`ifdef FOLDSIM_VMEM_MASTERS
    // Master j: controllers j, j + M, j + 2M, ... Each controller has at
    // most QUEUE writes and VMEM_ID_READS reads per id in flight.
    localparam VMEM_M = `FOLDSIM_VMEM_MASTERS;
    wire                 vmm_awvalid [VMEM_M], vmm_awready [VMEM_M];
    wire [63:0]          vmm_awaddr  [VMEM_M];
    wire [VMEM_ID_W-1:0] vmm_awid    [VMEM_M];
    wire [7:0]           vmm_awlen   [VMEM_M];
    wire                 vmm_wvalid  [VMEM_M], vmm_wready [VMEM_M], vmm_wlast [VMEM_M];
    wire [511:0]         vmm_wdata   [VMEM_M];
    wire [63:0]          vmm_wstrb   [VMEM_M];
    wire                 vmm_bvalid  [VMEM_M], vmm_bready [VMEM_M];
    wire [VMEM_ID_W-1:0] vmm_bid     [VMEM_M];
    wire                 vmm_arvalid [VMEM_M], vmm_arready [VMEM_M];
    wire [63:0]          vmm_araddr  [VMEM_M];
    wire [VMEM_ID_W-1:0] vmm_arid    [VMEM_M];
    wire [7:0]           vmm_arlen   [VMEM_M];
    wire                 vmm_rvalid  [VMEM_M], vmm_rready [VMEM_M], vmm_rlast [VMEM_M];
    wire [511:0]         vmm_rdata   [VMEM_M];
    wire [VMEM_ID_W-1:0] vmm_rid     [VMEM_M];
    for (genvar j = 0; j < VMEM_M; ++j) begin : g_vmem_master
        localparam G = (VMEM_N - j + VMEM_M - 1) / VMEM_M;
        wire                 s_awvalid [G], s_awready [G];
        wire [63:0]          s_awaddr  [G];
        wire [VMEM_ID_W-1:0] s_awid    [G];
        wire [7:0]           s_awlen   [G];
        wire                 s_wvalid  [G], s_wready [G], s_wlast [G];
        wire [511:0]         s_wdata   [G];
        wire [63:0]          s_wstrb   [G];
        wire                 s_bvalid  [G], s_bready [G];
        wire [VMEM_ID_W-1:0] s_bid     [G];
        wire                 s_arvalid [G], s_arready [G];
        wire [63:0]          s_araddr  [G];
        wire [VMEM_ID_W-1:0] s_arid    [G];
        wire [7:0]           s_arlen   [G];
        wire                 s_rvalid  [G], s_rready [G], s_rlast [G];
        wire [511:0]         s_rdata   [G];
        wire [VMEM_ID_W-1:0] s_rid     [G];
        for (genvar i = 0; i < G; ++i) begin : g_ctrl
            localparam K = j + i * VMEM_M;
            assign s_awvalid[i] = vmc_awvalid[K];
            assign vmc_awready[K] = s_awready[i];
            assign s_awaddr[i]  = vmc_awaddr[K];
            assign s_awid[i]    = vmc_awid[K];
            assign s_awlen[i]   = vmc_awlen[K];
            assign s_wvalid[i]  = vmc_wvalid[K];
            assign vmc_wready[K] = s_wready[i];
            assign s_wdata[i]   = vmc_wdata[K];
            assign s_wstrb[i]   = vmc_wstrb[K];
            assign s_wlast[i]   = vmc_wlast[K];
            assign vmc_bvalid[K] = s_bvalid[i];
            assign s_bready[i]  = vmc_bready[K];
            assign vmc_bid[K]   = s_bid[i];
            assign s_arvalid[i] = vmc_arvalid[K];
            assign vmc_arready[K] = s_arready[i];
            assign s_araddr[i]  = vmc_araddr[K];
            assign s_arid[i]    = vmc_arid[K];
            assign s_arlen[i]   = vmc_arlen[K];
            assign vmc_rvalid[K] = s_rvalid[i];
            assign s_rready[i]  = vmc_rready[K];
            assign vmc_rdata[K] = s_rdata[i];
            assign vmc_rid[K]   = s_rid[i];
            assign vmc_rlast[K] = s_rlast[i];
        end
        foldsim_vmem_share #(
            .N      (G),
            .ID_W   (VMEM_ID_W),
            .RDEPTH (G * VMEM_ID_READS),
            .WDEPTH (G * VMEM_QUEUE)
        ) share (
            .clk       (ap_clk),
            .s_awvalid (s_awvalid), .s_awready (s_awready), .s_awaddr (s_awaddr), .s_awid (s_awid),
            .s_awlen   (s_awlen),
            .s_wvalid  (s_wvalid),  .s_wready  (s_wready),  .s_wdata  (s_wdata),  .s_wstrb (s_wstrb),
            .s_wlast   (s_wlast),
            .s_bvalid  (s_bvalid),  .s_bready  (s_bready),  .s_bid    (s_bid),
            .s_arvalid (s_arvalid), .s_arready (s_arready), .s_araddr (s_araddr), .s_arid  (s_arid),
            .s_arlen   (s_arlen),
            .s_rvalid  (s_rvalid),  .s_rready  (s_rready),  .s_rdata  (s_rdata),  .s_rid   (s_rid),
            .s_rlast   (s_rlast),
            .m_awvalid (vmm_awvalid[j]), .m_awready (vmm_awready[j]), .m_awaddr (vmm_awaddr[j]),
            .m_awid    (vmm_awid[j]),    .m_awlen   (vmm_awlen[j]),
            .m_wvalid  (vmm_wvalid[j]),  .m_wready  (vmm_wready[j]),  .m_wdata  (vmm_wdata[j]),
            .m_wstrb   (vmm_wstrb[j]),   .m_wlast   (vmm_wlast[j]),
            .m_bvalid  (vmm_bvalid[j]),  .m_bready  (vmm_bready[j]),  .m_bid    (vmm_bid[j]),
            .m_arvalid (vmm_arvalid[j]), .m_arready (vmm_arready[j]), .m_araddr (vmm_araddr[j]),
            .m_arid    (vmm_arid[j]),    .m_arlen   (vmm_arlen[j]),
            .m_rvalid  (vmm_rvalid[j]),  .m_rready  (vmm_rready[j]),  .m_rdata  (vmm_rdata[j]),
            .m_rid     (vmm_rid[j]),     .m_rlast   (vmm_rlast[j])
        );
    end
`define FOLDSIM_VMEM_PORT(p, j) \
    assign m_axi_mem_``p``_awvalid = vmm_awvalid[j]; \
    assign vmm_awready[j]          = m_axi_mem_``p``_awready; \
    assign m_axi_mem_``p``_awaddr  = vmm_awaddr[j]; \
    assign m_axi_mem_``p``_awid    = vmm_awid[j]; \
    assign m_axi_mem_``p``_awlen   = vmm_awlen[j]; \
    assign m_axi_mem_``p``_wvalid  = vmm_wvalid[j]; \
    assign vmm_wready[j]           = m_axi_mem_``p``_wready; \
    assign m_axi_mem_``p``_wdata   = vmm_wdata[j]; \
    assign m_axi_mem_``p``_wstrb   = vmm_wstrb[j]; \
    assign m_axi_mem_``p``_wlast   = vmm_wlast[j]; \
    assign vmm_bvalid[j]           = m_axi_mem_``p``_bvalid; \
    assign m_axi_mem_``p``_bready  = vmm_bready[j]; \
    assign vmm_bid[j]              = m_axi_mem_``p``_bid; \
    assign m_axi_mem_``p``_arvalid = vmm_arvalid[j]; \
    assign vmm_arready[j]          = m_axi_mem_``p``_arready; \
    assign m_axi_mem_``p``_araddr  = vmm_araddr[j]; \
    assign m_axi_mem_``p``_arid    = vmm_arid[j]; \
    assign m_axi_mem_``p``_arlen   = vmm_arlen[j]; \
    assign vmm_rvalid[j]           = m_axi_mem_``p``_rvalid; \
    assign m_axi_mem_``p``_rready  = vmm_rready[j]; \
    assign vmm_rdata[j]            = m_axi_mem_``p``_rdata; \
    assign vmm_rid[j]              = m_axi_mem_``p``_rid; \
    assign vmm_rlast[j]            = m_axi_mem_``p``_rlast
    `FOLDSIM_VMEM_MASTER_REPEAT(FOLDSIM_VMEM_PORT, FOLDSIM_SEMI);
`undef FOLDSIM_VMEM_PORT
`endif

`ifdef FOLDSIM_VMEM_ON_PORTS
    // The memory ports' master p: its endpoint, then controllers p,
    // p + MEM_MASTERS, ... The endpoint's transactions are single beats of id
    // 0: the route of R holds as many of its reads in flight (more only wait
    // for room).
    localparam VMEM_PORT_READS = 64;
    wire                       pm_awvalid [MEM_MASTERS], pm_awready [MEM_MASTERS];
    wire [63:0]                pm_awaddr  [MEM_MASTERS];
    wire [VMEM_ID_W-1:0]       pm_awid    [MEM_MASTERS];
    wire [7:0]                 pm_awlen   [MEM_MASTERS];
    wire                       pm_wvalid  [MEM_MASTERS], pm_wready [MEM_MASTERS], pm_wlast [MEM_MASTERS];
    wire [DATA_W-1:0]          pm_wdata   [MEM_MASTERS];
    wire [DATA_W/8-1:0]        pm_wstrb   [MEM_MASTERS];
    wire                       pm_bvalid  [MEM_MASTERS], pm_bready [MEM_MASTERS];
    wire [VMEM_ID_W-1:0]       pm_bid     [MEM_MASTERS];
    wire                       pm_arvalid [MEM_MASTERS], pm_arready [MEM_MASTERS];
    wire [63:0]                pm_araddr  [MEM_MASTERS];
    wire [VMEM_ID_W-1:0]       pm_arid    [MEM_MASTERS];
    wire [7:0]                 pm_arlen   [MEM_MASTERS];
    wire                       pm_rvalid  [MEM_MASTERS], pm_rready [MEM_MASTERS], pm_rlast [MEM_MASTERS];
    wire [DATA_W-1:0]          pm_rdata   [MEM_MASTERS];
    wire [VMEM_ID_W-1:0]       pm_rid     [MEM_MASTERS];
    for (genvar p = 0; p < MEM_MASTERS; ++p) begin : g_vmem_port
        localparam G = (p < VMEM_N) ? (VMEM_N - p + MEM_MASTERS - 1) / MEM_MASTERS : 0;
        if (G == 0) begin : g_alone
            assign pm_awvalid[p]  = hbm_awvalid[p];
            assign hbm_awready[p] = pm_awready[p];
            assign pm_awaddr[p]   = hbm_awaddr[p];
            assign pm_awid[p]     = '0;
            assign pm_awlen[p]    = hbm_awlen[p];
            assign pm_wvalid[p]   = hbm_wvalid[p];
            assign hbm_wready[p]  = pm_wready[p];
            assign pm_wdata[p]    = hbm_wdata[p];
            assign pm_wstrb[p]    = hbm_wstrb[p];
            assign pm_wlast[p]    = hbm_wlast[p];
            assign hbm_bvalid[p]  = pm_bvalid[p];
            assign pm_bready[p]   = hbm_bready[p];
            assign pm_arvalid[p]  = hbm_arvalid[p];
            assign hbm_arready[p] = pm_arready[p];
            assign pm_araddr[p]   = hbm_araddr[p];
            assign pm_arid[p]     = '0;
            assign pm_arlen[p]    = hbm_arlen[p];
            assign hbm_rvalid[p]  = pm_rvalid[p];
            assign pm_rready[p]   = hbm_rready[p];
            assign hbm_rdata[p]   = pm_rdata[p];
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused = (|pm_bid[p]) | (|pm_rid[p]) | pm_rlast[p];
            /* verilator lint_on UNUSEDSIGNAL */
        end else begin : g_shared
            wire                 s_awvalid [G+1], s_awready [G+1];
            wire [63:0]          s_awaddr  [G+1];
            wire [VMEM_ID_W-1:0] s_awid    [G+1];
            wire [7:0]           s_awlen   [G+1];
            wire                 s_wvalid  [G+1], s_wready [G+1], s_wlast [G+1];
            wire [511:0]         s_wdata   [G+1];
            wire [63:0]          s_wstrb   [G+1];
            wire                 s_bvalid  [G+1], s_bready [G+1];
            wire [VMEM_ID_W-1:0] s_bid     [G+1];
            wire                 s_arvalid [G+1], s_arready [G+1];
            wire [63:0]          s_araddr  [G+1];
            wire [VMEM_ID_W-1:0] s_arid    [G+1];
            wire [7:0]           s_arlen   [G+1];
            wire                 s_rvalid  [G+1], s_rready [G+1], s_rlast [G+1];
            wire [511:0]         s_rdata   [G+1];
            wire [VMEM_ID_W-1:0] s_rid     [G+1];
            assign s_awvalid[0]   = hbm_awvalid[p];
            assign hbm_awready[p] = s_awready[0];
            assign s_awaddr[0]    = hbm_awaddr[p];
            assign s_awid[0]      = '0;
            assign s_awlen[0]     = hbm_awlen[p];
            assign s_wvalid[0]    = hbm_wvalid[p];
            assign hbm_wready[p]  = s_wready[0];
            assign s_wdata[0]     = hbm_wdata[p];
            assign s_wstrb[0]     = hbm_wstrb[p];
            assign s_wlast[0]     = hbm_wlast[p];
            assign hbm_bvalid[p]  = s_bvalid[0];
            assign s_bready[0]    = hbm_bready[p];
            assign s_arvalid[0]   = hbm_arvalid[p];
            assign hbm_arready[p] = s_arready[0];
            assign s_araddr[0]    = hbm_araddr[p];
            assign s_arid[0]      = '0;
            assign s_arlen[0]     = hbm_arlen[p];
            assign hbm_rvalid[p]  = s_rvalid[0];
            assign s_rready[0]    = hbm_rready[p];
            assign hbm_rdata[p]   = s_rdata[0];
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused = (|s_bid[0]) | (|s_rid[0]) | s_rlast[0];
            /* verilator lint_on UNUSEDSIGNAL */
            for (genvar i = 0; i < G; ++i) begin : g_ctrl
                localparam K = p + i * MEM_MASTERS;
                assign s_awvalid[i+1] = vmc_awvalid[K];
                assign vmc_awready[K] = s_awready[i+1];
                assign s_awaddr[i+1]  = vmc_awaddr[K];
                assign s_awid[i+1]    = vmc_awid[K];
                assign s_awlen[i+1]   = vmc_awlen[K];
                assign s_wvalid[i+1]  = vmc_wvalid[K];
                assign vmc_wready[K]  = s_wready[i+1];
                assign s_wdata[i+1]   = vmc_wdata[K];
                assign s_wstrb[i+1]   = vmc_wstrb[K];
                assign s_wlast[i+1]   = vmc_wlast[K];
                assign vmc_bvalid[K]  = s_bvalid[i+1];
                assign s_bready[i+1]  = vmc_bready[K];
                assign vmc_bid[K]     = s_bid[i+1];
                assign s_arvalid[i+1] = vmc_arvalid[K];
                assign vmc_arready[K] = s_arready[i+1];
                assign s_araddr[i+1]  = vmc_araddr[K];
                assign s_arid[i+1]    = vmc_arid[K];
                assign s_arlen[i+1]   = vmc_arlen[K];
                assign vmc_rvalid[K]  = s_rvalid[i+1];
                assign s_rready[i+1]  = vmc_rready[K];
                assign vmc_rdata[K]   = s_rdata[i+1];
                assign vmc_rid[K]     = s_rid[i+1];
                assign vmc_rlast[K]   = s_rlast[i+1];
            end
            foldsim_vmem_share #(
                .N      (G + 1),
                .ID_W   (VMEM_ID_W),
                .RDEPTH (VMEM_PORT_READS + G * VMEM_ID_READS),
                .WDEPTH ((G + 1) * VMEM_QUEUE)
            ) share (
                .clk       (ap_clk),
                .s_awvalid (s_awvalid), .s_awready (s_awready), .s_awaddr (s_awaddr), .s_awid (s_awid),
                .s_awlen   (s_awlen),
                .s_wvalid  (s_wvalid),  .s_wready  (s_wready),  .s_wdata  (s_wdata),  .s_wstrb (s_wstrb),
                .s_wlast   (s_wlast),
                .s_bvalid  (s_bvalid),  .s_bready  (s_bready),  .s_bid    (s_bid),
                .s_arvalid (s_arvalid), .s_arready (s_arready), .s_araddr (s_araddr), .s_arid  (s_arid),
                .s_arlen   (s_arlen),
                .s_rvalid  (s_rvalid),  .s_rready  (s_rready),  .s_rdata  (s_rdata),  .s_rid   (s_rid),
                .s_rlast   (s_rlast),
                .m_awvalid (pm_awvalid[p]), .m_awready (pm_awready[p]), .m_awaddr (pm_awaddr[p]),
                .m_awid    (pm_awid[p]),    .m_awlen   (pm_awlen[p]),
                .m_wvalid  (pm_wvalid[p]),  .m_wready  (pm_wready[p]),  .m_wdata  (pm_wdata[p]),
                .m_wstrb   (pm_wstrb[p]),   .m_wlast   (pm_wlast[p]),
                .m_bvalid  (pm_bvalid[p]),  .m_bready  (pm_bready[p]),  .m_bid    (pm_bid[p]),
                .m_arvalid (pm_arvalid[p]), .m_arready (pm_arready[p]), .m_araddr (pm_araddr[p]),
                .m_arid    (pm_arid[p]),    .m_arlen   (pm_arlen[p]),
                .m_rvalid  (pm_rvalid[p]),  .m_rready  (pm_rready[p]),  .m_rdata  (pm_rdata[p]),
                .m_rid     (pm_rid[p]),     .m_rlast   (pm_rlast[p])
            );
        end
    end
`define FOLDSIM_PORT_HBM(i) \
    assign m_axi_mem_``i``_awvalid = pm_awvalid[i]; \
    assign pm_awready[i]           = m_axi_mem_``i``_awready; \
    assign m_axi_mem_``i``_awaddr  = pm_awaddr[i]; \
    assign m_axi_mem_``i``_awid    = pm_awid[i]; \
    assign m_axi_mem_``i``_awlen   = pm_awlen[i]; \
    assign m_axi_mem_``i``_wvalid  = pm_wvalid[i]; \
    assign pm_wready[i]            = m_axi_mem_``i``_wready; \
    assign m_axi_mem_``i``_wdata   = pm_wdata[i]; \
    assign m_axi_mem_``i``_wstrb   = pm_wstrb[i]; \
    assign m_axi_mem_``i``_wlast   = pm_wlast[i]; \
    assign pm_bvalid[i]            = m_axi_mem_``i``_bvalid; \
    assign m_axi_mem_``i``_bready  = pm_bready[i]; \
    assign pm_bid[i]               = m_axi_mem_``i``_bid; \
    assign m_axi_mem_``i``_arvalid = pm_arvalid[i]; \
    assign pm_arready[i]           = m_axi_mem_``i``_arready; \
    assign m_axi_mem_``i``_araddr  = pm_araddr[i]; \
    assign m_axi_mem_``i``_arid    = pm_arid[i]; \
    assign m_axi_mem_``i``_arlen   = pm_arlen[i]; \
    assign pm_rvalid[i]            = m_axi_mem_``i``_rvalid; \
    assign m_axi_mem_``i``_rready  = pm_rready[i]; \
    assign pm_rdata[i]             = m_axi_mem_``i``_rdata; \
    assign pm_rid[i]               = m_axi_mem_``i``_rid; \
    assign pm_rlast[i]             = m_axi_mem_``i``_rlast
    `FOLDSIM_MEM_REPEAT0(FOLDSIM_PORT_HBM, FOLDSIM_SEMI)
    `FOLDSIM_MEM_REPEAT1(FOLDSIM_PORT_HBM, FOLDSIM_SEMI)
    `FOLDSIM_MEM_REPEAT2(FOLDSIM_PORT_HBM, FOLDSIM_SEMI)
    `FOLDSIM_MEM_REPEAT3(FOLDSIM_PORT_HBM, FOLDSIM_SEMI);
`undef FOLDSIM_PORT_HBM
`endif

`define FOLDSIM_SIM_VMEM(p, k) \
    .vm``k``_rsp_valid (vm``k``_rsp_valid), .vm``k``_rsp_inst (vm``k``_rsp_inst), \
    .vm``k``_rsp_data (vm``k``_rsp_data), .vm``k``_free (vm``k``_free), \
    .vm``k``_rd_valid (vm``k``_rd_valid), .vm``k``_rd_inst (vm``k``_rd_inst), .vm``k``_rd_row (vm``k``_rd_row), \
    .vm``k``_wr_valid (vm``k``_wr_valid), .vm``k``_wr_inst (vm``k``_wr_inst), .vm``k``_wr_row (vm``k``_wr_row), \
    .vm``k``_wr_data (vm``k``_wr_data), .vm``k``_wr_mask (vm``k``_wr_mask)
`endif

`ifdef FOLDSIM_PORTS
    // Folded memory ports: one record a port and target cycle each way
    // (PORTFOLD_IFACE.md), served by the folded memory endpoint.
    localparam PF_CW = `FOLDSIM_PORTS_CW;
    wire                          pf_req_valid, pf_req_ready, pf_req_treset;
    wire [PF_CW-1:0]              pf_req_chan;
    wire [`FOLDSIM_PORTS_REQ_W-1:0] pf_req_bits;
    wire                          pf_rsp_valid, pf_rsp_ready;
    wire [PF_CW-1:0]              pf_rsp_chan;
    wire [`FOLDSIM_PORTS_RSP_W-1:0] pf_rsp_bits;
`endif
    foldsim_sim sim (
        .hostClock (ap_clk),
        .hostReset (reset),
        .profile_finish (profile_finish),
`ifdef FOLDSIM_ADVANCE
        .advance (advance),
`endif
`ifdef FOLDSIM_VMEM_COUNT
`ifdef FOLDSIM_VMEM_GROUPS
`define FOLDSIM_SIM_VMEM_M(g, i, k) `FOLDSIM_SIM_VMEM(g, k)
        `FOLDSIM_VMEM_MEMBERS(FOLDSIM_SIM_VMEM_M, FOLDSIM_COMMA),
`undef FOLDSIM_SIM_VMEM_M
`else
        `FOLDSIM_VMEM_REPEAT(FOLDSIM_SIM_VMEM, FOLDSIM_COMMA),
`endif
`undef FOLDSIM_SIM_VMEM
`endif
`ifdef FOLDSIM_PORTS
        // Folded memory ports (foldsim --fold-ports): the record streams.
        .req_valid (pf_req_valid), .req_ready (pf_req_ready), .req_chan (pf_req_chan),
        .req_treset (pf_req_treset), .req_bits (pf_req_bits),
        .rsp_valid (pf_rsp_valid), .rsp_ready (pf_rsp_ready), .rsp_chan (pf_rsp_chan),
        .rsp_bits (pf_rsp_bits),
`else
        `FOLDSIM_MEM_REPEAT0(FOLDSIM_SIM_MEM, FOLDSIM_COMMA)
        `FOLDSIM_MEM_REPEAT1(FOLDSIM_SIM_MEM, FOLDSIM_COMMA)
        `FOLDSIM_MEM_REPEAT2(FOLDSIM_SIM_MEM, FOLDSIM_COMMA)
        `FOLDSIM_MEM_REPEAT3(FOLDSIM_SIM_MEM, FOLDSIM_COMMA),
`endif
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
    localparam MEM_STAGED = 0;
    localparam [31:0] MEM_CAPS = {12'd0, 8'(MEM_SLOTS), 4'(MEM_HLOG2 - 4), 8'(PORTS)};

    wire [63:0] mem_base;
    wire [31:0] mem_clock, mem_queue;
    wire [15:0] mem_latency, mem_wmark;
    wire [12:0] mem_geom;
    wire [18*12-1:0] mem_timing;
    wire [PORTS*8*64-1:0] mem_stats;
    wire [PORTS*8*32-1:0] mem_axi;
    wire [PORTS*32-1:0]   mem_dbg;
    wire [1:0]  mem_refresh;
    wire [15:0] mem_refi, mem_rfc;
    wire [PORTS-1:0] mem_idle, mem_error, mem_stall;
    wire [PORTS-1:0] mem_reset_valid, mem_reset_ready, mem_reset_bits;

`define FOLDSIM_MEM(name) \
    .name``_valid (mem_``name``_valid[p]), .name``_ready (mem_``name``_ready[p]), \
    .name``_bits (mem_``name``_bits[p])

`ifdef FOLDSIM_PORTS
    // The ports' reset tokens ride on the request records (req_treset): the
    // control endpoint's per-port reset channels are taken as they come.
    assign mem_reset_ready = '1;
    /* verilator lint_off UNUSEDSIGNAL */
    wire pf_unused_reset = &{1'b0, mem_reset_valid, mem_reset_bits};
    /* verilator lint_on UNUSEDSIGNAL */
    wire pf_idle, pf_error, pf_stall;
    assign mem_idle  = {PORTS{pf_idle}};
    assign mem_error = {PORTS{pf_error}};
    assign mem_stall = {PORTS{pf_stall}};
    // One memory endpoint threaded PORTS ways (foldsim_mem_fold, the threaded
    // module from a portfold_gen.sh snapshot made for these parameters) and
    // one AXI master (m_axi_mem_0): every port's data at base plus its target
    // address, as before.
    wire                pf_awvalid, pf_awready, pf_wvalid, pf_wready, pf_wlast, pf_bvalid, pf_bready;
    wire                pf_arvalid, pf_arready, pf_rvalid, pf_rready;
    wire [63:0]         pf_awaddr, pf_araddr;
    wire [7:0]          pf_awlen, pf_arlen;
    wire [DATA_W-1:0]   pf_wdata, pf_rdata;
    wire [DATA_W/8-1:0] pf_wstrb;
    foldsim_mem_fold #(
        .PORTS    (PORTS),
        .ID_W     (TARGET_ID_W),
        .ADDR_W   (TARGET_ADDR_W),
        .DATA_W   (DATA_W),
        .SLOTS    (MEM_SLOTS),
        .HLOG2    (MEM_HLOG2),
        .STAGED   (MEM_STAGED),
        .SPEC     (MEM_SPEC),
        .CW       (PF_CW),
        .REQ_W    (`FOLDSIM_PORTS_REQ_W),
        .RSP_W    (`FOLDSIM_PORTS_RSP_W)
    ) mem_fold (
        .clk           (ap_clk),
        .reset         (reset),
        .base          (mem_base),
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
        .stats         (mem_stats),
        .axi_stats     (mem_axi),
        .dbg           (mem_dbg),
        .idle          (pf_idle),
        .error         (pf_error),
        .stall         (pf_stall),
        .req_valid     (pf_req_valid),
        .req_ready     (pf_req_ready),
        .req_chan      (pf_req_chan),
        .req_treset    (pf_req_treset),
        .req_bits      (pf_req_bits),
        .rsp_valid     (pf_rsp_valid),
        .rsp_ready     (pf_rsp_ready),
        .rsp_chan      (pf_rsp_chan),
        .rsp_bits      (pf_rsp_bits),
        .m_awvalid (pf_awvalid), .m_awready (pf_awready), .m_awaddr (pf_awaddr), .m_awlen (pf_awlen),
        .m_wvalid  (pf_wvalid),  .m_wready  (pf_wready),  .m_wdata  (pf_wdata),  .m_wstrb (pf_wstrb),
        .m_wlast   (pf_wlast),   .m_bvalid  (pf_bvalid),  .m_bready  (pf_bready),
        .m_arvalid (pf_arvalid), .m_arready (pf_arready), .m_araddr  (pf_araddr), .m_arlen (pf_arlen),
        .m_rvalid  (pf_rvalid),  .m_rready  (pf_rready),  .m_rdata   (pf_rdata)
    );
    assign hbm_awvalid[0] = pf_awvalid;
    assign pf_awready     = hbm_awready[0];
    assign hbm_awaddr[0]  = pf_awaddr;
    assign hbm_awlen[0]   = pf_awlen;
    assign hbm_wvalid[0]  = pf_wvalid;
    assign pf_wready      = hbm_wready[0];
    assign hbm_wdata[0]   = pf_wdata;
    assign hbm_wstrb[0]   = pf_wstrb;
    assign hbm_wlast[0]   = pf_wlast;
    assign pf_bvalid      = hbm_bvalid[0];
    assign hbm_bready[0]  = pf_bready;
    assign hbm_arvalid[0] = pf_arvalid;
    assign pf_arready     = hbm_arready[0];
    assign hbm_araddr[0]  = pf_araddr;
    assign hbm_arlen[0]   = pf_arlen;
    assign pf_rvalid      = hbm_rvalid[0];
    assign hbm_rready[0]  = pf_rready;
    assign pf_rdata       = hbm_rdata[0];
`else
    for (genvar p = 0; p < PORTS; ++p) begin : g_mem
        foldsim_mem #(
            .ID_W     (TARGET_ID_W),
            .ADDR_W   (TARGET_ADDR_W),
            .DATA_W   (DATA_W),
            .SLOTS    (MEM_SLOTS),
            .HLOG2    (MEM_HLOG2),
            .STAGED   (MEM_STAGED),
            .CH_SHIFT (`FOLDSIM_MEM_CH_SHIFT),
            .SPEC     (MEM_SPEC)
        ) mem (
            .clk           (ap_clk),
            .reset         (reset),
            .base          (mem_base),
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
            .axi_stats     (mem_axi[p*8*32 +: 8*32]),
            .dbg           (mem_dbg[p*32 +: 32]),
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
`endif
`undef FOLDSIM_MEM

    foldsim_ctrl #(
        .DCR_ADDR_W  (DCR_ADDR_W),
        .DCR_DATA_W  (DCR_DATA_W),
        .NUM_CLASSES (`FOLDSIM_NUM_CLASSES),
        .MEM_PORTS   (PORTS),
        .MEM_CAPS    (MEM_CAPS),
`ifdef FOLDSIM_VMEM_COUNT
        .VMEM_BYTES  (VMEM_BYTES),
`endif
        .BUBBLES     (BUBBLES)
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
        .mem_base          (mem_base),
        .mem_clock         (mem_clock),
        .mem_latency       (mem_latency),
        .mem_queue         (mem_queue),
        .mem_geom          (mem_geom),
        .mem_wmark         (mem_wmark),
        .mem_timing        (mem_timing),
        .mem_refresh       (mem_refresh),
        .mem_refi          (mem_refi),
        .mem_rfc           (mem_rfc),
        .bubble            (bubble),
`ifdef FOLDSIM_VMEM_COUNT
        .vmem_base         (vmem_base),
        .vmem_on           (vmem_on),
        .vmem_zero         (vmem_zero),
        .vmem_move         (vmem_move),
        .vmem_moving       (|vmem_moving),
`else
        // Without virtual memories, the region is never used.
        /* verilator lint_off PINCONNECTEMPTY */
        .vmem_base         (),
        .vmem_on           (),
        .vmem_zero         (),
        .vmem_move         (),
        /* verilator lint_on PINCONNECTEMPTY */
        .vmem_moving       (1'b0),
`endif
        .mem_stats         (mem_stats),
        .mem_axi           (mem_axi),
        .mem_dbg           (mem_dbg),
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

`undef FOLDSIM_COMMA
`undef FOLDSIM_SEMI
