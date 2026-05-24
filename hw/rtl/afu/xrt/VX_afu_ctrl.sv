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

`include "vortex_afu.vh"

// ============================================================================
// VX_afu_ctrl — slim AXI-Lite slave for the xrt AFU shell.
//
// After the Command-Processor migration the AFU's command path is the CP
// regfile (host AXI-Lite 0x1000+). This module retains only the legacy
// 0x000-page essentials:
//   * 0x00 — a minimal `ap_ctrl` stub. The XRT framework expects an
//            `ap_ctrl` register at offset 0; the kernel is CP-driven, so
//            this stub simply reports the kernel permanently idle and
//            accepts writes inertly.
//   * 0x28/0x2C — the SCOPE bit-serial debug register pair (`ifdef SCOPE`).
//            SCOPE stays an independent sideband (proposal §10.6).
//
// The legacy launch FSM, legacy DCR registers, dev_caps/isa_caps copies,
// and the GIE/IER/ISR interrupt block were removed in Phase 4.
// ============================================================================

module VX_afu_ctrl #(
    parameter S_AXI_ADDR_WIDTH = 8,
    parameter S_AXI_DATA_WIDTH = 32
) (
    // axi4 lite slave signals
    input  wire                         clk,
    input  wire                         reset,

    input  wire                         s_axi_awvalid,
    input  wire [S_AXI_ADDR_WIDTH-1:0]  s_axi_awaddr,
    output wire                         s_axi_awready,

    input  wire                         s_axi_wvalid,
    input  wire [S_AXI_DATA_WIDTH-1:0]  s_axi_wdata,
    input  wire [S_AXI_DATA_WIDTH/8-1:0]s_axi_wstrb,
    output wire                         s_axi_wready,

    output wire                         s_axi_bvalid,
    output wire [1:0]                   s_axi_bresp,
    input  wire                         s_axi_bready,

    input  wire                         s_axi_arvalid,
    input  wire [S_AXI_ADDR_WIDTH-1:0]  s_axi_araddr,
    output wire                         s_axi_arready,

    output wire                         s_axi_rvalid,
    output wire [S_AXI_DATA_WIDTH-1:0]  s_axi_rdata,
    output wire [1:0]                   s_axi_rresp,
    input  wire                         s_axi_rready

`ifdef SCOPE
  , input  wire                         scope_bus_in,
    output wire                         scope_bus_out
`endif
);

    // Address map
    // 0x00       : ap_ctrl stub  (read: bit 2 = ap_idle = 1; writes inert)
    // 0x28/0x2C  : SCOPE bit-serial register pair (`ifdef SCOPE`)
    localparam
        ADDR_AP_CTRL    = 8'h00,
    `ifdef SCOPE
        ADDR_SCP_0      = 8'h28,
        ADDR_SCP_1      = 8'h2C,
    `endif
        ADDR_BITS       = 8;

    localparam
        WSTATE_ADDR     = 2'd0,
        WSTATE_DATA     = 2'd1,
        WSTATE_RESP     = 2'd2,
        WSTATE_WIDTH    = 2;

    localparam
        RSTATE_ADDR     = 2'd0,
        RSTATE_DATA     = 2'd1,
        RSTATE_RESP     = 2'd2,
        RSTATE_WIDTH    = 2;

    reg [WSTATE_WIDTH-1:0] wstate;
    reg [ADDR_BITS-1:0] waddr;
    wire        s_axi_aw_fire;
    wire        s_axi_w_fire;
    wire        s_axi_b_fire;

    logic [RSTATE_WIDTH-1:0] rstate;
    reg [31:0]  rdata;
    reg [ADDR_BITS-1:0] raddr;
    wire        s_axi_ar_fire;
    wire        s_axi_r_fire;

    logic wready_stall;
    logic rvalid_stall;

    `UNUSED_VAR (s_axi_wstrb)
    `UNUSED_VAR (s_axi_wdata)
`ifndef SCOPE
    `UNUSED_VAR (waddr)
`endif

`ifdef SCOPE

    // O-2: bounded SCOPE serial-bus watchdog. After SCOPE_RSTALL_TIMEOUT
    // cycles of `rvalid_stall` held high (host AXI-Lite read waiting for a
    // tap response that never arrived — desynced switch, dead tap, etc.),
    // force scope_rdata_valid=1 with a 64-bit sentinel pattern so the host
    // read returns instead of hanging the entire runtime. The runtime
    // (sw/runtime/common/scope.cpp) recognizes the sentinel and skips the
    // affected tap-this-pass rather than treating it as data. Mirrors the
    // bounded-stall pattern used by Xilinx AXI Firewall and ARM CoreSight
    // ITM debug buses.
    //
    // The threshold is sized to ~4 serial-bus round-trips (4 * 64 = 256
    // clocks), comfortably longer than the worst-case good-tap response.
    localparam int SCOPE_RSTALL_TIMEOUT = 256;
    localparam int SCOPE_RSTALL_CTR_W   = $clog2(SCOPE_RSTALL_TIMEOUT + 1);
    localparam logic [63:0] SCOPE_BUS_SENTINEL = 64'hDEAD_DEAD_DEAD_DEAD;

    reg [63:0] scope_bus_wdata, scope_bus_rdata;
    reg [5:0] scope_bus_ctr;

    reg cmd_scope_writing, cmd_scope_reading;
    reg scope_bus_out_r;
    reg scope_rdata_valid;

    reg is_scope_waddr, is_scope_raddr;

    // Watchdog state: counts cycles of stalled rvalid; resets on a new AR
    // (next read starts fresh) or when scope_rdata_valid asserts.
    reg [SCOPE_RSTALL_CTR_W-1:0] scope_rstall_ctr;
    wire scope_rvalid_stall_q;  // pre-watchdog stall condition (forward decl)

    always @(posedge clk) begin
        if (reset) begin
            cmd_scope_reading <= 0;
            cmd_scope_writing <= 0;
            scope_bus_ctr <= '0;
            scope_bus_out_r <= 0;
            is_scope_waddr <= 0;
            is_scope_raddr <= 0;
            scope_bus_rdata <= '0;
            scope_rdata_valid <= 0;
            scope_rstall_ctr <= '0;
        end else begin
            scope_bus_out_r <= 0;
            if (s_axi_aw_fire) begin
                is_scope_waddr <= (s_axi_awaddr[ADDR_BITS-1:0] == ADDR_SCP_0)
                               || (s_axi_awaddr[ADDR_BITS-1:0] == ADDR_SCP_1);
            end
            if (s_axi_ar_fire) begin
                is_scope_raddr <= (s_axi_araddr[ADDR_BITS-1:0] == ADDR_SCP_0)
                               || (s_axi_araddr[ADDR_BITS-1:0] == ADDR_SCP_1);
                scope_rstall_ctr <= '0;   // new read window: watchdog starts fresh
            end
            if (s_axi_w_fire && waddr == ADDR_SCP_0) begin
                scope_bus_wdata[31:0] <= s_axi_wdata;
            end
            if (s_axi_w_fire && waddr == ADDR_SCP_1) begin
                scope_bus_wdata[63:32] <= s_axi_wdata;
                cmd_scope_writing <= 1;
                scope_rdata_valid <= 0;
                scope_bus_out_r   <= 1;
                scope_bus_ctr     <= 63;
            end
            if (scope_bus_in) begin
                cmd_scope_reading <= 1;
                scope_bus_rdata   <= '0;
                scope_bus_ctr     <= 63;
            end
            if (cmd_scope_reading) begin
                scope_bus_rdata <= {scope_bus_rdata[62:0], scope_bus_in};
                scope_bus_ctr   <= scope_bus_ctr - 1;
                if (scope_bus_ctr == 0) begin
                    cmd_scope_reading <= 0;
                    scope_rdata_valid <= 1;
                    scope_bus_ctr     <= 0;
                end
            end
            if (cmd_scope_writing) begin
                scope_bus_out_r <= scope_bus_wdata[scope_bus_ctr];
                scope_bus_ctr <= scope_bus_ctr - 1;
                if (scope_bus_ctr == 0) begin
                    cmd_scope_writing <= 0;
                    scope_bus_ctr <= 0;
                end
            end
            // ---- Watchdog tick ----
            // Count up only while the host is genuinely waiting for a
            // SCOPE read (rvalid_stall asserted = host stalled). On
            // overflow, force scope_rdata_valid=1 with the sentinel and
            // tear down any in-flight serial-read state so the next
            // legitimate transaction starts clean.
            if (scope_rvalid_stall_q
                && scope_rstall_ctr != SCOPE_RSTALL_CTR_W'(SCOPE_RSTALL_TIMEOUT)) begin
                scope_rstall_ctr <= scope_rstall_ctr + 1;
            end
            if (scope_rvalid_stall_q
                && scope_rstall_ctr == SCOPE_RSTALL_CTR_W'(SCOPE_RSTALL_TIMEOUT)) begin
                scope_bus_rdata   <= SCOPE_BUS_SENTINEL;
                scope_rdata_valid <= 1;
                cmd_scope_reading <= 0;
                scope_bus_ctr     <= 0;
            end
        end
    end

    assign scope_bus_out = scope_bus_out_r;

    assign wready_stall = is_scope_waddr && cmd_scope_writing;
    // Pre-watchdog stall: the raw "host waiting" signal that drives the
    // counter. The publicly-visible `rvalid_stall` is the same expression
    // — the watchdog drops it from inside by raising scope_rdata_valid.
    assign scope_rvalid_stall_q = is_scope_raddr && ~scope_rdata_valid;
    assign rvalid_stall = scope_rvalid_stall_q;

`else

    assign wready_stall = 0;
    assign rvalid_stall = 0;

`endif

    // AXI Write Request
    assign s_axi_awready = (wstate == WSTATE_ADDR);
    assign s_axi_wready  = (wstate == WSTATE_DATA) && ~wready_stall;

    // AXI Write Response
    assign s_axi_bvalid  = (wstate == WSTATE_RESP);
    assign s_axi_bresp   = 2'b00;  // OKAY

    assign s_axi_aw_fire = s_axi_awvalid && s_axi_awready;
    assign s_axi_w_fire  = s_axi_wvalid && s_axi_wready;
    assign s_axi_b_fire  = s_axi_bvalid && s_axi_bready;

    // wstate
    always @(posedge clk) begin
        if (reset) begin
            wstate <= WSTATE_ADDR;
        end else begin
            case (wstate)
            WSTATE_ADDR: wstate <= s_axi_aw_fire ? WSTATE_DATA : WSTATE_ADDR;
            WSTATE_DATA: wstate <= s_axi_w_fire ? WSTATE_RESP : WSTATE_DATA;
            WSTATE_RESP: wstate <= s_axi_b_fire ? WSTATE_ADDR : WSTATE_RESP;
            default:     wstate <= WSTATE_ADDR;
            endcase
        end
    end

    // waddr — write-address latch (used by the SCOPE register decode)
    always @(posedge clk) begin
        if (s_axi_aw_fire) begin
            waddr <= s_axi_awaddr[ADDR_BITS-1:0];
        end
    end

    // ap_ctrl writes are accepted but inert — the kernel is CP-driven.

    // AXI Read Request
    assign s_axi_arready = (rstate == RSTATE_ADDR);

    // AXI Read Response
    assign s_axi_rvalid  = (rstate == RSTATE_RESP);
    assign s_axi_rdata   = rdata;
    assign s_axi_rresp   = 2'b00;  // OKAY

    assign s_axi_ar_fire = s_axi_arvalid && s_axi_arready;
    assign s_axi_r_fire  = s_axi_rvalid && s_axi_rready;

    // rstate
    always @(posedge clk) begin
        if (reset) begin
            rstate <= RSTATE_ADDR;
        end else begin
            case (rstate)
            RSTATE_ADDR: rstate <= s_axi_ar_fire ? RSTATE_DATA : RSTATE_ADDR;
            RSTATE_DATA: rstate <= rvalid_stall ? RSTATE_DATA : RSTATE_RESP;
            RSTATE_RESP: rstate <= s_axi_r_fire ? RSTATE_ADDR : RSTATE_RESP;
            default:     rstate <= RSTATE_ADDR;
            endcase
        end
    end

    // raddr
    always @(posedge clk) begin
        if (s_axi_ar_fire) begin
            raddr <= s_axi_araddr[ADDR_BITS-1:0];
        end
    end

    // rdata — the ap_ctrl stub reports the kernel permanently idle (bit 2);
    // the CP, not ap_ctrl, drives execution. SCOPE returns its serial word.
    always @(posedge clk) begin
        rdata <= '0;
        case (raddr)
            ADDR_AP_CTRL: begin
                rdata[2] <= 1'b1;   // ap_idle = 1
            end
        `ifdef SCOPE
            ADDR_SCP_0: begin
                rdata <= scope_bus_rdata[31:0];
            end
            ADDR_SCP_1: begin
                rdata <= scope_bus_rdata[63:32];
            end
        `endif
            default:;
        endcase
    end

endmodule
