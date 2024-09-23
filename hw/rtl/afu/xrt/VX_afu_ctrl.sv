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

module VX_afu_ctrl #(
    parameter S_AXI_ADDR_WIDTH = 8,
    parameter S_AXI_DATA_WIDTH = 32,
    parameter M_AXI_ADDR_WIDTH = 25
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
    input  wire                         s_axi_rready,

    output wire                         ap_reset,
    output wire                         ap_start,
    input  wire                         ap_done,
    input  wire                         ap_ready,
    input  wire                         ap_idle,
    output wire                         interrupt,

`ifdef SCOPE
    input wire                          scope_bus_in,
    output wire                         scope_bus_out,
`endif

    output wire                         dcr_wr_valid,
    output wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr,
    output wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data
);

    // Address Info
    // 0x00 : Control signals
    //        bit 0  - ap_start (Read/Write/COH)
    //        bit 1  - ap_done (Read/COR)
    //        bit 2  - ap_idle (Read)
    //        bit 3  - ap_ready (Read)
    //        bit 4  - ap_reset (Write)
    //        bit 7  - auto_restart (Read/Write)
    //        others - reserved
    // 0x04 : Global Interrupt Enable Register
    //        bit 0  - Global Interrupt Enable (Read/Write)
    //        others - reserved
    // 0x08 : IP Interrupt Enable Register (Read/Write)
    //        bit 0  - Channel 0 (ap_done)
    //        bit 1  - Channel 1 (ap_ready)
    //        others - reserved
    // 0x0c : IP Interrupt Status Register (Read/TOW)
    //        bit 0  - Channel 0 (ap_done)
    //        bit 1  - Channel 1 (ap_ready)
    //        others - reserved
    // 0x10 : Low 32-bit Data signal of DEV_CAPS
    // 0x14 : High 32-bit Data signal of DEV_CAPS
    // 0x18 : Control signal of DEV_CAPS
    // 0x1C : Low 32-bit Data signal of ISA_CAPS
    // 0x20 : High 32-bit Data signal of ISA_CAPS
    // 0x24 : Control signal of ISA_CAPS
    // 0x28 : Low 32-bit Data signal of DCR
    // 0x2C : High 32-bit Data signal of DCR
    // 0x30 : Control signal of DCR
    // 0x34 : Low 32-bit Data signal of SCP
    // 0x38 : High 32-bit Data signal of SCP
    // 0x3C : Control signal of SCP
    // 0x40 : Low 32-bit Data signal of MEM
    // 0x44 : High 32-bit Data signal of MEM
    // 0x48 : Control signal of MEM
    // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

    // Parameters
    localparam
        ADDR_AP_CTRL    = 8'h00,
        ADDR_GIE        = 8'h04,
        ADDR_IER        = 8'h08,
        ADDR_ISR        = 8'h0C,

        ADDR_DEV_0      = 8'h10,
        ADDR_DEV_1      = 8'h14,

        ADDR_ISA_0      = 8'h18,
        ADDR_ISA_1      = 8'h1C,

        ADDR_DCR_0      = 8'h20,
        ADDR_DCR_1      = 8'h24,

    `ifdef SCOPE
        ADDR_SCP_0      = 8'h28,
        ADDR_SCP_1      = 8'h2C,
    `endif

        ADDR_MEM_0      = 8'h30,
        ADDR_MEM_1      = 8'h34,

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

    // device caps
    wire [63:0] dev_caps = {8'b0,
                            5'(M_AXI_ADDR_WIDTH-16),
                            3'(`CLOG2(`PLATFORM_MEMORY_BANKS)),
                            8'(`LMEM_ENABLED ? `LMEM_LOG_SIZE : 0),
                            16'(`NUM_CORES * `NUM_CLUSTERS),
                            8'(`NUM_WARPS),
                            8'(`NUM_THREADS),
                            8'(`IMPLEMENTATION_ID)};

    wire [63:0] isa_caps = {32'(`MISA_EXT),
                            2'(`CLOG2(`XLEN)-4),
                            30'(`MISA_STD)};

    reg [WSTATE_WIDTH-1:0] wstate;
    reg [ADDR_BITS-1:0] waddr;
    wire [31:0] wmask;
    wire        s_axi_aw_fire;
    wire        s_axi_w_fire;
    wire        s_axi_b_fire;

    logic [RSTATE_WIDTH-1:0] rstate;
    reg [31:0]  rdata;
    reg [ADDR_BITS-1:0] raddr;
    wire        s_axi_ar_fire;
    wire        s_axi_r_fire;

    reg         ap_reset_r;
    reg         ap_start_r;
    reg         auto_restart_r;
    reg         gie_r;
    reg [1:0]   ier_r;
    reg [1:0]   isr_r;
    reg [31:0]  dcra_r;
    reg [31:0]  dcrv_r;
    reg         dcr_wr_valid_r;

    logic wready_stall;
    logic rvalid_stall;

`ifdef SCOPE

    reg [63:0] scope_bus_wdata, scope_bus_rdata;
    reg [5:0] scope_bus_ctr;

    reg cmd_scope_writing, cmd_scope_reading;
    reg scope_bus_out_r;
    reg scope_rdata_valid;

    reg is_scope_waddr, is_scope_raddr;

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
        end else begin
            scope_bus_out_r <= 0;
            if (s_axi_aw_fire) begin
                is_scope_waddr <= (s_axi_awaddr[ADDR_BITS-1:0] == ADDR_SCP_0)
                               || (s_axi_awaddr[ADDR_BITS-1:0] == ADDR_SCP_1);
            end
            if (s_axi_ar_fire) begin
                is_scope_raddr <= (s_axi_araddr[ADDR_BITS-1:0] == ADDR_SCP_0)
                               || (s_axi_araddr[ADDR_BITS-1:0] == ADDR_SCP_1);
            end
            if (s_axi_w_fire && waddr == ADDR_SCP_0) begin
                scope_bus_wdata[31:0] <= (s_axi_wdata & wmask) | (scope_bus_wdata[31:0] & ~wmask);
            end
            if (s_axi_w_fire && waddr == ADDR_SCP_1) begin
                scope_bus_wdata[63:32] <= (s_axi_wdata & wmask) | (scope_bus_wdata[63:32] & ~wmask);
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
        end
    end

    assign scope_bus_out = scope_bus_out_r;

    assign wready_stall = is_scope_waddr && cmd_scope_writing;
    assign rvalid_stall = is_scope_raddr && ~scope_rdata_valid;

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

    for (genvar i = 0; i < 4; ++i) begin : g_wmask
        assign wmask[8 * i +: 8] = {8{s_axi_wstrb[i]}};
    end

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

    // waddr
    always @(posedge clk) begin
        if (s_axi_aw_fire) begin
            waddr <= s_axi_awaddr[ADDR_BITS-1:0];
        end
    end

    // wdata
    always @(posedge clk) begin
        if (reset) begin
            ap_start_r <= 0;
            ap_reset_r <= 0;
            auto_restart_r <= 0;

            gie_r <= 0;
            ier_r <= '0;
            isr_r <= '0;

            dcra_r <= '0;
            dcrv_r <= '0;
            dcr_wr_valid_r <= 0;
        end else begin
            dcr_wr_valid_r <= 0;
            ap_reset_r <= 0;

            if (ap_ready)
                ap_start_r <= auto_restart_r;

            if (s_axi_w_fire) begin
                case (waddr)
                ADDR_AP_CTRL: begin
                    if (s_axi_wstrb[0]) begin
                        if (s_axi_wdata[0])
                            ap_start_r <= 1;
                        if (s_axi_wdata[4])
                            ap_reset_r <= 1;
                        if (s_axi_wdata[7])
                            auto_restart_r <= 1;
                    end
                end
                ADDR_GIE: begin
                    if (s_axi_wstrb[0])
                        gie_r <= s_axi_wdata[0];
                end
                ADDR_IER: begin
                    if (s_axi_wstrb[0])
                        ier_r <= s_axi_wdata[1:0];
                end
                ADDR_ISR: begin
                    if (s_axi_wstrb[0])
                        isr_r <= isr_r ^ s_axi_wdata[1:0];
                end
                ADDR_DCR_0: begin
                    dcra_r <= (s_axi_wdata & wmask) | (dcra_r & ~wmask);
                end
                ADDR_DCR_1: begin
                    dcrv_r <= (s_axi_wdata & wmask) | (dcrv_r & ~wmask);
                    dcr_wr_valid_r <= 1;
                end
                default:;
                endcase

                if (ier_r[0] & ap_done)
                    isr_r[0] <= 1'b1;
                if (ier_r[1] & ap_ready)
                    isr_r[1] <= 1'b1;
            end
        end
    end

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
            RSTATE_DATA: rstate <= (~rvalid_stall) ? RSTATE_RESP : RSTATE_DATA;
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

    // rdata
    always @(posedge clk) begin
        rdata <= '0;
        case (raddr)
            ADDR_AP_CTRL: begin
                rdata[0] <= ap_start_r;
                rdata[1] <= ap_done;
                rdata[2] <= ap_idle;
                rdata[3] <= ap_ready;
                rdata[7] <= auto_restart_r;
            end
            ADDR_GIE: begin
                rdata <= 32'(gie_r);
            end
            ADDR_IER: begin
                rdata <= 32'(ier_r);
            end
            ADDR_ISR: begin
                rdata <= 32'(isr_r);
            end
            ADDR_DEV_0: begin
                rdata <= dev_caps[31:0];
            end
            ADDR_DEV_1: begin
                rdata <= dev_caps[63:32];
            end
            ADDR_ISA_0: begin
                rdata <= isa_caps[31:0];
            end
            ADDR_ISA_1: begin
                rdata <= isa_caps[63:32];
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

    assign ap_reset  = ap_reset_r;
    assign ap_start  = ap_start_r;
    assign interrupt = gie_r & (| isr_r);

    assign dcr_wr_valid = dcr_wr_valid_r;
    assign dcr_wr_addr  = `VX_DCR_ADDR_WIDTH'(dcra_r);
    assign dcr_wr_data  = `VX_DCR_DATA_WIDTH'(dcrv_r);

endmodule
