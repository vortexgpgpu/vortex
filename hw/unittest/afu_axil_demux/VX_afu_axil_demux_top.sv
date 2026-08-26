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

// Harness for VX_afu_axil_demux, wired exactly as VX_afu_wrap wires it:
//   port 0 = the real VX_afu_ctrl
//   port 1 = a model of VX_cp_axil_regfile's write channel, whose ready
//            equations are copied verbatim from that file — the CP accepts a
//            W beat whenever its data buffer is empty, with no AW required,
//            which is what made a misrouted W beat vanish silently.

module VX_afu_axil_demux_top (
    input  wire        clk,
    input  wire        reset,

    input  wire        s_awvalid,
    output wire        s_awready,
    input  wire [15:0] s_awaddr,

    input  wire        s_wvalid,
    output wire        s_wready,
    input  wire [31:0] s_wdata,
    input  wire [3:0]  s_wstrb,

    output wire        s_bvalid,
    input  wire        s_bready,
    output wire [1:0]  s_bresp,

    input  wire        s_arvalid,
    output wire        s_arready,
    input  wire [15:0] s_araddr,

    output wire        s_rvalid,
    input  wire        s_rready,
    output wire [31:0] s_rdata,
    output wire [1:0]  s_rresp,

    output wire        ap_reset_out,
    output wire        dbg_cp_wr_data_buf_valid,
    output wire        dbg_cp_wr_addr_buf_valid
);
    wire        lg_awvalid, lg_awready;
    wire [15:0] lg_awaddr;
    wire        lg_wvalid, lg_wready;
    wire [31:0] lg_wdata;
    wire [3:0]  lg_wstrb;
    wire        lg_bvalid, lg_bready;
    wire [1:0]  lg_bresp;
    wire        lg_arvalid, lg_arready;
    wire [15:0] lg_araddr;
    wire        lg_rvalid, lg_rready;
    wire [31:0] lg_rdata;
    wire [1:0]  lg_rresp;

    wire        cp_awvalid, cp_awready;
    wire [15:0] cp_awaddr;
    wire        cp_wvalid, cp_wready;
    wire [31:0] cp_wdata;
    wire [3:0]  cp_wstrb;
    wire        cp_bvalid, cp_bready;
    wire [1:0]  cp_bresp;
    wire        cp_arvalid, cp_arready;
    wire [15:0] cp_araddr;
    wire        cp_rvalid, cp_rready;
    wire [31:0] cp_rdata;
    wire [1:0]  cp_rresp;

    VX_afu_axil_demux #(
        .ADDR_WIDTH (16),
        .DATA_WIDTH (32),
        .SEL_BIT    (12)
    ) demux (
        .clk (clk), .reset (reset),
        .s_awvalid (s_awvalid), .s_awready (s_awready), .s_awaddr (s_awaddr),
        .s_wvalid (s_wvalid), .s_wready (s_wready), .s_wdata (s_wdata), .s_wstrb (s_wstrb),
        .s_bvalid (s_bvalid), .s_bready (s_bready), .s_bresp (s_bresp),
        .s_arvalid (s_arvalid), .s_arready (s_arready), .s_araddr (s_araddr),
        .s_rvalid (s_rvalid), .s_rready (s_rready), .s_rdata (s_rdata), .s_rresp (s_rresp),

        .m0_awvalid (lg_awvalid), .m0_awready (lg_awready), .m0_awaddr (lg_awaddr),
        .m0_wvalid (lg_wvalid), .m0_wready (lg_wready), .m0_wdata (lg_wdata), .m0_wstrb (lg_wstrb),
        .m0_bvalid (lg_bvalid), .m0_bready (lg_bready), .m0_bresp (lg_bresp),
        .m0_arvalid (lg_arvalid), .m0_arready (lg_arready), .m0_araddr (lg_araddr),
        .m0_rvalid (lg_rvalid), .m0_rready (lg_rready), .m0_rdata (lg_rdata), .m0_rresp (lg_rresp),

        .m1_awvalid (cp_awvalid), .m1_awready (cp_awready), .m1_awaddr (cp_awaddr),
        .m1_wvalid (cp_wvalid), .m1_wready (cp_wready), .m1_wdata (cp_wdata), .m1_wstrb (cp_wstrb),
        .m1_bvalid (cp_bvalid), .m1_bready (cp_bready), .m1_bresp (cp_bresp),
        .m1_arvalid (cp_arvalid), .m1_arready (cp_arready), .m1_araddr (cp_araddr),
        .m1_rvalid (cp_rvalid), .m1_rready (cp_rready), .m1_rdata (cp_rdata), .m1_rresp (cp_rresp)
    );

    // ---- port 0: the real legacy slave ----
    VX_afu_ctrl #(
        .S_AXI_ADDR_WIDTH (8),
        .S_AXI_DATA_WIDTH (32)
    ) afu_ctrl (
        .clk (clk), .reset (reset),
        .s_axi_awvalid (lg_awvalid), .s_axi_awready (lg_awready), .s_axi_awaddr (lg_awaddr[7:0]),
        .s_axi_wvalid (lg_wvalid), .s_axi_wready (lg_wready), .s_axi_wdata (lg_wdata), .s_axi_wstrb (lg_wstrb),
        .s_axi_arvalid (lg_arvalid), .s_axi_arready (lg_arready), .s_axi_araddr (lg_araddr[7:0]),
        .s_axi_rvalid (lg_rvalid), .s_axi_rready (lg_rready), .s_axi_rdata (lg_rdata), .s_axi_rresp (lg_rresp),
        .s_axi_bvalid (lg_bvalid), .s_axi_bready (lg_bready), .s_axi_bresp (lg_bresp),
        .ap_reset (ap_reset_out), .soft_reset_busy (1'b0)
    );

    // ---- port 1: VX_cp_axil_regfile write-channel equations, verbatim ----
    reg        wr_addr_buf_valid;
    reg [15:0] wr_addr_buf;
    reg        wr_data_buf_valid;
    reg [31:0] wr_data_buf;
    reg        cp_bvalid_r;

    assign cp_awready = !wr_addr_buf_valid;
    assign cp_wready  = !wr_data_buf_valid;
    wire wr_commit = wr_addr_buf_valid && wr_data_buf_valid && !cp_bvalid_r;

    always @(posedge clk) begin
        if (reset) begin
            wr_addr_buf_valid <= 1'b0;
            wr_data_buf_valid <= 1'b0;
            wr_addr_buf       <= '0;
            wr_data_buf       <= '0;
        end else begin
            if (cp_awvalid && cp_awready) begin
                wr_addr_buf       <= cp_awaddr;
                wr_addr_buf_valid <= 1'b1;
            end
            if (cp_wvalid && cp_wready) begin
                wr_data_buf       <= cp_wdata;
                wr_data_buf_valid <= 1'b1;
            end
            if (wr_commit) begin
                wr_addr_buf_valid <= 1'b0;
                wr_data_buf_valid <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            cp_bvalid_r <= 1'b0;
        end else if (wr_commit) begin
            cp_bvalid_r <= 1'b1;
        end else if (cp_bvalid_r && cp_bready) begin
            cp_bvalid_r <= 1'b0;
        end
    end
    assign cp_bvalid = cp_bvalid_r;
    assign cp_bresp  = 2'b00;

    reg cp_rvalid_r;
    always @(posedge clk) begin
        if (reset) begin
            cp_rvalid_r <= 1'b0;
        end else if (cp_arvalid && cp_arready) begin
            cp_rvalid_r <= 1'b1;
        end else if (cp_rvalid_r && cp_rready) begin
            cp_rvalid_r <= 1'b0;
        end
    end
    assign cp_arready = !cp_rvalid_r;
    assign cp_rvalid  = cp_rvalid_r;
    assign cp_rdata   = 32'hC0DE0000;
    assign cp_rresp   = 2'b00;

    assign dbg_cp_wr_data_buf_valid = wr_data_buf_valid;
    assign dbg_cp_wr_addr_buf_valid = wr_addr_buf_valid;

    // AFU_ctrl decodes only the low 8 bits of its window.
    `UNUSED_VAR (lg_awaddr[15:8])
    `UNUSED_VAR (lg_araddr[15:8])
    `UNUSED_VAR (cp_wstrb)
    `UNUSED_VAR (wr_addr_buf)
    `UNUSED_VAR (wr_data_buf)
    `UNUSED_VAR (cp_araddr)

endmodule
