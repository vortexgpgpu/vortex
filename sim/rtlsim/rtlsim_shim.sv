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

`include "VX_define.vh"

module rtlsim_shim import VX_gpu_pkg::*; #(
    parameter MEM_DATA_WIDTH = `PLATFORM_MEMORY_DATA_WIDTH,
    parameter MEM_ADDR_WIDTH = `PLATFORM_MEMORY_ADDR_WIDTH,
    parameter MEM_NUM_BANKS  = `PLATFORM_MEMORY_BANKS,
    parameter MEM_TAG_WIDTH  = 64
) (
    `SCOPE_IO_DECL

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid [MEM_NUM_BANKS],
    output wire                             mem_req_rw [MEM_NUM_BANKS],
    output wire [(MEM_DATA_WIDTH/8)-1:0]    mem_req_byteen [MEM_NUM_BANKS],
    output wire [MEM_ADDR_WIDTH-1:0]        mem_req_addr [MEM_NUM_BANKS],
    output wire [MEM_DATA_WIDTH-1:0]        mem_req_data [MEM_NUM_BANKS],
    output wire [MEM_TAG_WIDTH-1:0]         mem_req_tag [MEM_NUM_BANKS],
    input  wire                             mem_req_ready [MEM_NUM_BANKS],

    // Memory response
    input wire                              mem_rsp_valid [MEM_NUM_BANKS],
    input wire [MEM_DATA_WIDTH-1:0]         mem_rsp_data [MEM_NUM_BANKS],
    input wire [MEM_TAG_WIDTH-1:0]          mem_rsp_tag [MEM_NUM_BANKS],
    output wire                             mem_rsp_ready [MEM_NUM_BANKS],

    // DCR write request
    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,

    // Status
    output wire                             busy
);
    localparam DST_LDATAW = `CLOG2(MEM_DATA_WIDTH);
    localparam SRC_LDATAW = `CLOG2(`VX_MEM_DATA_WIDTH);
    localparam SUB_LDATAW = DST_LDATAW - SRC_LDATAW;
    localparam VX_MEM_TAG_A_WIDTH  = `VX_MEM_TAG_WIDTH + `MAX(SUB_LDATAW, 0);
    localparam VX_MEM_ADDR_A_WIDTH = `VX_MEM_ADDR_WIDTH - SUB_LDATAW;

    wire                            vx_mem_req_valid [`VX_MEM_PORTS];
    wire                            vx_mem_req_rw [`VX_MEM_PORTS];
    wire [`VX_MEM_BYTEEN_WIDTH-1:0] vx_mem_req_byteen [`VX_MEM_PORTS];
    wire [`VX_MEM_ADDR_WIDTH-1:0]   vx_mem_req_addr [`VX_MEM_PORTS];
    wire [`VX_MEM_DATA_WIDTH-1:0]   vx_mem_req_data [`VX_MEM_PORTS];
    wire [`VX_MEM_TAG_WIDTH-1:0]    vx_mem_req_tag [`VX_MEM_PORTS];
    wire                            vx_mem_req_ready [`VX_MEM_PORTS];

    wire                            vx_mem_rsp_valid [`VX_MEM_PORTS];
    wire [`VX_MEM_DATA_WIDTH-1:0]   vx_mem_rsp_data [`VX_MEM_PORTS];
    wire [`VX_MEM_TAG_WIDTH-1:0]    vx_mem_rsp_tag [`VX_MEM_PORTS];
    wire                            vx_mem_rsp_ready [`VX_MEM_PORTS];

    `SCOPE_IO_SWITCH (1);

    Vortex vortex (
        `SCOPE_IO_BIND  (0)

        .clk            (clk),
        .reset          (reset),

        .mem_req_valid  (vx_mem_req_valid),
        .mem_req_rw     (vx_mem_req_rw),
        .mem_req_byteen (vx_mem_req_byteen),
        .mem_req_addr   (vx_mem_req_addr),
        .mem_req_data   (vx_mem_req_data),
        .mem_req_tag    (vx_mem_req_tag),
        .mem_req_ready  (vx_mem_req_ready),

        .mem_rsp_valid  (vx_mem_rsp_valid),
        .mem_rsp_data   (vx_mem_rsp_data),
        .mem_rsp_tag    (vx_mem_rsp_tag),
        .mem_rsp_ready  (vx_mem_rsp_ready),

        .dcr_wr_valid   (dcr_wr_valid),
        .dcr_wr_addr    (dcr_wr_addr),
        .dcr_wr_data    (dcr_wr_data),

        .busy           (busy)
    );

    wire                            mem_req_valid_a [`VX_MEM_PORTS];
    wire                            mem_req_rw_a [`VX_MEM_PORTS];
    wire [(MEM_DATA_WIDTH/8)-1:0]   mem_req_byteen_a [`VX_MEM_PORTS];
    wire [VX_MEM_ADDR_A_WIDTH-1:0]  mem_req_addr_a [`VX_MEM_PORTS];
    wire [MEM_DATA_WIDTH-1:0]       mem_req_data_a [`VX_MEM_PORTS];
    wire [VX_MEM_TAG_A_WIDTH-1:0]   mem_req_tag_a [`VX_MEM_PORTS];
    wire                            mem_req_ready_a [`VX_MEM_PORTS];

    wire                            mem_rsp_valid_a [`VX_MEM_PORTS];
    wire [MEM_DATA_WIDTH-1:0]       mem_rsp_data_a [`VX_MEM_PORTS];
    wire [VX_MEM_TAG_A_WIDTH-1:0]   mem_rsp_tag_a [`VX_MEM_PORTS];
    wire                            mem_rsp_ready_a [`VX_MEM_PORTS];

    // Adjust memory data width to match AXI interface
    for (genvar i = 0; i < `VX_MEM_PORTS; i++) begin : g_mem_adapter
        VX_mem_data_adapter #(
            .SRC_DATA_WIDTH (`VX_MEM_DATA_WIDTH),
            .DST_DATA_WIDTH (MEM_DATA_WIDTH),
            .SRC_ADDR_WIDTH (`VX_MEM_ADDR_WIDTH),
            .DST_ADDR_WIDTH (VX_MEM_ADDR_A_WIDTH),
            .SRC_TAG_WIDTH  (`VX_MEM_TAG_WIDTH),
            .DST_TAG_WIDTH  (VX_MEM_TAG_A_WIDTH),
            .REQ_OUT_BUF    (0),
            .RSP_OUT_BUF    (0)
        ) mem_data_adapter (
            .clk                (clk),
            .reset              (reset),

            .mem_req_valid_in   (vx_mem_req_valid[i]),
            .mem_req_addr_in    (vx_mem_req_addr[i]),
            .mem_req_rw_in      (vx_mem_req_rw[i]),
            .mem_req_byteen_in  (vx_mem_req_byteen[i]),
            .mem_req_data_in    (vx_mem_req_data[i]),
            .mem_req_tag_in     (vx_mem_req_tag[i]),
            .mem_req_ready_in   (vx_mem_req_ready[i]),

            .mem_rsp_valid_in   (vx_mem_rsp_valid[i]),
            .mem_rsp_data_in    (vx_mem_rsp_data[i]),
            .mem_rsp_tag_in     (vx_mem_rsp_tag[i]),
            .mem_rsp_ready_in   (vx_mem_rsp_ready[i]),

            .mem_req_valid_out  (mem_req_valid_a[i]),
            .mem_req_addr_out   (mem_req_addr_a[i]),
            .mem_req_rw_out     (mem_req_rw_a[i]),
            .mem_req_byteen_out (mem_req_byteen_a[i]),
            .mem_req_data_out   (mem_req_data_a[i]),
            .mem_req_tag_out    (mem_req_tag_a[i]),
            .mem_req_ready_out  (mem_req_ready_a[i]),

            .mem_rsp_valid_out  (mem_rsp_valid_a[i]),
            .mem_rsp_data_out   (mem_rsp_data_a[i]),
            .mem_rsp_tag_out    (mem_rsp_tag_a[i]),
            .mem_rsp_ready_out  (mem_rsp_ready_a[i])
        );
    end

    VX_mem_bank_adapter #(
        .DATA_WIDTH     (MEM_DATA_WIDTH),
        .ADDR_WIDTH_IN  (VX_MEM_ADDR_A_WIDTH),
        .ADDR_WIDTH_OUT (MEM_ADDR_WIDTH),
        .TAG_WIDTH_IN   (VX_MEM_TAG_A_WIDTH),
        .TAG_WIDTH_OUT  (MEM_TAG_WIDTH),
        .NUM_PORTS_IN   (`VX_MEM_PORTS),
        .NUM_BANKS_OUT  (MEM_NUM_BANKS),
        .INTERLEAVE     (0),
        .REQ_OUT_BUF    ((`VX_MEM_PORTS > 1) ? 2 : 0),
        .RSP_OUT_BUF    ((`VX_MEM_PORTS > 1 || MEM_NUM_BANKS > 1) ? 2 : 0)
    ) mem_bank_adapter (
        .clk                (clk),
        .reset              (reset),

        .mem_req_valid_in   (mem_req_valid_a),
        .mem_req_rw_in      (mem_req_rw_a),
        .mem_req_byteen_in  (mem_req_byteen_a),
        .mem_req_addr_in    (mem_req_addr_a),
        .mem_req_data_in    (mem_req_data_a),
        .mem_req_tag_in     (mem_req_tag_a),
        .mem_req_ready_in   (mem_req_ready_a),

        .mem_rsp_valid_in   (mem_rsp_valid_a),
        .mem_rsp_data_in    (mem_rsp_data_a),
        .mem_rsp_tag_in     (mem_rsp_tag_a),
        .mem_rsp_ready_in   (mem_rsp_ready_a),

        .mem_req_valid_out  (mem_req_valid),
        .mem_req_rw_out     (mem_req_rw),
        .mem_req_byteen_out (mem_req_byteen),
        .mem_req_addr_out   (mem_req_addr),
        .mem_req_data_out   (mem_req_data),
        .mem_req_tag_out    (mem_req_tag),
        .mem_req_ready_out  (mem_req_ready),

        .mem_rsp_valid_out  (mem_rsp_valid),
        .mem_rsp_data_out   (mem_rsp_data),
        .mem_rsp_tag_out    (mem_rsp_tag),
        .mem_rsp_ready_out  (mem_rsp_ready)
    );

endmodule
