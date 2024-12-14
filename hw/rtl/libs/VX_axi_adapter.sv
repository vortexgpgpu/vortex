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

`include "VX_platform.vh"

`TRACING_OFF
module VX_axi_adapter #(
    parameter DATA_WIDTH     = 512,
    parameter ADDR_WIDTH_IN  = 26, // word-addressable
    parameter ADDR_WIDTH_OUT = 32, // byte-addressable
    parameter TAG_WIDTH_IN   = 8,
    parameter TAG_WIDTH_OUT  = 8,
    parameter NUM_PORTS_IN   = 1,
    parameter NUM_PORTS_OUT  = 1,
    parameter INTERLEAVE     = 0,
    parameter TAG_BUFFER_SIZE= 32,
    parameter ARBITER        = "R",
    parameter REQ_OUT_BUF    = 1,
    parameter RSP_OUT_BUF    = 1,
    parameter DATA_SIZE      = DATA_WIDTH/8
 ) (
    input  wire                     clk,
    input  wire                     reset,

    // Vortex request
    input wire                      mem_req_valid [NUM_PORTS_IN],
    input wire                      mem_req_rw [NUM_PORTS_IN],
    input wire [DATA_SIZE-1:0]      mem_req_byteen [NUM_PORTS_IN],
    input wire [ADDR_WIDTH_IN-1:0]  mem_req_addr [NUM_PORTS_IN],
    input wire [DATA_WIDTH-1:0]     mem_req_data [NUM_PORTS_IN],
    input wire [TAG_WIDTH_IN-1:0]   mem_req_tag [NUM_PORTS_IN],
    output wire                     mem_req_ready [NUM_PORTS_IN],

    // Vortex response
    output wire                     mem_rsp_valid [NUM_PORTS_IN],
    output wire [DATA_WIDTH-1:0]    mem_rsp_data [NUM_PORTS_IN],
    output wire [TAG_WIDTH_IN-1:0]  mem_rsp_tag [NUM_PORTS_IN],
    input wire                      mem_rsp_ready [NUM_PORTS_IN],

    // AXI write request address channel
    output wire                     m_axi_awvalid [NUM_PORTS_OUT],
    input wire                      m_axi_awready [NUM_PORTS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] m_axi_awaddr [NUM_PORTS_OUT],
    output wire [TAG_WIDTH_OUT-1:0] m_axi_awid [NUM_PORTS_OUT],
    output wire [7:0]               m_axi_awlen [NUM_PORTS_OUT],
    output wire [2:0]               m_axi_awsize [NUM_PORTS_OUT],
    output wire [1:0]               m_axi_awburst [NUM_PORTS_OUT],
    output wire [1:0]               m_axi_awlock [NUM_PORTS_OUT],
    output wire [3:0]               m_axi_awcache [NUM_PORTS_OUT],
    output wire [2:0]               m_axi_awprot [NUM_PORTS_OUT],
    output wire [3:0]               m_axi_awqos [NUM_PORTS_OUT],
    output wire [3:0]               m_axi_awregion [NUM_PORTS_OUT],

    // AXI write request data channel
    output wire                     m_axi_wvalid [NUM_PORTS_OUT],
    input wire                      m_axi_wready [NUM_PORTS_OUT],
    output wire [DATA_WIDTH-1:0]    m_axi_wdata [NUM_PORTS_OUT],
    output wire [DATA_SIZE-1:0]     m_axi_wstrb [NUM_PORTS_OUT],
    output wire                     m_axi_wlast [NUM_PORTS_OUT],

    // AXI write response channel
    input wire                      m_axi_bvalid [NUM_PORTS_OUT],
    output wire                     m_axi_bready [NUM_PORTS_OUT],
    input wire [TAG_WIDTH_OUT-1:0]  m_axi_bid [NUM_PORTS_OUT],
    input wire [1:0]                m_axi_bresp [NUM_PORTS_OUT],

    // AXI read address channel
    output wire                     m_axi_arvalid [NUM_PORTS_OUT],
    input wire                      m_axi_arready [NUM_PORTS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] m_axi_araddr [NUM_PORTS_OUT],
    output wire [TAG_WIDTH_OUT-1:0] m_axi_arid [NUM_PORTS_OUT],
    output wire [7:0]               m_axi_arlen [NUM_PORTS_OUT],
    output wire [2:0]               m_axi_arsize [NUM_PORTS_OUT],
    output wire [1:0]               m_axi_arburst [NUM_PORTS_OUT],
    output wire [1:0]               m_axi_arlock [NUM_PORTS_OUT],
    output wire [3:0]               m_axi_arcache [NUM_PORTS_OUT],
    output wire [2:0]               m_axi_arprot [NUM_PORTS_OUT],
    output wire [3:0]               m_axi_arqos [NUM_PORTS_OUT],
    output wire [3:0]               m_axi_arregion [NUM_PORTS_OUT],

    // AXI read response channel
    input wire                      m_axi_rvalid [NUM_PORTS_OUT],
    output wire                     m_axi_rready [NUM_PORTS_OUT],
    input wire [DATA_WIDTH-1:0]     m_axi_rdata [NUM_PORTS_OUT],
    input wire                      m_axi_rlast [NUM_PORTS_OUT],
    input wire [TAG_WIDTH_OUT-1:0]  m_axi_rid [NUM_PORTS_OUT],
    input wire [1:0]                m_axi_rresp [NUM_PORTS_OUT]
);
    localparam LOG2_DATA_SIZE = `CLOG2(DATA_SIZE);
    localparam PORT_SEL_BITS  = `CLOG2(NUM_PORTS_OUT);
    localparam PORT_SEL_WIDTH = `UP(PORT_SEL_BITS);
    localparam DST_ADDR_WDITH = (ADDR_WIDTH_OUT - LOG2_DATA_SIZE) + PORT_SEL_BITS; // convert output addresss to byte-addressable input space
    localparam PORT_OFFSETW   = DST_ADDR_WDITH - PORT_SEL_BITS;
    localparam NUM_PORTS_IN_BITS = `CLOG2(NUM_PORTS_IN);
    localparam NUM_PORTS_IN_WIDTH = `UP(NUM_PORTS_IN_BITS);
    localparam TAG_BUFFER_ADDRW = `CLOG2(TAG_BUFFER_SIZE);
    localparam NEEDED_TAG_WIDTH = TAG_WIDTH_IN + NUM_PORTS_IN_BITS;
    localparam READ_TAG_WIDTH = (NEEDED_TAG_WIDTH > TAG_WIDTH_OUT) ? TAG_BUFFER_ADDRW : TAG_WIDTH_IN;
    localparam READ_FULL_TAG_WIDTH = READ_TAG_WIDTH + PORT_SEL_BITS;
    localparam WRITE_TAG_WIDTH = `MIN(TAG_WIDTH_IN, TAG_WIDTH_OUT);
    localparam DST_TAG_WIDTH  = `MAX(READ_FULL_TAG_WIDTH, WRITE_TAG_WIDTH);
    localparam ARB_TAG_WIDTH  = `MAX(READ_TAG_WIDTH, WRITE_TAG_WIDTH);
    localparam ARB_DATAW      = 1 + PORT_OFFSETW + DATA_SIZE + DATA_WIDTH + ARB_TAG_WIDTH;

    `STATIC_ASSERT ((DST_ADDR_WDITH >= ADDR_WIDTH_IN), ("invalid address width: current=%0d, expected=%0d", DST_ADDR_WDITH, ADDR_WIDTH_IN))
    `STATIC_ASSERT ((TAG_WIDTH_OUT >= DST_TAG_WIDTH), ("invalid output tag width: current=%0d, expected=%0d", TAG_WIDTH_OUT, DST_TAG_WIDTH))

    // Ports selection
    wire [NUM_PORTS_IN-1:0][PORT_SEL_WIDTH-1:0] req_port_out_sel;
    wire [NUM_PORTS_IN-1:0][PORT_OFFSETW-1:0] req_port_out_off;

    if (NUM_PORTS_OUT > 1) begin : g_port_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            wire [DST_ADDR_WDITH-1:0] mem_req_addr_out = DST_ADDR_WDITH'(mem_req_addr[i]);
            if (INTERLEAVE) begin : g_interleave
                assign req_port_out_sel[i] = mem_req_addr_out[PORT_SEL_BITS-1:0];
                assign req_port_out_off[i] = mem_req_addr_out[PORT_SEL_BITS +: PORT_OFFSETW];
            end else begin : g_no_interleave
                assign req_port_out_sel[i] = mem_req_addr_out[PORT_OFFSETW +: PORT_SEL_BITS];
                assign req_port_out_off[i] = mem_req_addr_out[PORT_OFFSETW-1:0];
            end
        end
    end else begin : g_no_port_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            assign req_port_out_sel[i] = '0;
            assign req_port_out_off[i] = DST_ADDR_WDITH'(mem_req_addr[i]);
        end
    end

    // Tag handling logic
    wire [NUM_PORTS_IN-1:0] mem_rd_req_tag_ready;
    wire [NUM_PORTS_IN-1:0][READ_TAG_WIDTH-1:0] mem_rd_req_tag;
    wire [NUM_PORTS_IN-1:0][READ_TAG_WIDTH-1:0] mem_rd_rsp_tag;

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_tag_buf
        if (NEEDED_TAG_WIDTH > TAG_WIDTH_OUT) begin : g_enabled
            wire [TAG_BUFFER_ADDRW-1:0] tbuf_waddr, tbuf_raddr;
            wire tbuf_full;
            VX_index_buffer #(
                .DATAW (TAG_WIDTH_IN),
                .SIZE  (TAG_BUFFER_SIZE)
            ) tag_buf (
                .clk        (clk),
                .reset      (reset),
                .acquire_en (mem_req_valid[i] && ~mem_req_rw[i] && mem_req_ready[i]),
                .write_addr (tbuf_waddr),
                .write_data (mem_req_tag[i]),
                .read_data  (mem_rsp_tag[i]),
                .read_addr  (tbuf_raddr),
                .release_en (mem_rsp_valid[i] && mem_rsp_ready[i]),
                .full       (tbuf_full),
                `UNUSED_PIN (empty)
            );
            assign mem_rd_req_tag_ready[i] = ~tbuf_full;
            assign mem_rd_req_tag[i] = tbuf_waddr;
            assign tbuf_raddr = mem_rd_rsp_tag[i];
        end else begin : g_none
            assign mem_rd_req_tag_ready[i] = 1;
            assign mem_rd_req_tag[i] = mem_req_tag[i];
            assign mem_rsp_tag[i] = mem_rd_rsp_tag[i];
        end
    end

    // AXi write request synchronization
    wire [NUM_PORTS_OUT-1:0] m_axi_awvalid_w, m_axi_wvalid_w;
    wire [NUM_PORTS_OUT-1:0] m_axi_awready_w, m_axi_wready_w;
    reg [NUM_PORTS_OUT-1:0] m_axi_aw_ack, m_axi_w_ack, axi_write_ready;

    for (genvar i = 0; i < NUM_PORTS_OUT; ++i) begin : g_axi_write_ready
        VX_axi_write_ack axi_write_ack (
            .clk    (clk),
            .reset  (reset),
            .awvalid(m_axi_awvalid_w[i]),
            .awready(m_axi_awready_w[i]),
            .wvalid (m_axi_wvalid_w[i]),
            .wready (m_axi_wready_w[i]),
            .aw_ack (m_axi_aw_ack[i]),
            .w_ack  (m_axi_w_ack[i]),
            .tx_rdy (axi_write_ready[i]),
            `UNUSED_PIN (tx_ack)
        );
    end

    // Request ack

    wire [NUM_PORTS_OUT-1:0][NUM_PORTS_IN-1:0] arb_ready_in;
    wire [NUM_PORTS_IN-1:0][NUM_PORTS_OUT-1:0] arb_ready_in_w;

    VX_transpose #(
        .N (NUM_PORTS_OUT),
        .M (NUM_PORTS_IN)
    ) rdy_in_transpose (
        .data_in  (arb_ready_in),
        .data_out (arb_ready_in_w)
    );

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_ready_in
        assign mem_req_ready[i] = | arb_ready_in_w[i];
    end

    // AXI request handling

    for (genvar i = 0; i < NUM_PORTS_OUT; ++i) begin : g_axi_write_req

        wire [PORT_OFFSETW-1:0] arb_addr_out, buf_addr_r_out, buf_addr_w_out;
        wire [ARB_TAG_WIDTH-1:0] arb_tag_out;
        wire [WRITE_TAG_WIDTH-1:0] buf_tag_w_out;
        wire [READ_TAG_WIDTH-1:0] buf_tag_r_out;
        wire [NUM_PORTS_IN_WIDTH-1:0] arb_sel_out, buf_sel_out;
        wire [DATA_WIDTH-1:0] arb_data_out;
        wire [DATA_SIZE-1:0] arb_byteen_out;
        wire arb_valid_out, arb_ready_out;
        wire arb_rw_out;

        wire [NUM_PORTS_IN-1:0][ARB_DATAW-1:0] arb_data_in;
        wire [NUM_PORTS_IN-1:0] arb_valid_in;

        for (genvar j = 0; j < NUM_PORTS_IN; ++j) begin : g_valid_in
            wire tag_ready = mem_req_rw[j] || mem_rd_req_tag_ready[j];
            assign arb_valid_in[j] = mem_req_valid[j] && tag_ready && (req_port_out_sel[j] == i);
        end

        for (genvar j = 0; j < NUM_PORTS_IN; ++j) begin : g_data_in
            wire [ARB_TAG_WIDTH-1:0] tag_value = mem_req_rw[j] ? ARB_TAG_WIDTH'(mem_req_tag[j]) : ARB_TAG_WIDTH'(mem_rd_req_tag[j]);
            assign arb_data_in[j] = {mem_req_rw[j], req_port_out_off[j], mem_req_byteen[j], mem_req_data[j], tag_value};
        end

        VX_stream_arb #(
            .NUM_INPUTS (NUM_PORTS_IN),
            .NUM_OUTPUTS(1),
            .DATAW      (ARB_DATAW),
            .ARBITER    (ARBITER)
        ) aw_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arb_valid_in),
            .ready_in  (arb_ready_in[i]),
            .data_in   (arb_data_in),
            .data_out  ({arb_rw_out, arb_addr_out, arb_byteen_out, arb_data_out, arb_tag_out}),
            .valid_out (arb_valid_out),
            .ready_out (arb_ready_out),
            .sel_out   (arb_sel_out)
        );

        wire m_axi_arready_w;

        assign arb_ready_out = axi_write_ready[i] || m_axi_arready_w;

        // AXI write address channel

        assign m_axi_awvalid_w[i] = arb_valid_out && arb_rw_out && ~m_axi_aw_ack[i];

        VX_elastic_buffer #(
            .DATAW   (PORT_OFFSETW + WRITE_TAG_WIDTH),
            .SIZE    (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(REQ_OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(REQ_OUT_BUF))
        ) aw_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (m_axi_awvalid_w[i]),
            .ready_in  (m_axi_awready_w[i]),
            .data_in   ({arb_addr_out, WRITE_TAG_WIDTH'(arb_tag_out)}),
            .data_out  ({buf_addr_w_out, buf_tag_w_out}),
            .valid_out (m_axi_awvalid[i]),
            .ready_out (m_axi_awready[i])
        );

        assign m_axi_awaddr[i]  = ADDR_WIDTH_OUT'(buf_addr_w_out) << LOG2_DATA_SIZE;
        assign m_axi_awid[i]    = TAG_WIDTH_OUT'(buf_tag_w_out);
        assign m_axi_awlen[i]   = 8'b00000000;
        assign m_axi_awsize[i]  = 3'(LOG2_DATA_SIZE);
        assign m_axi_awburst[i] = 2'b00;
        assign m_axi_awlock[i]  = 2'b00;
        assign m_axi_awcache[i] = 4'b0000;
        assign m_axi_awprot[i]  = 3'b000;
        assign m_axi_awqos[i]   = 4'b0000;
        assign m_axi_awregion[i]= 4'b0000;

        // AXI write data channel

        assign m_axi_wvalid_w[i] = arb_valid_out && arb_rw_out && ~m_axi_w_ack[i];

        VX_elastic_buffer #(
            .DATAW   (DATA_SIZE + DATA_WIDTH),
            .SIZE    (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(REQ_OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(REQ_OUT_BUF))
        ) w_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (m_axi_wvalid_w[i]),
            .ready_in  (m_axi_wready_w[i]),
            .data_in   ({arb_byteen_out, arb_data_out}),
            .data_out  ({m_axi_wstrb[i], m_axi_wdata[i]}),
            .valid_out (m_axi_wvalid[i]),
            .ready_out (m_axi_wready[i])
        );

        assign m_axi_wlast[i] = 1'b1;

        // AXI read address channel

        VX_elastic_buffer #(
            .DATAW   (PORT_OFFSETW + READ_TAG_WIDTH + NUM_PORTS_IN_WIDTH),
            .SIZE    (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(REQ_OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(REQ_OUT_BUF))
        ) ar_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arb_valid_out && ~arb_rw_out),
            .ready_in  (m_axi_arready_w),
            .data_in   ({arb_addr_out, READ_TAG_WIDTH'(arb_tag_out), arb_sel_out}),
            .data_out  ({buf_addr_r_out, buf_tag_r_out, buf_sel_out}),
            .valid_out (m_axi_arvalid[i]),
            .ready_out (m_axi_arready[i])
        );

        assign m_axi_araddr[i] = ADDR_WIDTH_OUT'(buf_addr_r_out) << LOG2_DATA_SIZE;

        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign m_axi_arid[i] = TAG_WIDTH_OUT'({buf_tag_r_out, buf_sel_out});
        end else begin : g_no_input_sel
            `UNUSED_VAR (buf_sel_out)
            assign m_axi_arid[i] = TAG_WIDTH_OUT'(buf_tag_r_out);
        end

        assign m_axi_arlen[i]   = 8'b00000000;
        assign m_axi_arsize[i]  = 3'(LOG2_DATA_SIZE);
        assign m_axi_arburst[i] = 2'b00;
        assign m_axi_arlock[i]  = 2'b00;
        assign m_axi_arcache[i] = 4'b0000;
        assign m_axi_arprot[i]  = 3'b000;
        assign m_axi_arqos[i]   = 4'b0000;
        assign m_axi_arregion[i]= 4'b0000;
    end

    // AXI write response channel (ignore)

    for (genvar i = 0; i < NUM_PORTS_OUT; ++i) begin : g_axi_write_rsp
        `UNUSED_VAR (m_axi_bvalid[i])
        `UNUSED_VAR (m_axi_bid[i])
        `UNUSED_VAR (m_axi_bresp[i])
        assign m_axi_bready[i] = 1'b1;
        `RUNTIME_ASSERT(~m_axi_bvalid[i] || m_axi_bresp[i] == 0, ("%t: *** AXI response error", $time))
    end

    // AXI read response channel

    wire [NUM_PORTS_OUT-1:0] rd_rsp_valid_in;
    wire [NUM_PORTS_OUT-1:0][DATA_WIDTH+READ_TAG_WIDTH-1:0] rd_rsp_data_in;
    wire [NUM_PORTS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] rd_rsp_sel_in;
    wire [NUM_PORTS_OUT-1:0] rd_rsp_ready_in;

    for (genvar i = 0; i < NUM_PORTS_OUT; ++i) begin : g_rd_rsp_data_in
        assign rd_rsp_valid_in[i] = m_axi_rvalid[i];
        assign rd_rsp_data_in[i] = {m_axi_rdata[i], m_axi_rid[i][NUM_PORTS_IN_BITS +: READ_TAG_WIDTH]};
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign rd_rsp_sel_in[i] = m_axi_rid[i][0 +: NUM_PORTS_IN_BITS];
        end else begin : g_no_input_sel
            assign rd_rsp_sel_in[i] = 0;
        end
        assign m_axi_rready[i] = rd_rsp_ready_in[i];
        `RUNTIME_ASSERT(~(m_axi_rvalid[i] && m_axi_rlast[i] == 0), ("%t: *** AXI response error", $time))
        `RUNTIME_ASSERT(~(m_axi_rvalid[i] && m_axi_rresp[i] != 0), ("%t: *** AXI response error", $time))
    end

    wire [NUM_PORTS_IN-1:0] rd_rsp_valid_out;
    wire [NUM_PORTS_IN-1:0][DATA_WIDTH+READ_TAG_WIDTH-1:0] rd_rsp_data_out;
    wire [NUM_PORTS_IN-1:0] rd_rsp_ready_out;

    VX_stream_xbar #(
        .NUM_INPUTS (NUM_PORTS_OUT),
        .NUM_OUTPUTS(NUM_PORTS_IN),
        .DATAW      (DATA_WIDTH + READ_TAG_WIDTH),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rd_rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rd_rsp_valid_in),
        .data_in   (rd_rsp_data_in),
        .ready_in  (rd_rsp_ready_in),
        .sel_in    (rd_rsp_sel_in),
        .data_out  (rd_rsp_data_out),
        .valid_out (rd_rsp_valid_out),
        .ready_out (rd_rsp_ready_out),
        `UNUSED_PIN (collisions),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_rd_rsp_data_out
        assign mem_rsp_valid[i] = rd_rsp_valid_out[i];
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign {mem_rsp_data[i], mem_rd_rsp_tag[i]} = rd_rsp_data_out[i];
        end else begin : g_no_input_sel
            assign {mem_rsp_data[i], mem_rd_rsp_tag[i]} = rd_rsp_data_out[i];
        end
        assign rd_rsp_ready_out[i] = mem_rsp_ready[i];
    end

endmodule
`TRACING_ON
