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
module VX_mem_adapter #(
    parameter SRC_DATA_WIDTH = 1,
    parameter SRC_ADDR_WIDTH = 1,
    parameter DST_DATA_WIDTH = 1,
    parameter DST_ADDR_WIDTH = 1,
    parameter SRC_TAG_WIDTH  = 1,
    parameter DST_TAG_WIDTH  = 1,
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0
) (
    input wire                          clk,
    input wire                          reset,

    input wire                          mem_req_valid_in,
    input wire [SRC_ADDR_WIDTH-1:0]     mem_req_addr_in,
    input wire                          mem_req_rw_in,
    input wire [SRC_DATA_WIDTH/8-1:0]   mem_req_byteen_in,
    input wire [SRC_DATA_WIDTH-1:0]     mem_req_data_in,
    input wire [SRC_TAG_WIDTH-1:0]      mem_req_tag_in,
    output wire                         mem_req_ready_in,

    output wire                         mem_rsp_valid_in,
    output wire [SRC_DATA_WIDTH-1:0]    mem_rsp_data_in,
    output wire [SRC_TAG_WIDTH-1:0]     mem_rsp_tag_in,
    input wire                          mem_rsp_ready_in,

    output wire                         mem_req_valid_out,
    output wire [DST_ADDR_WIDTH-1:0]    mem_req_addr_out,
    output wire                         mem_req_rw_out,
    output wire [DST_DATA_WIDTH/8-1:0]  mem_req_byteen_out,
    output wire [DST_DATA_WIDTH-1:0]    mem_req_data_out,
    output wire [DST_TAG_WIDTH-1:0]     mem_req_tag_out,
    input wire                          mem_req_ready_out,

    input wire                          mem_rsp_valid_out,
    input wire [DST_DATA_WIDTH-1:0]     mem_rsp_data_out,
    input wire [DST_TAG_WIDTH-1:0]      mem_rsp_tag_out,
    output wire                         mem_rsp_ready_out
);
    `STATIC_ASSERT ((DST_TAG_WIDTH >= SRC_TAG_WIDTH), ("oops!"))

    localparam DST_DATA_SIZE = (DST_DATA_WIDTH / 8);
    localparam DST_LDATAW = `CLOG2(DST_DATA_WIDTH);
    localparam SRC_LDATAW = `CLOG2(SRC_DATA_WIDTH);
    localparam D = `ABS(DST_LDATAW - SRC_LDATAW);
    localparam P = 2**D;

    wire                         mem_req_valid_out_w;
    wire [DST_ADDR_WIDTH-1:0]    mem_req_addr_out_w;
    wire                         mem_req_rw_out_w;
    wire [DST_DATA_WIDTH/8-1:0]  mem_req_byteen_out_w;
    wire [DST_DATA_WIDTH-1:0]    mem_req_data_out_w;
    wire [DST_TAG_WIDTH-1:0]     mem_req_tag_out_w;
    wire                         mem_req_ready_out_w;

    wire                         mem_rsp_valid_in_w;
    wire [SRC_DATA_WIDTH-1:0]    mem_rsp_data_in_w;
    wire [SRC_TAG_WIDTH-1:0]     mem_rsp_tag_in_w;
    wire                         mem_rsp_ready_in_w;

    `UNUSED_VAR (mem_rsp_tag_out)

    if (DST_LDATAW > SRC_LDATAW) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        wire [D-1:0] req_idx = mem_req_addr_in[D-1:0];
        wire [D-1:0] rsp_idx = mem_rsp_tag_out[D-1:0];

        wire [SRC_ADDR_WIDTH-D-1:0] mem_req_addr_in_qual = mem_req_addr_in[SRC_ADDR_WIDTH-1:D];

        wire [P-1:0][SRC_DATA_WIDTH-1:0] mem_rsp_data_out_w = mem_rsp_data_out;

        if (DST_ADDR_WIDTH < (SRC_ADDR_WIDTH - D)) begin
            `UNUSED_VAR (mem_req_addr_in_qual)
            assign mem_req_addr_out_w = mem_req_addr_in_qual[DST_ADDR_WIDTH-1:0];
        end else if (DST_ADDR_WIDTH > (SRC_ADDR_WIDTH - D)) begin
            assign mem_req_addr_out_w = DST_ADDR_WIDTH'(mem_req_addr_in_qual);
        end else begin
            assign mem_req_addr_out_w = mem_req_addr_in_qual;
        end

        assign mem_req_valid_out_w  = mem_req_valid_in;
        assign mem_req_rw_out_w     = mem_req_rw_in;
        assign mem_req_byteen_out_w = DST_DATA_SIZE'(mem_req_byteen_in) << ((DST_LDATAW-3)'(req_idx) << (SRC_LDATAW-3));
        assign mem_req_data_out_w   = DST_DATA_WIDTH'(mem_req_data_in) << ((DST_LDATAW'(req_idx)) << SRC_LDATAW);
        assign mem_req_tag_out_w    = DST_TAG_WIDTH'({mem_req_tag_in, req_idx});
        assign mem_req_ready_in     = mem_req_ready_out_w;

        assign mem_rsp_valid_in_w   = mem_rsp_valid_out;
        assign mem_rsp_data_in_w    = mem_rsp_data_out_w[rsp_idx];
        assign mem_rsp_tag_in_w     = SRC_TAG_WIDTH'(mem_rsp_tag_out[SRC_TAG_WIDTH+D-1:D]);
        assign mem_rsp_ready_out    = mem_rsp_ready_in_w;

    end else if (DST_LDATAW < SRC_LDATAW) begin

        reg [D-1:0] req_ctr, rsp_ctr;

        reg [P-1:0][DST_DATA_WIDTH-1:0] mem_rsp_data_out_r, mem_rsp_data_out_n;

        wire mem_req_out_fire = mem_req_valid_out && mem_req_ready_out;
        wire mem_rsp_in_fire = mem_rsp_valid_out && mem_rsp_ready_out;

        wire [P-1:0][DST_DATA_WIDTH-1:0] mem_req_data_in_w = mem_req_data_in;
        wire [P-1:0][DST_DATA_SIZE-1:0] mem_req_byteen_in_w = mem_req_byteen_in;

        always @(*) begin
            mem_rsp_data_out_n = mem_rsp_data_out_r;
            if (mem_rsp_in_fire) begin
                mem_rsp_data_out_n[rsp_ctr] = mem_rsp_data_out;
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                req_ctr <= '0;
                rsp_ctr <= '0;
            end else begin
                if (mem_req_out_fire) begin
                    req_ctr <= req_ctr + 1;
                end
                if (mem_rsp_in_fire) begin
                    rsp_ctr <= rsp_ctr + 1;
                end
            end
            mem_rsp_data_out_r <= mem_rsp_data_out_n;
        end

        reg [DST_TAG_WIDTH-1:0] mem_rsp_tag_in_r;
        wire [DST_TAG_WIDTH-1:0] mem_rsp_tag_in_x;

        always @(posedge clk) begin
            if (mem_rsp_in_fire) begin
                mem_rsp_tag_in_r <= mem_rsp_tag_out;
            end
        end
        assign mem_rsp_tag_in_x = (rsp_ctr != 0) ? mem_rsp_tag_in_r : mem_rsp_tag_out;
        `RUNTIME_ASSERT(!mem_rsp_in_fire || (mem_rsp_tag_in_x == mem_rsp_tag_out),
            ("%t: *** out-of-order memory reponse! cur=%d, expected=%d", $time, mem_rsp_tag_in_x, mem_rsp_tag_out))

        wire [SRC_ADDR_WIDTH+D-1:0] mem_req_addr_in_qual = {mem_req_addr_in, req_ctr};

        if (DST_ADDR_WIDTH < (SRC_ADDR_WIDTH + D)) begin
            `UNUSED_VAR (mem_req_addr_in_qual)
            assign mem_req_addr_out_w = mem_req_addr_in_qual[DST_ADDR_WIDTH-1:0];
        end else if (DST_ADDR_WIDTH > (SRC_ADDR_WIDTH + D)) begin
            assign mem_req_addr_out_w = DST_ADDR_WIDTH'(mem_req_addr_in_qual);
        end else begin
            assign mem_req_addr_out_w = mem_req_addr_in_qual;
        end

        assign mem_req_valid_out_w  = mem_req_valid_in;
        assign mem_req_rw_out_w     = mem_req_rw_in;
        assign mem_req_byteen_out_w = mem_req_byteen_in_w[req_ctr];
        assign mem_req_data_out_w   = mem_req_data_in_w[req_ctr];
        assign mem_req_tag_out_w    = DST_TAG_WIDTH'(mem_req_tag_in);
        assign mem_req_ready_in     = mem_req_ready_out_w && (req_ctr == (P-1));

        assign mem_rsp_valid_in_w   = mem_rsp_valid_out && (rsp_ctr == (P-1));
        assign mem_rsp_data_in_w    = mem_rsp_data_out_n;
        assign mem_rsp_tag_in_w     = SRC_TAG_WIDTH'(mem_rsp_tag_out);
        assign mem_rsp_ready_out    = mem_rsp_ready_in_w;

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        if (DST_ADDR_WIDTH < SRC_ADDR_WIDTH) begin
            `UNUSED_VAR (mem_req_addr_in)
            assign mem_req_addr_out_w = mem_req_addr_in[DST_ADDR_WIDTH-1:0];
        end else if (DST_ADDR_WIDTH > SRC_ADDR_WIDTH) begin
            assign mem_req_addr_out_w = DST_ADDR_WIDTH'(mem_req_addr_in);
        end else begin
            assign mem_req_addr_out_w = mem_req_addr_in;
        end

        assign mem_req_valid_out_w  = mem_req_valid_in;
        assign mem_req_rw_out_w     = mem_req_rw_in;
        assign mem_req_byteen_out_w = mem_req_byteen_in;
        assign mem_req_data_out_w   = mem_req_data_in;
        assign mem_req_tag_out_w    = DST_TAG_WIDTH'(mem_req_tag_in);
        assign mem_req_ready_in     = mem_req_ready_out_w;

        assign mem_rsp_valid_in_w   = mem_rsp_valid_out;
        assign mem_rsp_data_in_w    = mem_rsp_data_out;
        assign mem_rsp_tag_in_w     = SRC_TAG_WIDTH'(mem_rsp_tag_out);
        assign mem_rsp_ready_out    = mem_rsp_ready_in_w;

    end

    VX_elastic_buffer #(
        .DATAW    (1 + DST_DATA_SIZE + DST_ADDR_WIDTH + DST_DATA_WIDTH + DST_TAG_WIDTH),
        .SIZE     (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
        .OUT_REG  (`TO_OUT_BUF_REG(REQ_OUT_BUF))
    ) req_out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_valid_out_w),
        .ready_in  (mem_req_ready_out_w),
        .data_in   ({mem_req_rw_out_w, mem_req_byteen_out_w, mem_req_addr_out_w, mem_req_data_out_w, mem_req_tag_out_w}),
        .data_out  ({mem_req_rw_out,   mem_req_byteen_out,   mem_req_addr_out,   mem_req_data_out,   mem_req_tag_out}),
        .valid_out (mem_req_valid_out),
        .ready_out (mem_req_ready_out)
    );

    VX_elastic_buffer #(
        .DATAW    (SRC_DATA_WIDTH + SRC_TAG_WIDTH),
        .SIZE     (`TO_OUT_BUF_SIZE(RSP_OUT_BUF)),
        .OUT_REG  (`TO_OUT_BUF_REG(RSP_OUT_BUF))
    ) rsp_in_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_valid_in_w),
        .ready_in  (mem_rsp_ready_in_w),
        .data_in   ({mem_rsp_data_in_w, mem_rsp_tag_in_w}),
        .data_out  ({mem_rsp_data_in,   mem_rsp_tag_in}),
        .valid_out (mem_rsp_valid_in),
        .ready_out (mem_rsp_ready_in)
    );

endmodule
`TRACING_ON
