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
module VX_mem_coalescer #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS      = 1,
    parameter ADDR_WIDTH    = 32,
    parameter FLAGS_WIDTH   = 0,
    parameter DATA_IN_SIZE  = 4,
    parameter DATA_OUT_SIZE = 64,
    parameter TAG_WIDTH     = 8,
    parameter UUID_WIDTH    = 0, // upper section of the request tag contains the UUID
    parameter QUEUE_SIZE    = 8,

    parameter DATA_IN_WIDTH = DATA_IN_SIZE * 8,
    parameter DATA_OUT_WIDTH= DATA_OUT_SIZE * 8,
    parameter DATA_RATIO    = DATA_OUT_SIZE / DATA_IN_SIZE,
    parameter DATA_RATIO_W  = `LOG2UP(DATA_RATIO),
    parameter OUT_REQS      = NUM_REQS / DATA_RATIO,
    parameter OUT_ADDR_WIDTH= ADDR_WIDTH - DATA_RATIO_W,
    parameter QUEUE_ADDRW   = `CLOG2(QUEUE_SIZE),
    parameter OUT_TAG_WIDTH = UUID_WIDTH + QUEUE_ADDRW
) (
    input wire clk,
    input wire reset,

    // Input request
    input wire                          in_req_valid,
    input wire                          in_req_rw,
    input wire [NUM_REQS-1:0]           in_req_mask,
    input wire [NUM_REQS-1:0][DATA_IN_SIZE-1:0] in_req_byteen,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] in_req_addr,
    input wire [NUM_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] in_req_flags,
    input wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_req_data,
    input wire [TAG_WIDTH-1:0]          in_req_tag,
    output wire                         in_req_ready,

    // Input response
    output wire                         in_rsp_valid,
    output wire [NUM_REQS-1:0]          in_rsp_mask,
    output wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_rsp_data,
    output wire [TAG_WIDTH-1:0]         in_rsp_tag,
    input wire                          in_rsp_ready,

    // Output request
    output wire                         out_req_valid,
    output wire                         out_req_rw,
    output wire [OUT_REQS-1:0]          out_req_mask,
    output wire [OUT_REQS-1:0][DATA_OUT_SIZE-1:0] out_req_byteen,
    output wire [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] out_req_addr,
    output wire [OUT_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] out_req_flags,
    output wire [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_req_data,
    output wire [OUT_TAG_WIDTH-1:0]     out_req_tag,
    input wire 	                        out_req_ready,

    // Output response
    input wire                          out_rsp_valid,
    input wire [OUT_REQS-1:0]           out_rsp_mask,
    input wire [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_rsp_data,
    input wire [OUT_TAG_WIDTH-1:0]      out_rsp_tag,
    output wire                         out_rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT ((NUM_REQS > 1), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(NUM_REQS * DATA_IN_WIDTH, DATA_OUT_WIDTH), ("invalid parameter"))
    `STATIC_ASSERT ((NUM_REQS * DATA_IN_WIDTH >= DATA_OUT_WIDTH), ("invalid parameter"))
    `RUNTIME_ASSERT ((~in_req_valid || in_req_mask != 0), ("%t: invalid request mask", $time))
    `RUNTIME_ASSERT ((~out_rsp_valid || out_rsp_mask != 0), ("%t: invalid request mask", $time))

    localparam TAG_ID_WIDTH = TAG_WIDTH - UUID_WIDTH;
    //                           tag          + mask     + offest
    localparam IBUF_DATA_WIDTH = TAG_ID_WIDTH + NUM_REQS + (NUM_REQS * DATA_RATIO_W);

    localparam STATE_WAIT = 0;
    localparam STATE_SEND = 1;

    logic state_r, state_n;

    logic out_req_valid_r, out_req_valid_n;
    logic out_req_rw_r, out_req_rw_n;
    logic [OUT_REQS-1:0] out_req_mask_r, out_req_mask_n;
    logic [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] out_req_addr_r, out_req_addr_n;
    logic [OUT_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] out_req_flags_r, out_req_flags_n;
    logic [OUT_REQS-1:0][DATA_RATIO-1:0][DATA_IN_SIZE-1:0] out_req_byteen_r, out_req_byteen_n;
    logic [OUT_REQS-1:0][DATA_RATIO-1:0][DATA_IN_WIDTH-1:0] out_req_data_r, out_req_data_n;
    logic [OUT_TAG_WIDTH-1:0] out_req_tag_r, out_req_tag_n;

    reg in_req_ready_n;

    wire                        ibuf_push;
    wire                        ibuf_pop;
    wire [QUEUE_ADDRW-1:0]      ibuf_waddr;
    wire [QUEUE_ADDRW-1:0]      ibuf_raddr;
    wire                        ibuf_full;
    wire                        ibuf_empty;
    wire [IBUF_DATA_WIDTH-1:0]  ibuf_din;
    wire [IBUF_DATA_WIDTH-1:0]  ibuf_dout;

    logic [OUT_REQS-1:0] batch_valid_r, batch_valid_n;
    logic [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] seed_addr_r, seed_addr_n;
    logic [OUT_REQS-1:0][`UP(FLAGS_WIDTH)-1:0] seed_flags_r, seed_flags_n;
    logic [NUM_REQS-1:0] addr_matches_r, addr_matches_n;
    logic [NUM_REQS-1:0] req_rem_mask_r, req_rem_mask_n;

    wire [NUM_REQS-1:0][DATA_RATIO_W-1:0] in_addr_offset;
    for (genvar i = 0; i < NUM_REQS; i++) begin : g_in_addr_offset
        assign in_addr_offset[i] = in_req_addr[i][DATA_RATIO_W-1:0];
    end

    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_seed_gen
        wire [DATA_RATIO-1:0] batch_mask;
        wire [DATA_RATIO_W-1:0] batch_idx;

        assign batch_mask = in_req_mask[i * DATA_RATIO +: DATA_RATIO] & req_rem_mask_r[i * DATA_RATIO +: DATA_RATIO];

        VX_priority_encoder #(
            .N (DATA_RATIO)
        ) priority_encoder (
            .data_in    (batch_mask),
            .index_out  (batch_idx),
            `UNUSED_PIN (onehot_out),
            .valid_out  (batch_valid_n[i])
        );

        wire [DATA_RATIO-1:0][OUT_ADDR_WIDTH-1:0] addr_base;
        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_addr_base
            assign addr_base[j] = in_req_addr[DATA_RATIO * i + j][ADDR_WIDTH-1:DATA_RATIO_W];
        end

        wire [DATA_RATIO-1:0][`UP(FLAGS_WIDTH)-1:0] req_flags;
        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_req_flags
            assign req_flags[j] = in_req_flags[DATA_RATIO * i + j];
        end

        assign seed_addr_n[i]  = addr_base[batch_idx];
        assign seed_flags_n[i] = req_flags[batch_idx];

        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_addr_matches_n
            assign addr_matches_n[i * DATA_RATIO + j] = (addr_base[j] == seed_addr_n[i]);
        end
    end

    wire [NUM_REQS-1:0] current_pmask = in_req_mask & addr_matches_r;

    wire [OUT_REQS-1:0][DATA_RATIO-1:0][DATA_IN_SIZE-1:0] req_byteen_merged;
    wire [OUT_REQS-1:0][DATA_RATIO-1:0][DATA_IN_WIDTH-1:0] req_data_merged;

    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_data_merged
        reg [DATA_RATIO-1:0][DATA_IN_SIZE-1:0] byteen_merged;
        reg [DATA_RATIO-1:0][DATA_IN_WIDTH-1:0] data_merged;
        always @(*) begin
            byteen_merged = '0;
            data_merged = 'x;
            for (integer j = 0; j < DATA_RATIO; ++j) begin
                for (integer k = 0; k < DATA_IN_SIZE; ++k) begin
                    // perform byte-level merge since each thread may have different bytes enabled
                    if (current_pmask[i * DATA_RATIO + j] && in_req_byteen[DATA_RATIO * i + j][k]) begin
                        byteen_merged[in_addr_offset[DATA_RATIO * i + j]][k] = 1'b1;
                        data_merged[in_addr_offset[DATA_RATIO * i + j]][k * 8 +: 8] = in_req_data[DATA_RATIO * i + j][k * 8 +: 8];
                    end
                end
            end
        end
        assign req_byteen_merged[i] = byteen_merged;
        assign req_data_merged[i]   = data_merged;
    end

    wire is_last_batch = ~(| (in_req_mask & ~addr_matches_r & req_rem_mask_r));

    wire out_req_fire = out_req_valid && out_req_ready;

    always @(*) begin
        state_n          = state_r;
        out_req_valid_n  = out_req_valid_r;
        out_req_mask_n   = out_req_mask_r;
        out_req_rw_n     = out_req_rw_r;
        out_req_addr_n   = out_req_addr_r;
        out_req_flags_n  = out_req_flags_r;
        out_req_byteen_n = out_req_byteen_r;
        out_req_data_n   = out_req_data_r;
        out_req_tag_n    = out_req_tag_r;
        req_rem_mask_n   = req_rem_mask_r;
        in_req_ready_n   = 0;

        case (state_r)
        STATE_WAIT: begin
            // wait for pending outgoing request to submit
            if (out_req_fire) begin
                out_req_valid_n = 0;
            end
            if (in_req_valid && ~out_req_valid_n && ~ibuf_full) begin
                state_n = STATE_SEND;
            end
        end
        default/*STATE_SEND*/: begin
            state_n         = STATE_WAIT;
            out_req_valid_n = 1;
            out_req_mask_n  = batch_valid_r;
            out_req_rw_n    = in_req_rw;
            out_req_addr_n  = seed_addr_r;
            out_req_flags_n = seed_flags_r;
            out_req_byteen_n= req_byteen_merged;
            out_req_data_n  = req_data_merged;
            out_req_tag_n   = {in_req_tag[TAG_WIDTH-1 -: UUID_WIDTH], ibuf_waddr};
            req_rem_mask_n  = is_last_batch ? '1 : (req_rem_mask_r & ~current_pmask);
            in_req_ready_n  = is_last_batch;
        end
        endcase
    end

    VX_pipe_register #(
        .DATAW  (1 + NUM_REQS + 1 + 1 + NUM_REQS + OUT_REQS * (1 + 1 + OUT_ADDR_WIDTH + `UP(FLAGS_WIDTH) + OUT_ADDR_WIDTH + `UP(FLAGS_WIDTH) + DATA_OUT_SIZE + DATA_OUT_WIDTH) + OUT_TAG_WIDTH),
        .RESETW (1 + NUM_REQS + 1),
        .INIT_VALUE ({1'b0, {NUM_REQS{1'b1}}, 1'b0})
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({state_n, req_rem_mask_n, out_req_valid_n, out_req_rw_n, addr_matches_n, batch_valid_n, out_req_mask_n, seed_addr_n, seed_flags_n, out_req_addr_n, out_req_flags_n, out_req_byteen_n, out_req_data_n, out_req_tag_n}),
        .data_out ({state_r, req_rem_mask_r, out_req_valid_r, out_req_rw_r, addr_matches_r, batch_valid_r, out_req_mask_r, seed_addr_r, seed_flags_r, out_req_addr_r, out_req_flags_r, out_req_byteen_r, out_req_data_r, out_req_tag_r})
    );

    wire out_rsp_fire = out_rsp_valid && out_rsp_ready;

    wire out_rsp_eop;

    wire req_sent = (state_r == STATE_SEND);

    assign ibuf_push  = req_sent && ~in_req_rw;
    assign ibuf_pop   = out_rsp_fire && out_rsp_eop;
    assign ibuf_raddr = out_rsp_tag[QUEUE_ADDRW-1:0];

    wire [TAG_ID_WIDTH-1:0] ibuf_din_tag = in_req_tag[TAG_ID_WIDTH-1:0];
    wire [NUM_REQS-1:0][DATA_RATIO_W-1:0] ibuf_din_offset = in_addr_offset;
    wire [NUM_REQS-1:0] ibuf_din_pmask = current_pmask;

    assign ibuf_din = {ibuf_din_tag, ibuf_din_pmask, ibuf_din_offset};

    VX_index_buffer #(
        .DATAW (IBUF_DATA_WIDTH),
        .SIZE  (QUEUE_SIZE)
    ) req_ibuf (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (ibuf_push),
        .write_addr   (ibuf_waddr),
        .write_data   (ibuf_din),
        .read_data    (ibuf_dout),
        .read_addr    (ibuf_raddr),
        .release_en   (ibuf_pop),
        .full         (ibuf_full),
        .empty        (ibuf_empty)
    );
    `UNUSED_VAR (ibuf_empty)

    assign out_req_valid  = out_req_valid_r;
    assign out_req_rw     = out_req_rw_r;
    assign out_req_mask   = out_req_mask_r;
    assign out_req_byteen = out_req_byteen_r;
    assign out_req_addr   = out_req_addr_r;
    if (FLAGS_WIDTH != 0) begin : g_out_req_flags
        assign out_req_flags = out_req_flags_r;
    end else begin : g_out_req_flags_0
        `UNUSED_VAR (out_req_flags_r)
        assign out_req_flags = '0;
    end
    assign out_req_data   = out_req_data_r;
    assign out_req_tag    = out_req_tag_r;

    assign in_req_ready = in_req_ready_n;

    // unmerge responses

    reg [QUEUE_SIZE-1:0][OUT_REQS-1:0] rsp_rem_mask;
    wire [OUT_REQS-1:0] rsp_rem_mask_n = rsp_rem_mask[ibuf_raddr] & ~out_rsp_mask;
    assign out_rsp_eop = ~(| rsp_rem_mask_n);

    always @(posedge clk) begin
        if (ibuf_push) begin
            rsp_rem_mask[ibuf_waddr] <= batch_valid_r;
        end
        if (out_rsp_fire) begin
            rsp_rem_mask[ibuf_raddr] <= rsp_rem_mask_n;
        end
    end

    wire [NUM_REQS-1:0][DATA_RATIO_W-1:0] ibuf_dout_offset;
    wire [NUM_REQS-1:0] ibuf_dout_pmask;
    wire [TAG_ID_WIDTH-1:0] ibuf_dout_tag;

    assign {ibuf_dout_tag, ibuf_dout_pmask, ibuf_dout_offset} = ibuf_dout;

    wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_rsp_data_n;
    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_in_rsp_data_n
        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_j
            assign in_rsp_data_n[i * DATA_RATIO + j] = out_rsp_data[i][ibuf_dout_offset[i * DATA_RATIO + j] * DATA_IN_WIDTH +: DATA_IN_WIDTH];
        end
    end

    wire [NUM_REQS-1:0] in_rsp_mask_n;
    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_in_rsp_mask_n
        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_j
            assign in_rsp_mask_n[i * DATA_RATIO + j] = out_rsp_mask[i] && ibuf_dout_pmask[i * DATA_RATIO + j];
        end
    end

    assign in_rsp_valid  = out_rsp_valid;
    assign in_rsp_mask   = in_rsp_mask_n;
    assign in_rsp_data   = in_rsp_data_n;
    assign in_rsp_tag    = {out_rsp_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH], ibuf_dout_tag};
    assign out_rsp_ready = in_rsp_ready;

`ifdef DBG_TRACE_MEM
    wire [`UP(UUID_WIDTH)-1:0] out_req_uuid;
    wire [`UP(UUID_WIDTH)-1:0] out_rsp_uuid;

    if (UUID_WIDTH != 0) begin : g_out_req_uuid
        assign out_req_uuid = out_req_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_out_req_uuid_0
        assign out_req_uuid = '0;
    end

    if (UUID_WIDTH != 0) begin : g_out_rsp_uuid
        assign out_rsp_uuid = out_rsp_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_out_rsp_uuid_0
        assign out_rsp_uuid = '0;
    end

    reg [NUM_REQS-1:0][DATA_RATIO_W-1:0] out_req_offset;
    reg [NUM_REQS-1:0] out_req_pmask;

    always @(posedge clk) begin
        if (req_sent) begin
            out_req_offset <= ibuf_din_offset;
            out_req_pmask <= ibuf_din_pmask;
        end
    end

    always @(posedge clk) begin
        if (out_req_fire) begin
            if (out_req_rw) begin
                `TRACE(2, ("%t: %s out-req-wr: valid=%b, addr=", $time, INSTANCE_ID, out_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", out_req_addr, OUT_REQS)
                `TRACE(2, (", flags="))
                `TRACE_ARRAY1D(2, "%b", out_req_flags, OUT_REQS)
                `TRACE(2, (", byteen="))
                `TRACE_ARRAY1D(2, "0x%h", out_req_byteen, OUT_REQS)
                `TRACE(2, (", data="))
                `TRACE_ARRAY1D(2, "0x%0h", out_req_data, OUT_REQS)
            end else begin
                `TRACE(2,  ("%d: %s out-req-rd: valid=%b, addr=", $time, INSTANCE_ID, out_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", out_req_addr, OUT_REQS)
                `TRACE(2, (", flags="))
                `TRACE_ARRAY1D(2, "%b", out_req_flags, OUT_REQS)
            end
            `TRACE(2, (", offset="))
            `TRACE_ARRAY1D(2, "%0d", out_req_offset, NUM_REQS)
            `TRACE(2, (", pmask=%b, coalesced=%0d, tag=0x%0h (#%0d)\n", out_req_pmask, $countones(out_req_pmask), out_req_tag, out_req_uuid))
        end
        if (out_rsp_fire) begin
            `TRACE(2, ("%t: %s out-rsp: valid=%b, data=", $time, INSTANCE_ID, out_rsp_mask))
            `TRACE_ARRAY1D(2, "0x%0h", out_rsp_data, OUT_REQS)
            `TRACE(2, (", offset="))
            `TRACE_ARRAY1D(2, "%0d", ibuf_dout_offset, NUM_REQS)
            `TRACE(2, (", eop=%b, pmask=%b, tag=0x%0h (#%0d)\n", out_rsp_eop, ibuf_dout_pmask, out_rsp_tag, out_rsp_uuid))
        end
    end
`endif

endmodule
`TRACING_ON
