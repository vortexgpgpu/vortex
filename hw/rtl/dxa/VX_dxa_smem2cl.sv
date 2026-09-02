// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef VX_CFG_EXT_DXA_S2G_ENABLE

module VX_dxa_smem2cl import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_LINE_SIZE  = `VX_CFG_L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE),
    parameter SMEM_WORD_SIZE  = DXA_LMEM_WORD_SIZE,
    parameter SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W
) (
    input wire clk,
    input wire reset,

    input wire                       in_valid,
    output wire                      in_ready,
    input wire [GMEM_ADDR_WIDTH-1:0] in_cl_addr,
    input wire [DXA_SMEM_ADDR_W-1:0] in_smem_byte_addr,
    input wire [GMEM_OFF_BITS-1:0]   in_byte_offset,
    input wire [GMEM_OFF_BITS:0]     in_valid_length,
    input wire                       in_oob,
    input wire                       in_last,

    input wire [NC_WIDTH-1:0]        active_core_id,
    input wire [UUID_WIDTH-1:0]      active_uuid,

    output wire                       out_valid,
    input wire                        out_ready,
    output wire [GMEM_ADDR_WIDTH-1:0] out_cl_addr,
    output wire [GMEM_LINE_SIZE*8-1:0] out_data,
    output wire [GMEM_LINE_SIZE-1:0]  out_byteen,
    output wire                       out_last,

    VX_mem_bus_if.master smem_bus_if,

    output wire        read_outstanding,
    output wire [31:0] read_count,
    output wire        source_done,
    output wire        source_done_has_store
);
    localparam GMEM_OFF_BITS = `CLOG2(GMEM_LINE_SIZE);
    localparam SMEM_OFF_BITS = `CLOG2(SMEM_WORD_SIZE);
    localparam SMEM_DATAW    = SMEM_WORD_SIZE * 8;
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;
    localparam MAX_WORDS = (GMEM_LINE_SIZE + 2 * SMEM_WORD_SIZE - 2) / SMEM_WORD_SIZE;
    localparam WORD_INDEX_W = `CLOG2(MAX_WORDS);
    localparam WORD_COUNT_W = `CLOG2(MAX_WORDS + 1);
    localparam SOURCE_DATAW = MAX_WORDS * SMEM_DATAW;
    localparam SOURCE_BYTE_W = `CLOG2(MAX_WORDS * SMEM_WORD_SIZE);

    localparam STATE_IDLE  = 3'd0;
    localparam STATE_REQ   = 3'd1;
    localparam STATE_WAIT  = 3'd2;
    localparam STATE_EMIT  = 3'd3;

    reg [2:0] state_r;
    reg [GMEM_ADDR_WIDTH-1:0] cl_addr_r;
    reg [SMEM_ADDR_WIDTH-1:0] smem_word_addr_r;
    reg [SMEM_OFF_BITS-1:0] smem_byte_offset_r;
    reg [GMEM_OFF_BITS-1:0] byte_offset_r;
    reg [GMEM_OFF_BITS:0] valid_length_r;
    reg last_r;
    reg [WORD_INDEX_W-1:0] word_index_r;
    reg [WORD_COUNT_W-1:0] word_count_r;
    reg [NC_WIDTH-1:0] core_id_r;
    reg [UUID_WIDTH-1:0] uuid_r;
    reg [SOURCE_DATAW-1:0] source_words_r;
    reg [31:0] read_count_r;
    reg source_done_r;
    reg source_done_has_store_r;

    `STATIC_ASSERT(`IS_POW2(GMEM_LINE_SIZE), ("GMEM line size must be a power of two"))
    `STATIC_ASSERT(`IS_POW2(SMEM_WORD_SIZE), ("LMEM DMA word size must be a power of two"))
    `STATIC_ASSERT(MAX_WORDS >= 1, ("S2G packer must buffer at least one LMEM word"))
    `STATIC_ASSERT(SMEM_TAG_VALUE_W == (`CLOG2(`VX_CFG_NUM_CORES) + 1),
        ("S2G LMEM tag must contain the engine bit and core route"))
    `STATIC_ASSERT($bits(smem_bus_if.req_data.tag) == DXA_LMEM_TAG_W,
        ("S2G LMEM request tag width mismatch"))
    `STATIC_ASSERT($bits(smem_bus_if.rsp_data.tag) == DXA_LMEM_TAG_W,
        ("S2G LMEM response tag width mismatch"))

    wire token_fire = in_valid && in_ready;
    wire [GMEM_OFF_BITS:0] input_source_end =
        (GMEM_OFF_BITS+1)'(in_smem_byte_addr[SMEM_OFF_BITS-1:0])
        + in_valid_length;
    wire [GMEM_OFF_BITS:0] input_word_count =
        (input_source_end + (GMEM_OFF_BITS+1)'(SMEM_WORD_SIZE - 1)) >> SMEM_OFF_BITS;

    `RUNTIME_ASSERT(!token_fire || in_oob || (in_valid_length != 0),
        ("S2G source token has zero valid length"))
    `RUNTIME_ASSERT(!token_fire || in_oob
        || ({1'b0, in_byte_offset} + in_valid_length <= (GMEM_OFF_BITS+1)'(GMEM_LINE_SIZE)),
        ("S2G destination byte range exceeds the global line"))
    `RUNTIME_ASSERT(!token_fire || in_oob || (input_word_count <= (GMEM_OFF_BITS+1)'(MAX_WORDS)),
        ("S2G source token exceeds the packer capacity"))

    assign in_ready = (state_r == STATE_IDLE);

    wire request_valid = (state_r == STATE_REQ);
    wire [SMEM_TAG_VALUE_W-1:0] request_tag_value =
        (SMEM_TAG_VALUE_W'(core_id_r) << 1) | SMEM_TAG_VALUE_W'(1);

    assign smem_bus_if.req_valid          = request_valid;
    assign smem_bus_if.req_data.rw        = 1'b0;
    assign smem_bus_if.req_data.addr      = smem_word_addr_r + SMEM_ADDR_WIDTH'(word_index_r);
    assign smem_bus_if.req_data.data      = '0;
    assign smem_bus_if.req_data.byteen    = '0;
    assign smem_bus_if.req_data.attr      = '0;
    assign smem_bus_if.req_data.tag.uuid  = uuid_r;
    assign smem_bus_if.req_data.tag.value = request_tag_value;

    wire waiting_response = (state_r == STATE_WAIT);
    assign smem_bus_if.rsp_ready = waiting_response;

    wire request_fire = smem_bus_if.req_valid && smem_bus_if.req_ready;
    wire response_fire = smem_bus_if.rsp_valid && smem_bus_if.rsp_ready;
    wire [SMEM_TAG_VALUE_W-1:0] expected_tag_value =
        (SMEM_TAG_VALUE_W'(core_id_r) << 1) | SMEM_TAG_VALUE_W'(1);

    `RUNTIME_ASSERT(!smem_bus_if.rsp_valid || waiting_response,
        ("unexpected S2G LMEM response"))
    `RUNTIME_ASSERT(!response_fire || (smem_bus_if.rsp_data.tag.uuid == uuid_r),
        ("S2G LMEM response UUID mismatch"))
    `RUNTIME_ASSERT(!response_fire || (smem_bus_if.rsp_data.tag.value == expected_tag_value),
        ("S2G LMEM response route tag mismatch"))

    assign out_valid   = (state_r == STATE_EMIT);
    assign out_cl_addr = cl_addr_r;
    assign out_last    = last_r;

    wire [GMEM_LINE_SIZE:0] byteen_seed =
        ((GMEM_LINE_SIZE+1)'(1) << valid_length_r) - (GMEM_LINE_SIZE+1)'(1);
    assign out_byteen = GMEM_LINE_SIZE'(byteen_seed) << byte_offset_r;
    `UNUSED_VAR (byteen_seed[GMEM_LINE_SIZE])

    for (genvar i = 0; i < GMEM_LINE_SIZE; ++i) begin : g_pack_byte
        wire [SOURCE_BYTE_W-1:0] source_byte =
            SOURCE_BYTE_W'(smem_byte_offset_r)
            + SOURCE_BYTE_W'((GMEM_OFF_BITS+1)'(i) - {1'b0, byte_offset_r});
        assign out_data[i*8 +: 8] = out_byteen[i]
            ? source_words_r[source_byte*8 +: 8]
            : 8'h00;
    end

    assign read_outstanding = waiting_response;
    assign read_count = read_count_r;
    assign source_done = source_done_r;
    assign source_done_has_store = source_done_has_store_r;

    always @(posedge clk) begin
        if (reset) begin
            state_r <= STATE_IDLE;
            cl_addr_r <= '0;
            smem_word_addr_r <= '0;
            smem_byte_offset_r <= '0;
            byte_offset_r <= '0;
            valid_length_r <= '0;
            last_r <= 1'b0;
            word_index_r <= '0;
            word_count_r <= '0;
            core_id_r <= '0;
            uuid_r <= '0;
            source_words_r <= '0;
            read_count_r <= '0;
            source_done_r <= 1'b0;
            source_done_has_store_r <= 1'b0;
        end else begin
            source_done_r <= 1'b0;
            source_done_has_store_r <= 1'b0;

            case (state_r)
                STATE_IDLE: begin
                    if (token_fire) begin
                        if (in_oob) begin
                            if (in_last) begin
                                source_done_r <= 1'b1;
                                source_done_has_store_r <= 1'b0;
                            end
                        end else begin
                            cl_addr_r <= in_cl_addr;
                            smem_word_addr_r <= SMEM_ADDR_WIDTH'(in_smem_byte_addr >> SMEM_OFF_BITS);
                            smem_byte_offset_r <= in_smem_byte_addr[SMEM_OFF_BITS-1:0];
                            byte_offset_r <= in_byte_offset;
                            valid_length_r <= in_valid_length;
                            last_r <= in_last;
                            word_index_r <= '0;
                            word_count_r <= WORD_COUNT_W'(input_word_count);
                            core_id_r <= active_core_id;
                            uuid_r <= active_uuid;
                            source_words_r <= '0;
                            state_r <= STATE_REQ;
                        end
                    end
                end
                STATE_REQ: begin
                    if (request_fire) begin
                        read_count_r <= read_count_r + 32'd1;
                        state_r <= STATE_WAIT;
                    end
                end
                STATE_WAIT: begin
                    if (response_fire) begin
                        source_words_r[word_index_r*SMEM_DATAW +: SMEM_DATAW]
                            <= smem_bus_if.rsp_data.data;
                        if (WORD_COUNT_W'(word_index_r) + WORD_COUNT_W'(1) < word_count_r) begin
                            word_index_r <= word_index_r + WORD_INDEX_W'(1);
                            state_r <= STATE_REQ;
                        end else begin
                            if (last_r) begin
                                source_done_r <= 1'b1;
                                source_done_has_store_r <= 1'b1;
                            end
                            state_r <= STATE_EMIT;
                        end
                    end
                end
                STATE_EMIT: begin
                    if (out_ready) begin
                        state_r <= STATE_IDLE;
                    end
                end
                default: begin
                    state_r <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule

`endif
