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

// One pass engine of the warp-wide local memory. It takes a whole warp request
// and gives it to the banks in passes: in a pass every bank goes to the lowest
// pending lane that maps to it, and a plain read also serves every pending lane
// reading that lane's word. Lanes left over replay in the next passes, and the
// request retires before the next one starts.
//
//   S0  the engine's input queue of two requests; the head's lanes' words are
//       compared pairwise
//   S1  the request, its comparison matrices, its pending lanes and the next
//       pass's bank winners; a pass moves the winners' rows and payloads into
//       the bank registers
//   then the banks access their SRAM, two cycles after the pass each served
//   lane takes its bank's word into the response register, and from there the
//   response enters the response queue, or passes it when it is empty, on its
//   way to its block's response register.
//
// The pair matrices are computed once per request, and each pass's bank
// winners and whether it is the last are registered ahead of it, so nothing
// combinational sits in front of the bank registers but the winners' row and
// payload mux. The first pass's winners come from S0; a replay finds the next
// winners in the cycle after a pass, so a conflicting request replays at one
// pass per two cycles. The last pass frees S1 in its own cycle.
module VX_lmem_engine import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES       = 4,
    parameter NUM_BANKS       = 4,
    parameter WORD_SIZE       = 4,
    // word address width
    parameter ADDR_WIDTH      = 12,
    parameter TAG_WIDTH       = 8,
    // per-lane write payload carried to the banks (data, byte enables, atomic sideband)
    parameter PAYLOAD_WIDTH   = 1,
    // request attributes common to every lane
    parameter ATTR_WIDTH      = 1,
    parameter NUM_BLOCKS      = 1,
    parameter RSP_QUEUE_SIZE  = 8,

    parameter WORD_WIDTH      = WORD_SIZE * 8,
    parameter BLOCK_WIDTH     = `UP(`CLOG2(NUM_BLOCKS)),
    parameter DONE_WIDTH      = `CLOG2(NUM_BLOCKS + 1),
    parameter BANK_ADDR_WIDTH = ADDR_WIDTH - `CLOG2(NUM_BANKS)
) (
    input wire                                       clk,
    input wire                                       reset,

    // request
    input wire                                       req_valid,
    input wire [BLOCK_WIDTH-1:0]                     req_block,
    // reads and atomics respond, stores do not
    input wire                                       req_rw,
    // a plain read: its lanes of one word share a bank access
    input wire                                       req_read,
    input wire                                       req_amo,
    input wire [ATTR_WIDTH-1:0]                      req_attr,
    input wire [NUM_LANES-1:0]                       req_mask,
    input wire [NUM_LANES-1:0][ADDR_WIDTH-1:0]       req_addr,
    input wire [NUM_LANES-1:0][PAYLOAD_WIDTH-1:0]    req_payload,
    input wire [TAG_WIDTH-1:0]                       req_tag,
    output wire                                      req_ready,

    // the engine holds a request; the one it holds is a store or an atomic
    output wire                                      busy,
    output wire                                      holds_write,

    // the banks are free for a pass this cycle
    input wire                                       pass_en,
    // an atomic pass went out
    output wire                                      pass_amo,

    // bank requests, registered
    output wire [NUM_BANKS-1:0]                      bank_valid,
    output wire                                      bank_rw,
    output wire                                      bank_amo,
    output wire [ATTR_WIDTH-1:0]                     bank_attr,
    output wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_addr,
    output wire [NUM_BANKS-1:0][PAYLOAD_WIDTH-1:0]   bank_payload,

    // the banks' read data, two cycles after the pass
    input wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]       bank_rdata,

    // the response queue's head, taken into a block's response register;
    // rsp_done counts this engine's responses the blocks took this cycle
    output wire                                      rsp_valid,
    output wire [BLOCK_WIDTH-1:0]                    rsp_block,
    output wire [NUM_LANES-1:0]                      rsp_mask,
    output wire [NUM_LANES-1:0][WORD_WIDTH-1:0]      rsp_data,
    output wire [TAG_WIDTH-1:0]                      rsp_tag,
    input wire                                       rsp_take,
    input wire [DONE_WIDTH-1:0]                      rsp_done,

    // perf: lanes served by this cycle's pass, whether they store, and
    // whether lanes were left over
    output wire [NUM_LANES-1:0]                      perf_served,
    output wire                                      perf_store,
    output wire                                      perf_replay
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BANK_SEL_BITS  = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam CREDIT_WIDTH   = `CLOG2(RSP_QUEUE_SIZE + 1);
    localparam RSP_DATAW      = BLOCK_WIDTH + NUM_LANES + NUM_LANES * WORD_WIDTH + TAG_WIDTH;

    `STATIC_ASSERT(ADDR_WIDTH > BANK_SEL_BITS, ("invalid parameter"))

    // S0: the engine's input queue. Holding two requests, it takes one in any
    // cycle that starts with at most one, so its ready is a register and the
    // head stays put while S1 is busy.

    localparam S0_DATAW = BLOCK_WIDTH + 3 + ATTR_WIDTH + NUM_LANES
                        + NUM_LANES * (ADDR_WIDTH + PAYLOAD_WIDTH) + TAG_WIDTH;

    wire                                    s0_valid;
    wire [BLOCK_WIDTH-1:0]                  s0_block;
    wire                                    s0_rw;
    wire                                    s0_read;
    wire                                    s0_amo;
    wire [ATTR_WIDTH-1:0]                   s0_attr;
    wire [NUM_LANES-1:0]                    s0_mask;
    wire [NUM_LANES-1:0][ADDR_WIDTH-1:0]    s0_addr;
    wire [NUM_LANES-1:0][PAYLOAD_WIDTH-1:0] s0_payload;
    wire [TAG_WIDTH-1:0]                    s0_tag;
    wire                                    s1_free;

    VX_elastic_buffer #(
        .DATAW   (S0_DATAW),
        .SIZE    (2),
        .OUT_REG (1)
    ) s0_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid),
        .ready_in  (req_ready),
        .data_in   ({req_block, req_rw, req_read, req_amo, req_attr, req_mask, req_addr, req_payload, req_tag}),
        .data_out  ({s0_block, s0_rw, s0_read, s0_amo, s0_attr, s0_mask, s0_addr, s0_payload, s0_tag}),
        .valid_out (s0_valid),
        .ready_out (s1_free)
    );

    // pairwise comparison: same_bank[i][j] and same_word[i][j] for lower lanes j < i

    wire [NUM_LANES-1:0][NUM_LANES-1:0] s0_same_bank;
    wire [NUM_LANES-1:0][NUM_LANES-1:0] s0_same_word;
    wire [NUM_LANES-1:0][NUM_LANES-1:0] s0_conflict;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cmp_i
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_cmp_j
            if (j < i) begin : g_lower
                if (BANK_SEL_BITS != 0) begin : g_banks
                    assign s0_same_bank[i][j] = (s0_addr[i][BANK_SEL_BITS-1:0] == s0_addr[j][BANK_SEL_BITS-1:0]);
                end else begin : g_one_bank
                    assign s0_same_bank[i][j] = 1'b1;
                end
                assign s0_same_word[i][j] = (s0_addr[i] == s0_addr[j]);
                // two lanes that need different passes
                assign s0_conflict[i][j] = s0_mask[i] && s0_mask[j] && s0_same_bank[i][j]
                                        && ~(s0_read && s0_same_word[i][j]);
            end else begin : g_none
                assign s0_same_bank[i][j] = 1'b0;
                assign s0_same_word[i][j] = 1'b0;
                assign s0_conflict[i][j]  = 1'b0;
            end
        end
    end
    wire s0_one_pass = ~(| s0_conflict);

    // the first pass's bank winners: each bank's lowest lane
    wire [NUM_LANES-1:0] s0_first;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_s0_first
        assign s0_first[i] = s0_mask[i] && ~(| (s0_mask & s0_same_bank[i]));
    end

    // S1

    reg                          s1_valid;
    // the bank winners of the next pass are in s1_winners
    reg                          s1_ready;
    // the next pass serves every pending lane
    reg                                    s1_last;
    reg [NUM_LANES-1:0]                    s1_winners;
    reg [BLOCK_WIDTH-1:0]                  s1_block;
    reg                                    s1_rw;
    reg                                    s1_read;
    reg                                    s1_amo;
    reg [ATTR_WIDTH-1:0]                   s1_attr;
    reg [NUM_LANES-1:0]                    s1_pending;
    reg [NUM_LANES-1:0][ADDR_WIDTH-1:0]    s1_addr;
    reg [NUM_LANES-1:0][PAYLOAD_WIDTH-1:0] s1_payload;
    reg [TAG_WIDTH-1:0]                    s1_tag;
    reg [NUM_LANES-1:0][NUM_LANES-1:0]     s1_same_bank;
    reg [NUM_LANES-1:0][NUM_LANES-1:0]     s1_same_word;

    // A pass serves the registered bank winners and the pending read lanes of
    // their words. After a pass that leaves lanes pending, the next cycle finds
    // the next winners among them and whether the pass after serves them all:
    // a request replays at one pass per two cycles, and its last pass frees S1.

    wire [NUM_LANES-1:0] served;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_served
        assign served[i] = s1_winners[i] || (s1_read && s1_pending[i] && (| (s1_winners & s1_same_word[i])));
    end

    wire [NUM_LANES-1:0]                winners_n;
    wire [NUM_LANES-1:0][NUM_LANES-1:0] pending_conflict;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_winners_n
        assign winners_n[i] = s1_pending[i] && ~(| (s1_pending & s1_same_bank[i]));
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_conflict
            assign pending_conflict[i][j] = s1_pending[i] && s1_pending[j] && s1_same_bank[i][j]
                                         && ~(s1_read && s1_same_word[i][j]);
        end
    end
    wire last_n = ~(| pending_conflict);

    reg [CREDIT_WIDTH-1:0] credits;
    wire has_rsp    = ~s1_rw;
    wire pass_fire  = s1_valid && s1_ready && pass_en && (~has_rsp || (credits != '0));
    wire s1_release = pass_fire && s1_last;
    assign s1_free  = ~s1_valid || s1_release;
    wire   s1_load  = s0_valid && s1_free;

    always @(posedge clk) begin
        if (reset) begin
            s1_valid <= 1'b0;
        end else if (s1_load) begin
            s1_valid <= 1'b1;
        end else if (s1_release) begin
            s1_valid <= 1'b0;
        end
        if (s1_load) begin
            s1_ready     <= 1'b1;
            s1_last      <= s0_one_pass;
            s1_winners   <= s0_first;
            s1_pending   <= s0_mask;
            s1_block     <= s0_block;
            s1_rw        <= s0_rw;
            s1_read      <= s0_read;
            s1_amo       <= s0_amo;
            s1_attr      <= s0_attr;
            s1_addr      <= s0_addr;
            s1_payload   <= s0_payload;
            s1_tag       <= s0_tag;
            s1_same_bank <= s0_same_bank;
            s1_same_word <= s0_same_word;
        end else if (pass_fire) begin
            s1_ready     <= 1'b0;
            s1_pending   <= s1_pending & ~served;
        end else if (~s1_ready) begin
            s1_ready     <= 1'b1;
            s1_last      <= last_n;
            s1_winners   <= winners_n;
        end
    end

    // stores and atomics held between S0's two entries and S1
    reg [1:0] num_writes;
    wire write_in  = req_valid && req_ready && ~req_read;
    wire write_out = s1_release && ~s1_read;
    always @(posedge clk) begin
        if (reset) begin
            num_writes <= '0;
        end else begin
            num_writes <= num_writes + 2'(write_in) - 2'(write_out);
        end
    end

    assign busy = s0_valid || s1_valid;
    assign holds_write = (num_writes != '0);
    assign pass_amo = pass_fire && s1_amo;

    // bank registers: each bank takes its first lane's row and payload

    wire [NUM_LANES-1:0][BANK_SEL_WIDTH-1:0]                s1_bank;
    wire [NUM_LANES-1:0][BANK_ADDR_WIDTH+PAYLOAD_WIDTH-1:0] s1_bank_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_addr
        if (BANK_SEL_BITS != 0) begin : g_banks
            assign s1_bank[i] = s1_addr[i][BANK_SEL_BITS-1:0];
        end else begin : g_one_bank
            assign s1_bank[i] = '0;
        end
        assign s1_bank_data[i] = {s1_addr[i][BANK_SEL_BITS +: BANK_ADDR_WIDTH], s1_payload[i]};
    end

    reg [NUM_BANKS-1:0]                      bank_valid_r;
    reg                                      bank_rw_r;
    reg                                      bank_amo_r;
    reg [ATTR_WIDTH-1:0]                     bank_attr_r;
    reg [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_addr_r;
    reg [NUM_BANKS-1:0][PAYLOAD_WIDTH-1:0]   bank_payload_r;

    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_banks
        wire [NUM_LANES-1:0] sel;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sel
            assign sel[i] = s1_winners[i] && (s1_bank[i] == BANK_SEL_WIDTH'(b));
        end

        wire [BANK_ADDR_WIDTH-1:0] sel_addr;
        wire [PAYLOAD_WIDTH-1:0]   sel_payload;

        VX_onehot_mux #(
            .DATAW (BANK_ADDR_WIDTH + PAYLOAD_WIDTH),
            .N     (NUM_LANES)
        ) bank_mux (
            .data_in  (s1_bank_data),
            .sel_in   (sel),
            .data_out ({sel_addr, sel_payload})
        );

        always @(posedge clk) begin
            if (reset) begin
                bank_valid_r[b] <= 1'b0;
            end else begin
                bank_valid_r[b] <= pass_fire && (| sel);
            end
        end

        // The bank's address and payload are read only with its valid, so on
        // ASIC they load only when the bank is used.
        wire bank_load = pass_fire && (| sel);
    `ifdef ASIC
        wire bank_clk;
        VX_clockgate bank_cg (.clk_in (clk), .en (bank_load), .clk_out (bank_clk));
        always @(posedge bank_clk) begin
            if (bank_load) begin
                bank_addr_r[b]    <= sel_addr;
                bank_payload_r[b] <= sel_payload;
            end
        end
    `else
        `UNUSED_VAR (bank_load)
        always @(posedge clk) begin
            bank_addr_r[b]    <= sel_addr;
            bank_payload_r[b] <= sel_payload;
        end
    `endif
    end

`ifdef ASIC
    wire pass_clk;
    VX_clockgate pass_cg (.clk_in (clk), .en (pass_fire), .clk_out (pass_clk));
    always @(posedge pass_clk) begin
        if (pass_fire) begin
            bank_rw_r   <= s1_rw;
            bank_amo_r  <= s1_amo;
            bank_attr_r <= s1_attr;
        end
    end
`else
    always @(posedge clk) begin
        bank_rw_r   <= s1_rw;
        bank_amo_r  <= s1_amo;
        bank_attr_r <= s1_attr;
    end
`endif

    assign bank_valid   = bank_valid_r;
    assign bank_rw      = bank_rw_r;
    assign bank_amo     = bank_amo_r;
    assign bank_attr    = bank_attr_r;
    assign bank_addr    = bank_addr_r;
    assign bank_payload = bank_payload_r;

    // response: the pass's lanes and banks follow it to the SRAM output

    wire                                     m2_valid;
    wire [BLOCK_WIDTH-1:0]                   m2_block;
    wire [NUM_LANES-1:0]                     m2_mask;
    wire [NUM_LANES-1:0][BANK_SEL_WIDTH-1:0] m2_bank;
    wire [TAG_WIDTH-1:0]                     m2_tag;

    VX_pipe_register #(
        .DATAW  (1 + BLOCK_WIDTH + NUM_LANES + NUM_LANES * BANK_SEL_WIDTH + TAG_WIDTH),
        .RESETW (1),
        .DATA_GATE (1),
        .DEPTH  (2)
    ) rsp_meta_pipe (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({pass_fire && has_rsp, s1_block, served, s1_bank, s1_tag}),
        .data_out ({m2_valid, m2_block, m2_mask, m2_bank, m2_tag})
    );

    wire [NUM_LANES-1:0][WORD_WIDTH-1:0] m2_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        assign m2_data[i] = bank_rdata[m2_bank[i]];
    end

    // The lane mux reads the SRAM outputs; its result is registered before it
    // travels to the response queue.
    wire                                 m3_valid;
    wire [BLOCK_WIDTH-1:0]               m3_block;
    wire [NUM_LANES-1:0]                 m3_mask;
    wire [NUM_LANES-1:0][WORD_WIDTH-1:0] m3_data;
    wire [TAG_WIDTH-1:0]                 m3_tag;

    VX_pipe_register #(
        .DATAW  (RSP_DATAW + 1),
        .RESETW (1),
        .DATA_GATE (1)
    ) rsp_data_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({m2_valid, m2_block, m2_mask, m2_data, m2_tag}),
        .data_out ({m3_valid, m3_block, m3_mask, m3_data, m3_tag})
    );

    // The queue has a slot for every pass that responds and is in flight
    // toward its block, so it never fills. A response that finds the queue
    // empty passes straight on to its block's register.
    wire rsp_empty;
    wire [RSP_DATAW-1:0] rsp_head;
    wire rsp_push = m3_valid && ~(rsp_empty && rsp_take);
    wire rsp_pop  = ~rsp_empty && rsp_take;

    VX_fifo_queue #(
        .DATAW   (RSP_DATAW),
        .DEPTH   (RSP_QUEUE_SIZE),
        .OUT_REG (0),
        .LUTRAM  (1)
    ) rsp_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (rsp_push),
        .pop      (rsp_pop),
        .data_in  ({m3_block, m3_mask, m3_data, m3_tag}),
        .data_out (rsp_head),
        .empty    (rsp_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    assign rsp_valid = ~rsp_empty || m3_valid;
    assign {rsp_block, rsp_mask, rsp_data, rsp_tag} = rsp_empty ? {m3_block, m3_mask, m3_data, m3_tag} : rsp_head;

    // a response's slot is free again once its block has taken it on
    wire credit_take = pass_fire && has_rsp;
    always @(posedge clk) begin
        if (reset) begin
            credits <= CREDIT_WIDTH'(RSP_QUEUE_SIZE);
        end else begin
            credits <= credits - CREDIT_WIDTH'(credit_take) + CREDIT_WIDTH'(rsp_done);
        end
    end

    assign perf_served = pass_fire ? served : '0;
    assign perf_store  = s1_rw;
    assign perf_replay = pass_fire && ~s1_last;

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (pass_fire) begin
            `TRACE(2, ("%t: %s pass: rw=%b, amo=%b, pending=%b, served=%b, tag=0x%0h\n",
                $time, INSTANCE_ID, s1_rw, s1_amo, s1_pending, served, s1_tag))
        end
    end
`endif

endmodule
