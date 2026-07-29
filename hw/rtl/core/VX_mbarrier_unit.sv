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

`ifdef VX_CFG_EXT_MBAR_ENABLE

module VX_mbarrier_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    input wire              req_valid,
    input mbarrier_t        req_data,
    output wire             req_ready,
    output wire             req_phase,

`ifdef VX_CFG_DXA_MBAR_ENABLE
    VX_mbar_completion_if.slave completion_if,
`endif

    output wire             unlock_valid,
    output wire [`VX_CFG_NUM_WARPS-1:0] unlock_mask,
    output wire             busy,

    VX_mem_bus_if.master    lmem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    localparam NUM_ENTRIES = `VX_CFG_MBAR_CACHE_SIZE;
    localparam ENTRY_W = `LOG2UP(NUM_ENTRIES);
    localparam ARRIVAL_W = `CLOG2(`VX_CFG_NUM_WARPS + 1);
    localparam TX_W = `CLOG2(`VX_CFG_MAX_BAR_EVENTS + 1);
    localparam STATE_W = 1 + 2 * ARRIVAL_W + TX_W;
    localparam WORDS_PER_LINE = LMEM_DMA_DATA_SIZE / MBAR_OBJECT_SIZE;
    localparam WORD_INDEX_W = `LOG2UP(WORDS_PER_LINE);
    localparam LINE_BITS = LMEM_DMA_DATA_SIZE * 8;
    localparam LINE_SHIFT = `CLOG2(LMEM_DMA_DATA_SIZE);

    `STATIC_ASSERT(`IS_POW2(NUM_ENTRIES),
        ("MBAR_CACHE_SIZE must be a power of two"))
    `STATIC_ASSERT(MBAR_OBJECT_SIZE == 4,
        ("mbarrier backing object must be one word"))
    `STATIC_ASSERT((LMEM_DMA_DATA_SIZE % MBAR_OBJECT_SIZE) == 0,
        ("LMEM DMA line must contain whole mbarrier objects"))
    `STATIC_ASSERT(STATE_W <= 32,
        ("mbarrier state must fit in one word"))

    reg [NUM_ENTRIES-1:0] valid_r;
    reg [NUM_ENTRIES-1:0][MBAR_ADDR_W-1:0] addr_r;
    reg [NUM_ENTRIES-1:0] phase_r;
    reg [NUM_ENTRIES-1:0][ARRIVAL_W-1:0] pending_arrivals_r;
    reg [NUM_ENTRIES-1:0][ARRIVAL_W-1:0] expected_arrivals_r;
    reg [NUM_ENTRIES-1:0][TX_W-1:0] pending_transactions_r;
    reg [ENTRY_W-1:0] victim_ptr_r;

    reg [`VX_CFG_NUM_WARPS-1:0] waiter_valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0][MBAR_ADDR_W-1:0] waiter_addr_r;

    reg write_valid_r;
    reg [MBAR_ADDR_W-1:0] write_addr_r;
    reg [31:0] write_state_r;

    reg read_valid_r;
    reg read_pending_r;
    reg [MBAR_ADDR_W-1:0] read_addr_r;
    reg [ENTRY_W-1:0] read_entry_r;

    reg unlock_valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0] unlock_mask_r;

`ifdef VX_CFG_DXA_MBAR_ENABLE
    wire current_is_completion = completion_if.valid;
    wire current_valid = completion_if.valid || req_valid;
    wire [MBAR_ADDR_W-1:0] current_addr =
        current_is_completion ? completion_if.addr : req_data.addr;
    wire [NW_WIDTH-1:0] current_wid =
        current_is_completion ? '0 : req_data.wid;
    wire [31:0] current_value =
        current_is_completion ? 32'd1 : req_data.value;
    mbarrier_op_e current_op;
    assign current_op =
        current_is_completion ? MBAR_OP_COMPLETE_TX : req_data.op;
`else
    wire current_is_completion = 1'b0;
    wire current_valid = req_valid;
    wire [MBAR_ADDR_W-1:0] current_addr = req_data.addr;
    wire [NW_WIDTH-1:0] current_wid = req_data.wid;
    wire [31:0] current_value = req_data.value;
    mbarrier_op_e current_op;
    assign current_op = req_data.op;
`endif

    logic [NUM_ENTRIES-1:0] hit_vec;
    logic [ENTRY_W-1:0] hit_entry;
    logic [ENTRY_W-1:0] victim_entry;
    logic victim_found;

    always @(*) begin
        hit_vec = '0;
        hit_entry = '0;
        victim_entry = victim_ptr_r;
        victim_found = 0;
        for (integer i = 0; i < NUM_ENTRIES; ++i) begin
            hit_vec[i] = valid_r[i] && (addr_r[i] == current_addr);
            if (hit_vec[i])
                hit_entry = ENTRY_W'(i);
            if (!victim_found && !valid_r[i]) begin
                victim_found = 1;
                victim_entry = ENTRY_W'(i);
            end
        end
        if (!victim_found)
            victim_found = 1;
    end

    wire hit = |hit_vec;
    wire memory_idle = !write_valid_r && !read_valid_r && !read_pending_r;
    wire operation_ready = memory_idle
                        && current_valid
                        && (hit || (current_op == MBAR_OP_INIT));
    wire operation_fire = operation_ready;
    wire miss_start = memory_idle
                   && current_valid
                   && !hit
                   && (current_op != MBAR_OP_INIT);

    assign req_ready = operation_fire && !current_is_completion;
    assign req_phase = hit ? phase_r[hit_entry] : 1'b0;
`ifdef VX_CFG_DXA_MBAR_ENABLE
    assign completion_if.ready = operation_fire && current_is_completion;
`endif

    wire write_fire = write_valid_r && lmem_if.req_ready;
    wire read_fire = read_valid_r && !write_valid_r && lmem_if.req_ready;

    wire [`VX_CFG_LMEM_LOG_SIZE-1:0] write_byte_addr =
        {write_addr_r, {MBAR_OBJECT_LG2{1'b0}}};
    wire [`VX_CFG_LMEM_LOG_SIZE-1:0] read_byte_addr =
        {read_addr_r, {MBAR_OBJECT_LG2{1'b0}}};
    wire [`VX_CFG_LMEM_LOG_SIZE-1:0] memory_byte_addr =
        write_valid_r ? write_byte_addr : read_byte_addr;

    wire [WORD_INDEX_W-1:0] write_word_index =
        WORD_INDEX_W'(write_byte_addr >> MBAR_OBJECT_LG2);
    wire [WORD_INDEX_W-1:0] read_word_index =
        WORD_INDEX_W'(read_byte_addr >> MBAR_OBJECT_LG2);

    wire [LINE_BITS-1:0] write_line_data;
    wire [LMEM_DMA_DATA_SIZE-1:0] write_line_byteen;
    for (genvar i = 0; i < WORDS_PER_LINE; ++i) begin : g_write_line
        wire selected = (write_word_index == WORD_INDEX_W'(i));
        assign write_line_data[i * 32 +: 32] =
            selected ? write_state_r : '0;
        assign write_line_byteen[i * 4 +: 4] =
            {4{selected}};
    end

    wire [31:0] read_state =
        lmem_if.rsp_data.data[read_word_index * 32 +: 32];

    assign lmem_if.req_valid = write_valid_r || read_valid_r;
    assign lmem_if.req_data.rw = write_valid_r;
    assign lmem_if.req_data.addr =
        LMEM_DMA_ADDR_WIDTH'(memory_byte_addr >> LINE_SHIFT);
    assign lmem_if.req_data.data = write_line_data;
    assign lmem_if.req_data.byteen =
        write_valid_r ? write_line_byteen : '0;
    assign lmem_if.req_data.attr = '0;
    assign lmem_if.req_data.tag = '0;
    assign lmem_if.rsp_ready = 1'b1;

    wire refill_phase = read_state[0];
    wire [ARRIVAL_W-1:0] refill_pending_arrivals =
        read_state[1 +: ARRIVAL_W];
    wire [ARRIVAL_W-1:0] refill_expected_arrivals =
        read_state[1 + ARRIVAL_W +: ARRIVAL_W];
    wire [TX_W-1:0] refill_pending_transactions =
        read_state[1 + 2 * ARRIVAL_W +: TX_W];

    wire [ENTRY_W-1:0] victim_next =
        (victim_entry == ENTRY_W'(NUM_ENTRIES - 1))
            ? '0 : victim_entry + ENTRY_W'(1);

    always @(posedge clk) begin
        if (reset) begin
            valid_r <= '0;
            victim_ptr_r <= '0;
            waiter_valid_r <= '0;
            write_valid_r <= 0;
            read_valid_r <= 0;
            read_pending_r <= 0;
            unlock_valid_r <= 0;
        end else begin
            unlock_valid_r <= 0;

            if (write_fire)
                write_valid_r <= 0;

            if (read_fire) begin
                read_valid_r <= 0;
                read_pending_r <= 1;
            end

            if (lmem_if.rsp_valid) begin
                valid_r[read_entry_r] <= 1;
                addr_r[read_entry_r] <= read_addr_r;
                phase_r[read_entry_r] <= refill_phase;
                pending_arrivals_r[read_entry_r] <= refill_pending_arrivals;
                expected_arrivals_r[read_entry_r] <= refill_expected_arrivals;
                pending_transactions_r[read_entry_r]
                    <= refill_pending_transactions;
                read_pending_r <= 0;
            end

            if (miss_start) begin
                read_valid_r <= 1;
                read_addr_r <= current_addr;
                read_entry_r <= victim_entry;
                victim_ptr_r <= victim_next;
            end

            if (operation_fire) begin
                automatic logic [ENTRY_W-1:0] entry;
                automatic logic next_phase;
                automatic logic [ARRIVAL_W-1:0] next_pending_arrivals;
                automatic logic [ARRIVAL_W-1:0] next_expected_arrivals;
                automatic logic [TX_W-1:0] next_pending_transactions;
                automatic logic phase_complete;
                automatic logic [`VX_CFG_NUM_WARPS-1:0] wake_mask;

                entry = hit ? hit_entry : victim_entry;
                next_phase = hit ? phase_r[hit_entry] : 0;
                next_pending_arrivals =
                    hit ? pending_arrivals_r[hit_entry] : '0;
                next_expected_arrivals =
                    hit ? expected_arrivals_r[hit_entry] : '0;
                next_pending_transactions =
                    hit ? pending_transactions_r[hit_entry] : '0;
                phase_complete = 0;
                wake_mask = '0;

                case (current_op)
                MBAR_OP_INIT: begin
                    valid_r[entry] <= 1;
                    addr_r[entry] <= current_addr;
                    next_phase = 0;
                    next_pending_arrivals = ARRIVAL_W'(current_value);
                    next_expected_arrivals = ARRIVAL_W'(current_value);
                    next_pending_transactions = 0;
                    victim_ptr_r <= victim_next;
                end
                MBAR_OP_ARRIVE: begin
                    next_pending_arrivals =
                        pending_arrivals_r[entry] - ARRIVAL_W'(current_value);
                    phase_complete =
                        (next_pending_arrivals == 0)
                     && (pending_transactions_r[entry] == 0);
                end
                MBAR_OP_EXPECT_TX: begin
                    next_pending_transactions =
                        pending_transactions_r[entry] + TX_W'(current_value);
                end
                MBAR_OP_WAIT: begin
                    if (current_value[0] == phase_r[entry]) begin
                        waiter_valid_r[current_wid] <= 1;
                        waiter_addr_r[current_wid] <= current_addr;
                    end else begin
                        wake_mask[current_wid] = 1;
                        unlock_valid_r <= 1;
                        unlock_mask_r <= wake_mask;
                    end
                end
                MBAR_OP_COMPLETE_TX: begin
                    next_pending_transactions =
                        pending_transactions_r[entry] - TX_W'(current_value);
                    phase_complete =
                        (pending_arrivals_r[entry] == 0)
                     && (next_pending_transactions == 0);
                end
                default: begin
                end
                endcase

                if (phase_complete) begin
                    next_phase = ~phase_r[entry];
                    next_pending_arrivals = expected_arrivals_r[entry];
                    for (integer wid = 0;
                         wid < `VX_CFG_NUM_WARPS; ++wid) begin
                        if (waiter_valid_r[wid]
                         && (waiter_addr_r[wid] == current_addr)) begin
                            wake_mask[wid] = 1;
                            waiter_valid_r[wid] <= 0;
                        end
                    end
                    if (|wake_mask) begin
                        unlock_valid_r <= 1;
                        unlock_mask_r <= wake_mask;
                    end
                end

                if (current_op != MBAR_OP_WAIT) begin
                    phase_r[entry] <= next_phase;
                    pending_arrivals_r[entry] <= next_pending_arrivals;
                    expected_arrivals_r[entry] <= next_expected_arrivals;
                    pending_transactions_r[entry]
                        <= next_pending_transactions;
                    write_valid_r <= 1;
                    write_addr_r <= current_addr;
                    write_state_r <= {
                        {(32 - STATE_W){1'b0}},
                        next_pending_transactions,
                        next_expected_arrivals,
                        next_pending_arrivals,
                        next_phase
                    };
                end
            end
        end
    end

    assign unlock_valid = unlock_valid_r;
    assign unlock_mask = unlock_mask_r;
    assign busy = write_valid_r || read_valid_r || read_pending_r;

    `RUNTIME_ASSERT(!lmem_if.rsp_valid || read_pending_r,
        ("unexpected mbarrier LMEM response"))
    `RUNTIME_ASSERT(!operation_fire
                 || (current_op != MBAR_OP_INIT)
                 || (current_value > 0
                     && current_value <= `VX_CFG_NUM_WARPS),
        ("invalid mbarrier initialization count"))
    `RUNTIME_ASSERT(!operation_fire
                 || (current_op != MBAR_OP_ARRIVE)
                 || (current_value > 0
                     && current_value <= pending_arrivals_r[hit_entry]),
        ("invalid mbarrier arrival count"))
    `RUNTIME_ASSERT(!operation_fire
                 || (current_op != MBAR_OP_EXPECT_TX)
                 || (current_value > 0
                     && current_value
                        <= (32'(`VX_CFG_MAX_BAR_EVENTS)
                            - 32'(pending_transactions_r[hit_entry]))),
        ("invalid mbarrier transaction expectation"))
    `RUNTIME_ASSERT(!operation_fire
                 || (current_op != MBAR_OP_COMPLETE_TX)
                 || (current_value > 0
                     && current_value
                        <= pending_transactions_r[hit_entry]),
        ("invalid mbarrier transaction completion"))
    `RUNTIME_ASSERT(!lmem_if.rsp_valid
                 || (refill_pending_arrivals
                     <= refill_expected_arrivals),
        ("invalid mbarrier backing arrival state"))
    `RUNTIME_ASSERT(!lmem_if.rsp_valid
                 || (read_state[31:STATE_W] == 0),
        ("mbarrier backing word has reserved bits set"))

endmodule

`endif
