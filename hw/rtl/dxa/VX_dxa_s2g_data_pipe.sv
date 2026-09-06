// Pipelined S2G data path.  This is an optional implementation selected by
// VX_CFG_DXA_S2G_PIPELINED; the legacy one-line-at-a-time path remains the
// default.  A small line FIFO decouples source gathers from global stores.

`include "VX_define.vh"

`ifdef VX_CFG_EXT_DXA_S2G_ENABLE
`ifdef VX_CFG_DXA_S2G_PIPELINED

module VX_dxa_s2g_data_pipe import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_LINE_SIZE = `VX_CFG_L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE),
    parameter GMEM_OFF_BITS = `CLOG2(GMEM_LINE_SIZE),
    parameter SLOTS = 4
) (
    input wire clk,
    input wire reset,
    input wire transfer_active,
    input wire pipeline_start,
    input wire zero_length,
    input wire                       ag_valid,
    output wire                      ag_ready,
    input wire [GMEM_ADDR_WIDTH-1:0] ag_cl_addr,
    input wire [DXA_SMEM_ADDR_W-1:0] ag_smem_byte_addr,
    input wire [GMEM_OFF_BITS-1:0]   ag_byte_offset,
    input wire [GMEM_OFF_BITS:0]     ag_valid_length,
    input wire                       ag_oob,
    input wire                       ag_last,
    input wire [NC_WIDTH-1:0]        active_core_id,
    input wire [UUID_WIDTH-1:0]      active_uuid,
    input wire [NW_WIDTH-1:0]        active_wid,
    input wire [DXA_GROUP_EPOCH_W-1:0] active_epoch,
    input wire [DXA_GROUP_SEQ_W-1:0] active_group_seq,
    input wire [DXA_GROUP_OPID_W-1:0] active_op_id,
    VX_mem_bus_if.master gmem_bus_if,
    VX_mem_bus_if.master smem_bus_if,
    VX_dxa_group_completion_if.master completion_if,
    output wire transfer_done,
    output wire [31:0] sent_count,
    output wire [31:0] read_count
);
    localparam GMEM_BYTES = GMEM_LINE_SIZE;
    localparam GMEM_DATAW = GMEM_BYTES * 8;
    localparam SMEM_BYTES = DXA_LMEM_WORD_SIZE;
    localparam SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W;
    localparam SMEM_OFF_BITS = `CLOG2(SMEM_BYTES);
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;
    localparam MAX_WORDS = (GMEM_BYTES + 2 * SMEM_BYTES - 2) / SMEM_BYTES;
    localparam WORD_INDEX_W = `CLOG2(MAX_WORDS);
    localparam WORD_COUNT_W = `CLOG2(MAX_WORDS + 1);
    localparam SOURCE_DATAW = MAX_WORDS * SMEM_BYTES * 8;
    localparam SOURCE_BYTE_W = `CLOG2(MAX_WORDS * SMEM_BYTES);
    localparam SLOT_W = `UP(`CLOG2(SLOTS));
    localparam SLOT_COUNT_W = `CLOG2(SLOTS + 1);

    typedef struct packed {
        logic [GMEM_ADDR_WIDTH-1:0] cl_addr;
        logic [SMEM_ADDR_WIDTH-1:0] smem_word_addr;
        logic [SMEM_OFF_BITS-1:0] smem_byte_offset;
        logic [GMEM_OFF_BITS-1:0] byte_offset;
        logic [GMEM_OFF_BITS:0] valid_length;
        logic last;
        logic oob;
        logic complete;
        logic emitted;
        logic [WORD_INDEX_W-1:0] words_done;
        logic [WORD_COUNT_W-1:0] word_count;
        logic [SOURCE_DATAW-1:0] data;
    } slot_t;

    slot_t slots_r [SLOTS];
    reg [SLOTS-1:0] valid_r;
    reg [SLOT_W-1:0] alloc_ptr_r;
    reg [SLOT_COUNT_W-1:0] valid_count_r;
    reg read_active_r;
    reg [SLOT_W-1:0] read_slot_r;
    reg [WORD_INDEX_W-1:0] read_word_r;
    reg source_done_r;
    reg source_signaled_r;
    reg last_token_seen_r;
    reg last_token_has_store_r;
    reg completion_pending_r;
    reg completion_sent_r;
    reg destination_done_r;
    reg [31:0] sent_count_r;
    reg [31:0] read_count_r;

    wire [GMEM_OFF_BITS:0] input_source_end =
        (GMEM_OFF_BITS+1)'(ag_smem_byte_addr[SMEM_OFF_BITS-1:0]) + ag_valid_length;
    wire [WORD_COUNT_W-1:0] input_word_count =
        (input_source_end + (GMEM_OFF_BITS+1)'(SMEM_BYTES - 1)) >> SMEM_OFF_BITS;
    wire token_fire = ag_valid && ag_ready;
    wire have_free = valid_count_r < SLOT_COUNT_W'(SLOTS);
    assign ag_ready = have_free && !pipeline_start;

    wire [SLOT_W-1:0] alloc_slot = alloc_ptr_r;
    wire [SLOT_W-1:0] read_slot = read_slot_r;
    wire read_request_valid = read_active_r;
    wire [SMEM_TAG_VALUE_W-1:0] route_tag =
        (SMEM_TAG_VALUE_W'(active_core_id) << 1) | SMEM_TAG_VALUE_W'(1);
    assign smem_bus_if.req_valid = read_request_valid;
    assign smem_bus_if.req_data.rw = 1'b0;
    assign smem_bus_if.req_data.addr = slots_r[read_slot].smem_word_addr + SMEM_ADDR_WIDTH'(read_word_r);
    assign smem_bus_if.req_data.data = '0;
    assign smem_bus_if.req_data.byteen = '0;
    assign smem_bus_if.req_data.attr = '0;
    assign smem_bus_if.req_data.tag.uuid = active_uuid;
    assign smem_bus_if.req_data.tag.value = route_tag;
    wire read_request_fire = smem_bus_if.req_valid && smem_bus_if.req_ready;
    assign smem_bus_if.rsp_ready = read_active_r;
    wire read_response_fire = smem_bus_if.rsp_valid && smem_bus_if.rsp_ready;

    reg have_read_candidate;
    reg [SLOT_W-1:0] read_candidate;
    integer ri;
    always @(*) begin
        have_read_candidate = 1'b0;
        read_candidate = '0;
        for (ri = 0; ri < SLOTS; ++ri) begin
            if (!have_read_candidate && valid_r[ri] && !slots_r[ri].complete) begin
                have_read_candidate = 1'b1;
                read_candidate = SLOT_W'(ri);
            end
        end
    end

    reg have_emit_candidate;
    reg [SLOT_W-1:0] emit_slot;
    integer ei;
    always @(*) begin
        have_emit_candidate = 1'b0;
        emit_slot = '0;
        for (ei = 0; ei < SLOTS; ++ei) begin
            if (!have_emit_candidate && valid_r[ei] && slots_r[ei].complete && !slots_r[ei].emitted) begin
                have_emit_candidate = 1'b1;
                emit_slot = SLOT_W'(ei);
            end
        end
    end

    reg source_pending_any;
    integer pi;
    always @(*) begin
        source_pending_any = 1'b0;
        for (pi = 0; pi < SLOTS; ++pi)
            source_pending_any = source_pending_any || (valid_r[pi] && !slots_r[pi].complete);
    end

    wire out_valid = have_emit_candidate;
    wire [SOURCE_DATAW-1:0] emit_data = slots_r[emit_slot].data;
    wire [GMEM_BYTES:0] byteen_seed =
        ((GMEM_BYTES+1)'(1) << slots_r[emit_slot].valid_length) - (GMEM_BYTES+1)'(1);
    wire [GMEM_BYTES-1:0] emit_byteen = GMEM_BYTES'(byteen_seed) << slots_r[emit_slot].byte_offset;
    `UNUSED_VAR (byteen_seed[GMEM_BYTES])
    wire store_fire = out_valid && gmem_bus_if.req_ready;
    wire last_store_fire = store_fire && slots_r[emit_slot].last;
    assign gmem_bus_if.req_valid = out_valid;
    assign gmem_bus_if.req_data.rw = 1'b1;
    assign gmem_bus_if.req_data.addr = slots_r[emit_slot].cl_addr;
    assign gmem_bus_if.req_data.byteen = emit_byteen;
    assign gmem_bus_if.req_data.attr = '0;
    assign gmem_bus_if.req_data.tag.uuid = active_uuid;
    assign gmem_bus_if.req_data.tag.value = '0;
    assign gmem_bus_if.rsp_ready = 1'b0;
    wire [GMEM_DATAW-1:0] emit_bus_data;
    assign gmem_bus_if.req_data.data = emit_bus_data;
    genvar bi;
    generate for (bi = 0; bi < GMEM_BYTES; ++bi) begin : g_emit_byte
        wire [SOURCE_BYTE_W-1:0] source_byte =
            SOURCE_BYTE_W'(slots_r[emit_slot].smem_byte_offset)
            + SOURCE_BYTE_W'((GMEM_OFF_BITS+1)'(bi) - {1'b0, slots_r[emit_slot].byte_offset});
        assign emit_bus_data[bi*8 +: 8] = emit_byteen[bi]
            ? emit_data[source_byte*8 +: 8] : 8'h00;
    end endgenerate

    wire source_event_done = completion_sent_r || completion_if.valid && completion_if.ready;
    wire destination_event_done = destination_done_r || last_store_fire;
    assign completion_if.valid = completion_pending_r && !pipeline_start;
    assign completion_if.core_id = active_core_id;
    assign completion_if.data.wid = active_wid;
    assign completion_if.data.epoch = active_epoch;
    assign completion_if.data.group_seq = active_group_seq;
    assign completion_if.data.op_id = active_op_id;
    wire completion_fire = completion_if.valid && completion_if.ready;
    assign transfer_done = transfer_active && !pipeline_start
        && (zero_length || (source_event_done && destination_event_done));
    assign sent_count = sent_count_r;
    assign read_count = read_count_r;

    integer si;
    always @(posedge clk) begin
        if (reset || pipeline_start) begin
            valid_r <= '0;
            alloc_ptr_r <= '0;
            valid_count_r <= '0;
            read_active_r <= 1'b0;
            read_slot_r <= '0;
            read_word_r <= '0;
            source_done_r <= zero_length;
            source_signaled_r <= 1'b0;
            last_token_seen_r <= 1'b0;
            last_token_has_store_r <= 1'b0;
            completion_pending_r <= 1'b0;
            completion_sent_r <= 1'b0;
            destination_done_r <= zero_length;
            sent_count_r <= '0;
            read_count_r <= '0;
            for (si = 0; si < SLOTS; ++si) slots_r[si] <= '0;
            if (pipeline_start && zero_length) begin
                source_done_r <= 1'b1;
                source_signaled_r <= 1'b1;
                completion_pending_r <= 1'b1;
                destination_done_r <= 1'b1;
            end
        end else begin
            source_done_r <= 1'b0;
            if (!read_active_r && have_read_candidate) begin
                read_active_r <= 1'b1;
                read_slot_r <= read_candidate;
                read_word_r <= slots_r[read_candidate].words_done;
            end
            if (read_request_fire) begin
                read_count_r <= read_count_r + 32'd1;
            end
            if (read_response_fire) begin
                slots_r[read_slot].data[slots_r[read_slot].words_done*SMEM_BYTES*8 +: SMEM_BYTES*8]
                    <= SOURCE_DATAW'(smem_bus_if.rsp_data.data);
                read_active_r <= 1'b0;
                if (slots_r[read_slot].words_done + 1 < slots_r[read_slot].word_count) begin
                    slots_r[read_slot].words_done <= slots_r[read_slot].words_done + 1'b1;
                end else begin
                    slots_r[read_slot].complete <= 1'b1;
                end
            end
            if (token_fire) begin
                if (ag_oob) begin
                    // Out-of-bounds address-generator tokens carry no source
                    // bytes and therefore do not consume a payload slot.
                    if (ag_last) begin
                        last_token_seen_r <= 1'b1;
                        last_token_has_store_r <= 1'b0;
                    end
                end else begin
                    valid_r[alloc_slot] <= 1'b1;
                    slots_r[alloc_slot].cl_addr <= ag_cl_addr;
                    slots_r[alloc_slot].smem_word_addr <= SMEM_ADDR_WIDTH'(ag_smem_byte_addr >> SMEM_OFF_BITS);
                    slots_r[alloc_slot].smem_byte_offset <= ag_smem_byte_addr[SMEM_OFF_BITS-1:0];
                    slots_r[alloc_slot].byte_offset <= ag_byte_offset;
                    slots_r[alloc_slot].valid_length <= ag_valid_length;
                    slots_r[alloc_slot].last <= ag_last;
                    slots_r[alloc_slot].oob <= 1'b0;
                    slots_r[alloc_slot].words_done <= '0;
                    slots_r[alloc_slot].word_count <= WORD_COUNT_W'(input_word_count);
                    slots_r[alloc_slot].emitted <= 1'b0;
                    slots_r[alloc_slot].data <= '0;
                    slots_r[alloc_slot].complete <= 1'b0;
                    alloc_ptr_r <= (alloc_slot + 1 >= SLOTS) ? '0 : alloc_slot + 1'b1;
                    if (ag_last) begin
                        last_token_seen_r <= 1'b1;
                        last_token_has_store_r <= 1'b1;
                    end
                end
            end
            // The final address-generator token may be OOB or may arrive
            // while earlier lines are still gathering.  Delay the
            // SOURCE_CONSUMED event until every accepted line is complete.
            if (!source_signaled_r && (last_token_seen_r || (token_fire && ag_last))
                && !(token_fire && !ag_oob) && !read_active_r && !source_pending_any) begin
                source_done_r <= 1'b1;
                source_signaled_r <= 1'b1;
                completion_pending_r <= 1'b1;
                if (!(last_token_has_store_r || (token_fire && ag_last && !ag_oob)))
                    destination_done_r <= 1'b1;
            end
            if (store_fire) begin
                slots_r[emit_slot].emitted <= 1'b1;
                valid_r[emit_slot] <= 1'b0;
                sent_count_r <= sent_count_r + 1'b1;
                if (last_store_fire || (source_signaled_r && (valid_count_r == 1)))
                    destination_done_r <= 1'b1;
            end
            if (completion_fire) begin
                completion_pending_r <= 1'b0;
                completion_sent_r <= 1'b1;
            end
            case ({token_fire && !ag_oob, store_fire})
                2'b10: valid_count_r <= valid_count_r + 1'b1;
                2'b01: valid_count_r <= valid_count_r - 1'b1;
                default: valid_count_r <= valid_count_r;
            endcase
        end
    end

    `RUNTIME_ASSERT(!gmem_bus_if.rsp_valid, ("unexpected response to S2G store"))
    `RUNTIME_ASSERT(!read_response_fire || (smem_bus_if.rsp_data.tag.uuid == active_uuid),
        ("S2G LMEM response UUID mismatch"))
    `RUNTIME_ASSERT(!read_response_fire || (smem_bus_if.rsp_data.tag.value == route_tag),
        ("S2G LMEM response route tag mismatch"))

endmodule

`endif
`endif
