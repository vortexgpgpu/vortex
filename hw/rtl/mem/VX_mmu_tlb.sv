// Copyright 2024
// TLB: CAM-based address translation

`include "VX_define.vh"
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */

module VX_mmu_tlb import VX_gpu_pkg::*; #(
    parameter NUM_REQS       = DCACHE_NUM_REQS,
    parameter DATA_SIZE      = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH_IN   = DCACHE_TAG_WIDTH,
    parameter TAG_WIDTH_OUT  = TAG_WIDTH_IN + `UP(`CLOG2(NUM_REQS)),
    parameter ADDR_WIDTH     = DCACHE_ADDR_WIDTH,
    parameter FLAGS_WIDTH    = MEM_FLAGS_WIDTH
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    VX_mem_bus_if.slave  tlb_in_if [NUM_REQS],
    VX_mem_bus_if.master tlb_out_if [NUM_REQS],

    output wire          miss_valid,
    input  wire          miss_ready,
    output wire [31:0]   miss_vaddr,

    input  wire          fill_valid,
    output wire          fill_ready,
    input  wire [31:0]   fill_vaddr,
    input  wire [31:0]   fill_paddr,
    input  wire [7:0]    fill_flags
);

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam DATA_WIDTH    = DATA_SIZE * 8;
    localparam REQ_DATAW_IN  = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + FLAGS_WIDTH + TAG_WIDTH_IN;
    localparam REQ_DATAW_OUT = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + FLAGS_WIDTH + TAG_WIDTH_OUT;
    localparam RSP_DATAW_IN  = DATA_WIDTH + TAG_WIDTH_IN;
    localparam RSP_DATAW_OUT = DATA_WIDTH + TAG_WIDTH_OUT;
    localparam SOURCE_BITS   = `UP(`CLOG2(NUM_REQS));

    // =========================================================================
    // Request Serialize (NUM_REQS-to-1)
    // =========================================================================

    wire [NUM_REQS-1:0]                 req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW_IN-1:0] req_data_in;
    wire [NUM_REQS-1:0]                 req_ready_in;

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_req_in
        assign req_valid_in[i] = tlb_in_if[i].req_valid;
        assign req_data_in[i]  = {
            tlb_in_if[i].req_data.rw,
            tlb_in_if[i].req_data.addr,
            tlb_in_if[i].req_data.data,
            tlb_in_if[i].req_data.byteen,
            tlb_in_if[i].req_data.flags[FLAGS_WIDTH-1:0],
            tlb_in_if[i].req_data.tag[TAG_WIDTH_IN-1:0]
        };
        assign tlb_in_if[i].req_ready = req_ready_in[i];
    end

    wire                      ser_req_valid;
    wire [REQ_DATAW_IN-1:0]   ser_req_data;
    wire [SOURCE_BITS-1:0]    ser_req_sel;
    wire                      ser_req_ready;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        .DATAW       (REQ_DATAW_IN),
        .ARBITER     ("R"),
        .OUT_BUF     (0)
    ) req_serialize_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .ready_in  (req_ready_in),
        .valid_out (ser_req_valid),
        .data_out  (ser_req_data),
        .sel_out   (ser_req_sel),
        .ready_out (ser_req_ready)
    );

    // =========================================================================
    // TLB Logic
    // =========================================================================

    localparam TLB_SIZE       = 32;
    localparam TLB_INDEX_BITS = 5;
    localparam PAGE_OFFSET_BITS = 12 - `CLOG2(DATA_SIZE);
    localparam VPN_WIDTH = 20;
    localparam PPN_WIDTH = VPN_WIDTH;
    localparam SUPERPAGE_OFFSET_BITS = 22 - `CLOG2(DATA_SIZE);

    typedef struct packed {
        logic                 valid;
        logic                 mru;
        logic [1:0]           page_level;
        logic [VPN_WIDTH-1:0] vpn;
        logic [PPN_WIDTH-1:0] ppn;
        logic [7:0]           flags;
    } tlb_entry_t;

    tlb_entry_t tlb_entries [TLB_SIZE-1:0];

    typedef enum logic [1:0] {
        TLB_IDLE,
        TLB_READY,
        TLB_PTW_WAIT,
        TLB_REPLAY
    } tlb_state_t;

    tlb_state_t state;

    reg [REQ_DATAW_IN-1:0]   miss_buffer;
    reg [SOURCE_BITS-1:0]    miss_sel;
    reg [31:0]               miss_fill_paddr;
    reg miss_sent;
    reg [TLB_INDEX_BITS-1:0] victim_index;

    localparam ADDR_LSB_IN = TAG_WIDTH_IN + FLAGS_WIDTH + DATA_SIZE + DATA_WIDTH;
    localparam ADDR_LSB = TAG_WIDTH_OUT + FLAGS_WIDTH + DATA_SIZE + DATA_WIDTH;

    wire use_miss_buffer = (state == TLB_REPLAY);
    wire [REQ_DATAW_IN-1:0] lookup_data = use_miss_buffer ? miss_buffer : ser_req_data;
    wire [SOURCE_BITS-1:0]  lookup_sel  = use_miss_buffer ? miss_sel : ser_req_sel;
    wire [ADDR_WIDTH-1:0] lookup_addr = lookup_data[ADDR_LSB_IN +: ADDR_WIDTH];
    wire [VPN_WIDTH-1:0] lookup_vpn = lookup_addr[ADDR_WIDTH-1:PAGE_OFFSET_BITS];
    wire [TAG_WIDTH_IN-1:0] lookup_tag = lookup_data[TAG_WIDTH_IN-1:0];
    wire [TAG_WIDTH_OUT-1:0] lookup_tag_encoded;

    VX_bits_insert #(
        .N   (TAG_WIDTH_IN),
        .S   (SOURCE_BITS),
        .POS (0)
    ) tag_encode (
        .data_in  (lookup_tag),
        .ins_in   (lookup_sel),
        .data_out (lookup_tag_encoded)
    );

    wire [REQ_DATAW_OUT-1:0] lookup_data_encoded = {
        lookup_data[REQ_DATAW_IN-1:TAG_WIDTH_IN],
        lookup_tag_encoded
    };

    // CAM Lookup
    function automatic [VPN_WIDTH-1:0] vpn_mask(input [1:0] level);
        case (level)
            2'd0:    vpn_mask = 20'hFFFFF;
            2'd1:    vpn_mask = 20'hFFC00;
            2'd2:    vpn_mask = 20'h00000;
            default: vpn_mask = 20'hFFFFF;
        endcase
    endfunction

    wire [TLB_SIZE-1:0] cam_hit;
    for (genvar i = 0; i < TLB_SIZE; i++) begin : g_cam
        wire [VPN_WIDTH-1:0] mask_i = vpn_mask(tlb_entries[i].page_level);
        assign cam_hit[i] = tlb_entries[i].valid &&
                            ((tlb_entries[i].vpn & mask_i) == (lookup_vpn & mask_i));
    end

    wire tlb_hit = |cam_hit;

    reg [TLB_INDEX_BITS-1:0] hit_index;
    always_comb begin
        hit_index = '0;
        for (int j = TLB_SIZE-1; j >= 0; j--) begin
            if (cam_hit[j]) hit_index = j[TLB_INDEX_BITS-1:0];
        end
    end

    // Victim Selection (MRU-based)
    reg [TLB_INDEX_BITS-1:0] victim_candidate;
    reg found_invalid;
    wire all_mru;

    always_comb begin
        victim_candidate = '0;
        found_invalid = 1'b0;
        for (int j = TLB_SIZE-1; j >= 0; j--) begin
            if (!tlb_entries[j].valid) begin
                victim_candidate = j[TLB_INDEX_BITS-1:0];
                found_invalid = 1'b1;
            end
        end
        if (!found_invalid) begin
            for (int j = TLB_SIZE-1; j >= 0; j--) begin
                if (tlb_entries[j].valid && !tlb_entries[j].mru)
                    victim_candidate = j[TLB_INDEX_BITS-1:0];
            end
        end
    end

    wire [TLB_SIZE-1:0] entry_mru;
    for (genvar i = 0; i < TLB_SIZE; i++) begin : g_mru_check
        assign entry_mru[i] = tlb_entries[i].valid ? tlb_entries[i].mru : 1'b0;
    end
    assign all_mru = &entry_mru;

    // Address Translation
    wire [PPN_WIDTH-1:0] hit_ppn   = tlb_entries[hit_index].ppn;
    wire [1:0]           hit_level = tlb_entries[hit_index].page_level;

    reg [ADDR_WIDTH-1:0] cam_translated_addr;
    always_comb begin
        case (hit_level)
            2'd0:    cam_translated_addr = {hit_ppn, lookup_addr[PAGE_OFFSET_BITS-1:0]};
            2'd1:    cam_translated_addr = {hit_ppn[VPN_WIDTH-1:10], lookup_addr[SUPERPAGE_OFFSET_BITS-1:0]};
            2'd2:    cam_translated_addr = lookup_addr;
            default: cam_translated_addr = {hit_ppn, lookup_addr[PAGE_OFFSET_BITS-1:0]};
        endcase
    end

    wire [ADDR_WIDTH-1:0] replay_paddr = {miss_fill_paddr[31:12], lookup_addr[PAGE_OFFSET_BITS-1:0]};
    wire [ADDR_WIDTH-1:0] translated_addr = use_miss_buffer ? replay_paddr : cam_translated_addr;

    // State Machine
    wire input_handshake = ser_req_valid && ser_req_ready;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= TLB_IDLE;
            miss_buffer <= '0;
            miss_sel <= '0;
            miss_fill_paddr <= '0;
            miss_sent <= 1'b0;
            victim_index <= '0;
            for (int i = 0; i < TLB_SIZE; i++) begin
                tlb_entries[i].valid      <= 1'b0;
                tlb_entries[i].mru        <= 1'b0;
                tlb_entries[i].page_level <= 2'd0;
                tlb_entries[i].vpn        <= '0;
                tlb_entries[i].ppn        <= '0;
                tlb_entries[i].flags      <= '0;
            end
        end else begin
            case (state)
                TLB_IDLE: begin
                    state <= TLB_READY;
                end

                TLB_READY: begin
                    if (input_handshake) begin
                        if (tlb_hit) begin
                            tlb_entries[hit_index].mru <= 1'b1;
                            if (all_mru) begin
                                for (int i = 0; i < TLB_SIZE; i++) begin
                                    if (i[TLB_INDEX_BITS-1:0] != hit_index)
                                        tlb_entries[i].mru <= 1'b0;
                                end
                            end
                        end else begin
                            miss_buffer <= ser_req_data;
                            miss_sel <= ser_req_sel;
                            victim_index <= victim_candidate;
                            state <= TLB_PTW_WAIT;
                        end
                    end
                end

                TLB_PTW_WAIT: begin
                    if (miss_valid && miss_ready) miss_sent <= 1'b1;

                    if (fill_valid && fill_ready) begin
                        miss_fill_paddr <= fill_paddr;
                        tlb_entries[victim_index].valid      <= 1'b1;
                        tlb_entries[victim_index].mru        <= 1'b1;
                        tlb_entries[victim_index].page_level <= 2'd0;
                        tlb_entries[victim_index].vpn        <= fill_vaddr[31:12];
                        tlb_entries[victim_index].ppn        <= fill_paddr[31:12];
                        tlb_entries[victim_index].flags      <= fill_flags;

                        if (all_mru) begin
                            for (int i = 0; i < TLB_SIZE; i++) begin
                                if (i[TLB_INDEX_BITS-1:0] != victim_index)
                                    tlb_entries[i].mru <= 1'b0;
                            end
                        end
                        state <= TLB_REPLAY;
                        miss_sent <= 1'b0;
                    end
                end

                TLB_REPLAY: begin
                    if (output_handshake) state <= TLB_READY;
                end

                default: state <= TLB_IDLE;
            endcase
        end
    end

    // Control Signals
    assign ser_req_ready = (state == TLB_READY) && deser_req_ready;

    wire tlb_out_valid = (state == TLB_READY && input_handshake && tlb_hit) ||
                         (state == TLB_REPLAY);

    wire output_handshake = tlb_out_valid && deser_req_ready;

    wire [REQ_DATAW_OUT-1:0] tlb_out_data = {
        lookup_data_encoded[REQ_DATAW_OUT-1],
        translated_addr,
        lookup_data_encoded[ADDR_LSB-1:0]
    };

    // =========================================================================
    // Request Deserialize (1-to-NUM_REQS)
    // =========================================================================

    wire [SOURCE_BITS-1:0] deser_req_sel = tlb_out_data[SOURCE_BITS-1:0];

    wire                      deser_req_ready;
    wire [NUM_REQS-1:0]       deser_req_valid_out;
    wire [NUM_REQS-1:0][REQ_DATAW_OUT-1:0] deser_req_data_out;
    wire [NUM_REQS-1:0]       deser_req_ready_out;

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (REQ_DATAW_OUT),
        .OUT_BUF     (1)
    ) req_deserialize_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (deser_req_sel),
        .valid_in  (tlb_out_valid),
        .data_in   (tlb_out_data),
        .ready_in  (deser_req_ready),
        .valid_out (deser_req_valid_out),
        .data_out  (deser_req_data_out),
        .ready_out (deser_req_ready_out)
    );

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_req_out
        assign tlb_out_if[i].req_valid = deser_req_valid_out[i];
        assign tlb_out_if[i].req_data.rw     = deser_req_data_out[i][REQ_DATAW_OUT-1];
        assign tlb_out_if[i].req_data.addr   = deser_req_data_out[i][REQ_DATAW_OUT-2 -: ADDR_WIDTH];
        assign tlb_out_if[i].req_data.data   = deser_req_data_out[i][REQ_DATAW_OUT-2-ADDR_WIDTH -: DATA_WIDTH];
        assign tlb_out_if[i].req_data.byteen = deser_req_data_out[i][REQ_DATAW_OUT-2-ADDR_WIDTH-DATA_WIDTH -: DATA_SIZE];
        assign tlb_out_if[i].req_data.flags  = deser_req_data_out[i][REQ_DATAW_OUT-2-ADDR_WIDTH-DATA_WIDTH-DATA_SIZE -: FLAGS_WIDTH];
        assign tlb_out_if[i].req_data.tag    = deser_req_data_out[i][TAG_WIDTH_OUT-1:0];
        assign deser_req_ready_out[i]  = tlb_out_if[i].req_ready;
    end

    // =========================================================================
    // Response Serialize (NUM_REQS-to-1)
    // =========================================================================

    wire [NUM_REQS-1:0]                  rsp_valid_in;
    wire [NUM_REQS-1:0][RSP_DATAW_OUT-1:0] rsp_data_in;
    wire [NUM_REQS-1:0]                  rsp_ready_in;

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_rsp_in
        assign rsp_valid_in[i] = tlb_out_if[i].rsp_valid;
        assign rsp_data_in[i]  = {
            tlb_out_if[i].rsp_data.data,
            tlb_out_if[i].rsp_data.tag[TAG_WIDTH_OUT-1:0]
        };
        assign tlb_out_if[i].rsp_ready = rsp_ready_in[i];
    end

    wire                      ser_rsp_valid;
    wire [RSP_DATAW_OUT-1:0]  ser_rsp_data;
    wire                      ser_rsp_ready;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        .DATAW       (RSP_DATAW_OUT),
        .ARBITER     ("R"),
        .OUT_BUF     (0)
    ) rsp_serialize_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid_in),
        .data_in   (rsp_data_in),
        .ready_in  (rsp_ready_in),
        .valid_out (ser_rsp_valid),
        .data_out  (ser_rsp_data),
        `UNUSED_PIN (sel_out),
        .ready_out (ser_rsp_ready)
    );

    // =========================================================================
    // Response Deserialize (1-to-NUM_REQS)
    // =========================================================================

    wire [TAG_WIDTH_OUT-1:0] ser_rsp_tag = ser_rsp_data[TAG_WIDTH_OUT-1:0];
    wire [SOURCE_BITS-1:0]   rsp_source;
    wire [TAG_WIDTH_IN-1:0]  rsp_tag_restored;

    VX_bits_remove #(
        .N   (TAG_WIDTH_OUT),
        .S   (SOURCE_BITS),
        .POS (0)
    ) rsp_tag_decode (
        .data_in  (ser_rsp_tag),
        .sel_out  (rsp_source),
        .data_out (rsp_tag_restored)
    );

    wire [RSP_DATAW_IN-1:0] ser_rsp_data_restored = {
        ser_rsp_data[RSP_DATAW_OUT-1:TAG_WIDTH_OUT],
        rsp_tag_restored
    };

    wire                     deser_rsp_ready;
    wire [NUM_REQS-1:0]      deser_rsp_valid_out;
    wire [NUM_REQS-1:0][RSP_DATAW_IN-1:0] deser_rsp_data_out;
    wire [NUM_REQS-1:0]      deser_rsp_ready_out;

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (RSP_DATAW_IN),
        .OUT_BUF     (0)
    ) rsp_deserialize_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (rsp_source),
        .valid_in  (ser_rsp_valid),
        .data_in   (ser_rsp_data_restored),
        .ready_in  (deser_rsp_ready),
        .valid_out (deser_rsp_valid_out),
        .data_out  (deser_rsp_data_out),
        .ready_out (deser_rsp_ready_out)
    );

    assign ser_rsp_ready = deser_rsp_ready;

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_rsp_out
        assign tlb_in_if[i].rsp_valid = deser_rsp_valid_out[i];
        assign tlb_in_if[i].rsp_data.data = deser_rsp_data_out[i][RSP_DATAW_IN-1 -: DATA_WIDTH];
        assign tlb_in_if[i].rsp_data.tag  = deser_rsp_data_out[i][TAG_WIDTH_IN-1:0];
        assign deser_rsp_ready_out[i] = tlb_in_if[i].rsp_ready;
    end

    // =========================================================================
    // Miss/Fill Interface
    // =========================================================================

    wire [ADDR_WIDTH-1:0] miss_buffer_addr = miss_buffer[ADDR_LSB_IN +: ADDR_WIDTH];
    wire [VPN_WIDTH-1:0] miss_buffer_vpn = miss_buffer_addr[ADDR_WIDTH-1:PAGE_OFFSET_BITS];

    assign miss_valid = (state == TLB_PTW_WAIT) && !miss_sent;
    assign miss_vaddr = {miss_buffer_vpn, 12'b0};
    assign fill_ready = (state == TLB_PTW_WAIT) && miss_sent;

    // =========================================================================
    // Performance Counters
    // =========================================================================

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_tlb_reads;
    reg [PERF_CTR_BITS-1:0] perf_tlb_hits;
    reg [PERF_CTR_BITS-1:0] perf_tlb_misses;
    reg [PERF_CTR_BITS-1:0] perf_tlb_evictions;
    wire victim_was_valid = tlb_entries[victim_index].valid;

    always @(posedge clk) begin
        if (reset) begin
            perf_tlb_reads     <= '0;
            perf_tlb_hits      <= '0;
            perf_tlb_misses    <= '0;
            perf_tlb_evictions <= '0;
        end else begin
            if (state == TLB_READY && input_handshake)
                perf_tlb_reads <= perf_tlb_reads + PERF_CTR_BITS'(1);
            if (state == TLB_READY && input_handshake && tlb_hit)
                perf_tlb_hits <= perf_tlb_hits + PERF_CTR_BITS'(1);
            if (miss_valid && miss_ready)
                perf_tlb_misses <= perf_tlb_misses + PERF_CTR_BITS'(1);
            if (fill_valid && fill_ready && victim_was_valid)
                perf_tlb_evictions <= perf_tlb_evictions + PERF_CTR_BITS'(1);
        end
    end

    assign mmu_perf.tlb_reads     = perf_tlb_reads;
    assign mmu_perf.tlb_hits      = perf_tlb_hits;
    assign mmu_perf.tlb_misses    = perf_tlb_misses;
    assign mmu_perf.tlb_evictions = perf_tlb_evictions;
    assign mmu_perf.ptw_walks     = perf_tlb_misses;
    assign mmu_perf.ptw_latency   = '0;
`endif

endmodule
