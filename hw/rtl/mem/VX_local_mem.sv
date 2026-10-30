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

module VX_local_mem import VX_gpu_pkg::*; #(
    parameter `STRING  INSTANCE_ID = "",

    // Size of cache in bytes
    parameter SIZE              = (1024*16*8),

    // Number of Word requests per cycle
    parameter NUM_REQS          = 4,
    // Number of banks
    parameter NUM_BANKS         = 4,

    // Address width
    parameter ADDR_WIDTH        = `CLOG2(SIZE),
    // Size of a word in bytes
    parameter WORD_SIZE         = `VX_CFG_XLEN/8,

    // Request tag size
    parameter TAG_WIDTH         = 16,

    // Enable DMA port
    parameter DMA_ENABLE        = 0,
    parameter DMA_TAG_WIDTH     = 1,

    // Synthesize atomic-op + LR/SC logic in the banks
    parameter AMO_ENABLE        = 0,

    // Response buffer
    parameter OUT_BUF           = 0
 ) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    output lmem_perf_t lmem_perf,
`endif

    // LSU read/write port
    VX_mem_bus_if.slave lsu_bus_if [NUM_REQS],

    // DMA read/write port
    VX_mem_bus_if.slave dma_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam REQ_SEL_BITS    = `CLOG2(NUM_REQS);
    localparam REQ_SEL_WIDTH   = `UP(REQ_SEL_BITS);
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam NUM_WORDS       = SIZE / WORD_SIZE;
    localparam WORDS_PER_BANK  = NUM_WORDS / NUM_BANKS;
    localparam BANK_ADDR_WIDTH = `CLOG2(WORDS_PER_BANK);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);
    // AMO sideband carried to the banks: valid + op + unsigned + hart_id. The
    // operand width comes from the request's byteen; hart_id identifies the
    // reserver so a store-conditional can tell its own reservation from
    // another hart's.
    // Compare-and-swap adds a comparand: the request's data word already
    // carries the swap value, so the third operand needs its own room.
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    localparam AMO_CMP_WIDTH   = `VX_CFG_XLEN;
`else
    localparam AMO_CMP_WIDTH   = 0;
`endif
    localparam AMO_SB_WIDTH    = 1 + $bits(amo_op_e) + 1 + HART_ID_WIDTH + AMO_CMP_WIDTH;
    localparam REQ_DATAW       = 1 + BANK_ADDR_WIDTH + WORD_SIZE + WORD_WIDTH + TAG_WIDTH + AMO_SB_WIDTH;
    localparam RSP_DATAW       = WORD_WIDTH + TAG_WIDTH;

    `STATIC_ASSERT(ADDR_WIDTH == (BANK_ADDR_WIDTH + `CLOG2(NUM_BANKS)), ("invalid parameter"))

    // bank selection

    wire [NUM_REQS-1:0][BANK_SEL_WIDTH-1:0] req_bank_idx;
    if (NUM_BANKS > 1) begin : g_req_bank_idx
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_bank_idxs
            assign req_bank_idx[i] = lsu_bus_if[i].req_data.addr[0 +: BANK_SEL_BITS];
        end
    end else begin : g_req_bank_idx_0
        assign req_bank_idx = 0;
    end

    // bank addressing

    wire [NUM_REQS-1:0][BANK_ADDR_WIDTH-1:0] req_bank_addr;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_bank_addr
        assign req_bank_addr[i] = lsu_bus_if[i].req_data.addr[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    end

    // AMO sideband per request, from the shared mem-bus attr.
    wire [NUM_REQS-1:0]                  req_amo_valid;
    wire [NUM_REQS-1:0][$bits(amo_op_e)-1:0] req_amo_op;
    wire [NUM_REQS-1:0]                  req_amo_unsigned;
    wire [NUM_REQS-1:0][HART_ID_WIDTH-1:0] req_amo_hart_id;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [NUM_REQS-1:0][AMO_CMP_WIDTH-1:0] req_amo_cmp;
`endif
    // Gated by AMO_ENABLE: the widths stay constant and the zeros constant-fold
    // through the crossbar payload (the VX_cache_bank pattern).
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_amo
        mem_bus_attr_t lane_attr;
        assign lane_attr = mem_bus_attr_t'(lsu_bus_if[i].req_data.attr);
        assign req_amo_valid[i]    = AMO_ENABLE ? lane_attr.amo.amo_valid : 1'b0;
        assign req_amo_op[i]       = AMO_ENABLE ? lane_attr.amo.amo_op : '0;
        assign req_amo_unsigned[i] = AMO_ENABLE ? lane_attr.amo.amo_unsigned : 1'b0;
        assign req_amo_hart_id[i]  = AMO_ENABLE ? lane_attr.amo.hart_id : '0;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        assign req_amo_cmp[i]      = AMO_ENABLE ? lane_attr.amo.amo_cmp : '0;
    `endif
        `UNUSED_VAR (lane_attr)
    end

    // bank requests dispatch

    wire [NUM_BANKS-1:0]                    per_bank_req_valid;
    wire [NUM_BANKS-1:0]                    per_bank_req_rw;
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] per_bank_req_addr;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_req_byteen;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]    per_bank_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag;
    wire [NUM_BANKS-1:0]                    per_bank_req_amo_valid;
    wire [NUM_BANKS-1:0][$bits(amo_op_e)-1:0] per_bank_req_amo_op;
    wire [NUM_BANKS-1:0]                    per_bank_req_amo_unsigned;
    wire [NUM_BANKS-1:0][HART_ID_WIDTH-1:0] per_bank_req_amo_hart_id;
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [NUM_BANKS-1:0][AMO_CMP_WIDTH-1:0] per_bank_req_amo_cmp;
`endif
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_req_idx;
    wire [NUM_BANKS-1:0]                    per_bank_req_ready;

    wire [NUM_BANKS-1:0][REQ_DATAW-1:0]     per_bank_req_data_aos;

    wire [NUM_REQS-1:0]                 req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_REQS-1:0]                 req_ready_in;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] perf_collisions;
`endif

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_data_in
        assign req_valid_in[i] = lsu_bus_if[i].req_valid;
        assign req_data_in[i] = {
            req_amo_valid[i],
            req_amo_op[i],
            req_amo_unsigned[i],
            req_amo_hart_id[i],
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            req_amo_cmp[i],
        `endif
            lsu_bus_if[i].req_data.rw,
            req_bank_addr[i],
            lsu_bus_if[i].req_data.data,
            lsu_bus_if[i].req_data.byteen,
            lsu_bus_if[i].req_data.tag
        };
        assign lsu_bus_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (REQ_DATAW),
        .PERF_CTR_BITS (PERF_CTR_BITS),
        .ARBITER     ("P"),
        // over more than 12 ports the arbiters are a tree; slices of four keep
        // its depth and shorten the grant each request waits on
        .MAX_FANOUT  ((NUM_REQS > 12) ? 4 : `MAX_FANOUT),
        .OUT_BUF     (3) // output should be registered for the data_store addressing
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
    `ifdef PERF_ENABLE
        .collisions (perf_collisions),
    `else
        `UNUSED_PIN (collisions),
    `endif
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .sel_in    (req_bank_idx),
        .ready_in  (req_ready_in),
        .valid_out (per_bank_req_valid),
        .data_out  (per_bank_req_data_aos),
        .sel_out   (per_bank_req_idx),
        .ready_out (per_bank_req_ready)
    );

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_req_data_soa
        assign {
            per_bank_req_amo_valid[i],
            per_bank_req_amo_op[i],
            per_bank_req_amo_unsigned[i],
            per_bank_req_amo_hart_id[i],
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            per_bank_req_amo_cmp[i],
        `endif
            per_bank_req_rw[i],
            per_bank_req_addr[i],
            per_bank_req_data[i],
            per_bank_req_byteen[i],
            per_bank_req_tag[i]
        } = per_bank_req_data_aos[i];
    end

    // banks access (declared here so g_dma_enable can reference per_bank_rsp_data)

    wire [NUM_BANKS-1:0]                 per_bank_rsp_valid;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] per_bank_rsp_data;
    // Back-pressure-safe copy of the LSU response data (see VX_local_mem_bank).
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] bank_lsu_rsp_data;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_rsp_idx;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]  per_bank_rsp_tag;
    wire [NUM_BANKS-1:0]                 per_bank_rsp_ready;

    // DMA port handshake
    //   rw=0 reads  : accepted when the response pipe-buffer has space.
    //   rw=1 writes : always accepted; no response issued.
    //   DMA has priority over LSU at every bank SRAM.

    wire dma_rsp_buf_ready; // driven by pipe-buffer or tied 0 when disabled

    if (DMA_ENABLE) begin : g_dma_enable
        `UNUSED_VAR (dma_bus_if.req_data.attr)

        assign dma_bus_if.req_ready = dma_bus_if.req_data.rw || dma_rsp_buf_ready;

        wire dma_rd_fire = dma_bus_if.req_valid && ~dma_bus_if.req_data.rw && dma_rsp_buf_ready;

        // Delay tag by 1 cycle to align with SRAM OUT_REG latency
        VX_pipe_buffer #(
            .DATAW (DMA_TAG_WIDTH)
        ) dma_rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (dma_rd_fire),
            .ready_in  (dma_rsp_buf_ready),
            .data_in   (dma_bus_if.req_data.tag),
            .valid_out (dma_bus_if.rsp_valid),
            .data_out  (dma_bus_if.rsp_data.tag),
            .ready_out (dma_bus_if.rsp_ready)
        );

        // Pack all bank SRAM outputs into the read response.
        //
        // Same back-pressure hazard as the LSU side: dma_rsp_buf buffers only
        // the tag/valid; the data (all banks) is the live SRAM OUT_REG. While a
        // DMA read response is back-pressured, an interleaving LSU read
        // (~dma_active) re-drives a bank's OUT_REG and corrupts the held DMA
        // line. Latch the whole line on the first response-valid cycle (rdata
        // is still valid then) and serve the latched copy while stalled. One
        // DMA response spans all banks, so this is a single full-line hold.
        reg  [NUM_BANKS-1:0][WORD_WIDTH-1:0] dma_rsp_hold_data_r;
        reg                                  dma_rsp_hold_valid_r;
        wire dma_rsp_consumed = dma_bus_if.rsp_valid && dma_bus_if.rsp_ready;
        always @(posedge clk) begin
            if (reset) begin
                dma_rsp_hold_valid_r <= 1'b0;
            end else if (dma_rsp_consumed) begin
                dma_rsp_hold_valid_r <= 1'b0;
            end else if (dma_bus_if.rsp_valid && ~dma_rsp_hold_valid_r) begin
                dma_rsp_hold_data_r  <= per_bank_rsp_data;
                dma_rsp_hold_valid_r <= 1'b1;
            end
        end
        for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_dma_rsp_data
            assign dma_bus_if.rsp_data.data[i*WORD_WIDTH +: WORD_WIDTH] =
                dma_rsp_hold_valid_r ? dma_rsp_hold_data_r[i] : per_bank_rsp_data[i];
        end

    end else begin : g_no_dma
        assign dma_rsp_buf_ready    = 1'b0;
        assign dma_bus_if.req_ready = 1'b0;
        assign dma_bus_if.rsp_valid = 1'b0;
        assign dma_bus_if.rsp_data  = '0;
        `UNUSED_VAR (dma_bus_if.req_valid)
        `UNUSED_VAR (dma_bus_if.req_data)
        `UNUSED_VAR (dma_bus_if.rsp_ready)
        `UNUSED_VAR (per_bank_rsp_data)
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_data_store
        VX_local_mem_bank #(
            .WORD_SIZE     (WORD_SIZE),
            .NUM_WORDS     (WORDS_PER_BANK),
            .TAG_WIDTH     (TAG_WIDTH),
            .REQ_SEL_WIDTH (REQ_SEL_WIDTH),
            .DMA_ENABLE    (DMA_ENABLE),
            .AMO_ENABLE    (AMO_ENABLE)
        ) bank (
            .clk               (clk),
            .reset             (reset),
            .req_valid         (per_bank_req_valid[i]),
            .req_rw            (per_bank_req_rw[i]),
            .req_addr          (per_bank_req_addr[i]),
            .req_byteen        (per_bank_req_byteen[i]),
            .req_data          (per_bank_req_data[i]),
            .req_tag           (per_bank_req_tag[i]),
            .req_idx           (per_bank_req_idx[i]),
            .req_amo_valid     (per_bank_req_amo_valid[i]),
            .req_amo_op        (per_bank_req_amo_op[i]),
            .req_amo_unsigned  (per_bank_req_amo_unsigned[i]),
            .req_amo_hart_id   (per_bank_req_amo_hart_id[i]),
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            .req_amo_cmp       (per_bank_req_amo_cmp[i]),
        `endif
            .req_ready         (per_bank_req_ready[i]),
            .rsp_valid         (per_bank_rsp_valid[i]),
            .rsp_data          (bank_lsu_rsp_data[i]),
            .rsp_tag           (per_bank_rsp_tag[i]),
            .rsp_idx           (per_bank_rsp_idx[i]),
            .rsp_ready         (per_bank_rsp_ready[i]),
            .rsp_rdata         (per_bank_rsp_data[i]),
            .dma_req_valid     (dma_bus_if.req_valid),
            .dma_req_rw        (dma_bus_if.req_data.rw),
            .dma_req_addr      (BANK_ADDR_WIDTH'(dma_bus_if.req_data.addr)),
            .dma_req_byteen    (dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]),
            .dma_req_data      (dma_bus_if.req_data.data[i*WORD_WIDTH +: WORD_WIDTH]),
            .dma_rsp_buf_ready (dma_rsp_buf_ready)
        );
    end

    // bank responses gather

    wire [NUM_BANKS-1:0][RSP_DATAW-1:0] per_bank_rsp_data_aos;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_rsp_data_aos
        assign per_bank_rsp_data_aos[i] = {bank_lsu_rsp_data[i], per_bank_rsp_tag[i]};
    end

    wire [NUM_REQS-1:0]                 rsp_valid_out;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0]  rsp_data_out;
    wire [NUM_REQS-1:0]                 rsp_ready_out;

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (RSP_DATAW),
        .ARBITER     ("P"), // this priority arbiter has negligeable impact on performance
        .OUT_BUF     (OUT_BUF)
    ) rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .sel_in    (per_bank_rsp_idx),
        .valid_in  (per_bank_rsp_valid),
        .data_in   (per_bank_rsp_data_aos),
        .ready_in  (per_bank_rsp_ready),
        .valid_out (rsp_valid_out),
        .data_out  (rsp_data_out),
        .ready_out (rsp_ready_out),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_lsu_bus_if
        assign lsu_bus_if[i].rsp_valid = rsp_valid_out[i];
        assign lsu_bus_if[i].rsp_data  = rsp_data_out[i];
        assign rsp_ready_out[i] = lsu_bus_if[i].rsp_ready;
    end

`ifdef PERF_ENABLE
    // per cycle: reads, writes
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_reads_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_writes_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_crsp_stall_per_cycle;

    wire [NUM_REQS-1:0] req_rw;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_rw
        assign req_rw[i] = lsu_bus_if[i].req_data.rw;
    end

    wire [NUM_REQS-1:0] perf_reads_per_req, perf_writes_per_req;
    wire [NUM_REQS-1:0] perf_crsp_stall_per_req = rsp_valid_out & ~rsp_ready_out;

    `BUFFER(perf_reads_per_req, req_valid_in & req_ready_in & ~req_rw);
    `BUFFER(perf_writes_per_req, req_valid_in & req_ready_in & req_rw);

    `POP_COUNT(perf_reads_per_cycle, perf_reads_per_req);
    `POP_COUNT(perf_writes_per_cycle, perf_writes_per_req);
    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);

    reg [PERF_CTR_BITS-1:0] perf_reads;
    reg [PERF_CTR_BITS-1:0] perf_writes;
    reg [PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_reads       <= '0;
            perf_writes      <= '0;
            perf_crsp_stalls <= '0;
        end else begin
            perf_reads       <= perf_reads  + PERF_CTR_BITS'(perf_reads_per_cycle);
            perf_writes      <= perf_writes + PERF_CTR_BITS'(perf_writes_per_cycle);
            perf_crsp_stalls <= perf_crsp_stalls + PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign lmem_perf.reads       = perf_reads;
    assign lmem_perf.writes      = perf_writes;
    assign lmem_perf.bank_stalls = perf_collisions;
    assign lmem_perf.crsp_stalls = perf_crsp_stalls;

`endif

`ifdef DBG_TRACE_MEM

    wire [NUM_BANKS-1:0][TAG_WIDTH-UUID_WIDTH-1:0] per_bank_req_tag_value;
    wire [NUM_BANKS-1:0][UUID_WIDTH-1:0] per_bank_req_uuid;

    wire [NUM_BANKS-1:0][TAG_WIDTH-UUID_WIDTH-1:0] per_bank_rsp_tag_value;
    wire [NUM_BANKS-1:0][UUID_WIDTH-1:0] per_bank_rsp_uuid;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_req_uuid
        assign per_bank_req_tag_value[i] = per_bank_req_tag[i][TAG_WIDTH-UUID_WIDTH-1:0];
        assign per_bank_rsp_tag_value[i] = per_bank_rsp_tag[i][TAG_WIDTH-UUID_WIDTH-1:0];
        if (UUID_WIDTH != 0) begin : g_uuid
            assign per_bank_req_uuid[i] = per_bank_req_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
            assign per_bank_rsp_uuid[i] = per_bank_rsp_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin : g_no_uuid
            assign per_bank_req_uuid[i] = 0;
            assign per_bank_rsp_uuid[i] = 0;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_trace
        always @(posedge clk) begin
            if (lsu_bus_if[i].req_valid && lsu_bus_if[i].req_ready) begin
                if (lsu_bus_if[i].req_data.rw) begin
                    `TRACE(2, ("%t: %s core-wr-req[%0d]: addr=0x%0h, byteen=0x%h, data=0x%h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, lsu_bus_if[i].req_data.addr, lsu_bus_if[i].req_data.byteen, lsu_bus_if[i].req_data.data, lsu_bus_if[i].req_data.tag.value, lsu_bus_if[i].req_data.tag.uuid))
                end else begin
                    `TRACE(2, ("%t: %s core-rd-req[%0d]: addr=0x%0h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, lsu_bus_if[i].req_data.addr, lsu_bus_if[i].req_data.tag.value, lsu_bus_if[i].req_data.tag.uuid))
                end
            end
            if (lsu_bus_if[i].rsp_valid && lsu_bus_if[i].rsp_ready) begin
                `TRACE(2, ("%t: %s core-rd-rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n",
                    $time, INSTANCE_ID, i, lsu_bus_if[i].rsp_data.data, lsu_bus_if[i].rsp_data.tag.value, lsu_bus_if[i].rsp_data.tag.uuid))
            end
        end
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_bank_trace
        always @(posedge clk) begin
            if (per_bank_req_valid[i] && per_bank_req_ready[i]) begin
                if (per_bank_req_rw[i]) begin
                    `TRACE(2, ("%t: %s bank-wr-req[%0d]: addr=0x%0h, byteen=0x%h, data=0x%h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, per_bank_req_addr[i], per_bank_req_byteen[i], per_bank_req_data[i], per_bank_req_tag_value[i], per_bank_req_uuid[i]))
                end else begin
                    `TRACE(2, ("%t: %s bank-rd-req[%0d]: addr=0x%0h, tag=0x%0h (#%0d)\n",
                        $time, INSTANCE_ID, i, per_bank_req_addr[i], per_bank_req_tag_value[i], per_bank_req_uuid[i]))
                end
            end
            if (per_bank_rsp_valid[i] && per_bank_rsp_ready[i]) begin
                `TRACE(2, ("%t: %s bank-rd-rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n",
                    $time, INSTANCE_ID, i, per_bank_rsp_data[i], per_bank_rsp_tag_value[i], per_bank_rsp_uuid[i]))
            end
        end
    end

`endif

endmodule
