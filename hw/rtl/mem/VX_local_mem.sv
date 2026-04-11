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
    parameter WORD_SIZE         = `XLEN/8,

    // Request tag size
    parameter TAG_WIDTH         = 16,

    // Enable DMA port
    parameter DMA_ENABLE        = 0,
    parameter DMA_TAG_WIDTH     = 1,

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
    localparam REQ_DATAW       = 1 + BANK_ADDR_WIDTH + WORD_SIZE + WORD_WIDTH + TAG_WIDTH;
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
        `UNUSED_VAR (lsu_bus_if[i].req_data.flags)
    end

    // bank requests dispatch

    wire [NUM_BANKS-1:0]                    per_bank_req_valid;
    wire [NUM_BANKS-1:0]                    per_bank_req_rw;
    wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0] per_bank_req_addr;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]     per_bank_req_byteen;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]    per_bank_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]     per_bank_req_tag;
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
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_rsp_idx;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]  per_bank_rsp_tag;
    wire [NUM_BANKS-1:0]                 per_bank_rsp_ready;

    // DMA port handshake
    //   rw=0 reads  : accepted when the response pipe-buffer has space.
    //   rw=1 writes : always accepted; no response issued.
    //   DMA has priority over LSU at every bank SRAM.

    wire dma_rsp_buf_ready; // driven by pipe-buffer or tied 0 when disabled

    if (DMA_ENABLE) begin : g_dma_enable
        `UNUSED_VAR (dma_bus_if.req_data.flags)

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

        // Pack all bank SRAM outputs into the read response
        for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_dma_rsp_data
            assign dma_bus_if.rsp_data.data[i*WORD_WIDTH +: WORD_WIDTH] = per_bank_rsp_data[i];
        end

    end else begin : g_no_dma
        assign dma_rsp_buf_ready    = 1'b0;
        assign dma_bus_if.req_ready = 1'b0;
        assign dma_bus_if.rsp_valid = 1'b0;
        assign dma_bus_if.rsp_data  = '0;
        `UNUSED_VAR (dma_bus_if.req_valid)
        `UNUSED_VAR (dma_bus_if.req_data)
        `UNUSED_VAR (dma_bus_if.rsp_ready)
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_data_store
        wire bank_rsp_valid, bank_rsp_ready;

        // DMA active signals (priority over LSU)
        wire dma_wr_b = DMA_ENABLE
                     && dma_bus_if.req_valid
                     && dma_bus_if.req_data.rw
                     && (|dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]);

        wire dma_rd_b = DMA_ENABLE
                     && dma_bus_if.req_valid
                     && ~dma_bus_if.req_data.rw
                     && dma_rsp_buf_ready;

        wire dma_active = dma_wr_b | dma_rd_b;

        // SRAM address / write-data / write-enable mux (DMA has priority)

        wire [BANK_ADDR_WIDTH-1:0] bank_sram_addr;
        wire [WORD_WIDTH-1:0]      bank_sram_wdata;
        wire [WORD_SIZE-1:0]       bank_sram_wren;

        assign bank_sram_addr  = dma_active ? BANK_ADDR_WIDTH'(dma_bus_if.req_data.addr)
                                            : per_bank_req_addr[i];
        assign bank_sram_wdata = dma_wr_b   ? dma_bus_if.req_data.data[i*WORD_WIDTH +: WORD_WIDTH]
                                            : per_bank_req_data[i];
        assign bank_sram_wren  = dma_wr_b   ? dma_bus_if.req_data.byteen[i*WORD_SIZE +: WORD_SIZE]
                                            : per_bank_req_byteen[i];

        wire lsu_active = per_bank_req_valid[i] && per_bank_req_ready[i];

        VX_sp_ram #(
            .DATAW (WORD_WIDTH),
            .SIZE  (WORDS_PER_BANK),
            .WRENW (WORD_SIZE),
            .OUT_REG (1),
            .RDW_MODE ("R")
        ) lmem_store (
            .clk   (clk),
            .reset (reset),
            .read  (dma_rd_b || (lsu_active && ~per_bank_req_rw[i])),
            .write (dma_wr_b || (lsu_active &&  per_bank_req_rw[i])),
            .wren  (bank_sram_wren),
            .addr  (bank_sram_addr),
            .wdata (bank_sram_wdata),
            .rdata (per_bank_rsp_data[i])
        );

        // Read-during-write hazard: stalls LSU reads to an address written last cycle
        // (SRAM OUT_REG + RDW_MODE="R" returns stale data on same-cycle read-after-write).
        // DMA reads bypass this check.

        reg [BANK_ADDR_WIDTH-1:0] last_wr_addr;
        reg last_wr_valid;
        always @(posedge clk) begin
            if (reset) begin
                last_wr_valid <= 0;
            end else begin
                last_wr_valid <= dma_wr_b || (lsu_active && per_bank_req_rw[i]);
            end
            last_wr_addr <= bank_sram_addr;
        end
        wire is_rdw_hazard = last_wr_valid && ~per_bank_req_rw[i] && (per_bank_req_addr[i] == last_wr_addr);

        // LSU response valid / request ready — blocked by DMA and RDW hazards

        assign bank_rsp_valid = per_bank_req_valid[i]
                             && ~dma_active
                             && ~per_bank_req_rw[i]
                             && ~is_rdw_hazard;

        assign per_bank_req_ready[i] = ~dma_active
                                    && (bank_rsp_ready || per_bank_req_rw[i])
                                    && ~is_rdw_hazard;

        // Delay tag/idx to align with SRAM 1-cycle output latency
        VX_pipe_buffer #(
            .DATAW (REQ_SEL_WIDTH + TAG_WIDTH)
        ) bram_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (bank_rsp_valid),
            .ready_in  (bank_rsp_ready),
            .data_in   ({per_bank_req_idx[i], per_bank_req_tag[i]}),
            .data_out  ({per_bank_rsp_idx[i], per_bank_rsp_tag[i]}),
            .valid_out (per_bank_rsp_valid[i]),
            .ready_out (per_bank_rsp_ready[i])
        );
    end

    // bank responses gather

    wire [NUM_BANKS-1:0][RSP_DATAW-1:0] per_bank_rsp_data_aos;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_rsp_data_aos
        assign per_bank_rsp_data_aos[i] = {per_bank_rsp_data[i], per_bank_rsp_tag[i]};
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
