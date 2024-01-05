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

module VX_shared_mem import VX_gpu_pkg::*; #(
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

    // Request debug identifier
    parameter UUID_WIDTH        = 0,

    // Request tag size
    parameter TAG_WIDTH         = 16
 ) (    
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    output cache_perf_t cache_perf,
`endif

    // Core request    
    input wire [NUM_REQS-1:0]                   req_valid,
    input wire [NUM_REQS-1:0]                   req_rw,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]   req_addr,
    input wire [NUM_REQS-1:0][WORD_SIZE-1:0]    req_byteen,
    input wire [NUM_REQS-1:0][WORD_SIZE*8-1:0]  req_data,
    input wire [NUM_REQS-1:0][TAG_WIDTH-1:0]    req_tag,
    output wire [NUM_REQS-1:0]                  req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                  rsp_valid,
    output wire [NUM_REQS-1:0][WORD_SIZE*8-1:0] rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   rsp_tag,
    input  wire [NUM_REQS-1:0]                  rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (UUID_WIDTH)

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
    if (NUM_BANKS > 1) begin
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign req_bank_idx[i] = req_addr[i][0 +: BANK_SEL_BITS];
        end
    end else begin
        assign req_bank_idx = 0;
    end   

    // bank addressing

    wire [NUM_REQS-1:0][BANK_ADDR_WIDTH-1:0] req_bank_addr;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign req_bank_addr[i] = req_addr[i][BANK_SEL_BITS +: BANK_ADDR_WIDTH];
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

    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;    
    wire [NUM_BANKS-1:0][REQ_DATAW-1:0] req_data_out;

`ifdef PERF_ENABLE
    wire [`PERF_CTR_BITS-1:0] perf_collisions;
`endif

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign req_data_in[i] = {
            req_rw[i],        
            req_bank_addr[i],
            req_byteen[i],
            req_data[i],
            req_tag[i]};
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (REQ_DATAW),
        .PERF_CTR_BITS (`PERF_CTR_BITS),
        .OUT_REG     (3) // output should be registered for the data_store addressing
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
    `ifdef PERF_ENABLE
        .collisions (perf_collisions),
    `else
        `UNUSED_PIN (collisions),
    `endif
        .valid_in  (req_valid),
        .data_in   (req_data_in),
        .sel_in    (req_bank_idx),
        .ready_in  (req_ready),
        .valid_out (per_bank_req_valid),
        .data_out  (req_data_out),
        .sel_out   (per_bank_req_idx),
        .ready_out (per_bank_req_ready)
    );

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign {
            per_bank_req_rw[i],        
            per_bank_req_addr[i],
            per_bank_req_byteen[i],        
            per_bank_req_data[i],        
            per_bank_req_tag[i]} = req_data_out[i];
    end

    // banks access

    wire [NUM_BANKS-1:0]                per_bank_rsp_valid;
    wire [NUM_BANKS-1:0][WORD_WIDTH-1:0] per_bank_rsp_data;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] per_bank_rsp_idx; 
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0] per_bank_rsp_tag;   
    wire [NUM_BANKS-1:0]                per_bank_rsp_ready;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin        
        VX_sp_ram #(
            .DATAW (WORD_WIDTH),
            .SIZE  (WORDS_PER_BANK),
            .WRENW (WORD_SIZE)
        ) data_store (
            .clk   (clk),
            .read  (1'b1),
            .write (per_bank_req_valid[i] && per_bank_req_ready[i] && per_bank_req_rw[i]),
            .wren  (per_bank_req_byteen[i]),
            .addr  (per_bank_req_addr[i]),            
            .wdata (per_bank_req_data[i]),
            .rdata (per_bank_rsp_data[i])
        );

        // drop write response
        wire per_bank_req_valid_w, per_bank_req_ready_w;
        assign per_bank_req_valid_w = per_bank_req_valid[i] && ~per_bank_req_rw[i];
        assign per_bank_req_ready[i] = per_bank_req_ready_w || per_bank_req_rw[i];

        VX_elastic_buffer #(
            .DATAW (REQ_SEL_WIDTH + TAG_WIDTH),
            .SIZE  (0)
        ) bank_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (per_bank_req_valid_w),
            .ready_in  (per_bank_req_ready_w),
            .data_in   ({per_bank_req_idx[i], per_bank_req_tag[i]}),
            .data_out  ({per_bank_rsp_idx[i], per_bank_rsp_tag[i]}),
            .valid_out (per_bank_rsp_valid[i]),
            .ready_out (per_bank_rsp_ready[i])
        );
    end

    // bank responses gather

    wire [NUM_BANKS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_out;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign rsp_data_in[i] = {per_bank_rsp_data[i], per_bank_rsp_tag[i]};
    end

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (RSP_DATAW),
        .OUT_REG     (2)
    ) rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (collisions),
        .sel_in    (per_bank_rsp_idx),
        .valid_in  (per_bank_rsp_valid),
        .ready_in  (per_bank_rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (rsp_data_out),
        .valid_out (rsp_valid),
        .ready_out (rsp_ready),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign {rsp_data[i], rsp_tag[i]} = rsp_data_out[i];
    end

`ifdef PERF_ENABLE
    // per cycle: reads, writes
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_reads_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_writes_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_crsp_stall_per_cycle;

    wire [NUM_REQS-1:0] perf_reads_per_req, perf_writes_per_req;
    wire [NUM_REQS-1:0] perf_crsp_stall_per_req = rsp_valid & ~rsp_ready;

    `BUFFER(perf_reads_per_req, req_valid & req_ready & ~req_rw);
    `BUFFER(perf_writes_per_req, req_valid & req_ready & req_rw);

    `POP_COUNT(perf_reads_per_cycle, perf_reads_per_req);
    `POP_COUNT(perf_writes_per_cycle, perf_writes_per_req);
    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);

    reg [`PERF_CTR_BITS-1:0] perf_reads;
    reg [`PERF_CTR_BITS-1:0] perf_writes;
    reg [`PERF_CTR_BITS-1:0] perf_crsp_stalls;

    always @(posedge clk) begin
        if (reset) begin
            perf_reads       <= '0;
            perf_writes      <= '0;
            perf_crsp_stalls <= '0;
        end else begin
            perf_reads       <= perf_reads  + `PERF_CTR_BITS'(perf_reads_per_cycle);
            perf_writes      <= perf_writes + `PERF_CTR_BITS'(perf_writes_per_cycle);
            perf_crsp_stalls <= perf_crsp_stalls + `PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign cache_perf.reads        = perf_reads;
    assign cache_perf.writes       = perf_writes;
    assign cache_perf.read_misses  = '0;
    assign cache_perf.write_misses = '0;
    assign cache_perf.bank_stalls  = perf_collisions;
    assign cache_perf.mshr_stalls  = '0;
    assign cache_perf.mem_stalls   = '0;
    assign cache_perf.crsp_stalls  = perf_crsp_stalls;

`endif

`ifdef DBG_TRACE_CACHE_BANK

    wire [NUM_REQS-1:0][`UP(UUID_WIDTH)-1:0] req_uuid;
    wire [NUM_REQS-1:0][`UP(UUID_WIDTH)-1:0] rsp_uuid;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (UUID_WIDTH != 0) begin
            assign req_uuid[i] = req_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
            assign rsp_uuid[i] = rsp_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin
            assign req_uuid[i] = 0;
            assign rsp_uuid[i] = 0;
        end
    end

    wire [NUM_BANKS-1:0][`UP(UUID_WIDTH)-1:0] per_bank_req_uuid;
    wire [NUM_BANKS-1:0][`UP(UUID_WIDTH)-1:0] per_bank_rsp_uuid;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        if (UUID_WIDTH != 0) begin
            assign per_bank_req_uuid[i] = per_bank_req_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
            assign per_bank_rsp_uuid[i] = per_bank_rsp_tag[i][TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin
            assign per_bank_req_uuid[i] = 0;
            assign per_bank_rsp_uuid[i] = 0;
        end
    end

    always @(posedge clk) begin        
        for (integer i = 0; i < NUM_REQS; ++i) begin
            if (req_valid[i] && req_ready[i]) begin
                if (req_rw[i]) begin
                    `TRACE(1, ("%d: %s wr-req: req_idx=%0d, addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", 
                        $time, INSTANCE_ID, i, req_addr[i], req_tag[i], req_byteen[i], req_data[i], req_uuid[i]));
                end else begin
                    `TRACE(1, ("%d: %s rd-req: req_idx=%0d, addr=0x%0h, tag=0x%0h (#%0d)\n", 
                        $time, INSTANCE_ID, i, req_addr[i], req_tag[i], req_uuid[i]));
                end
            end
            if (rsp_valid[i] && rsp_ready[i]) begin
                `TRACE(1, ("%d: %s rd-rsp: req_idx=%0d, tag=0x%0h, data=0x%0h (#%0d)\n", 
                    $time, INSTANCE_ID, i, rsp_tag[i], rsp_data[i], rsp_uuid[i]));
            end
        end

        for (integer i = 0; i < NUM_BANKS; ++i) begin
            if (per_bank_req_valid[i] && per_bank_req_ready[i]) begin
                if (per_bank_req_rw[i]) begin
                    `TRACE(2, ("%d: %s-bank%0d wr-req: addr=0x%0h, tag=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", 
                        $time, INSTANCE_ID, i, per_bank_req_addr[i], per_bank_req_tag[i], per_bank_req_byteen[i], per_bank_req_data[i], per_bank_req_uuid[i]));
                end else begin
                    `TRACE(2, ("%d: %s-bank%0d rd-req: addr=0x%0h, tag=0x%0h (#%0d)\n", 
                        $time, INSTANCE_ID, i, per_bank_req_addr[i], per_bank_req_tag[i], per_bank_req_uuid[i]));
                end
            end
            if (per_bank_rsp_valid[i] && per_bank_rsp_ready[i]) begin
                `TRACE(2, ("%d: %s-bank%0d rd-rsp: tag=0x%0h, data=0x%0h (#%0d)\n", 
                    $time, INSTANCE_ID, i, per_bank_rsp_tag[i], per_bank_rsp_data[i], per_bank_rsp_uuid[i]));
            end
        end
    end

`endif

endmodule
