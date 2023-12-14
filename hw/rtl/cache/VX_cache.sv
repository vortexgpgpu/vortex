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

`include "VX_cache_define.vh"

module VX_cache import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID    = "",

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 4096, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64, 
    // Number of banks
    parameter NUM_BANKS             = 1,
    // Number of associative ways
    parameter NUM_WAYS              = 1,
    // Size of a word in bytes
    parameter WORD_SIZE             = `XLEN/8,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 2,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 8, 
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 0,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = UUID_WIDTH + 1,

    // Core response output register
    parameter CORE_OUT_REG          = 0,

    // Memory request output register
    parameter MEM_OUT_REG           = 0
 ) (    
    // PERF
`ifdef PERF_ENABLE
    output cache_perf_t     cache_perf,
`endif
    
    input wire clk,
    input wire reset,

    VX_mem_bus_if.slave     core_bus_if [NUM_REQS],
    VX_mem_bus_if.master    mem_bus_if
);

    `STATIC_ASSERT(NUM_BANKS <= NUM_REQS, ("invalid parameter"))    
    `STATIC_ASSERT(NUM_BANKS == (1 << `CLOG2(NUM_BANKS)), ("invalid parameter"))

    localparam REQ_SEL_WIDTH   = `UP(`CS_REQ_SEL_BITS);
    localparam WORD_SEL_WIDTH  = `UP(`CS_WORD_SEL_BITS);
    localparam MSHR_ADDR_WIDTH = `LOG2UP(MSHR_SIZE);
    localparam MEM_TAG_WIDTH   = MSHR_ADDR_WIDTH + `CS_BANK_SEL_BITS;
    localparam WORDS_PER_LINE  = LINE_SIZE / WORD_SIZE;
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam WORD_SEL_BITS   = `CLOG2(WORDS_PER_LINE);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);
    localparam LINE_ADDR_WIDTH = (`CS_WORD_ADDR_WIDTH - BANK_SEL_BITS - WORD_SEL_BITS);
    localparam CORE_REQ_DATAW  = LINE_ADDR_WIDTH + 1 + WORD_SEL_WIDTH + WORD_SIZE + WORD_WIDTH + TAG_WIDTH;
    localparam CORE_RSP_DATAW  = WORD_WIDTH + TAG_WIDTH;

    localparam CORE_REQ_BUF_ENABLE = (NUM_BANKS != 1) || (NUM_REQS != 1);
    localparam MEM_REQ_BUF_ENABLE  = (NUM_BANKS != 1);

`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_read_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_write_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_mshr_stall_per_bank;
`endif

    wire [NUM_REQS-1:0]                     core_req_valid;
    wire [NUM_REQS-1:0][`CS_WORD_ADDR_WIDTH-1:0] core_req_addr;
    wire [NUM_REQS-1:0]                     core_req_rw;    
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]      core_req_byteen;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_req_data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]      core_req_tag;
    wire [NUM_REQS-1:0]                     core_req_ready;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_valid[i]  = core_bus_if[i].req_valid;
        assign core_req_addr[i]   = core_bus_if[i].req_data.addr;
        assign core_req_rw[i]     = core_bus_if[i].req_data.rw;
        assign core_req_byteen[i] = core_bus_if[i].req_data.byteen;
        assign core_req_data[i]   = core_bus_if[i].req_data.data;
        assign core_req_tag[i]    = core_bus_if[i].req_data.tag;
        assign core_bus_if[i].req_ready = core_req_ready[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    // Core response buffering
    wire [NUM_REQS-1:0]                  core_rsp_valid_s;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] core_rsp_data_s;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   core_rsp_tag_s;
    wire [NUM_REQS-1:0]                  core_rsp_ready_s;

    for (genvar i = 0; i < NUM_REQS; ++i) begin

        `RESET_RELAY (core_rsp_reset, reset);

        VX_elastic_buffer #(
            .DATAW   (`CS_WORD_WIDTH + TAG_WIDTH),
            .SIZE    (CORE_REQ_BUF_ENABLE ? `OUT_REG_TO_EB_SIZE(CORE_OUT_REG) : 0),
            .OUT_REG (`OUT_REG_TO_EB_REG(CORE_OUT_REG))
        ) core_rsp_buf (
            .clk       (clk),
            .reset     (core_rsp_reset),
            .valid_in  (core_rsp_valid_s[i]),
            .ready_in  (core_rsp_ready_s[i]),
            .data_in   ({core_rsp_data_s[i], core_rsp_tag_s[i]}),
            .data_out  ({core_bus_if[i].rsp_data.data, core_bus_if[i].rsp_data.tag}), 
            .valid_out (core_bus_if[i].rsp_valid),
            .ready_out (core_bus_if[i].rsp_ready)
        );
    end

    ///////////////////////////////////////////////////////////////////////////

    // Memory request buffering
    wire                             mem_req_valid_s;
    wire [`CS_MEM_ADDR_WIDTH-1:0]    mem_req_addr_s;
    wire                             mem_req_rw_s;
    wire [LINE_SIZE-1:0]             mem_req_byteen_s;
    wire [`CS_LINE_WIDTH-1:0]        mem_req_data_s;
    wire [MEM_TAG_WIDTH-1:0]         mem_req_tag_s;
    wire                             mem_req_ready_s;

    `RESET_RELAY (mem_req_buf_reset, reset);

    VX_elastic_buffer #(
        .DATAW   (1 + LINE_SIZE + `CS_MEM_ADDR_WIDTH + `CS_LINE_WIDTH + MEM_TAG_WIDTH),
        .SIZE    (MEM_REQ_BUF_ENABLE ? `OUT_REG_TO_EB_SIZE(MEM_OUT_REG) : 0),
        .OUT_REG (`OUT_REG_TO_EB_REG(MEM_OUT_REG))
    ) mem_req_buf (
        .clk       (clk),
        .reset     (mem_req_buf_reset),
        .valid_in  (mem_req_valid_s), 
        .ready_in  (mem_req_ready_s), 
        .data_in   ({mem_req_rw_s, mem_req_byteen_s, mem_req_addr_s, mem_req_data_s, mem_req_tag_s}),
        .data_out  ({mem_bus_if.req_data.rw, mem_bus_if.req_data.byteen, mem_bus_if.req_data.addr, mem_bus_if.req_data.data, mem_bus_if.req_data.tag}), 
        .valid_out (mem_bus_if.req_valid), 
        .ready_out (mem_bus_if.req_ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    // Memory response buffering
    wire                         mem_rsp_valid_s;
    wire [`CS_LINE_WIDTH-1:0]    mem_rsp_data_s;
    wire [MEM_TAG_WIDTH-1:0]     mem_rsp_tag_s;
    wire                         mem_rsp_ready_s;

    `RESET_RELAY (mem_rsp_reset, reset);
        
    VX_elastic_buffer #(
        .DATAW   (MEM_TAG_WIDTH + `CS_LINE_WIDTH), 
        .SIZE    (MRSQ_SIZE),
        .OUT_REG (MRSQ_SIZE > 2)
    ) mem_rsp_queue (
        .clk        (clk),
        .reset      (mem_rsp_reset),
        .valid_in   (mem_bus_if.rsp_valid),
        .ready_in   (mem_bus_if.rsp_ready),
        .data_in    ({mem_bus_if.rsp_data.tag, mem_bus_if.rsp_data.data}), 
        .data_out   ({mem_rsp_tag_s, mem_rsp_data_s}), 
        .valid_out  (mem_rsp_valid_s),
        .ready_out  (mem_rsp_ready_s)
    );

    ///////////////////////////////////////////////////////////////////////

    wire [`CS_LINE_SEL_BITS-1:0] init_line_sel;
    wire init_enable;

    `RESET_RELAY (init_reset, reset);

    VX_cache_init #( 
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE), 
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS)
    ) cache_init (
        .clk       (clk),
        .reset     (init_reset),
        .addr_out  (init_line_sel),
        .valid_out (init_enable)
    );

    ///////////////////////////////////////////////////////////////////////
    
    wire [NUM_BANKS-1:0]                        per_bank_core_req_valid;
    wire [NUM_BANKS-1:0][`CS_LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_rw;
    wire [NUM_BANKS-1:0][WORD_SEL_WIDTH-1:0]    per_bank_core_req_wsel;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]         per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_core_req_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]         per_bank_core_req_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     per_bank_core_req_idx;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_ready;
    
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0]         per_bank_core_rsp_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     per_bank_core_rsp_idx;
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_ready;

    wire [NUM_BANKS-1:0]                        per_bank_mem_req_valid;    
    wire [NUM_BANKS-1:0][`CS_MEM_ADDR_WIDTH-1:0] per_bank_mem_req_addr;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_rw;
    wire [NUM_BANKS-1:0][WORD_SEL_WIDTH-1:0]    per_bank_mem_req_wsel;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]         per_bank_mem_req_byteen;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_mem_req_data;
    wire [NUM_BANKS-1:0][MSHR_ADDR_WIDTH-1:0]   per_bank_mem_req_id;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_ready;

    wire [NUM_BANKS-1:0]                        per_bank_mem_rsp_ready;
    
    if (NUM_BANKS == 1) begin
        assign mem_rsp_ready_s = per_bank_mem_rsp_ready;
    end else begin
        assign mem_rsp_ready_s = per_bank_mem_rsp_ready[`CS_MEM_TAG_TO_BANK_ID(mem_rsp_tag_s)];
    end

    // Bank requests dispatch

    wire [NUM_REQS-1:0][CORE_REQ_DATAW-1:0]  core_req_data_in;    
    wire [NUM_BANKS-1:0][CORE_REQ_DATAW-1:0] core_req_data_out;
    wire [NUM_REQS-1:0][LINE_ADDR_WIDTH-1:0] core_req_line_addr;
    wire [NUM_REQS-1:0][BANK_SEL_WIDTH-1:0]  core_req_bid;
    wire [NUM_REQS-1:0][WORD_SEL_WIDTH-1:0]  core_req_wsel;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (WORDS_PER_LINE > 1) begin
            assign core_req_wsel[i] = core_req_addr[i][0 +: WORD_SEL_BITS];
        end else begin
            assign core_req_wsel[i] = '0;
        end
        assign core_req_line_addr[i] = core_req_addr[i][(BANK_SEL_BITS + WORD_SEL_BITS) +: LINE_ADDR_WIDTH];
    end

    if (NUM_BANKS > 1) begin
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign core_req_bid[i] = core_req_addr[i][WORD_SEL_BITS +: BANK_SEL_BITS];
        end
    end else begin
        assign core_req_bid = '0;
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign core_req_data_in[i] = {
            core_req_line_addr[i],
            core_req_rw[i],
            core_req_wsel[i],
            core_req_byteen[i],            
            core_req_data[i],
            core_req_tag[i]};
    end

`ifdef PERF_ENABLE
    wire [`PERF_CTR_BITS-1:0] perf_collisions;
`endif

    `RESET_RELAY (req_xbar_reset, reset);

     VX_stream_xbar #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (CORE_REQ_DATAW),
        .PERF_CTR_BITS (`PERF_CTR_BITS)
    ) req_xbar (
        .clk       (clk),
        .reset     (req_xbar_reset),
    `ifdef PERF_ENABLE
        .collisions(perf_collisions),
    `else
        `UNUSED_PIN(collisions),
    `endif
        .valid_in  (core_req_valid),
        .data_in   (core_req_data_in),
        .sel_in    (core_req_bid),
        .ready_in  (core_req_ready),
        .valid_out (per_bank_core_req_valid),
        .data_out  (core_req_data_out),
        .sel_out   (per_bank_core_req_idx),
        .ready_out (per_bank_core_req_ready)
    );

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign {
            per_bank_core_req_addr[i],
            per_bank_core_req_rw[i],
            per_bank_core_req_wsel[i],
            per_bank_core_req_byteen[i],            
            per_bank_core_req_data[i],
            per_bank_core_req_tag[i]} = core_req_data_out[i];
    end
    
    // Banks access
    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        wire [`CS_LINE_ADDR_WIDTH-1:0] curr_bank_mem_req_addr;
        wire curr_bank_mem_rsp_valid;

        if (NUM_BANKS == 1) begin
            assign curr_bank_mem_rsp_valid = mem_rsp_valid_s;
        end else begin
            assign curr_bank_mem_rsp_valid = mem_rsp_valid_s && (`CS_MEM_TAG_TO_BANK_ID(mem_rsp_tag_s) == i);
        end

        `RESET_RELAY (bank_reset, reset);
        
        VX_cache_bank #(                
            .BANK_ID      (i),
            .INSTANCE_ID  (INSTANCE_ID),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .WORD_SIZE    (WORD_SIZE),
            .NUM_REQS     (NUM_REQS),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .WRITE_ENABLE (WRITE_ENABLE),
            .UUID_WIDTH   (UUID_WIDTH),
            .TAG_WIDTH    (TAG_WIDTH),
            .CORE_OUT_REG (CORE_REQ_BUF_ENABLE ? 0 : CORE_OUT_REG),
            .MEM_OUT_REG  (MEM_REQ_BUF_ENABLE ? 0 : MEM_OUT_REG)
        ) bank (          
            .clk                (clk),
            .reset              (bank_reset),

        `ifdef PERF_ENABLE
            .perf_read_misses   (perf_read_miss_per_bank[i]),
            .perf_write_misses  (perf_write_miss_per_bank[i]),
            .perf_mshr_stalls   (perf_mshr_stall_per_bank[i]),
        `endif
                    
            // Core request
            .core_req_valid     (per_bank_core_req_valid[i]),
            .core_req_addr      (per_bank_core_req_addr[i]),
            .core_req_rw        (per_bank_core_req_rw[i]),
            .core_req_wsel      (per_bank_core_req_wsel[i]),
            .core_req_byteen    (per_bank_core_req_byteen[i]),
            .core_req_data      (per_bank_core_req_data[i]),
            .core_req_tag       (per_bank_core_req_tag[i]),
            .core_req_idx       (per_bank_core_req_idx[i]),
            .core_req_ready     (per_bank_core_req_ready[i]),

            // Core response                
            .core_rsp_valid     (per_bank_core_rsp_valid[i]),
            .core_rsp_data      (per_bank_core_rsp_data[i]),
            .core_rsp_tag       (per_bank_core_rsp_tag[i]),
            .core_rsp_idx       (per_bank_core_rsp_idx[i]),
            .core_rsp_ready     (per_bank_core_rsp_ready[i]),

            // Memory request
            .mem_req_valid      (per_bank_mem_req_valid[i]),
            .mem_req_addr       (curr_bank_mem_req_addr),
            .mem_req_rw         (per_bank_mem_req_rw[i]),
            .mem_req_wsel       (per_bank_mem_req_wsel[i]),
            .mem_req_byteen     (per_bank_mem_req_byteen[i]),
            .mem_req_data       (per_bank_mem_req_data[i]),
            .mem_req_id         (per_bank_mem_req_id[i]),
            .mem_req_ready      (per_bank_mem_req_ready[i]),

            // Memory response
            .mem_rsp_valid      (curr_bank_mem_rsp_valid),
            .mem_rsp_data       (mem_rsp_data_s),
            .mem_rsp_id         (`CS_MEM_TAG_TO_REQ_ID(mem_rsp_tag_s)),
            .mem_rsp_ready      (per_bank_mem_rsp_ready[i]),

            // initialization    
            .init_enable        (init_enable),
            .init_line_sel      (init_line_sel)
        );

        if (NUM_BANKS == 1) begin
            assign per_bank_mem_req_addr[i] = curr_bank_mem_req_addr;
        end else begin
            assign per_bank_mem_req_addr[i] = `CS_LINE_TO_MEM_ADDR(curr_bank_mem_req_addr, i);
        end
    end   

    // Bank responses gather

    wire [NUM_BANKS-1:0][CORE_RSP_DATAW-1:0] core_rsp_data_in;
    wire [NUM_REQS-1:0][CORE_RSP_DATAW-1:0]  core_rsp_data_out;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign core_rsp_data_in[i] = {per_bank_core_rsp_data[i], per_bank_core_rsp_tag[i]};
    end

    `RESET_RELAY (rsp_xbar_reset, reset);

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (CORE_RSP_DATAW)
    ) rsp_xbar (
        .clk       (clk),
        .reset     (rsp_xbar_reset),
        `UNUSED_PIN (collisions),
        .valid_in  (per_bank_core_rsp_valid),
        .data_in   (core_rsp_data_in),
        .sel_in    (per_bank_core_rsp_idx),
        .ready_in  (per_bank_core_rsp_ready),
        .valid_out (core_rsp_valid_s),
        .data_out  (core_rsp_data_out),
        .ready_out (core_rsp_ready_s),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign {core_rsp_data_s[i], core_rsp_tag_s[i]} = core_rsp_data_out[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    wire                        mem_req_valid_p;
    wire [`CS_MEM_ADDR_WIDTH-1:0] mem_req_addr_p;
    wire                        mem_req_rw_p;
    wire [WORD_SEL_WIDTH-1:0]   mem_req_wsel_p;
    wire [WORD_SIZE-1:0]        mem_req_byteen_p;
    wire [`CS_WORD_WIDTH-1:0]   mem_req_data_p;
    wire [MEM_TAG_WIDTH-1:0]    mem_req_tag_p;
    wire [MSHR_ADDR_WIDTH-1:0]  mem_req_id_p;
    wire                        mem_req_ready_p;

    // Memory request arbitration

    wire [NUM_BANKS-1:0][(`CS_MEM_ADDR_WIDTH + MSHR_ADDR_WIDTH + 1 + WORD_SIZE + WORD_SEL_WIDTH + `CS_WORD_WIDTH)-1:0] data_in;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign data_in[i] = {per_bank_mem_req_addr[i],
                             per_bank_mem_req_rw[i],
                             per_bank_mem_req_wsel[i], 
                             per_bank_mem_req_byteen[i],                              
                             per_bank_mem_req_data[i],
                             per_bank_mem_req_id[i]};
    end

    `RESET_RELAY (mem_req_arb_reset, reset);

    VX_stream_arb #(
        .NUM_INPUTS (NUM_BANKS),
        .DATAW      (`CS_MEM_ADDR_WIDTH + 1  + WORD_SEL_WIDTH + WORD_SIZE + `CS_WORD_WIDTH + MSHR_ADDR_WIDTH),
        .ARBITER    ("R")
    ) mem_req_arb (
        .clk       (clk),
        .reset     (mem_req_arb_reset),
        .valid_in  (per_bank_mem_req_valid),
        .ready_in  (per_bank_mem_req_ready),
        .data_in   (data_in),
        .data_out  ({mem_req_addr_p, mem_req_rw_p, mem_req_wsel_p, mem_req_byteen_p, mem_req_data_p, mem_req_id_p}),
        .valid_out (mem_req_valid_p),
        .ready_out (mem_req_ready_p),
        `UNUSED_PIN (sel_out)
    );

    if (NUM_BANKS > 1) begin
        wire [`CS_BANK_SEL_BITS-1:0] mem_req_bank_id = `CS_MEM_ADDR_TO_BANK_ID(mem_req_addr_p);
        assign mem_req_tag_p = MEM_TAG_WIDTH'({mem_req_bank_id, mem_req_id_p});            
    end else begin
        assign mem_req_tag_p = MEM_TAG_WIDTH'(mem_req_id_p);
    end   

    // Memory request multi-port handling

    assign mem_req_valid_s = mem_req_valid_p;
    assign mem_req_addr_s  = mem_req_addr_p;
    assign mem_req_tag_s   = mem_req_tag_p;
    assign mem_req_ready_p = mem_req_ready_s;

    if (WRITE_ENABLE != 0) begin
        if (`CS_WORDS_PER_LINE > 1) begin
            reg [LINE_SIZE-1:0]      mem_req_byteen_r;
            reg [`CS_LINE_WIDTH-1:0] mem_req_data_r;

            always @(*) begin
                mem_req_byteen_r = '0;
                mem_req_data_r   = 'x;
                mem_req_byteen_r[mem_req_wsel_p * WORD_SIZE +: WORD_SIZE] = mem_req_byteen_p;
                mem_req_data_r[mem_req_wsel_p * `CS_WORD_WIDTH +: `CS_WORD_WIDTH] = mem_req_data_p;
            end            
            assign mem_req_rw_s     = mem_req_rw_p;
            assign mem_req_byteen_s = mem_req_byteen_r;
            assign mem_req_data_s   = mem_req_data_r;
        end else begin
            `UNUSED_VAR (mem_req_wsel_p)
            assign mem_req_rw_s     = mem_req_rw_p;
            assign mem_req_byteen_s = mem_req_byteen_p;            
            assign mem_req_data_s   = mem_req_data_p;            
        end
    end else begin
        `UNUSED_VAR (mem_req_byteen_p)
        `UNUSED_VAR (mem_req_wsel_p)
        `UNUSED_VAR (mem_req_data_p)
        `UNUSED_VAR (mem_req_rw_p)
        
        assign mem_req_rw_s     = 0;
        assign mem_req_byteen_s = {LINE_SIZE{1'b1}};
        assign mem_req_data_s   = '0;
    end

`ifdef PERF_ENABLE
    // per cycle: core_reads, core_writes
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_core_reads_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_core_writes_per_cycle;
    
    wire [NUM_REQS-1:0] perf_core_reads_per_req;
    wire [NUM_REQS-1:0] perf_core_writes_per_req;
        
    // per cycle: read misses, write misses, msrq stalls, pipeline stalls
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_read_miss_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_write_miss_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_mshr_stall_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_crsp_stall_per_cycle;

    `BUFFER(perf_core_reads_per_req, core_req_valid & core_req_ready & ~core_req_rw);
    `BUFFER(perf_core_writes_per_req, core_req_valid & core_req_ready & core_req_rw);
    
    `POP_COUNT(perf_core_reads_per_cycle, perf_core_reads_per_req);
    `POP_COUNT(perf_core_writes_per_cycle, perf_core_writes_per_req);
    `POP_COUNT(perf_read_miss_per_cycle, perf_read_miss_per_bank);
    `POP_COUNT(perf_write_miss_per_cycle, perf_write_miss_per_bank);
    `POP_COUNT(perf_mshr_stall_per_cycle, perf_mshr_stall_per_bank);
    
    wire [NUM_REQS-1:0] perf_crsp_stall_per_req;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign perf_crsp_stall_per_req[i] = core_bus_if[i].rsp_valid && ~core_bus_if[i].rsp_ready;
    end

    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);

    wire perf_mem_stall_per_cycle = mem_bus_if.req_valid && ~mem_bus_if.req_ready;

    reg [`PERF_CTR_BITS-1:0] perf_core_reads;
    reg [`PERF_CTR_BITS-1:0] perf_core_writes;
    reg [`PERF_CTR_BITS-1:0] perf_read_misses;
    reg [`PERF_CTR_BITS-1:0] perf_write_misses;
    reg [`PERF_CTR_BITS-1:0] perf_mshr_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_mem_stalls;
    reg [`PERF_CTR_BITS-1:0] perf_crsp_stalls;    

    always @(posedge clk) begin
        if (reset) begin
            perf_core_reads   <= '0;
            perf_core_writes  <= '0;
            perf_read_misses  <= '0;
            perf_write_misses <= '0;
            perf_mshr_stalls  <= '0;
            perf_mem_stalls   <= '0;
            perf_crsp_stalls  <= '0;
        end else begin
            perf_core_reads   <= perf_core_reads   + `PERF_CTR_BITS'(perf_core_reads_per_cycle);
            perf_core_writes  <= perf_core_writes  + `PERF_CTR_BITS'(perf_core_writes_per_cycle);
            perf_read_misses  <= perf_read_misses  + `PERF_CTR_BITS'(perf_read_miss_per_cycle);
            perf_write_misses <= perf_write_misses + `PERF_CTR_BITS'(perf_write_miss_per_cycle);
            perf_mshr_stalls  <= perf_mshr_stalls  + `PERF_CTR_BITS'(perf_mshr_stall_per_cycle);
            perf_mem_stalls   <= perf_mem_stalls   + `PERF_CTR_BITS'(perf_mem_stall_per_cycle);
            perf_crsp_stalls  <= perf_crsp_stalls  + `PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end

    assign cache_perf.reads        = perf_core_reads;
    assign cache_perf.writes       = perf_core_writes;
    assign cache_perf.read_misses  = perf_read_misses;
    assign cache_perf.write_misses = perf_write_misses;
    assign cache_perf.bank_stalls  = perf_collisions;
    assign cache_perf.mshr_stalls  = perf_mshr_stalls;
    assign cache_perf.mem_stalls   = perf_mem_stalls;
    assign cache_perf.crsp_stalls  = perf_crsp_stalls;
`endif

endmodule
