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

//`TRACING_OFF
module VX_mem_scheduler #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_REQS     = 1,
    parameter MEM_CHANNELS  = 1,
    parameter WORD_SIZE     = 4,
    parameter LINE_SIZE     = 4,
    parameter ADDR_WIDTH    = 32 - `CLOG2(WORD_SIZE),    
    parameter UUID_WIDTH    = 0, // upper section of the request tag contains the UUID
    parameter TAG_WIDTH     = 8,
    parameter QUEUE_SIZE    = 8,
    parameter RSP_PARTIAL   = 0,
    parameter CORE_OUT_BUF  = 0,
    parameter MEM_OUT_BUF   = 0,

    parameter WORD_WIDTH    = WORD_SIZE * 8,
    parameter LINE_WIDTH    = LINE_SIZE * 8,    
    parameter PER_LINE_REQS = LINE_SIZE / WORD_SIZE,
    parameter MERGED_REQS   = CORE_REQS / PER_LINE_REQS,
    parameter NUM_BATCHES   = (MERGED_REQS + MEM_CHANNELS - 1) / MEM_CHANNELS,
    parameter QUEUE_ADDRW   = `CLOG2(QUEUE_SIZE),
    parameter BATCH_SEL_BITS= `CLOG2(NUM_BATCHES),
    parameter TAG_ID_WIDTH  = TAG_WIDTH - UUID_WIDTH,
    parameter MEM_TAG_ID    = TAG_WIDTH - TAG_ID_WIDTH,
    parameter MEM_ADDR_WIDTH= ADDR_WIDTH - `CLOG2(PER_LINE_REQS),
    parameter REQQ_TAG_WIDTH= MEM_TAG_ID + QUEUE_ADDRW,
    parameter MEM_TAG_WIDTH = REQQ_TAG_WIDTH + BATCH_SEL_BITS
) (
    input wire clk,
    input wire reset,

    // Core request
    input wire                              core_req_valid,
    input wire                              core_req_rw,
    input wire [CORE_REQS-1:0]              core_req_mask,
    input wire [CORE_REQS-1:0][WORD_SIZE-1:0] core_req_byteen,
    input wire [CORE_REQS-1:0][ADDR_WIDTH-1:0] core_req_addr,
    input wire [CORE_REQS-1:0][WORD_WIDTH-1:0] core_req_data,
    input wire [TAG_WIDTH-1:0]              core_req_tag,
    output wire                             core_req_ready,
    output wire                             core_req_empty,    
    output wire                             core_req_sent,

    // Core response
    output wire                             core_rsp_valid,
    output wire [CORE_REQS-1:0]             core_rsp_mask,
    output wire [CORE_REQS-1:0][WORD_WIDTH-1:0] core_rsp_data,
    output wire [TAG_WIDTH-1:0]             core_rsp_tag,
    output wire                             core_rsp_sop,
    output wire                             core_rsp_eop,
    input wire                              core_rsp_ready,

    // Memory request
    output wire [MEM_CHANNELS-1:0]          mem_req_valid,
    output wire [MEM_CHANNELS-1:0]          mem_req_rw,
    output wire [MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen,
    output wire [MEM_CHANNELS-1:0][MEM_ADDR_WIDTH-1:0] mem_req_addr,
    output wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data,
    output wire [MEM_CHANNELS-1:0][MEM_TAG_WIDTH-1:0] mem_req_tag,
    input wire 	[MEM_CHANNELS-1:0]          mem_req_ready,

    // Memory response
    input wire [MEM_CHANNELS-1:0]           mem_rsp_valid,
    input wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_rsp_data,
    input wire [MEM_CHANNELS-1:0][MEM_TAG_WIDTH-1:0] mem_rsp_tag,    
    output wire [MEM_CHANNELS-1:0]          mem_rsp_ready
);
    localparam BATCH_SEL_WIDTH = `UP(BATCH_SEL_BITS);
    localparam STALL_TIMEOUT   = 10000000;

    `STATIC_ASSERT ((WORD_SIZE == LINE_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(CORE_REQS * WORD_SIZE, LINE_SIZE), ("invalid parameter"))
    `STATIC_ASSERT ((MEM_TAG_ID >= UUID_WIDTH), ("invalid parameter"))
    `STATIC_ASSERT ((0 == RSP_PARTIAL) || (1 == RSP_PARTIAL), ("invalid parameter"))
    `RUNTIME_ASSERT ((~core_req_valid || core_req_mask != 0), ("invalid request mask"));

    wire [MEM_CHANNELS-1:0]         mem_req_valid_s;
    wire [MEM_CHANNELS-1:0]         mem_req_mask_s;
    wire                            mem_req_rw_s;
    wire [MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen_s;
    wire [MEM_CHANNELS-1:0][MEM_ADDR_WIDTH-1:0] mem_req_addr_s;
    wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data_s;
    wire [MEM_TAG_WIDTH-1:0]        mem_req_tag_s;
    wire [MEM_CHANNELS-1:0]         mem_req_ready_s;

    wire                            mem_rsp_valid_s;
    wire [MEM_CHANNELS-1:0]         mem_rsp_mask_s;
    wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_rsp_data_s;
    wire [MEM_TAG_WIDTH-1:0]        mem_rsp_tag_s;
    wire                            mem_rsp_ready_s;
    wire                            mem_rsp_fire_s;

    wire                            reqq_valid;
    wire [CORE_REQS-1:0]            reqq_mask;
    wire                            reqq_rw;    
    wire [CORE_REQS-1:0][WORD_SIZE-1:0] reqq_byteen;
    wire [CORE_REQS-1:0][ADDR_WIDTH-1:0] reqq_addr;
    wire [CORE_REQS-1:0][WORD_WIDTH-1:0] reqq_data;
    wire [REQQ_TAG_WIDTH-1:0]       reqq_tag;
    wire                            reqq_ready;

    wire                            ibuf_push;
    wire                            ibuf_pop;
    wire [QUEUE_ADDRW-1:0]          ibuf_waddr;
    wire [QUEUE_ADDRW-1:0]          ibuf_raddr;
    wire                            ibuf_full;
    wire                            ibuf_empty;
    wire [TAG_ID_WIDTH-1:0]         ibuf_din;
    wire [TAG_ID_WIDTH-1:0]         ibuf_dout;

    wire                            crsp_valid;
    wire [CORE_REQS-1:0]            crsp_mask;
    wire [CORE_REQS-1:0][WORD_WIDTH-1:0] crsp_data;
    wire [REQQ_TAG_WIDTH-1:0]       crsp_tag;
    wire                            crsp_sop;
    wire                            crsp_eop;
    wire                            crsp_ready;

    wire                            reqq_valid_s;    
    wire [MERGED_REQS-1:0]          reqq_mask_s;
    wire                            reqq_rw_s;
    wire [MERGED_REQS-1:0][LINE_SIZE-1:0] reqq_byteen_s;
    wire [MERGED_REQS-1:0][MEM_ADDR_WIDTH-1:0] reqq_addr_s;
    wire [MERGED_REQS-1:0][LINE_WIDTH-1:0] reqq_data_s;
    wire [REQQ_TAG_WIDTH-1:0]       reqq_tag_s;
    wire                            reqq_ready_s;

    wire                            crsp_valid_s;
    wire [MERGED_REQS-1:0]          crsp_mask_s;
    wire [MERGED_REQS-1:0][LINE_WIDTH-1:0] crsp_data_s;
    wire [REQQ_TAG_WIDTH-1:0]       crsp_tag_s;
    wire                            crsp_sop_s;
    wire                            crsp_eop_s;
    wire                            crsp_ready_s;

    // Request queue //////////////////////////////////////////////////////////

    wire req_sent_all;

    wire ibuf_ready = (core_req_rw || ~ibuf_full);
    wire reqq_valid_in = core_req_valid && ibuf_ready;
    wire reqq_ready_in;
    
    wire [REQQ_TAG_WIDTH-1:0] reqq_tag_u;
    if (MEM_TAG_ID != 0) begin
        assign reqq_tag_u = {core_req_tag[TAG_WIDTH-1:TAG_ID_WIDTH], ibuf_waddr};
    end else begin
        assign reqq_tag_u = ibuf_waddr;
    end

    VX_elastic_buffer #(
        .DATAW   (1 + CORE_REQS * (1 + WORD_SIZE + ADDR_WIDTH + WORD_WIDTH) + REQQ_TAG_WIDTH),
        .SIZE	 (QUEUE_SIZE),
        .OUT_REG (1)
    ) req_queue (
        .clk      (clk),
        .reset    (reset),
        .valid_in (reqq_valid_in),
        .ready_in (reqq_ready_in),
        .data_in  ({core_req_rw, core_req_mask, core_req_byteen, core_req_addr, core_req_data, reqq_tag_u}),
        .data_out ({reqq_rw, reqq_mask, reqq_byteen, reqq_addr, reqq_data, reqq_tag}),
        .valid_out(reqq_valid),
        .ready_out(reqq_ready)
    );

    // can accept another request?
    assign core_req_ready = reqq_ready_in && ibuf_ready;

    // no pending requests
    assign core_req_empty = !reqq_valid && ibuf_empty;

    // notify request submisison 
    assign core_req_sent = reqq_valid && reqq_ready;

    // Index buffer ///////////////////////////////////////////////////////////

    assign ibuf_push  = core_req_valid && core_req_ready && ~core_req_rw;
    assign ibuf_pop   = crsp_valid && crsp_ready && crsp_eop;
    assign ibuf_raddr = crsp_tag[QUEUE_ADDRW-1:0];
    assign ibuf_din   = core_req_tag[TAG_ID_WIDTH-1:0];

    VX_index_buffer #(
        .DATAW (TAG_ID_WIDTH),
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

    wire [QUEUE_ADDRW-1:0] ibuf_waddr_s = reqq_tag_s[QUEUE_ADDRW-1:0];
    wire [QUEUE_ADDRW-1:0] ibuf_raddr_s = crsp_tag_s[QUEUE_ADDRW-1:0];
    
    assign reqq_valid_s = reqq_valid;
    assign reqq_mask_s  = reqq_mask;
    assign reqq_rw_s    = reqq_rw;        
    assign reqq_byteen_s= reqq_byteen;
    assign reqq_addr_s  = reqq_addr;        
    assign reqq_data_s  = reqq_data;
    assign reqq_tag_s   = reqq_tag;
    assign reqq_ready   = reqq_ready_s;

    assign crsp_valid   = crsp_valid_s;
    assign crsp_mask    = crsp_mask_s;
    assign crsp_data    = crsp_data_s;
    assign crsp_tag     = crsp_tag_s;
    assign crsp_sop     = crsp_sop_s;
    assign crsp_eop     = crsp_eop_s;
    assign crsp_ready_s = crsp_ready;

    // Handle memory requests /////////////////////////////////////////////////

    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0] mem_req_mask_b;
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen_b; 
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0][MEM_ADDR_WIDTH-1:0] mem_req_addr_b;
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data_b;
    
    wire [BATCH_SEL_WIDTH-1:0] req_batch_idx;

    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        for (genvar j = 0; j < MEM_CHANNELS; ++j) begin
            localparam r = i * MEM_CHANNELS + j;
            if (r < MERGED_REQS) begin
                assign mem_req_mask_b[i][j]   = reqq_mask_s[r];
                assign mem_req_byteen_b[i][j] = reqq_byteen_s[r];
                assign mem_req_addr_b[i][j]   = reqq_addr_s[r];
                assign mem_req_data_b[i][j]   = reqq_data_s[r];
            end else begin
                assign mem_req_mask_b[i][j]   = 0;
                assign mem_req_byteen_b[i][j] = '0;
                assign mem_req_addr_b[i][j]   = '0;
                assign mem_req_data_b[i][j]   = '0;
            end
        end
    end

    assign mem_req_mask_s   = mem_req_mask_b[req_batch_idx];
    assign mem_req_rw_s     = reqq_rw_s;
    assign mem_req_byteen_s = mem_req_byteen_b[req_batch_idx];
    assign mem_req_addr_s   = mem_req_addr_b[req_batch_idx];
    assign mem_req_data_s   = mem_req_data_b[req_batch_idx];

    reg [MEM_CHANNELS-1:0] batch_sent_mask;    
    wire [MEM_CHANNELS-1:0] batch_sent_mask_n = batch_sent_mask | mem_req_ready_s;    
    wire batch_sent_all = (mem_req_mask_s & ~batch_sent_mask_n) == 0;

    always @(posedge clk) begin
        if (reset) begin
            batch_sent_mask <= '0;
        end else begin
            if (reqq_valid_s) begin
                if (batch_sent_all) begin
                    batch_sent_mask <= '0;
                end else begin
                    batch_sent_mask <= batch_sent_mask_n;
                end
            end
        end
    end
    
    if (NUM_BATCHES != 1) begin
        reg [BATCH_SEL_BITS-1:0] req_batch_idx_r;
        always @(posedge clk) begin
            if (reset) begin
                req_batch_idx_r <= '0;
            end else begin
                if (reqq_valid_s && batch_sent_all) begin
                    if (req_sent_all) begin
                        req_batch_idx_r <= '0;
                    end else begin
                        req_batch_idx_r <= req_batch_idx_r + BATCH_SEL_BITS'(1);
                    end
                end
            end
        end

        wire [NUM_BATCHES-1:0] req_batch_valids;
        wire [NUM_BATCHES-1:0][BATCH_SEL_BITS-1:0] req_batch_idxs;
        wire [BATCH_SEL_BITS-1:0] req_batch_idx_last;

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin                 
            assign req_batch_valids[i] = (| mem_req_mask_b[i]);
            assign req_batch_idxs[i] = BATCH_SEL_BITS'(i);
        end  

        VX_find_first #(
            .N       (NUM_BATCHES),
            .DATAW   (BATCH_SEL_BITS),
            .REVERSE (1)
        ) find_last (
            .valid_in  (req_batch_valids),
            .data_in   (req_batch_idxs),
            .data_out  (req_batch_idx_last),
            `UNUSED_PIN (valid_out)
        );

        assign req_batch_idx = req_batch_idx_r;        
        assign req_sent_all  = batch_sent_all && (req_batch_idx_r == req_batch_idx_last);
        assign mem_req_tag_s = {reqq_tag_s, req_batch_idx};

    end else begin

        assign req_batch_idx = '0;
        assign req_sent_all  = batch_sent_all;
        assign mem_req_tag_s = reqq_tag_s;

    end

    assign mem_req_valid_s = {MEM_CHANNELS{reqq_valid_s}} & mem_req_mask_s & ~batch_sent_mask;
    assign reqq_ready_s = req_sent_all;

    for (genvar i = 0; i < MEM_CHANNELS; ++i) begin
        VX_elastic_buffer #(
            .DATAW   (1 + LINE_SIZE + MEM_ADDR_WIDTH + LINE_WIDTH + MEM_TAG_WIDTH),
            .SIZE    (`TO_OUT_BUF_SIZE(MEM_OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(MEM_OUT_BUF))
        ) mem_req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (mem_req_valid_s[i]),
            .ready_in  (mem_req_ready_s[i]),
            .data_in   ({mem_req_rw_s,  mem_req_byteen_s[i], mem_req_addr_s[i], mem_req_data_s[i], mem_req_tag_s}),
            .data_out  ({mem_req_rw[i], mem_req_byteen[i],   mem_req_addr[i],   mem_req_data[i],   mem_req_tag[i]}),
            .valid_out (mem_req_valid[i]),
            .ready_out (mem_req_ready[i])
        );
    end

    // Handle memory responses ////////////////////////////////////////////////

    reg [QUEUE_SIZE-1:0] pending_req_valids;
    reg [QUEUE_SIZE-1:0][MERGED_REQS-1:0] rsp_rem_mask;
    wire [MERGED_REQS-1:0] rsp_rem_mask_n, curr_mask;
    wire [BATCH_SEL_WIDTH-1:0] rsp_batch_idx;

    wire reqq_fire_s = reqq_valid_s && reqq_ready_s;
    wire reqq_rd_start_s = reqq_fire_s && ~reqq_rw_s && ~pending_req_valids[ibuf_waddr_s];

    // Select memory response
    VX_mem_rsp_sel #(
        .NUM_REQS     (MEM_CHANNELS),
        .DATA_WIDTH   (LINE_WIDTH),
        .TAG_WIDTH    (MEM_TAG_WIDTH),
        .TAG_SEL_BITS (MEM_TAG_WIDTH - MEM_TAG_ID),
        .OUT_BUF      (2)
    ) mem_rsp_sel (
        .clk           (clk),
        .reset         (reset),
        .rsp_valid_in  (mem_rsp_valid),
        .rsp_data_in   (mem_rsp_data),
        .rsp_tag_in    (mem_rsp_tag),
        .rsp_ready_in  (mem_rsp_ready),
        .rsp_valid_out (mem_rsp_valid_s),
        .rsp_mask_out  (mem_rsp_mask_s),
        .rsp_data_out  (mem_rsp_data_s),
        .rsp_tag_out   (mem_rsp_tag_s),
        .rsp_ready_out (mem_rsp_ready_s)
    );

    for (genvar r = 0; r < MERGED_REQS; ++r) begin
        localparam i = r / MEM_CHANNELS;
        localparam j = r % MEM_CHANNELS;
        assign curr_mask[r] = (BATCH_SEL_WIDTH'(i) == rsp_batch_idx) && mem_rsp_mask_s[j];
    end
    
    assign rsp_rem_mask_n = rsp_rem_mask[ibuf_raddr_s] & ~curr_mask;

    wire rsp_complete = ~(| rsp_rem_mask_n);

    if (NUM_BATCHES > 1) begin
        assign rsp_batch_idx = mem_rsp_tag_s[BATCH_SEL_BITS-1:0];
    end else begin
        assign rsp_batch_idx = '0;
    end

    always @(posedge clk) begin
        if (reset) begin
            pending_req_valids <= '0;
        end else begin
            if (reqq_rd_start_s) begin
                pending_req_valids[ibuf_waddr_s] <= 1;
            end
            if (mem_rsp_fire_s && rsp_complete) begin
                pending_req_valids[ibuf_raddr_s] <= 0;
            end
        end
        if (reqq_rd_start_s) begin
            rsp_rem_mask[ibuf_waddr_s] <= reqq_mask_s;
        end
        if (mem_rsp_fire_s) begin
            rsp_rem_mask[ibuf_raddr_s] <= rsp_rem_mask_n;
        end
    end

    assign mem_rsp_fire_s = mem_rsp_valid_s && mem_rsp_ready_s;

    if (RSP_PARTIAL == 1) begin

        reg [QUEUE_SIZE-1:0] rsp_sop_r;

        always @(posedge clk) begin
            if (reqq_rd_start_s) begin
                rsp_sop_r[ibuf_waddr_s] <= 1;
            end
            if (mem_rsp_fire_s) begin
                rsp_sop_r[ibuf_raddr_s] <= 0;
            end
        end

        assign mem_rsp_ready_s = crsp_ready_s;
        
        assign crsp_valid_s = mem_rsp_valid_s;

        assign crsp_mask_s = curr_mask;
        assign crsp_sop_s = rsp_sop_r[ibuf_raddr_s];

        for (genvar r = 0; r < MERGED_REQS; ++r) begin
            localparam j = r % MEM_CHANNELS;
            assign crsp_data_s[r] = mem_rsp_data_s[j];
        end

    end else begin

        reg [NUM_BATCHES*MEM_CHANNELS*LINE_WIDTH-1:0] rsp_store [QUEUE_SIZE-1:0];        
        reg [NUM_BATCHES*MEM_CHANNELS*LINE_WIDTH-1:0] rsp_store_n;        
        reg [MERGED_REQS-1:0] rsp_orig_mask [QUEUE_SIZE-1:0];    

        always @(*) begin
            rsp_store_n = rsp_store[ibuf_raddr_s];            
            for (integer i = 0; i < MEM_CHANNELS; ++i) begin
                if ((MEM_CHANNELS == 1) || mem_rsp_mask_s[i]) begin
                    rsp_store_n[(rsp_batch_idx * MEM_CHANNELS + i) * LINE_WIDTH +: LINE_WIDTH] = mem_rsp_data_s[i];
                end
            end
        end        
        
        always @(posedge clk) begin
            if (reqq_rd_start_s) begin
                rsp_orig_mask[ibuf_waddr_s] <= core_req_mask;
            end
            if (mem_rsp_valid_s) begin
                rsp_store[ibuf_raddr_s] <= rsp_store_n;
            end
        end

        assign mem_rsp_ready_s = crsp_ready_s || ~rsp_complete;      

        assign crsp_valid_s = mem_rsp_valid_s && rsp_complete;

        assign crsp_mask_s = rsp_orig_mask[ibuf_raddr_s];
        assign crsp_sop_s = 1'b1;

        for (genvar r = 0; r < MERGED_REQS; ++r) begin
            localparam i = r / MEM_CHANNELS;
            localparam j = r % MEM_CHANNELS;
            assign crsp_data_s[r] = rsp_store_n[(i * MEM_CHANNELS + j) * LINE_WIDTH +: LINE_WIDTH];
        end
    end

    assign crsp_tag_s = mem_rsp_tag_s[MEM_TAG_WIDTH-1 -: REQQ_TAG_WIDTH];

    assign crsp_eop_s = rsp_complete;

    // Send response to caller

    wire [TAG_WIDTH-1:0] crsp_tag_2;

    if (MEM_TAG_ID != 0) begin
        assign crsp_tag_2 = {crsp_tag[MEM_TAG_WIDTH-1 -: MEM_TAG_ID], ibuf_dout};
    end else begin
        assign crsp_tag_2 = ibuf_dout;
    end

    VX_elastic_buffer #(
        .DATAW   (CORE_REQS + 1 + 1 + (CORE_REQS * WORD_WIDTH) + TAG_WIDTH),
        .SIZE    (`TO_OUT_BUF_SIZE(CORE_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsp_valid),  
        .ready_in  (crsp_ready),
        .data_in   ({crsp_mask, crsp_sop, crsp_eop, crsp_data, crsp_tag_2}),
        .data_out  ({core_rsp_mask, core_rsp_sop, core_rsp_eop, core_rsp_data, core_rsp_tag}),
        .valid_out (core_rsp_valid),        
        .ready_out (core_rsp_ready)
    );

`ifdef SIMULATION
    wire [`UP(UUID_WIDTH)-1:0] req_dbg_uuid;
    wire [`UP(UUID_WIDTH)-1:0] rsp_dbg_uuid;
    wire [`UP(UUID_WIDTH)-1:0] mem_req_dbg_uuid;
    wire [`UP(UUID_WIDTH)-1:0] mem_rsp_dbg_uuid;

    if (UUID_WIDTH != 0) begin
        assign req_dbg_uuid = core_req_tag[TAG_WIDTH-1 -: UUID_WIDTH];
        assign rsp_dbg_uuid = core_rsp_tag[TAG_WIDTH-1 -: UUID_WIDTH];
        assign mem_req_dbg_uuid = reqq_tag_s[REQQ_TAG_WIDTH-1 -: UUID_WIDTH];
        assign mem_rsp_dbg_uuid = mem_rsp_tag_s[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_dbg_uuid = '0;
        assign rsp_dbg_uuid = '0;
        assign mem_req_dbg_uuid = '0;
        assign mem_rsp_dbg_uuid = '0;
    end
    
    `UNUSED_VAR (req_dbg_uuid)
    `UNUSED_VAR (rsp_dbg_uuid)
    `UNUSED_VAR (mem_req_dbg_uuid)
    `UNUSED_VAR (mem_rsp_dbg_uuid)

    reg [(`UP(UUID_WIDTH) + TAG_ID_WIDTH + 64)-1:0] pending_reqs_time [QUEUE_SIZE-1:0];

    always @(posedge clk) begin
        if (reqq_rd_start_s) begin            
            pending_reqs_time[ibuf_waddr_s] <= {req_dbg_uuid, ibuf_din, $time};
        end

        for (integer i = 0; i < QUEUE_SIZE; ++i) begin
            if (pending_req_valids[i]) begin
                `ASSERT(($time - pending_reqs_time[i][63:0]) < STALL_TIMEOUT,
                    ("%t: *** %s response timeout: remaining=%b, tag=0x%0h (#%0d)", 
                        $time, INSTANCE_ID, rsp_rem_mask[i], pending_reqs_time[i][64 +: TAG_ID_WIDTH], pending_reqs_time[i][64+TAG_ID_WIDTH +: `UP(UUID_WIDTH)]));
            end
        end
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

`ifndef NDEBUG
    wire [MEM_CHANNELS-1:0] mem_req_fire_s = mem_req_valid_s & mem_req_ready_s;
    always @(posedge clk) begin
        if (core_req_valid && core_req_ready) begin
            if (core_req_rw) begin
                `TRACE(1, ("%d: %s-core-req-wr: valid=%b, addr=", $time, INSTANCE_ID, core_req_mask));
                `TRACE_ARRAY1D(1, core_req_addr, CORE_REQS);                       
                `TRACE(1, (", byteen="));
                `TRACE_ARRAY1D(1, core_req_byteen, CORE_REQS);
                `TRACE(1, (", data="));
                `TRACE_ARRAY1D(1, core_req_data, CORE_REQS);         
            end else begin
                `TRACE(1, ("%d: %s-core-req-rd: valid=%b, addr=", $time, INSTANCE_ID, core_req_mask));
                `TRACE_ARRAY1D(1, core_req_addr, CORE_REQS);                
            end 
            `TRACE(1, (", tag=0x%0h (#%0d)\n", core_req_tag, req_dbg_uuid));           
        end
        if (core_rsp_valid && core_rsp_ready) begin
            `TRACE(1, ("%d: %s-rsp: valid=%b, sop=%b, eop=%b, data=", $time, INSTANCE_ID, core_rsp_mask, core_rsp_sop, core_rsp_eop));
            `TRACE_ARRAY1D(1, core_rsp_data, CORE_REQS);
            `TRACE(1, (", tag=0x%0h (#%0d)\n", core_rsp_tag, rsp_dbg_uuid));
        end
        if (| mem_req_fire_s) begin
            if (| mem_req_rw_s) begin
                `TRACE(1, ("%d: %s-mem-req-wr: valid=%b, addr=", $time, INSTANCE_ID, mem_req_fire_s));
                `TRACE_ARRAY1D(1, mem_req_addr_s, MEM_CHANNELS);
                `TRACE(1, (", byteen="));
                `TRACE_ARRAY1D(1, mem_req_byteen_s, MEM_CHANNELS);
                `TRACE(1, (", data="));
                `TRACE_ARRAY1D(1, mem_req_data_s, MEM_CHANNELS);           
            end else begin
                `TRACE(1, ("%d: %s-mem-req-rd: valid=%b, addr=", $time, INSTANCE_ID, mem_req_fire_s));
                `TRACE_ARRAY1D(1, mem_req_addr_s, MEM_CHANNELS);                
            end
            `TRACE(1, (", ibuf_idx=%0d, batch_idx=%0d (#%0d)\n", ibuf_waddr_s, req_batch_idx, mem_req_dbg_uuid));
        end 
        if (mem_rsp_fire_s) begin
            `TRACE(1, ("%d: %s-mem-rsp: valid=%b, data=", $time, INSTANCE_ID, mem_rsp_mask_s));                
            `TRACE_ARRAY1D(1, mem_rsp_data_s, MEM_CHANNELS);
            `TRACE(1, (", ibuf_idx=%0d, batch_idx=%0d (#%0d)\n", ibuf_raddr_s, rsp_batch_idx, mem_rsp_dbg_uuid));
        end
    end
`endif
  
endmodule
//`TRACING_ON
