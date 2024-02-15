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

`TRACING_OFF
module VX_mem_scheduler #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_REQS    = 1,
    parameter MEM_CHANNELS = 1,
    parameter ADDR_WIDTH   = 32,
    parameter WORD_SIZE    = 4,
    parameter LINE_SIZE    = 4,
    parameter TAG_WIDTH    = 8,
    parameter TAG_ID_WIDTH = 0, // lower section of the request tag contains the tag identifier
    parameter UUID_WIDTH   = 0, // upper section of the request tag contains the UUID
    parameter QUEUE_SIZE   = 8,
    parameter RSP_PARTIAL  = 0,
    parameter CORE_OUT_BUF = 0,
    parameter MEM_OUT_BUF  = 0,

    parameter WORD_WIDTH   = WORD_SIZE * 8,
    parameter LINE_WIDTH   = LINE_SIZE * 8,
    parameter MEM_REQS     = (CORE_REQS * WORD_SIZE) / LINE_SIZE,
    parameter NUM_BATCHES  = (MEM_REQS + MEM_CHANNELS - 1) / MEM_CHANNELS,
    parameter QUEUE_ADDRW  = `CLOG2(QUEUE_SIZE),
    parameter BATCH_SEL_BITS = `CLOG2(NUM_BATCHES),
    parameter MEM_TAG_ID   = TAG_WIDTH - TAG_ID_WIDTH,
    parameter MEM_TAGW     = MEM_TAG_ID + QUEUE_ADDRW + BATCH_SEL_BITS
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
    output wire [MEM_CHANNELS-1:0][ADDR_WIDTH-1:0] mem_req_addr,
    output wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data,
    output wire [MEM_CHANNELS-1:0][MEM_TAGW-1:0]mem_req_tag,
    input wire 	[MEM_CHANNELS-1:0]          mem_req_ready,

    // Memory response
    input wire [MEM_CHANNELS-1:0]           mem_rsp_valid,
    input wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_rsp_data,
    input wire [MEM_CHANNELS-1:0][MEM_TAGW-1:0] mem_rsp_tag,    
    output wire [MEM_CHANNELS-1:0]          mem_rsp_ready
);

    localparam REQQ_TAG_WIDTH  = MEM_TAG_ID + QUEUE_ADDRW;
    localparam BATCH_SEL_WIDTH = `UP(BATCH_SEL_BITS);
    localparam STALL_TIMEOUT   = 10000000;

    `STATIC_ASSERT ((WORD_SIZE == LINE_SIZE), ("invalid parameter"))
    `STATIC_ASSERT ((MEM_TAG_ID >= UUID_WIDTH), ("invalid parameter"))
    `STATIC_ASSERT ((0 == RSP_PARTIAL) || (1 == RSP_PARTIAL), ("invalid parameter"))
    `RUNTIME_ASSERT ((~core_req_valid || core_req_mask != 0), ("invalid request mask"));

    wire [MEM_CHANNELS-1:0]         mem_req_valid_s;
    wire [MEM_CHANNELS-1:0]         mem_req_mask_s;
    wire [MEM_CHANNELS-1:0]         mem_req_rw_s;
    wire [MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen_s;
    wire [MEM_CHANNELS-1:0][ADDR_WIDTH-1:0] mem_req_addr_s;
    wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data_s;
    wire [MEM_TAGW-1:0]             mem_req_tag_s;
    wire [MEM_CHANNELS-1:0]         mem_req_ready_s;

    wire                            mem_rsp_valid_s;
    wire [MEM_CHANNELS-1:0]         mem_rsp_mask_s;
    wire [MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_rsp_data_s;
    wire [MEM_TAGW-1:0]             mem_rsp_tag_s;
    wire                            mem_rsp_ready_s;
    wire                            mem_rsp_fire_s;

    wire                            reqq_push;
    wire                            reqq_pop;
    wire                            reqq_full;
    wire                            reqq_empty;
    wire                            reqq_rw;
    wire [CORE_REQS-1:0]            reqq_mask;
    wire [CORE_REQS-1:0][WORD_SIZE-1:0] reqq_byteen;
    wire [CORE_REQS-1:0][ADDR_WIDTH-1:0] reqq_addr;
    wire [CORE_REQS-1:0][WORD_WIDTH-1:0] reqq_data;
    wire [REQQ_TAG_WIDTH-1:0]       reqq_tag;

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
    wire [TAG_WIDTH-1:0]            crsp_tag;
    wire                            crsp_sop;
    wire                            crsp_eop;
    wire                            crsp_ready;

    // Request queue //////////////////////////////////////////////////////////

    wire req_sent_all;

    assign reqq_push = core_req_valid && core_req_ready;
    assign reqq_pop  = ~reqq_empty && req_sent_all;
    
    wire [REQQ_TAG_WIDTH-1:0] reqq_tag_u;
    if (MEM_TAG_ID != 0) begin
        assign reqq_tag_u = {core_req_tag[TAG_WIDTH-1:TAG_ID_WIDTH], ibuf_waddr};
    end else begin
        assign reqq_tag_u = ibuf_waddr;
    end

    wire [`CLOG2(QUEUE_SIZE+1)-1:0] reqq_size;
    `UNUSED_VAR (reqq_size)

    VX_fifo_queue #(
        .DATAW   (1 + CORE_REQS * (1 + WORD_SIZE + ADDR_WIDTH + WORD_WIDTH) + REQQ_TAG_WIDTH),
        .DEPTH	 (QUEUE_SIZE),
        .OUT_REG (1)
    ) req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (reqq_push),
        .pop        (reqq_pop),
        .data_in    ({core_req_rw, core_req_mask, core_req_byteen, core_req_addr, core_req_data, reqq_tag_u}),
        .data_out   ({reqq_rw, reqq_mask, reqq_byteen, reqq_addr, reqq_data, reqq_tag}),
        .full       (reqq_full),
        .empty      (reqq_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        .size       (reqq_size)
    );

    // can accept another request?
    assign core_req_ready = ~reqq_full && (core_req_rw || ~ibuf_full);

    // no pending requests
    assign core_req_empty = reqq_empty && ibuf_empty;

    // notify request submisison 
    assign core_req_sent = reqq_pop;

    // Index buffer ///////////////////////////////////////////////////////////

    wire rsp_complete;

    assign ibuf_push  = reqq_push && ~core_req_rw;
    assign ibuf_pop   = crsp_valid && crsp_ready && rsp_complete;
    assign ibuf_raddr = mem_rsp_tag_s[0 +: QUEUE_ADDRW];
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

    // Handle memory requests /////////////////////////////////////////////////

    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0] mem_req_mask_b;
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0] mem_req_rw_b;
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0][LINE_SIZE-1:0] mem_req_byteen_b; 
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0][ADDR_WIDTH-1:0] mem_req_addr_b;
    wire [NUM_BATCHES-1:0][MEM_CHANNELS-1:0][LINE_WIDTH-1:0] mem_req_data_b;
    
    wire [BATCH_SEL_WIDTH-1:0] req_batch_idx;

    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        for (genvar j = 0; j < MEM_CHANNELS; ++j) begin
            localparam r = i * MEM_CHANNELS + j;
            if (r < CORE_REQS) begin
                assign mem_req_mask_b[i][j]   = reqq_mask[r];
                assign mem_req_rw_b[i][j]     = reqq_rw;
                assign mem_req_byteen_b[i][j] = reqq_byteen[r];
                assign mem_req_addr_b[i][j]   = reqq_addr[r];
                assign mem_req_data_b[i][j]   = reqq_data[r];
            end else begin
                assign mem_req_mask_b[i][j]   = 0;
                assign mem_req_rw_b[i][j]     = '0;
                assign mem_req_byteen_b[i][j] = '0;
                assign mem_req_addr_b[i][j]   = '0;
                assign mem_req_data_b[i][j]   = '0;
            end
        end
    end

    assign mem_req_mask_s   = mem_req_mask_b[req_batch_idx];
    assign mem_req_rw_s     = mem_req_rw_b[req_batch_idx];
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
            if (~reqq_empty) begin
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
                if (~reqq_empty && batch_sent_all) begin
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
        assign mem_req_tag_s = {reqq_tag, req_batch_idx};

    end else begin

        assign req_batch_idx = '0;
        assign req_sent_all  = batch_sent_all;
        assign mem_req_tag_s = reqq_tag;

    end

    assign mem_req_valid_s = {MEM_CHANNELS{~reqq_empty}} & mem_req_mask_s & ~batch_sent_mask;  

    for (genvar i = 0; i < MEM_CHANNELS; ++i) begin
        VX_elastic_buffer #(
            .DATAW   (1 + LINE_SIZE + ADDR_WIDTH + LINE_WIDTH + MEM_TAGW),
            .SIZE    (`TO_OUT_BUF_SIZE(MEM_OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(MEM_OUT_BUF))
        ) mem_req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (mem_req_valid_s[i]),
            .ready_in  (mem_req_ready_s[i]),
            .data_in   ({mem_req_rw_s[i], mem_req_byteen_s[i], mem_req_addr_s[i], mem_req_data_s[i], mem_req_tag_s}),
            .data_out  ({mem_req_rw[i],   mem_req_byteen[i],   mem_req_addr[i],   mem_req_data[i],   mem_req_tag[i]}),
            .valid_out (mem_req_valid[i]),
            .ready_out (mem_req_ready[i])
        );
    end

    // Handle memory responses ////////////////////////////////////////////////

    reg [QUEUE_SIZE-1:0][CORE_REQS-1:0] rsp_rem_mask;
    wire [CORE_REQS-1:0] rsp_rem_mask_n, curr_mask;
    wire [BATCH_SEL_WIDTH-1:0] rsp_batch_idx;

    // Select memory response
    VX_mem_rsp_sel #(
        .NUM_REQS     (MEM_CHANNELS),
        .DATA_WIDTH   (LINE_WIDTH),
        .TAG_WIDTH    (MEM_TAGW),
        .TAG_SEL_BITS (MEM_TAGW - MEM_TAG_ID),
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

    for (genvar r = 0; r < CORE_REQS; ++r) begin
        localparam i = r / MEM_CHANNELS;
        localparam j = r % MEM_CHANNELS;
        assign curr_mask[r] = (BATCH_SEL_WIDTH'(i) == rsp_batch_idx) && mem_rsp_mask_s[j];
    end
    
    assign rsp_rem_mask_n = rsp_rem_mask[ibuf_raddr] & ~curr_mask;

    if (NUM_BATCHES > 1) begin
        assign rsp_batch_idx = mem_rsp_tag_s[BATCH_SEL_BITS-1:0];
    end else begin
        assign rsp_batch_idx = '0;
    end    

    assign rsp_complete = ~(| rsp_rem_mask_n);

    always @(posedge clk) begin
        if (ibuf_push) begin
            rsp_rem_mask[ibuf_waddr] <= core_req_mask;
        end
        if (mem_rsp_fire_s) begin
            rsp_rem_mask[ibuf_raddr] <= rsp_rem_mask_n;
        end
    end

    assign mem_rsp_fire_s = mem_rsp_valid_s && mem_rsp_ready_s;

    if (RSP_PARTIAL == 1) begin

        reg [QUEUE_SIZE-1:0] rsp_sop_r;

        always @(posedge clk) begin
            if (ibuf_push) begin
                rsp_sop_r[ibuf_waddr] <= 1;
            end
            if (mem_rsp_fire_s) begin
                rsp_sop_r[ibuf_raddr] <= 0;
            end
        end

        assign mem_rsp_ready_s = crsp_ready;
        
        assign crsp_valid = mem_rsp_valid_s;

        assign crsp_mask = curr_mask;
        assign crsp_sop = rsp_sop_r[ibuf_raddr];

        for (genvar r = 0; r < CORE_REQS; ++r) begin
            localparam j = r % MEM_CHANNELS;
            assign crsp_data[r] = mem_rsp_data_s[j];
        end

    end else begin

        reg [NUM_BATCHES*MEM_CHANNELS*WORD_WIDTH-1:0] rsp_store [QUEUE_SIZE-1:0];        
        reg [NUM_BATCHES*MEM_CHANNELS*WORD_WIDTH-1:0] rsp_store_n;        
        reg [CORE_REQS-1:0] rsp_orig_mask [QUEUE_SIZE-1:0];    

        always @(*) begin
            rsp_store_n = rsp_store[ibuf_raddr];            
            for (integer i = 0; i < MEM_CHANNELS; ++i) begin
                if ((MEM_CHANNELS == 1) || mem_rsp_mask_s[i]) begin
                    rsp_store_n[(rsp_batch_idx * MEM_CHANNELS + i) * WORD_WIDTH +: WORD_WIDTH] = mem_rsp_data_s[i];
                end
            end
        end        
        
        always @(posedge clk) begin
            if (ibuf_push) begin
                rsp_orig_mask[ibuf_waddr] <= core_req_mask;
            end
            if (mem_rsp_valid_s) begin
                rsp_store[ibuf_raddr] <= rsp_store_n;
            end
        end

        assign mem_rsp_ready_s = crsp_ready || ~rsp_complete;      

        assign crsp_valid = mem_rsp_valid_s && rsp_complete;

        assign crsp_mask = rsp_orig_mask[ibuf_raddr];
        assign crsp_sop = 1'b1;

        for (genvar r = 0; r < CORE_REQS; ++r) begin
            localparam i = r / MEM_CHANNELS;
            localparam j = r % MEM_CHANNELS;
            assign crsp_data[r] = rsp_store_n[(i * MEM_CHANNELS + j) * LINE_WIDTH +: WORD_WIDTH];
        end
    end

    if (MEM_TAG_ID != 0) begin
        assign crsp_tag = {mem_rsp_tag_s[MEM_TAGW-1 -: MEM_TAG_ID], ibuf_dout};
    end else begin
        assign crsp_tag = ibuf_dout;
    end

    assign crsp_eop  = ibuf_pop;

    // Send response to caller

    VX_elastic_buffer #(
        .DATAW   (CORE_REQS + 1 + 1 + (CORE_REQS * WORD_WIDTH) + TAG_WIDTH),
        .SIZE    (`TO_OUT_BUF_SIZE(CORE_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsp_valid),  
        .ready_in  (crsp_ready),
        .data_in   ({crsp_mask, crsp_sop, crsp_eop, crsp_data, crsp_tag}),
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
        assign mem_req_dbg_uuid = reqq_tag[REQQ_TAG_WIDTH-1 -: UUID_WIDTH];
        assign mem_rsp_dbg_uuid = mem_rsp_tag_s[MEM_TAGW-1 -: UUID_WIDTH];
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

    reg [(`UP(UUID_WIDTH) + TAG_ID_WIDTH + 64)-1:0] pending_reqs [QUEUE_SIZE-1:0];
    reg [QUEUE_SIZE-1:0] pending_req_valids;

    always @(posedge clk) begin
        if (reset) begin
            pending_req_valids <= '0;
        end else begin
            if (ibuf_push) begin
                pending_req_valids[ibuf_waddr] <= 1'b1;
            end
            if (ibuf_pop) begin
                pending_req_valids[ibuf_raddr] <= 1'b0;
            end
        end

        if (ibuf_push) begin            
            pending_reqs[ibuf_waddr] <= {req_dbg_uuid, ibuf_din, $time};
        end

        for (integer i = 0; i < QUEUE_SIZE; ++i) begin
            if (pending_req_valids[i]) begin
                `ASSERT(($time - pending_reqs[i][0 +: 64]) < STALL_TIMEOUT,
                    ("%t: *** %s response timeout: remaining=%b, tag=0x%0h (#%0d)", 
                        $time, INSTANCE_ID, rsp_rem_mask[i], pending_reqs[i][64 +: TAG_ID_WIDTH], pending_reqs[i][64+TAG_ID_WIDTH +: `UP(UUID_WIDTH)]));
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
            `TRACE(1, (", ibuf_idx=%0d, batch_idx=%0d (#%0d)\n", ibuf_waddr, req_batch_idx, mem_req_dbg_uuid));
        end 
        if (mem_rsp_fire_s) begin
            `TRACE(1, ("%d: %s-mem-rsp: valid=%b, data=", $time, INSTANCE_ID, mem_rsp_mask_s));                
            `TRACE_ARRAY1D(1, mem_rsp_data_s, MEM_CHANNELS);
            `TRACE(1, (", ibuf_idx=%0d, batch_idx=%0d (#%0d)\n", ibuf_raddr, rsp_batch_idx, mem_rsp_dbg_uuid));
        end
    end
`endif
  
endmodule
`TRACING_ON
