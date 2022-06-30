`include "VX_platform.vh"

`TRACING_OFF
module VX_mem_scheduler #(
    parameter string INSTANCE_ID = "",
    parameter NUM_REQS      = 4,
    parameter NUM_BANKS     = 4,
    parameter ADDR_WIDTH    = 32,
    parameter DATA_WIDTH    = 32,
    parameter TAG_WIDTH     = 32,
    parameter UUID_WIDTH    = 0,
    parameter QUEUE_SIZE    = 16,
    parameter RSP_PARTIAL   = 0,
    parameter CORE_OUT_REG  = 0,
    parameter MEM_OUT_REG   = 0,

    localparam BYTEENW      = DATA_WIDTH / 8,
    localparam NUM_BATCHES  = (NUM_REQS + NUM_BANKS - 1) / NUM_BANKS,
    localparam QUEUE_ADDRW  = `CLOG2(QUEUE_SIZE),
    localparam BATCH_SEL_BITS = `CLOG2(NUM_BATCHES),
    localparam MEM_TAGW     = UUID_WIDTH + QUEUE_ADDRW + BATCH_SEL_BITS
) (
    input wire clk,
    input wire reset,

    // Input request
    input wire                              req_valid,
    input wire                              req_rw,
    input wire [NUM_REQS-1:0]               req_mask,
    input wire [NUM_REQS-1:0][BYTEENW-1:0]  req_byteen,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] req_addr,
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0] req_data,
    input wire [TAG_WIDTH-1:0]              req_tag,
    output wire                             req_empty,
    output wire                             req_ready,

    // Output response
    output wire                             rsp_valid,
    output wire [NUM_REQS-1:0]              rsp_mask,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] rsp_data,
    output wire [TAG_WIDTH-1:0]             rsp_tag,
    output wire                             rsp_eop,
    input wire                              rsp_ready,

    // Memory request
    output wire [NUM_BANKS-1:0]             mem_req_valid,
    output wire [NUM_BANKS-1:0]             mem_req_rw,
    output wire [NUM_BANKS-1:0][BYTEENW-1:0] mem_req_byteen,
    output wire [NUM_BANKS-1:0][ADDR_WIDTH-1:0] mem_req_addr,
    output wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_req_data,
    output wire [NUM_BANKS-1:0][MEM_TAGW-1:0]mem_req_tag,
    input wire 	[NUM_BANKS-1:0]             mem_req_ready,

    // Memory response
    input wire [NUM_BANKS-1:0]              mem_rsp_valid,
    input wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_rsp_data,
    input wire [NUM_BANKS-1:0][MEM_TAGW-1:0] mem_rsp_tag,    
    output wire [NUM_BANKS-1:0]             mem_rsp_ready
  );

    localparam REM_BATCH_SIZE = NUM_REQS % NUM_BANKS;
    localparam BATCH_DATAW = NUM_BANKS * (1 + BYTEENW + ADDR_WIDTH + DATA_WIDTH);
    localparam REQ_SIZEW = $clog2(NUM_REQS + 1);
    localparam BANK_SIZEW = $clog2(NUM_BANKS + 1);
    localparam TAG_ONLY_WIDTH = TAG_WIDTH - UUID_WIDTH;

    `STATIC_ASSERT (DATA_WIDTH == 8 * (DATA_WIDTH / 8), ("invalid parameter"))
    `STATIC_ASSERT ((0 == RSP_PARTIAL) || (1 == RSP_PARTIAL), ("invalid parameter"))
    `RUNTIME_ASSERT ((~req_valid || req_mask != 0), ("invalid input"));

    wire [NUM_BANKS-1:0]             mem_req_valid_s;
    wire [NUM_BANKS-1:0]             mem_req_rw_s;
    wire [NUM_BANKS-1:0][BYTEENW-1:0] mem_req_byteen_s;
    wire [NUM_BANKS-1:0][ADDR_WIDTH-1:0] mem_req_addr_s;
    wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_req_data_s;
    wire [NUM_BANKS-1:0][MEM_TAGW-1:0]mem_req_tag_s;
    wire [NUM_BANKS-1:0]            mem_req_ready_s;

    wire                            mem_rsp_valid_s;
    wire [NUM_BANKS-1:0]            mem_rsp_mask_s;
    wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_rsp_data_s;
    wire [MEM_TAGW-1:0]             mem_rsp_tag_s;
    wire                            mem_rsp_ready_s;
    wire                            mem_rsp_fire;

    wire                            reqq_push;
    wire                            reqq_pop;
    wire                            reqq_full;
    wire                            reqq_empty;
    wire                            reqq_rw;
    wire [NUM_REQS-1:0]             reqq_mask;
    wire [NUM_REQS-1:0][BYTEENW-1:0] reqq_byteen;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] reqq_addr;
    wire [NUM_REQS-1:0][DATA_WIDTH-1:0] reqq_data;
    wire [QUEUE_ADDRW-1:0]          reqq_tag;
    wire [`UP(UUID_WIDTH)-1:0]      reqq_uuid;

    wire                            ibuf_push;
    wire                            ibuf_pop;
    wire [QUEUE_ADDRW-1:0]          ibuf_waddr;
    wire [QUEUE_ADDRW-1:0]          ibuf_raddr;
    wire                            ibuf_full;
    wire                            ibuf_empty;
    wire [TAG_ONLY_WIDTH-1:0]       ibuf_dout;

    wire                            crsp_valid;
    wire [NUM_REQS-1:0]             crsp_mask;
    wire [NUM_REQS-1:0][DATA_WIDTH-1:0] crsp_data;
    wire [TAG_WIDTH-1:0]            crsp_tag;
    wire                            crsp_ready;

    // Request queue //////////////////////////////////////////////////////////

    wire req_complete;

    assign reqq_push = req_valid && req_ready;
    assign reqq_pop  = ~reqq_empty && req_complete;
    
    wire [`UP(UUID_WIDTH)-1:0] req_uuid;
    if (UUID_WIDTH != 0) begin
        assign req_uuid = req_tag[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid = 0;
    end

    VX_fifo_queue #(
        .DATAW   (1 + NUM_REQS * (1 + BYTEENW + ADDR_WIDTH + DATA_WIDTH) + `UP(UUID_WIDTH) + QUEUE_ADDRW),
        .SIZE	 (QUEUE_SIZE),
        .OUT_REG (1)
    ) req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (reqq_push),
        .pop        (reqq_pop),
        .data_in    ({req_rw,  req_mask,  req_byteen,  req_addr,  req_data,  req_uuid,  ibuf_waddr}),
        .data_out   ({reqq_rw, reqq_mask, reqq_byteen, reqq_addr, reqq_data, reqq_uuid, reqq_tag}),
        .full       (reqq_full),
        .empty      (reqq_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    // can accept another request?
    assign req_ready = ~reqq_full && (req_rw || ~ibuf_full);

    // no pending requests
    assign req_empty = reqq_empty && ibuf_empty;

    // Index buffer ///////////////////////////////////////////////////////////

    wire rsp_complete;

    assign ibuf_push  = reqq_push && ~req_rw;
    assign ibuf_pop   = crsp_valid && crsp_ready && rsp_complete;
    assign ibuf_raddr = mem_rsp_tag_s[0 +: QUEUE_ADDRW];

    wire [TAG_ONLY_WIDTH-1:0] req_tag_only = req_tag[TAG_ONLY_WIDTH-1:0];

    VX_index_buffer #(
        .DATAW (TAG_ONLY_WIDTH),
        .SIZE  (QUEUE_SIZE)
    ) req_ibuf (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (ibuf_waddr),
        .acquire_slot (ibuf_push),
        .read_addr    (ibuf_raddr),
        .write_data   (req_tag_only),
        .read_data    (ibuf_dout),
        .release_addr (ibuf_raddr),
        .release_slot (ibuf_pop),
        .full         (ibuf_full),
        .empty        (ibuf_empty)
    );

    `UNUSED_VAR (ibuf_empty)

    assign rsp_eop = ibuf_pop;

    // Handle memory requests /////////////////////////////////////////////////

    wire [NUM_BATCHES-1:0][BATCH_DATAW-1:0] mem_req_data_b;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]   mem_req_mask_b;
    reg  [NUM_BANKS-1:0]           req_sent_mask;
    wire [NUM_BANKS-1:0]           req_sent_mask_n;
    reg  [`UP(BATCH_SEL_BITS)-1:0] req_batch_idx;
    
    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
        assign mem_req_data_b[i] = {
            {NUM_BANKS{reqq_rw}},
            (NUM_BANKS * BYTEENW)'(reqq_byteen[i * NUM_BANKS +: SIZE]),
            (NUM_BANKS * ADDR_WIDTH)'(reqq_addr[i * NUM_BANKS +: SIZE]),
            (NUM_BANKS * DATA_WIDTH)'(reqq_data[i * NUM_BANKS +: SIZE])
        };
    end    
    
    if (NUM_BATCHES > 1) begin
        wire [NUM_REQS-1:0] req_sent_mask_all;
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            localparam j = i / NUM_BANKS;
            localparam k = i % NUM_BANKS;
            wire req_sent_curr = (req_batch_idx == BATCH_SEL_BITS'(j)) && req_sent_mask_n[k];
            assign req_sent_mask_all[i] = ((j < (NUM_BATCHES-1)) && (req_batch_idx > BATCH_SEL_BITS'(j))) 
                                         | (req_sent_curr & reqq_mask[i]) 
                                         | (~req_sent_curr & ~reqq_mask[i]);
        end
        assign req_complete = (& req_sent_mask_all);
    end else begin
        assign req_complete = (req_sent_mask_n == reqq_mask);
    end

    wire [NUM_BANKS-1:0] mem_req_fire_s = mem_req_valid_s & mem_req_ready_s;

    assign mem_req_mask_b = (NUM_BATCHES * NUM_BANKS)'(reqq_mask);

    assign req_sent_mask_n = req_sent_mask | mem_req_fire_s;

    wire req_complete_b = ~reqq_empty && (req_sent_mask_n == mem_req_mask_b[req_batch_idx]);

    always @(posedge clk) begin
        if (reset) begin
            req_sent_mask <= '0;
            req_batch_idx <= 0;
        end else begin
            if (req_complete_b) begin
                if (req_complete 
                 || (req_batch_idx == `UP(BATCH_SEL_BITS)'(NUM_BATCHES-1))) begin
                    req_batch_idx <= 0;
                end else begin
                    req_batch_idx <= req_batch_idx + `UP(BATCH_SEL_BITS)'(1);
                end
                req_sent_mask <= 0;
            end else begin
                req_sent_mask <= req_sent_mask_n;
            end
        end
    end

    assign mem_req_valid_s = mem_req_mask_b[req_batch_idx] & ~req_sent_mask & {NUM_BANKS{~reqq_empty}};

    assign {mem_req_rw_s, mem_req_byteen_s, mem_req_addr_s, mem_req_data_s} = mem_req_data_b[req_batch_idx];

    if (UUID_WIDTH != 0) begin
        if (NUM_BATCHES > 1) begin
            assign mem_req_tag_s = {NUM_BANKS{{reqq_uuid, req_batch_idx, reqq_tag}}};
        end else begin
            assign mem_req_tag_s = {NUM_BANKS{{reqq_uuid, reqq_tag}}};
        end
    end else begin
        `UNUSED_VAR (reqq_uuid)
        if (NUM_BATCHES > 1) begin
            assign mem_req_tag_s = {NUM_BANKS{{req_batch_idx, reqq_tag}}};
        end else begin
            assign mem_req_tag_s = {NUM_BANKS{reqq_tag}};
        end
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        VX_generic_buffer #(
            .DATAW   (1 + BYTEENW + ADDR_WIDTH + DATA_WIDTH + MEM_TAGW),
            .SKID    (MEM_OUT_REG >> 1),
            .OUT_REG (MEM_OUT_REG & 1)
        ) mem_req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (mem_req_valid_s[i]),
            .ready_in  (mem_req_ready_s[i]),
            .data_in   ({mem_req_rw_s[i], mem_req_byteen_s[i], mem_req_addr_s[i], mem_req_data_s[i], mem_req_tag_s[i]}),
            .data_out  ({mem_req_rw[i],   mem_req_byteen[i],   mem_req_addr[i],   mem_req_data[i],   mem_req_tag[i]}),
            .valid_out (mem_req_valid[i]),
            .ready_out (mem_req_ready[i])
        );
    end

    // Handle memory responses ////////////////////////////////////////////////

    reg  [QUEUE_SIZE-1:0][REQ_SIZEW-1:0] rsp_rem_size;
    wire [REQ_SIZEW-1:0]                 rsp_rem_size_n;
    wire [`UP(BATCH_SEL_BITS)-1:0]       rsp_batch_idx;

    // Select memory response
    VX_mem_rsp_sel #(
        .NUM_REQS     (NUM_BANKS),
        .DATA_WIDTH   (DATA_WIDTH),
        .TAG_WIDTH    (MEM_TAGW),
        .TAG_SEL_BITS (MEM_TAGW - UUID_WIDTH),
        .OUT_REG      (3)
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

    wire [REQ_SIZEW-1:0] reqq_size;
    `POP_COUNT(reqq_size, reqq_mask);

    wire [BANK_SIZEW-1:0] mem_rsp_size;
    `POP_COUNT(mem_rsp_size, mem_rsp_mask_s);

    if (NUM_BATCHES > 1) begin
        assign rsp_batch_idx = mem_rsp_tag_s[QUEUE_ADDRW +: BATCH_SEL_BITS];
    end else begin
        assign rsp_batch_idx = 0;
    end

    assign rsp_rem_size_n = rsp_rem_size[ibuf_raddr] - REQ_SIZEW'(mem_rsp_size);

    assign rsp_complete = (0 == rsp_rem_size_n);

    always @(posedge clk) begin
        if (mem_req_fire_s != 0 && req_sent_mask == 0 && req_batch_idx == 0 && reqq_rw == 0) begin
            rsp_rem_size[reqq_tag] <= reqq_size;
        end
        if (mem_rsp_fire) begin
            rsp_rem_size[ibuf_raddr] <= rsp_rem_size_n;
        end
    end

    assign mem_rsp_fire = mem_rsp_valid_s & mem_rsp_ready_s;

    if (RSP_PARTIAL == 1) begin

        assign mem_rsp_ready_s = crsp_ready;

        assign crsp_valid = mem_rsp_valid_s;

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign crsp_mask[i * NUM_BANKS +: SIZE] = {SIZE{(i == rsp_batch_idx)}} & mem_rsp_mask_s[0 +: SIZE];
            assign crsp_data[i * NUM_BANKS +: SIZE] = mem_rsp_data_s[0 +: SIZE];
        end
    
    end else begin

        reg [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0] rsp_store;   
        reg [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0]                 rsp_store_n;
        reg [QUEUE_SIZE-1:0][NUM_REQS-1:0] rsp_orig_mask;
        wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_rsp_data_m;

        assign mem_rsp_ready_s = crsp_ready || ~rsp_complete;        

        assign crsp_valid = mem_rsp_valid_s & rsp_complete;

        assign crsp_mask = rsp_orig_mask[ibuf_raddr];        

        for (genvar i = 0; i < NUM_BANKS; ++i) begin
            assign mem_rsp_data_m[i] = {DATA_WIDTH{mem_rsp_mask_s[i]}} & mem_rsp_data_s[i];
        end        
        
        always @(*) begin
            rsp_store_n = rsp_store[ibuf_raddr];
            rsp_store_n[rsp_batch_idx] |= mem_rsp_data_m;
        end

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign crsp_data[i * NUM_BANKS +: SIZE] = rsp_store_n[i][0 +: SIZE];
        end
        
        always @(posedge clk) begin
            if (reset) begin
                rsp_store <= '0;
            end else begin
                if (ibuf_push) begin  
                    rsp_store[ibuf_waddr] <= '0;
                    rsp_orig_mask[ibuf_waddr] <= req_mask;
                end
                if (mem_rsp_fire) begin
                    rsp_store[ibuf_raddr] <= rsp_store_n;
                end
            end
        end
    end

    if (UUID_WIDTH != 0) begin
        assign crsp_tag = {mem_rsp_tag_s[MEM_TAGW-1 -: UUID_WIDTH], ibuf_dout};
    end else begin
        assign crsp_tag = ibuf_dout;
    end

    // Send response to caller

    VX_generic_buffer #(
        .DATAW   (NUM_REQS + (NUM_REQS * DATA_WIDTH) + TAG_WIDTH),
        .SKID    (CORE_OUT_REG >> 1),
        .OUT_REG (CORE_OUT_REG & 1)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsp_valid),  
        .ready_in  (crsp_ready),
        .data_in   ({crsp_mask, crsp_data, crsp_tag}),
        .data_out  ({rsp_mask,  rsp_data,  rsp_tag}),        
        .valid_out (rsp_valid),        
        .ready_out (rsp_ready)
    );  

`ifdef SIMULATION
    wire [`UP(`UUID_BITS)-1:0] req_dbg_uuid;
    wire [`UP(`UUID_BITS)-1:0] rsp_dbg_uuid;
    wire [`UP(`UUID_BITS)-1:0] mem_req_dbg_uuid;
    wire [`UP(`UUID_BITS)-1:0] mem_rsp_dbg_uuid;

    if (UUID_WIDTH != 0) begin
        assign req_dbg_uuid = req_tag[TAG_WIDTH-1 -: `UP(`UUID_BITS)];
        assign rsp_dbg_uuid = rsp_tag[TAG_WIDTH-1 -: `UP(`UUID_BITS)];
        assign mem_req_dbg_uuid = reqq_uuid[UUID_WIDTH-1 -: `UP(`UUID_BITS)];
        assign mem_rsp_dbg_uuid = mem_rsp_tag_s[MEM_TAGW-1 -: `UP(`UUID_BITS)];
    end else begin
        assign req_dbg_uuid = 0;
        assign rsp_dbg_uuid = 0;
        assign mem_req_dbg_uuid = 0;
        assign mem_rsp_dbg_uuid = 0;
    end
    
    `UNUSED_VAR (req_dbg_uuid)
    `UNUSED_VAR (rsp_dbg_uuid)
    `UNUSED_VAR (mem_req_dbg_uuid)
    `UNUSED_VAR (mem_rsp_dbg_uuid)

    reg [QUEUE_SIZE-1:0][(`UP(`UUID_BITS) + TAG_ONLY_WIDTH + 64 + 1)-1:0] pending_reqs;
    always @(posedge clk) begin
        if (reset) begin
            pending_reqs <= '0;
        end begin
            if (ibuf_push) begin            
                pending_reqs[ibuf_waddr] <= {req_dbg_uuid, req_tag_only, $time, 1'b1};
            end
            if (ibuf_pop) begin
                pending_reqs[ibuf_raddr] <= '0;
            end
        end

        for (integer i = 0; i < QUEUE_SIZE; ++i) begin
            if (pending_reqs[i][0]) begin
                `ASSERT(($time - pending_reqs[i][1 +: 64]) < `STALL_TIMEOUT, 
                    ("%t: *** %s response timeout: remaining=%0d, tag=0x%0h (#%0d)", 
                        $time, INSTANCE_ID, rsp_rem_size[i], pending_reqs[i][1+64 +: TAG_ONLY_WIDTH], pending_reqs[i][1+64+TAG_ONLY_WIDTH +: `UP(`UUID_BITS)]));
            end
        end
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

    /*always @(posedge clk) begin
        if (req_valid && req_ready) begin            
            dpi_trace(1, "%d: %s-req: rw=%b, mask=%b, byteen=", $time, INSTANCE_ID, req_rw, req_mask);
            `TRACE_ARRAY1D(1, req_byteen, NUM_REQS);
            dpi_trace(1, ", addr=");
            `TRACE_ARRAY1D(1, req_addr, NUM_REQS);
            dpi_trace(1, ", data=");
            `TRACE_ARRAY1D(1, req_data, NUM_REQS);
            dpi_trace(1, ", tag=0x%0h (#%0d)\n", req_tag, req_dbg_uuid);
        end
        if (rsp_valid && rsp_ready) begin
            dpi_trace(1, "%d: %s-rsp: mask=%b, data=", $time, INSTANCE_ID, rsp_mask);
             `TRACE_ARRAY1D(1, rsp_data, NUM_REQS);
            dpi_trace(1, ", tag=0x%0h (#%0d)\n", rsp_tag, rsp_dbg_uuid);
        end
        if (| mem_req_fire_s) begin
            if (| mem_req_rw_s) begin
                dpi_trace(1, "%d: %s-mem-wr: valid=%b, byteen=", $time, INSTANCE_ID, mem_req_fire_s);
                `TRACE_ARRAY1D(1, mem_req_byteen_s, NUM_BANKS);
                dpi_trace(1, ", addr=");
                `TRACE_ARRAY1D(1, mem_req_addr_s, NUM_BANKS);
                dpi_trace(1, ", data=");
                `TRACE_ARRAY1D(1, mem_req_data_s, NUM_BANKS);
                dpi_trace(1, ", tag=");
                `TRACE_ARRAY1D(1, ibuf_waddr, NUM_BANKS);
                dpi_trace(1, ", batch=%0d (#%0d)\n", req_batch_idx, mem_req_dbg_uuid);
            end else begin
                dpi_trace(1, "%d: %s-mem-rd: valid=%b, addr=", $time, INSTANCE_ID, mem_req_fire_s);
                `TRACE_ARRAY1D(1, mem_req_addr_s, NUM_BANKS);
                dpi_trace(1, ", tag=");
                `TRACE_ARRAY1D(1, ibuf_waddr, NUM_BANKS);
                dpi_trace(1, ", batch=%0d (#%0d)\n", req_batch_idx, mem_req_dbg_uuid);
            end
        end 
        if (mem_rsp_fire) begin
            dpi_trace(1, "%d: %s-mem-rsp: mask=%b, data=", $time, INSTANCE_ID, mem_rsp_mask_s);                
            `TRACE_ARRAY1D(1, mem_rsp_data_s, NUM_BANKS);
            dpi_trace(1, ", tag=0x%0h, batch=%0d (#%0d)\n", ibuf_raddr, rsp_batch_idx, mem_rsp_dbg_uuid);
        end
    end*/
  
endmodule
`TRACING_ON
