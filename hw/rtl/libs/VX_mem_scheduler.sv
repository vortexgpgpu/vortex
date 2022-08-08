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

    parameter BYTEENW      = DATA_WIDTH / 8,
    parameter NUM_BATCHES  = (NUM_REQS + NUM_BANKS - 1) / NUM_BANKS,
    parameter QUEUE_ADDRW  = `CLOG2(QUEUE_SIZE),
    parameter BATCH_SEL_BITS = `CLOG2(NUM_BATCHES),
    parameter MEM_TAGW     = UUID_WIDTH + QUEUE_ADDRW + BATCH_SEL_BITS
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
    output wire                             write_notify,

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
    `RUNTIME_ASSERT ((~req_valid || req_mask != 0), ("invalid request mask"));

    wire [NUM_BANKS-1:0]             mem_req_valid_s;
    wire [NUM_BANKS-1:0]             mem_req_mask_s;
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

    wire req_sent_all;

    assign reqq_push = req_valid && req_ready;
    assign reqq_pop  = ~reqq_empty && req_sent_all;
    
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

    // notify write submisison 
    assign write_notify = reqq_pop && reqq_rw;

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

    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0] mem_req_mask_b;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0] mem_req_rw_b;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][BYTEENW-1:0] mem_req_byteen_b; 
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][ADDR_WIDTH-1:0] mem_req_addr_b;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_req_data_b;
    
    wire [`UP(BATCH_SEL_BITS)-1:0] req_batch_idx;

    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        for (genvar j = 0; j < NUM_BANKS; ++j) begin
            localparam r = i * NUM_BANKS + j;
            if (r < NUM_REQS) begin
                assign mem_req_mask_b[i][j]   = reqq_mask[r];
                assign mem_req_rw_b[i][j]     = reqq_rw;
                assign mem_req_byteen_b[i][j] = reqq_byteen[r];
                assign mem_req_addr_b[i][j]   = reqq_addr[r];
                assign mem_req_data_b[i][j]   = reqq_data[r];
            end else begin
                assign mem_req_mask_b[i][j]   = 0;
                assign mem_req_rw_b[i][j]     = 'x;
                assign mem_req_byteen_b[i][j] = 'x;
                assign mem_req_addr_b[i][j]   = 'x;
                assign mem_req_data_b[i][j]   = 'x;
            end
        end
    end

    assign mem_req_mask_s   = mem_req_mask_b[req_batch_idx];
    assign mem_req_rw_s     = mem_req_rw_b[req_batch_idx];
    assign mem_req_byteen_s = mem_req_byteen_b[req_batch_idx];
    assign mem_req_addr_s   = mem_req_addr_b[req_batch_idx];
    assign mem_req_data_s   = mem_req_data_b[req_batch_idx];

    reg [NUM_BANKS-1:0] batch_sent_mask;
    
    wire [NUM_BANKS-1:0] batch_sent_mask_n = batch_sent_mask | mem_req_ready_s;
    
    wire req_sent_batch = (mem_req_mask_s & ~batch_sent_mask_n) == 0;

    always @(posedge clk) begin
        if (reset) begin
            batch_sent_mask <= '0;
        end else begin
            if (~reqq_empty) begin
                if (req_sent_batch) begin
                    batch_sent_mask <= '0;
                end else begin
                    batch_sent_mask <= batch_sent_mask_n;
                end
            end
        end
    end
    
    if (NUM_BATCHES > 1) begin
        reg [`UP(BATCH_SEL_BITS)-1:0] req_batch_idx_r;
        always @(posedge clk) begin
            if (reset) begin
                req_batch_idx_r <= 0;
            end else begin
                if (~reqq_empty && req_sent_batch) begin
                    if (req_sent_all 
                    || (req_batch_idx_r == `UP(BATCH_SEL_BITS)'(NUM_BATCHES-1))) begin
                        req_batch_idx_r <= 0;
                    end else begin
                        req_batch_idx_r <= req_batch_idx_r + `UP(BATCH_SEL_BITS)'(1);
                    end
                end
            end
        end

        wire [NUM_REQS-1:0] req_sent_mask;
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            localparam batch_idx = i / NUM_BANKS;
            localparam bank_idx  = i % NUM_BANKS;
            wire req_sent_curr = (req_batch_idx == BATCH_SEL_BITS'(batch_idx)) && batch_sent_mask_n[bank_idx];
            assign req_sent_mask[i] = ((batch_idx < (NUM_BATCHES-1)) && (req_batch_idx > BATCH_SEL_BITS'(batch_idx))) 
                                    | (reqq_mask[i] && req_sent_curr) 
                                    | ~reqq_mask[i];
        end
        assign req_batch_idx = req_batch_idx_r;
        assign req_sent_all  = (& req_sent_mask);    
    end else begin
        assign req_batch_idx = 0;
        assign req_sent_all  = req_sent_batch;
    end

    assign mem_req_valid_s = {NUM_BANKS{~reqq_empty}} & mem_req_mask_s & ~batch_sent_mask;

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

    reg  [REQ_SIZEW-1:0] rsp_rem_size [QUEUE_SIZE-1:0];
    wire [REQ_SIZEW-1:0] rsp_rem_size_n;
    wire [`UP(BATCH_SEL_BITS)-1:0] rsp_batch_idx;

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
    wire [NUM_BANKS-1:0] mem_rsp_mask_x;
    `POP_COUNT(reqq_size, reqq_mask);

    wire [BANK_SIZEW-1:0] mem_rsp_size;
    if (NUM_BANKS > 1) begin
        `POP_COUNT(mem_rsp_size, mem_rsp_mask_s);
        assign mem_rsp_mask_x = mem_rsp_mask_s;
    end else begin
        assign mem_rsp_size   = 1'b1;
        assign mem_rsp_mask_x = 1'b1;
        `UNUSED_VAR (mem_rsp_mask_s)
    end

    if (NUM_BATCHES > 1) begin
        assign rsp_batch_idx = mem_rsp_tag_s[QUEUE_ADDRW +: BATCH_SEL_BITS];
    end else begin
        assign rsp_batch_idx = 0;
    end

    assign rsp_rem_size_n = rsp_rem_size[ibuf_raddr] - REQ_SIZEW'(mem_rsp_size);

    assign rsp_complete = (0 == rsp_rem_size_n);

    always @(posedge clk) begin
        if (~reqq_empty && ~reqq_rw && req_batch_idx == 0 && batch_sent_mask == 0) begin
            rsp_rem_size[reqq_tag] <= reqq_size;
        end
        if (mem_rsp_fire) begin
            rsp_rem_size[ibuf_raddr] <= rsp_rem_size_n;
        end
    end

    assign mem_rsp_fire = mem_rsp_valid_s && mem_rsp_ready_s;

    if (RSP_PARTIAL == 1) begin

        assign mem_rsp_ready_s = crsp_ready;

        assign crsp_valid = mem_rsp_valid_s;

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign crsp_mask[i * NUM_BANKS +: SIZE] = {SIZE{(i == rsp_batch_idx)}} & mem_rsp_mask_x[SIZE-1:0];
            assign crsp_data[i * NUM_BANKS +: SIZE] = mem_rsp_data_s[SIZE-1:0];
        end
    
    end else begin

        reg [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0] rsp_store [QUEUE_SIZE-1:0];
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0] rsp_store_n;
        reg [NUM_REQS-1:0] rsp_orig_mask [QUEUE_SIZE-1:0];

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            for (genvar j = 0; j < NUM_BANKS; ++j) begin
                assign rsp_store_n[i][j] = (i == rsp_batch_idx && mem_rsp_mask_x[j]) ? mem_rsp_data_s[j] : rsp_store[ibuf_raddr][i][j];
            end
        end
        
        always @(posedge clk) begin
            if (ibuf_push) begin
                rsp_orig_mask[ibuf_waddr] <= req_mask;
            end
            if (mem_rsp_fire) begin
                rsp_store[ibuf_raddr] <= rsp_store_n;
            end
        end

        assign mem_rsp_ready_s = crsp_ready || ~rsp_complete;      

        assign crsp_valid = mem_rsp_valid_s && rsp_complete;

        assign crsp_mask = rsp_orig_mask[ibuf_raddr];

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign crsp_data[i * NUM_BANKS +: SIZE] = rsp_store_n[i][SIZE-1:0];
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

    reg [(`UP(`UUID_BITS) + TAG_ONLY_WIDTH + 64)-1:0] pending_reqs [QUEUE_SIZE-1:0];
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
            pending_reqs[ibuf_waddr] <= {req_dbg_uuid, req_tag_only, $time};
        end

        for (integer i = 0; i < QUEUE_SIZE; ++i) begin
            if (pending_req_valids[i]) begin
                `ASSERT(($time - pending_reqs[i][0 +: 64]) < `STALL_TIMEOUT,
                    ("%t: *** %s response timeout: remaining=%0d, tag=0x%0h (#%0d)", 
                        $time, INSTANCE_ID, rsp_rem_size[i], pending_reqs[i][64 +: TAG_ONLY_WIDTH], pending_reqs[i][64+TAG_ONLY_WIDTH +: `UP(`UUID_BITS)]));
            end
        end
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

`ifndef NDEBUG
    wire [NUM_BANKS-1:0] mem_req_fire_s = mem_req_valid_s & mem_req_ready_s;
    always @(posedge clk) begin
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
    end
`endif
  
endmodule
`TRACING_ON
