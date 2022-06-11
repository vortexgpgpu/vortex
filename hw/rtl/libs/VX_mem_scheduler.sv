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
    parameter PARTIAL_RESPONSE = 0,
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
    localparam TAG_ONLY_WIDTH = TAG_WIDTH - UUID_WIDTH;

    `STATIC_ASSERT (DATA_WIDTH == 8 * (DATA_WIDTH / 8), ("invalid parameter"))
    `STATIC_ASSERT ((0 == PARTIAL_RESPONSE) || (1 == PARTIAL_RESPONSE), ("invalid parameter"))
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

    wire                            sreq_push;
    wire                            sreq_pop;
    wire                            sreq_full;
    wire                            sreq_empty;
    wire                            sreq_rw;
    wire [NUM_REQS-1:0]             sreq_mask;
    wire [NUM_REQS-1:0][BYTEENW-1:0] sreq_byteen;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] sreq_addr;
    wire [NUM_REQS-1:0][DATA_WIDTH-1:0] sreq_data;
    wire [QUEUE_ADDRW-1:0]          sreq_tag;
    wire [`UP(UUID_WIDTH)-1:0]      sreq_uuid;

    wire                            stag_push;
    wire                            stag_pop;
    wire [QUEUE_ADDRW-1:0]          stag_waddr;
    wire [QUEUE_ADDRW-1:0]          stag_raddr;
    wire                            stag_full;
    wire                            stag_empty;
    wire [TAG_ONLY_WIDTH-1:0]       stag_dout;    

    wire                            crsp_valid;
    wire [NUM_REQS-1:0]             crsp_mask;
    wire [NUM_REQS-1:0][DATA_WIDTH-1:0] crsp_data;
    wire [TAG_WIDTH-1:0]            crsp_tag;
    wire                            crsp_ready;

    // Store request //////////////////////////////////////////////////////

    wire req_complete;

    assign sreq_push = req_valid && req_ready;
    assign sreq_pop  = ~sreq_empty && req_complete;
    
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
    ) req_store (
        .clk        (clk),
        .reset      (reset),
        .push       (sreq_push),
        .pop        (sreq_pop),
        .data_in    ({req_rw,  req_mask,  req_byteen,  req_addr,  req_data,  req_uuid,  stag_waddr}),
        .data_out   ({sreq_rw, sreq_mask, sreq_byteen, sreq_addr, sreq_data, sreq_uuid, sreq_tag}),
        .full       (sreq_full),
        .empty      (sreq_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    // can accept another request?
    assign req_ready = ~sreq_full && (req_rw || ~stag_full);

    // no pending requests
    assign req_empty = sreq_empty && stag_empty;

    // Tag store //////////////////////////////////////////////////////////////

    wire rsp_complete;

    assign stag_push  = sreq_push && ~req_rw;
    assign stag_pop   = crsp_valid && crsp_ready && rsp_complete;
    assign stag_raddr = mem_rsp_tag_s[0 +: QUEUE_ADDRW];

    wire [TAG_ONLY_WIDTH-1:0] req_tag_only = req_tag[TAG_ONLY_WIDTH-1:0];

    VX_index_buffer #(
        .DATAW (TAG_ONLY_WIDTH),
        .SIZE  (QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (stag_waddr),
        .acquire_slot (stag_push),
        .read_addr    (stag_raddr),
        .write_data   (req_tag_only),
        .read_data    (stag_dout),
        .release_addr (stag_raddr),
        .release_slot (stag_pop),
        .full         (stag_full),
        .empty        (stag_empty)
    );

    `UNUSED_VAR (stag_empty)

    assign rsp_eop = stag_pop;

    // Handle memory requests /////////////////////////////////////////////////

    wire [NUM_BATCHES-1:0][BATCH_DATAW-1:0] mem_req_data_b;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]   mem_req_mask_b;
    reg  [NUM_BANKS-1:0]           req_sent_mask;
    wire [NUM_BANKS-1:0]           req_sent_mask_n;
    reg  [`UP(BATCH_SEL_BITS)-1:0] req_batch_idx;
    
    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
        assign mem_req_data_b[i] = {
            {NUM_BANKS{sreq_rw}},
            (NUM_BANKS * BYTEENW)'(sreq_byteen[i * NUM_BANKS +: SIZE]),
            (NUM_BANKS * ADDR_WIDTH)'(sreq_addr[i * NUM_BANKS +: SIZE]),
            (NUM_BANKS * DATA_WIDTH)'(sreq_data[i * NUM_BANKS +: SIZE])
        };
    end    

    wire [NUM_REQS-1:0] req_sent_mask_all;
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        if (NUM_BATCHES > 1) begin
            localparam j = i / NUM_BANKS;
            localparam k = i % NUM_BANKS;
            wire [BATCH_SEL_BITS-1:0] batch_idx = BATCH_SEL_BITS'(i / NUM_BANKS);
            if (j < (NUM_BATCHES-1)) begin
                assign req_sent_mask_all[i] = (batch_idx < req_batch_idx) ? sreq_mask[i] : ((batch_idx == req_batch_idx) & req_sent_mask_n[k]);
            end else begin
                assign req_sent_mask_all[i] = (batch_idx == req_batch_idx) & req_sent_mask_n[k];
            end            
        end else begin
            assign req_sent_mask_all[i] = req_sent_mask_n[i];
        end
    end

    wire [NUM_BANKS-1:0] mem_req_fire_s = mem_req_valid_s & mem_req_ready_s;

    assign mem_req_mask_b = (NUM_BATCHES * NUM_BANKS)'(sreq_mask);

    assign req_sent_mask_n = req_sent_mask | mem_req_fire_s;
    
    assign req_complete = (req_sent_mask_all == sreq_mask);

    wire req_complete_b = ~sreq_empty && (req_sent_mask_n == mem_req_mask_b[req_batch_idx]);

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

    assign mem_req_valid_s = mem_req_mask_b[req_batch_idx] & ~req_sent_mask & {NUM_BANKS{~sreq_empty}};

    assign {mem_req_rw_s, mem_req_byteen_s, mem_req_addr_s, mem_req_data_s} = mem_req_data_b[req_batch_idx];

    if (UUID_WIDTH != 0) begin
        if (NUM_BATCHES > 1) begin
            assign mem_req_tag_s = {NUM_BANKS{{sreq_uuid, req_batch_idx, sreq_tag}}};
        end else begin
            assign mem_req_tag_s = {NUM_BANKS{{sreq_uuid, sreq_tag}}};
        end
    end else begin
        `UNUSED_VAR (sreq_uuid)
        if (NUM_BATCHES > 1) begin
            assign mem_req_tag_s = {NUM_BANKS{{req_batch_idx, sreq_tag}}};
        end else begin
            assign mem_req_tag_s = {NUM_BANKS{sreq_tag}};
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

    reg  [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0] rsp_rem_mask;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]                 rsp_rem_mask_n; 
    reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0] rsp_orig_mask;
    wire [`UP(BATCH_SEL_BITS)-1:0]      rsp_batch_idx;

    // Select memory response
    VX_mem_rsp_sel #(
        .NUM_REQS     (NUM_BANKS),
        .DATA_WIDTH   (DATA_WIDTH),
        .TAG_WIDTH    (MEM_TAGW),
        .TAG_SEL_BITS (MEM_TAGW - UUID_WIDTH),
        .OUT_REG      (1)
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

    if (NUM_BATCHES > 1) begin
        assign rsp_batch_idx = mem_rsp_tag_s[QUEUE_ADDRW +: BATCH_SEL_BITS];
        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            assign rsp_rem_mask_n[i] = rsp_rem_mask[stag_raddr][i] & ~({NUM_BANKS{(i == rsp_batch_idx)}} & mem_rsp_mask_s);
        end
    end else begin
        assign rsp_batch_idx = 0;
        assign rsp_rem_mask_n = rsp_rem_mask[stag_raddr] & ~mem_rsp_mask_s;
    end   

    assign rsp_complete = (0 == rsp_rem_mask_n);

    always @(posedge clk) begin
        if (reset) begin
            rsp_rem_mask <= '0;
        end else begin
            if (stag_push) begin
                rsp_orig_mask[stag_waddr] <= req_mask;
                rsp_rem_mask[stag_waddr]  <= (NUM_BATCHES * NUM_BANKS)'(req_mask);
            end
            if (mem_rsp_fire) begin
                rsp_rem_mask[stag_raddr] <= rsp_rem_mask_n;
            end
        end
    end

    assign mem_rsp_fire = mem_rsp_valid_s & mem_rsp_ready_s;

    if (PARTIAL_RESPONSE == 1) begin

        assign mem_rsp_ready_s = crsp_ready;

        assign crsp_valid = mem_rsp_valid_s;

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign crsp_mask[i] = {NUM_BANKS{(i == rsp_batch_idx)}} & mem_rsp_mask_s;
            assign crsp_data[i * NUM_BANKS +: SIZE] = mem_rsp_data_s[0 +: SIZE];
        end
    
    end else begin

        reg [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0] rsp_store;   
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATA_WIDTH-1:0]                rsp_store_n;
        wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] mem_rsp_data_m;

        assign mem_rsp_ready_s = crsp_ready || ~rsp_complete;        

        assign crsp_valid = mem_rsp_valid_s & rsp_complete;

        assign crsp_mask  = rsp_orig_mask[stag_raddr];        

        for (genvar i = 0; i < NUM_BANKS; ++i) begin
            assign mem_rsp_data_m[i] = {DATA_WIDTH{mem_rsp_mask_s[i]}} & mem_rsp_data_s[i];
        end

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign rsp_store_n[i] = rsp_store[stag_raddr][i] | ({NUM_BANKS * DATA_WIDTH{(i == rsp_batch_idx)}} & mem_rsp_data_m);
            assign crsp_data[i * NUM_BANKS +: SIZE] = rsp_store_n[i][0 +: SIZE];
        end
        
        always @(posedge clk) begin
            if (reset) begin
                rsp_store  <= '0;
            end else begin
                if (stag_push) begin                    
                    rsp_store[stag_waddr] <= '0;
                end
                if (mem_rsp_fire) begin
                    rsp_store[stag_raddr] <= rsp_store_n;
                end
            end
        end
    end

    if (UUID_WIDTH != 0) begin
        assign crsp_tag = {mem_rsp_tag_s[MEM_TAGW-1 -: UUID_WIDTH], stag_dout};
    end else begin
        assign crsp_tag = stag_dout;
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
        assign mem_req_dbg_uuid = sreq_uuid[UUID_WIDTH-1 -: `UP(`UUID_BITS)];
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
            if (stag_push) begin            
                pending_reqs[stag_waddr] <= {req_dbg_uuid, req_tag_only, $time, 1'b1};
            end
            if (stag_pop) begin
                pending_reqs[stag_raddr] <= '0;
            end
        end

        for (integer i = 0; i < QUEUE_SIZE; ++i) begin
            if (pending_reqs[i][0]) begin
                `ASSERT(($time - pending_reqs[i][1 +: 64]) < `STALL_TIMEOUT, 
                    ("%t: *** %s response timeout: remaining=%b, tag=0x%0h (#%0d)", 
                        $time, INSTANCE_ID, rsp_rem_mask[i], pending_reqs[i][1+64 +: TAG_ONLY_WIDTH], 
                                                pending_reqs[i][1+64+TAG_ONLY_WIDTH +: `UP(`UUID_BITS)]));
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
                `TRACE_ARRAY1D(1, stag_waddr, NUM_BANKS);
                dpi_trace(1, ", batch=%0d (#%0d)\n", req_batch_idx, mem_req_dbg_uuid);
            end else begin
                dpi_trace(1, "%d: %s-mem-rd: valid=%b, addr=", $time, INSTANCE_ID, mem_req_fire_s);
                `TRACE_ARRAY1D(1, mem_req_addr_s, NUM_BANKS);
                dpi_trace(1, ", tag=");
                `TRACE_ARRAY1D(1, stag_waddr, NUM_BANKS);
                dpi_trace(1, ", batch=%0d (#%0d)\n", req_batch_idx, mem_req_dbg_uuid);
            end
        end 
        if (mem_rsp_fire) begin
            dpi_trace(1, "%d: %s-mem-rsp: mask=%b, data=", $time, INSTANCE_ID, mem_rsp_mask_s);                
            `TRACE_ARRAY1D(1, mem_rsp_data_s, NUM_BANKS);
            dpi_trace(1, ", tag=0x%0h, batch=%0d (#%0d)\n", stag_raddr, rsp_batch_idx, mem_rsp_dbg_uuid);
        end
    end*/
  
endmodule
`TRACING_ON
