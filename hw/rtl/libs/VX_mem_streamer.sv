`include "VX_platform.vh"

`TRACING_OFF
module VX_mem_streamer #(
    parameter NUM_REQS      = 4,
    parameter NUM_BANKS     = 4,
    parameter ADDRW         = 32,
    parameter DATAW         = 32,
    parameter TAGW          = 32,
    parameter QUEUE_SIZE    = 16,
    parameter PARTIAL_RESPONSE = 0,
    parameter DUPLICATE_ADDR = 0,
    parameter OUT_REG       = 0,
    localparam BYTEENW      = DATAW / 8,
    localparam NUM_BATCHES  = (NUM_REQS + NUM_BANKS - 1) / NUM_BANKS,
    localparam QUEUE_ADDRW  = `CLOG2(QUEUE_SIZE),
    localparam BATCH_SEL_BITS = `CLOG2(NUM_BATCHES),
    localparam MEM_TAGW     = QUEUE_ADDRW + BATCH_SEL_BITS
) (
    input wire clk,
    input wire reset,

    // Input request
    input wire                              req_valid,
    input wire                              req_rw,
    input wire [NUM_REQS-1:0]               req_mask,
    input wire [NUM_REQS-1:0][BYTEENW-1:0]  req_byteen,
    input wire [NUM_REQS-1:0][ADDRW-1:0]    req_addr,
    input wire [NUM_REQS-1:0][DATAW-1:0]    req_data,
    input wire [TAGW-1:0]                   req_tag,
    output wire                             req_ready,

    // Output response
    output wire                             rsp_valid,
    output wire [NUM_REQS-1:0]              rsp_mask,
    output wire [NUM_REQS-1:0][DATAW-1:0]   rsp_data,
    output wire [TAGW-1:0]                  rsp_tag,
    input wire                              rsp_ready,

    // Memory request
    output wire [NUM_BANKS-1:0]             mem_req_valid,
    output wire [NUM_BANKS-1:0]             mem_req_rw,
    output wire [NUM_BANKS-1:0][BYTEENW-1:0] mem_req_byteen,
    output wire [NUM_BANKS-1:0][ADDRW-1:0]  mem_req_addr,
    output wire [NUM_BANKS-1:0][DATAW-1:0]  mem_req_data,
    output wire [NUM_BANKS-1:0][MEM_TAGW-1:0]mem_req_tag,
    input wire 	[NUM_BANKS-1:0]             mem_req_ready,

    // Memory response
    input wire [NUM_BANKS-1:0]              mem_rsp_valid,
    input wire [NUM_BANKS-1:0][DATAW-1:0]   mem_rsp_data,
    input wire [NUM_BANKS-1:0][MEM_TAGW-1:0] mem_rsp_tag,
    output wire [NUM_BANKS-1:0]             mem_rsp_ready
  );

    localparam REM_BATCH_SIZE = NUM_REQS % NUM_BANKS;
    localparam BATCH_DATAW = NUM_BANKS * (1 + BYTEENW + ADDRW + DATAW);

    `STATIC_ASSERT (DATAW == 8 * (DATAW / 8), ("invalid parameter"))
    `STATIC_ASSERT ((0 == PARTIAL_RESPONSE) || (1 == PARTIAL_RESPONSE), ("invalid parameter"))
    `STATIC_ASSERT ((0 == DUPLICATE_ADDR) || (1 == DUPLICATE_ADDR), ("invalid parameter"))
    `RUNTIME_ASSERT ((~req_valid || req_mask != 0), ("invalid input"));  

    wire                            sreq_push;
    wire                            sreq_pop;
    wire                            sreq_full;
    wire                            sreq_empty;
    wire                            sreq_rw;
    wire [NUM_REQS-1:0]             sreq_mask;
    wire [NUM_REQS-1:0][BYTEENW-1:0] sreq_byteen;
    wire [NUM_REQS-1:0][ADDRW-1:0]  sreq_addr;
    wire [NUM_REQS-1:0][DATAW-1:0]  sreq_data;
    wire [QUEUE_ADDRW-1:0]          sreq_tag;

    wire                            stag_push;
    wire                            stag_pop;
    wire [QUEUE_ADDRW-1:0]          stag_waddr;
    wire [QUEUE_ADDRW-1:0]          stag_raddr;
    wire                            stag_full;
    wire                            stag_empty;
    wire [TAGW-1:0]                 stag_dout;    

    wire                            crsp_valid;
    wire [NUM_REQS-1:0]             crsp_mask;
    wire [NUM_REQS-1:0][DATAW-1:0]  crsp_data;
    wire [TAGW-1:0]                 crsp_tag;
    
    // Duplicate address optimization /////////////////////////////////////////

    wire [NUM_REQS-1:0] req_dup_mask; 
    wire rsp_dup;   

    if (DUPLICATE_ADDR == 1) begin

        reg [QUEUE_SIZE-1:0] rsp_dups;
        wire [NUM_REQS-2:0] addr_matches;

        for (genvar i = 0; i < NUM_REQS-1; ++i) begin
            assign addr_matches[i] = (req_addr[i+1] == req_addr[0]) || ~req_mask[i+1];
        end

        wire req_dup = req_mask[0] && (& addr_matches);        

        always @(posedge clk) begin
            if (reset) begin
                rsp_dups <= '0;
            end else begin
                if (sreq_push) begin
                    rsp_dups[stag_waddr] <= req_dup;
                end
            end
        end
        
        assign req_dup_mask = req_mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};
        assign rsp_dup      = rsp_dups[stag_raddr];

    end else begin

        assign req_dup_mask = req_mask;
        assign rsp_dup      = 1'b0;

    end    

    // Store request //////////////////////////////////////////////////////

    wire [NUM_BANKS-1:0] mem_req_fire;
    wire                 req_complete;

    assign sreq_push = req_valid && !sreq_full && !stag_full;
    assign sreq_pop  = (| mem_req_fire) && req_complete;
    assign req_ready = !sreq_full && !stag_full;

    VX_fifo_queue #(
        .DATAW	 (1 + NUM_REQS * (1 + BYTEENW + ADDRW + DATAW) + QUEUE_ADDRW),
        .SIZE	 (QUEUE_SIZE),
        .OUT_REG (1)
    ) req_store (
        .clk        (clk),
        .reset      (reset),
        .push       (sreq_push),
        .pop        (sreq_pop),
        .data_in    ({req_rw,  req_dup_mask, req_byteen,  req_addr,  req_data,  stag_waddr}),
        .data_out   ({sreq_rw, sreq_mask,    sreq_byteen, sreq_addr, sreq_data, sreq_tag}),
        .full       (sreq_full),
        .empty      (sreq_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    // Tag store //////////////////////////////////////////////////////////////

    wire rsp_stall;
    wire rsp_complete;

    assign stag_push  = sreq_push && !req_rw;
    assign stag_pop   = crsp_valid && rsp_complete && ~rsp_stall;
    assign stag_raddr = mem_rsp_tag_s[BATCH_SEL_BITS +: QUEUE_ADDRW];

    VX_index_buffer #(
        .DATAW	(TAGW),
        .SIZE	(QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .write_addr   (stag_waddr),
        .acquire_slot (stag_push),
        .read_addr    (stag_raddr),
        .write_data   (req_tag),
        .read_data    (stag_dout),
        .release_addr (stag_raddr),
        .release_slot (stag_pop),
        .full         (stag_full),
        .empty        (stag_empty)
    );

    `UNUSED_VAR (stag_empty)    

    // Handle memory requests /////////////////////////////////////////////////

    wire [NUM_BATCHES-1:0][BATCH_DATAW-1:0] mem_req_data_b;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0] mem_req_mask_b;
    reg  [NUM_BATCHES-1:0][NUM_BANKS-1:0] req_sent_mask;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0] req_sent_mask_n;    
    reg  [`UP(BATCH_SEL_BITS)-1:0] req_batch_idx;

    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
        assign mem_req_mask_b[i] = NUM_BANKS'(sreq_mask[i * NUM_BANKS +: SIZE]);
        assign mem_req_data_b[i] = {
            {NUM_BANKS{sreq_rw}},
            (NUM_BANKS * BYTEENW)'(sreq_byteen[i * NUM_BANKS +: SIZE]),
            (NUM_BANKS * ADDRW)'(sreq_addr[i * NUM_BANKS +: SIZE]),
            (NUM_BANKS * DATAW)'(sreq_data[i * NUM_BANKS +: SIZE])
        };
    end

    for (genvar i = 0; i < NUM_BATCHES; ++i) begin        
        assign req_sent_mask_n[i] = (i == req_batch_idx) ? (req_sent_mask[i] | mem_req_fire) : req_sent_mask[i];
    end

    assign req_complete = (req_sent_mask_n == mem_req_mask_b);

    wire req_complete_b = (req_sent_mask_n[req_batch_idx] == mem_req_mask_b[req_batch_idx]);

    always @(posedge clk) begin
        if (reset) begin
            req_sent_mask <= '0;
            req_batch_idx <= 0;
        end else begin
            if (req_complete) begin
                req_sent_mask <= '0;
            end else begin
                req_sent_mask <= req_sent_mask_n;
            end
            if (req_complete_b) begin
                if (req_batch_idx == `UP(BATCH_SEL_BITS)'(NUM_BATCHES-1)) begin
                    req_batch_idx <= 0;
                end else begin
                    req_batch_idx <= req_batch_idx + 1;
                end
            end
        end
    end

    assign mem_req_valid = mem_req_mask_b[req_batch_idx] & ~req_sent_mask[req_batch_idx] & {NUM_BANKS{~sreq_empty}};

    assign {mem_req_rw, mem_req_byteen, mem_req_addr, mem_req_data} = mem_req_data_b[req_batch_idx];

    if (NUM_BATCHES > 1) begin
        assign mem_req_tag = {NUM_BANKS{{sreq_tag, req_batch_idx}}};
    end else begin
        assign mem_req_tag = {NUM_BANKS{sreq_tag}};
    end

    assign mem_req_fire = mem_req_valid & mem_req_ready; 

    // Handle memory responses ////////////////////////////////////////////////

    wire                            mem_rsp_valid_s;
    wire [NUM_BANKS-1:0]            mem_rsp_mask_s;
    wire [NUM_BANKS-1:0][DATAW-1:0] mem_rsp_data_s;
    wire [MEM_TAGW-1:0]             mem_rsp_tag_s;
    wire                            mem_rsp_ready_s;
    wire                            mem_rsp_fire;

    reg  [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0] rsp_rem_mask;
    wire [NUM_BATCHES-1:0][NUM_BANKS-1:0]                 rsp_rem_mask_n;

    reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0] rsp_orig_mask;
    wire [`UP(BATCH_SEL_BITS)-1:0]      rsp_batch_idx;

    // Select memory response
    VX_mem_rsp_sel #(
        .NUM_REQS     (NUM_BANKS),
        .DATA_WIDTH   (DATAW),
        .TAG_WIDTH    (MEM_TAGW),
        .TAG_SEL_BITS (MEM_TAGW),
        .OUT_REG      (1)
    ) mem_rsp_sel (
        .clk            (clk),
        .reset          (reset),
        .rsp_valid_in   (mem_rsp_valid),
        .rsp_data_in    (mem_rsp_data),
        .rsp_tag_in     (mem_rsp_tag),
        .rsp_ready_in   (mem_rsp_ready),
        .rsp_valid_out  (mem_rsp_valid_s),
        .rsp_tmask_out  (mem_rsp_mask_s),
        .rsp_data_out   (mem_rsp_data_s),
        .rsp_tag_out    (mem_rsp_tag_s),
        .rsp_ready_out  (mem_rsp_ready_s)
    );

    if (NUM_BATCHES > 1) begin
        assign rsp_batch_idx = mem_rsp_tag_s[0 +: BATCH_SEL_BITS];
    end else begin
        assign rsp_batch_idx = 0;
    end
    
    for (genvar i = 0; i < NUM_BATCHES; ++i) begin
        assign rsp_rem_mask_n[i] = (i == rsp_batch_idx) ? (rsp_rem_mask[stag_raddr][i] & ~mem_rsp_mask_s) : rsp_rem_mask[stag_raddr][i];
    end

    assign rsp_complete = (0 == rsp_rem_mask_n);

    always @(posedge clk) begin
        if (reset) begin
            rsp_orig_mask <= '0;
            rsp_rem_mask  <= '0;
        end else begin
            if (sreq_push) begin
                rsp_orig_mask[stag_waddr] <= req_mask;
                rsp_rem_mask[stag_waddr]  <= (NUM_BATCHES * NUM_BANKS)'(req_dup_mask);
            end
            if (mem_rsp_fire) begin
                rsp_rem_mask[stag_raddr] <= rsp_rem_mask_n;
            end
            if (stag_pop) begin
                rsp_orig_mask[stag_raddr] <= 0;
            end
        end
    end

    if (PARTIAL_RESPONSE == 1) begin

        assign mem_rsp_ready_s = ~rsp_stall;
        assign mem_rsp_fire    = mem_rsp_valid_s & mem_rsp_ready_s;

        assign crsp_valid = mem_rsp_valid_s;
        assign crsp_tag   = stag_dout;

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign crsp_mask[i] = rsp_dup ? rsp_orig_mask[stag_raddr][i]: ((i == rsp_batch_idx) ? mem_rsp_mask_s : '0);
            assign crsp_data[i * NUM_BANKS +: SIZE] = rsp_dup ? {SIZE{mem_rsp_data_s[0]}} : mem_rsp_data_s[0 +: SIZE];
        end
    
    end else begin

        reg [QUEUE_SIZE-1:0][NUM_BATCHES-1:0][NUM_BANKS-1:0][DATAW-1:0] rsp_store;   
        wire [NUM_BATCHES-1:0][NUM_BANKS-1:0][DATAW-1:0]                rsp_store_n;
        wire [NUM_BANKS-1:0][DATAW-1:0] mem_rsp_data_m;

        assign mem_rsp_ready_s = ~(rsp_stall && rsp_complete);
        assign mem_rsp_fire = mem_rsp_valid_s & mem_rsp_ready_s;

        assign crsp_valid = mem_rsp_valid_s & rsp_complete;
        assign crsp_mask  = rsp_orig_mask[stag_raddr];
        assign crsp_tag   = stag_dout;

        for (genvar i = 0; i < NUM_BANKS; ++i) begin
            assign mem_rsp_data_m[i] = mem_rsp_mask_s[i] ? mem_rsp_data_s[i] : '0;
        end

        for (genvar i = 0; i < NUM_BATCHES; ++i) begin
            localparam SIZE = ((i + 1) * NUM_BANKS > NUM_REQS) ? REM_BATCH_SIZE : NUM_BANKS;
            assign rsp_store_n[i] = rsp_store[stag_raddr][i] | ((i == rsp_batch_idx) ? mem_rsp_data_m : '0);
            assign crsp_data[i * NUM_BANKS +: SIZE] = rsp_dup ? {SIZE{mem_rsp_data_s[0]}} : rsp_store_n[i][0 +: SIZE];
        end
        
        always @(posedge clk) begin
            if (reset) begin
                rsp_store  <= '0;
            end else begin
                if (sreq_push) begin                    
                    rsp_store[stag_waddr] <= '0;
                end
                if (mem_rsp_fire) begin
                    rsp_store[stag_raddr] <= rsp_store_n;
                end
            end
        end 
    end

    // Send response to caller
    VX_pipe_register #(
        .DATAW	(1 + NUM_REQS + (NUM_REQS * DATAW) + TAGW),
        .RESETW (1),
        .DEPTH  (OUT_REG)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable	  (~rsp_stall),
        .data_in  ({crsp_valid, crsp_mask, crsp_data, crsp_tag}),
        .data_out ({rsp_valid,  rsp_mask,  rsp_data,  rsp_tag})
    );

    assign rsp_stall = rsp_valid & ~rsp_ready;   

    ///////////////////////////////////////////////////////////////////////////

    /*
    always @(posedge clk) begin
        if (req_valid && req_ready) begin            
            dpi_trace(1, "%d: memstream req: rw=%b, mask=%b, byteen=", $time, req_rw, req_mask);
            `TRACE_ARRAY1D(1, req_byteen, NUM_REQS);
            dpi_trace(1, ", addr=");
            `TRACE_ARRAY1D(1, req_addr, NUM_REQS);
            dpi_trace(1, ", data=");
            `TRACE_ARRAY1D(1, req_data, NUM_REQS);
            dpi_trace(1, ", tag=0x%0h\n", req_tag);
        end
        if (rsp_valid && rsp_ready) begin
            dpi_trace(1, "%d: memstream rsp: mask=%b, data=", $time, rsp_mask);
             `TRACE_ARRAY1D(1, rsp_data, NUM_REQS);
            dpi_trace(1, ", tag=0x%0h\n", rsp_tag);
        end
        if (| mem_req_fire) begin
            if (| mem_req_rw) begin
                dpi_trace(1, "%d: memstream mem-wr-req: valid=%b, byteen=", $time, mem_req_fire);
                `TRACE_ARRAY1D(1, mem_req_byteen, NUM_BANKS);
                dpi_trace(1, ", addr=");
                `TRACE_ARRAY1D(1, mem_req_addr, NUM_BANKS);
                dpi_trace(1, ", data=");
                `TRACE_ARRAY1D(1, mem_req_data, NUM_BANKS);
                dpi_trace(1, ", tag=");
                `TRACE_ARRAY1D(1, mem_req_tag, NUM_BANKS);
                dpi_trace(1, ", batch=%0d\n", req_batch_idx);
            end else begin
                dpi_trace(1, "%d: memstream mem-rd-req: valid=%b, addr=", $time, mem_req_fire);
                `TRACE_ARRAY1D(1, mem_req_addr, NUM_BANKS);
                dpi_trace(1, ", tag=");
                `TRACE_ARRAY1D(1, mem_req_tag, NUM_BANKS);
                dpi_trace(1, ", batch=%0d\n", req_batch_idx);
            end
        end 
        if (mem_rsp_fire) begin
            dpi_trace(1, "%d: memstream mem-rd-rsp: mask=%b, data=", $time, mem_rsp_mask_s);                
            `TRACE_ARRAY1D(1, mem_rsp_data_s, NUM_BANKS);
            dpi_trace(1, ", tag=0x%0h, batch=%0d\n", mem_rsp_tag_s, rsp_batch_idx);
        end
    end
    */

endmodule
`TRACING_ON