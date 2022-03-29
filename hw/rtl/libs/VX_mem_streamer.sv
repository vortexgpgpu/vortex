`include "VX_platform.vh"

// `TRACING_OFF
module VX_mem_streamer #(
    parameter NUM_REQS = 4,
    parameter ADDRW = 32,
    parameter DATAW = 32,
    parameter TAGW = 32,
    parameter WORD_SIZE = 4,
    parameter QUEUE_SIZE = 16,
    parameter PARTIAL_RESPONSE = 0,
    parameter DUPLICATE_ADDR = 0
) (
    input  wire clk,
    input  wire reset,

    // Input request
    input wire                           req_valid,
    input wire                           req_rw,
    input wire [NUM_REQS-1:0]            req_mask,
    input wire [WORD_SIZE-1:0]           req_byteen,
    input wire [NUM_REQS-1:0][ADDRW-1:0] req_addr,
    input wire [NUM_REQS-1:0][DATAW-1:0] req_data,
    input wire [TAGW-1:0]                req_tag,
    output wire                          req_ready,

    // Output request
    output wire [NUM_REQS-1:0]                  mem_req_valid,
    output wire [NUM_REQS-1:0]                  mem_req_rw,
    output wire [NUM_REQS-1:0][WORD_SIZE-1:0]   mem_req_byteen,
    output wire [NUM_REQS-1:0][ADDRW-1:0]       mem_req_addr,
    output wire [NUM_REQS-1:0][DATAW-1:0]       mem_req_data,
    output wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] mem_req_tag,
    input wire 	[NUM_REQS-1:0]                  mem_req_ready,

    // Input response
    input wire                           mem_rsp_valid,
    input wire [NUM_REQS-1:0]            mem_rsp_mask,
    input wire [NUM_REQS-1:0][DATAW-1:0] mem_rsp_data,
    input wire [QUEUE_ADDRW-1:0]         mem_rsp_tag,
    output wire                          mem_rsp_ready,

    // Output response
    output wire                           rsp_valid,
    output wire [NUM_REQS-1:0]            rsp_mask,
    output wire [NUM_REQS-1:0][DATAW-1:0] rsp_data,
    output wire [TAGW-1:0]                rsp_tag,
    input wire                            rsp_ready
  );
    localparam QUEUE_ADDRW = `CLOG2(QUEUE_SIZE);
   
    `STATIC_ASSERT ((0 == PARTIAL_RESPONSE) || (1 == PARTIAL_RESPONSE), ("invalid parameter"))
    `STATIC_ASSERT ((0 == DUPLICATE_ADDR) || (1 == DUPLICATE_ADDR), ("invalid parameter"))

    // Detect duplicate addresses
    wire [NUM_REQS-1:0] req_dup_mask;

    // Pending queue
    wire                           sreq_rw;
    wire [NUM_REQS-1:0]            sreq_mask;
    wire [WORD_SIZE-1:0]           sreq_byteen;
    wire [NUM_REQS-1:0][ADDRW-1:0] sreq_addr;
    wire [NUM_REQS-1:0][DATAW-1:0] sreq_data;
    wire [QUEUE_ADDRW-1:0]         sreq_tag;

    wire sreq_push;
    wire sreq_pop;
    wire sreq_full;
    wire sreq_empty;

    wire                   stag_push;
    wire                   stag_pop;
    wire [QUEUE_ADDRW-1:0] stag_waddr;
    wire [QUEUE_ADDRW-1:0] stag_raddr;
    wire [QUEUE_ADDRW-1:0] stag_pop_addr;
    wire                   stag_full;
    wire                   stag_empty;
    wire [TAGW-1:0]        stag_dout;

    // Memory request
    wire [NUM_REQS-1:0]                  mreq_valid;
    wire [NUM_REQS-1:0]                  mreq_rw;
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]   mreq_byteen;
    wire [NUM_REQS-1:0][ADDRW-1:0]       mreq_addr;
    wire [NUM_REQS-1:0][DATAW-1:0]       mreq_data;
    wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] mreq_tag;

    wire [NUM_REQS-1:0] mem_req_fire;
    reg  [NUM_REQS-1:0] req_sent_mask;
    wire [NUM_REQS-1:0] req_sent_mask_n;
    wire                req_sent_all;

    // Memory response
    wire                                mem_rsp_fire;
    wire                                rsp_fire;
    wire                                rsp_done;
    reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0] rsp_rem_mask;
    wire [NUM_REQS-1:0]                 rsp_rem_mask_n;
    wire                                crsp_valid;
    wire [NUM_REQS-1:0]                 crsp_mask;
    wire [NUM_REQS-1:0][DATAW-1:0]      crsp_data;
    wire [TAGW-1:0]                     crsp_tag;

    wire rsp_stall;

    //////////////////////////////////////////////////////////////////

    // Detect duplicate addresses
   
    if (DUPLICATE_ADDR == 1) begin
        assign req_dup_mask = req_mask;
    end else begin
        wire [NUM_REQS-2:0] addr_matches;
        wire req_dup;

        for(genvar i = 0; i < NUM_REQS-1; i++) begin
            assign addr_matches[i] = (req_addr[i+1] == req_addr[0]) || ~req_mask[i+1];
        end

        assign req_dup = req_mask[0] && (& addr_matches);
        assign req_dup_mask = req_mask & {{(NUM_REQS-1){~req_dup}}, 1'b1};
    end

    //////////////////////////////////////////////////////////////////

    // Save incoming requests into a pending queue

    assign sreq_push = req_valid && !sreq_full && !stag_full;
    assign sreq_pop  = (| mem_req_fire) && req_sent_all && !sreq_empty;
    assign req_ready = !sreq_full && !stag_full;

    VX_fifo_queue #(
        .DATAW	(1 + NUM_REQS + WORD_SIZE + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + QUEUE_ADDRW),
        .SIZE	(QUEUE_SIZE)
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

    // Reads only
    assign stag_push  = sreq_push && !req_rw;
    assign stag_pop   = rsp_fire && rsp_done && !stag_empty;
    assign stag_raddr = mem_rsp_tag;

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
        .release_addr (stag_pop_addr),
        .release_slot (stag_pop),
        .full         (stag_full),
        .empty        (stag_empty)
    );

    //////////////////////////////////////////////////////////////////

    // Memory response

    // Evaluate remaning responses
    assign rsp_rem_mask_n = rsp_rem_mask[stag_raddr] & ~mem_rsp_mask;

    always @(posedge clk) begin
        if (sreq_push)
            rsp_rem_mask[stag_waddr] <= req_dup_mask;
        if (mem_rsp_fire)
            rsp_rem_mask[stag_raddr] <= rsp_rem_mask_n;
    end

    if (PARTIAL_RESPONSE == 1) begin
        assign mem_rsp_ready = ~rsp_stall;
        assign mem_rsp_fire  = mem_rsp_valid & mem_rsp_ready;

        assign crsp_valid = mem_rsp_valid;
        assign crsp_mask  = mem_rsp_mask;
        assign crsp_data  = mem_rsp_data;
        assign crsp_tag   = stag_dout;
        
    end else begin

        reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0][DATAW-1:0] rsp_store;
        reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0] mask_store;
        reg  [QUEUE_SIZE-1:0] rsp_full;

        assign mem_rsp_ready = ~rsp_stall && ~(& rsp_full);
        assign mem_rsp_fire  = mem_rsp_valid & mem_rsp_ready;

        assign crsp_valid = mem_rsp_valid & (0 == rsp_rem_mask_n);
        assign crsp_mask  = mask_store[stag_raddr];
        assign crsp_data  = rsp_store[stag_raddr] | mem_rsp_data; 
        assign crsp_tag   = stag_dout;

        // Store response until ready to send
        always @(posedge clk) begin
            if (reset) begin
                rsp_store  <= 0;
                mask_store <= 0;
                rsp_full   <= 0;
            end else begin
                if (sreq_push) begin
                    rsp_store[stag_waddr]  <= 0;
                    mask_store[stag_waddr] <= req_dup_mask;
                end
                if (stag_pop) begin
                    rsp_full[stag_pop_addr]   <= 1'b0;
                    mask_store[stag_pop_addr] <= 0;
                end
                if (mem_rsp_fire) begin
                    rsp_store[stag_raddr] <= crsp_data;
                    rsp_full[stag_raddr]  <= 1'b1;
                end
            end
        end 

    end

    //////////////////////////////////////////////////////////////////

    // Memory request
    assign mreq_valid  = sreq_mask & ~req_sent_mask & {NUM_REQS{!sreq_empty}};
    assign mreq_rw     = {NUM_REQS{sreq_rw}};
    assign mreq_byteen = {NUM_REQS{sreq_byteen}};
    assign mreq_addr   = sreq_addr;
    assign mreq_data   = sreq_data;
    assign mreq_tag    = {NUM_REQS{sreq_tag}};

    assign mem_req_fire    = mem_req_valid & mem_req_ready;
    assign req_sent_mask_n = req_sent_mask | mem_req_fire;
    assign req_sent_all    = (req_sent_mask_n == mem_req_valid);

    always @(posedge clk) begin
        if (reset)
            req_sent_mask <= 0;
        else begin
            if (req_sent_all)
                req_sent_mask <= 0;
            else
                req_sent_mask <= req_sent_mask_n;
        end
    end

    //////////////////////////////////////////////////////////////////

    // Send request to memory
    VX_pipe_register #(
        .DATAW	(NUM_REQS + NUM_REQS + (NUM_REQS * WORD_SIZE) + (NUM_REQS * ADDRW) + (NUM_REQS * DATAW) + (NUM_REQS * QUEUE_ADDRW)),
        .RESETW (1)
    ) req_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable	  (1'b1),
        .data_in  ({mreq_valid,    mreq_rw,    mreq_byteen,    mreq_addr,    mreq_data,    mreq_tag}),
        .data_out ({mem_req_valid, mem_req_rw, mem_req_byteen, mem_req_addr, mem_req_data, mem_req_tag})
    );

    // Send response to caller
    VX_pipe_register #(
        .DATAW	(1 + NUM_REQS + (NUM_REQS * DATAW) + TAGW + QUEUE_ADDRW + 1),
        .RESETW (1)
    ) rsp_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable	  (1'b1),
        .data_in  ({crsp_valid, crsp_mask, crsp_data, crsp_tag, stag_raddr,    (0 == rsp_rem_mask_n)}),
        .data_out ({rsp_valid,  rsp_mask,  rsp_data,  rsp_tag,  stag_pop_addr, rsp_done})
    );

    assign rsp_fire = rsp_valid & rsp_ready;
    assign rsp_stall = rsp_valid & ~rsp_ready;

    //////////////////////////////////////////////////////////////////

endmodule
// `TRACING_ON
