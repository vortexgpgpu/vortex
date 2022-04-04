`include "VX_platform.vh"

//`TRACING_OFF
module VX_mem_streamer #(
    parameter NUM_REQS = 4,
    parameter ADDRW = 32,
    parameter DATAW = 32,
    parameter TAGW = 32,
    parameter QUEUE_SIZE = 16,
    parameter PARTIAL_RESPONSE = 0,
    parameter DUPLICATE_ADDR = 0,
    parameter OUT_REG = 0
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
    output wire [NUM_REQS-1:0]              mem_req_valid,
    output wire [NUM_REQS-1:0]              mem_req_rw,
    output wire [NUM_REQS-1:0][BYTEENW-1:0] mem_req_byteen,
    output wire [NUM_REQS-1:0][ADDRW-1:0]   mem_req_addr,
    output wire [NUM_REQS-1:0][DATAW-1:0]   mem_req_data,
    output wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] mem_req_tag,
    input wire 	[NUM_REQS-1:0]              mem_req_ready,

    // Memory response
    input wire [NUM_REQS-1:0]               mem_rsp_valid,
    input wire [NUM_REQS-1:0][DATAW-1:0]    mem_rsp_data,
    input wire [NUM_REQS-1:0][QUEUE_ADDRW-1:0] mem_rsp_tag,
    output wire [NUM_REQS-1:0]              mem_rsp_ready
  );

    localparam BYTEENW = DATAW / 8;
    localparam QUEUE_ADDRW = `CLOG2(QUEUE_SIZE);
   
    `STATIC_ASSERT (DATAW == 8 * (DATAW / 8), ("invalid parameter"))
    `STATIC_ASSERT ((0 == PARTIAL_RESPONSE) || (1 == PARTIAL_RESPONSE), ("invalid parameter"))
    `STATIC_ASSERT ((0 == DUPLICATE_ADDR) || (1 == DUPLICATE_ADDR), ("invalid parameter"))

    `RUNTIME_ASSERT((~req_valid || req_mask != 0), ("invalid input"));

    // Detect duplicate addresses
    wire [NUM_REQS-1:0] req_dup_mask;

    // Pending queue
    wire                           sreq_rw;
    wire [NUM_REQS-1:0]            sreq_mask;
    wire [NUM_REQS-1:0][BYTEENW-1:0] sreq_byteen;
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
    wire                   stag_full;
    wire                   stag_empty;
    wire [TAGW-1:0]        stag_dout;

    // Memory request
    wire [NUM_REQS-1:0] mem_req_fire;
    reg  [NUM_REQS-1:0] req_sent_mask;
    wire [NUM_REQS-1:0] req_sent_mask_n;
    wire                req_complete;

    // Memory response
    wire                            mem_rsp_valid_s;
    wire [NUM_REQS-1:0]             mem_rsp_mask_s;
    wire [NUM_REQS-1:0][DATAW-1:0]  mem_rsp_data_s;
    wire [QUEUE_ADDRW-1:0]          mem_rsp_tag_s;
    wire                            mem_rsp_ready_s;
    wire                            mem_rsp_fire;

    // Caller response
    wire                                rsp_stall;
    wire                                rsp_complete;
    reg  [QUEUE_SIZE-1:0][NUM_REQS-1:0] rsp_rem_mask;
    wire [NUM_REQS-1:0]                 rsp_rem_mask_n;
    wire                                crsp_valid;
    wire [NUM_REQS-1:0]                 crsp_mask;
    wire [NUM_REQS-1:0][DATAW-1:0]      crsp_data;
    wire [TAGW-1:0]                     crsp_tag;

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

    // Reads only
    assign stag_push  = sreq_push && !req_rw;
    assign stag_pop   = crsp_valid && rsp_complete && ~rsp_stall;
    assign stag_raddr = mem_rsp_tag_s;

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

    //////////////////////////////////////////////////////////////////

    // Memory response

    VX_mem_rsp_sel #(
        .NUM_REQS     (NUM_REQS),
        .DATA_WIDTH   (DATAW),
        .TAG_WIDTH    (QUEUE_ADDRW),
        .TAG_SEL_BITS (QUEUE_ADDRW),
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

    // Evaluate remaning responses
    assign rsp_rem_mask_n = rsp_rem_mask[stag_raddr] & ~mem_rsp_mask_s;
    assign rsp_complete = (0 == rsp_rem_mask_n);

    always @(posedge clk) begin
        if (sreq_push)
            rsp_rem_mask[stag_waddr] <= req_dup_mask;
        if (mem_rsp_fire)
            rsp_rem_mask[stag_raddr] <= rsp_rem_mask_n;
    end

    if (PARTIAL_RESPONSE == 1) begin
        assign mem_rsp_ready_s = ~rsp_stall;
        assign mem_rsp_fire  = mem_rsp_valid_s & mem_rsp_ready_s;

        assign crsp_valid = mem_rsp_valid_s;
        assign crsp_mask  = mem_rsp_mask_s;
        assign crsp_data  = mem_rsp_data_s;
        assign crsp_tag   = stag_dout;
        
    end else begin

        reg [QUEUE_SIZE-1:0][NUM_REQS-1:0][DATAW-1:0] rsp_store;
        reg [QUEUE_SIZE-1:0][NUM_REQS-1:0] mask_store;
        wire [NUM_REQS-1:0][DATAW-1:0] mem_rsp_data_m;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            assign mem_rsp_data_m[i] = mem_rsp_mask_s[i] ? mem_rsp_data_s[i] : DATAW'(0);
        end

        assign mem_rsp_ready_s = ~(rsp_stall && rsp_complete);
        assign mem_rsp_fire = mem_rsp_valid_s & mem_rsp_ready_s;

        assign crsp_valid = mem_rsp_valid_s & rsp_complete;
        assign crsp_mask  = mask_store[stag_raddr];
        assign crsp_data  = rsp_store[stag_raddr] | mem_rsp_data_m; 
        assign crsp_tag   = stag_dout;

        // Store response until ready to send
        always @(posedge clk) begin
            if (reset) begin
                rsp_store  <= '0;
                mask_store <= '0;
            end else begin
                if (sreq_push) begin
                    mask_store[stag_waddr] <= req_dup_mask;
                    rsp_store[stag_waddr]  <= '0;
                end
                if (mem_rsp_fire) begin
                    rsp_store[stag_raddr] <= crsp_data;
                end
                if (stag_pop) begin
                    mask_store[stag_raddr] <= 0;
                end
            end
        end 

    end

    //////////////////////////////////////////////////////////////////

    // Memory request
    assign mem_req_valid  = sreq_mask & ~req_sent_mask & {NUM_REQS{~sreq_empty}};
    assign mem_req_rw     = {NUM_REQS{sreq_rw}};
    assign mem_req_byteen = sreq_byteen;
    assign mem_req_addr   = sreq_addr;
    assign mem_req_data   = sreq_data;
    assign mem_req_tag    = {NUM_REQS{sreq_tag}};

    assign req_sent_mask_n = req_sent_mask | mem_req_fire;
    assign req_complete    = (req_sent_mask_n == sreq_mask);

    always @(posedge clk) begin
        if (reset)
            req_sent_mask <= 0;
        else begin
            if (req_complete)
                req_sent_mask <= 0;
            else
                req_sent_mask <= req_sent_mask_n;
        end
    end

    assign mem_req_fire = mem_req_valid & mem_req_ready;

    //////////////////////////////////////////////////////////////////

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

endmodule
// `TRACING_ON
